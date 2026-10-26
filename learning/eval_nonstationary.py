"""Evaluate saved policies under stationary or time-varying dynamics.

Run ``python -m learning.eval_nonstationary --help`` for options.
"""

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import functools
import itertools
import json
import pickle

from brax.training import types
from brax.training.acme import running_statistics
import flax
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from custom_envs import registry, dm_control_suite, locomotion, manipulation
from learning.agents.sampler_ppo import networks as samplerppo_networks
from learning.configs.dm_control_training_config import brax_ppo_config
from learning.configs.locomotion_training_config import locomotion_ppo_config
from learning.configs.manipulation_training_config import manipulation_ppo_config
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.evaluator import AdvEvaluator


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, jax.Array)):
            return np.asarray(obj).tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def load_ppo_config(task, work_dir):
    saved_path = Path(work_dir) / "cfg/ppo_config.yaml"
    if saved_path.exists():
        return OmegaConf.load(saved_path)
    if task in dm_control_suite.ALL_ENVS:
        config = brax_ppo_config(task)
    elif task in manipulation.ALL_ENVS:
        config = manipulation_ppo_config(task)
    elif task in locomotion.ALL_ENVS:
        config = locomotion_ppo_config(task)
    else:
        raise ValueError(f"No PPO configuration for task {task!r}.")
    return OmegaConf.create(config.to_dict())


def get_work_dir(cfg):
    if cfg.run_dir:
        return Path(cfg.run_dir).expanduser().resolve()
    base_path = Path(cfg.log_root) / cfg.task / str(cfg.seed) / cfg.policy
    if cfg.policy in ("gmmppo", "gbsppo", "flowppo"):
        # Both integer and float Hydra overrides are used by older run folders.
        suffix = f"_gamma={cfg.gamma}" if cfg.policy == "flowppo" else ""
        candidates = [
            base_path / f"beta={value}{suffix}"
            for value in (f"{cfg.beta:g}", str(float(cfg.beta)))
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    if cfg.policy == "epoptppo":
        return base_path / f"epsilon={cfg.epsilon}"
    if cfg.policy == "adrppo":
        return base_path / f"threshold={cfg.success_threshold}"
    if cfg.policy == "doraemonppo":
        return (
            base_path
            / f"threshold={cfg.success_threshold}_condition={cfg.success_rate_condition}"
        )
    return base_path


def distribution_label(cfg):
    label = cfg.dist_type
    if label == "random_walk":
        label += f"_rw={cfg.random_walk_scale}"
    elif label in ("jump", "jump_diffusion"):
        label += f"_jp={cfg.jump_prob}_drift={cfg.drift}"
    elif label == "langevin":
        label += f"_d={cfg.drift}_n={cfg.random_walk_scale}"
    elif label == "beta":
        label += f"_scale={cfg.beta_scale}"
    return f"{label}_ns={cfg.non_stationary}"


def load_gmm_log_prob(cfg):
    checkpoint_path = (
        Path(cfg.sampler_checkpoint)
        if cfg.sampler_checkpoint
        else Path(cfg.work_dir) / "results/GMM/checkpoints/sampler_state_latest.msgpack"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Langevin requires a saved GMM sampler: {checkpoint_path}. "
            "Pass --sampler_checkpoint to use a reference RSDR sampler."
        )
    env = registry.load(cfg.task, config_overrides={"impl": cfg.impl})
    ppo_params = load_ppo_config(cfg.task, cfg.work_dir)
    network, _, _, (_, _, _, template) = samplerppo_networks.make_samplerppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        dynamics_param_size=len(env.dr_range[0]),
        num_envs=int(ppo_params.num_envs),
        batch_size=int(ppo_params.batch_size),
        init_key=jax.random.PRNGKey(0),
        bound_info=env.dr_range,
        sampler_choice="GMM",
    )
    state = flax.serialization.from_bytes(template, checkpoint_path.read_bytes())
    return jax.jit(
        functools.partial(
            network.gmm_network.model.log_density, state.model_state.gmm_state
        )
    )


def evaluate(cfg, key, log_prob_fn=None):
    num_eval_envs = cfg.num_eval_envs
    num_runs = cfg.num_runs

    # Load environment
    env_cfg = registry.get_default_config(cfg.task)
    saved_env_path = Path(cfg.work_dir) / "cfg/env_config.yaml"
    if saved_env_path.exists():
        env_cfg.update(OmegaConf.to_container(OmegaConf.load(saved_env_path)))
    env_cfg.impl = cfg.impl
    env = registry.load(cfg.task, config=env_cfg)

    # Domain Randomization Setup
    randomizer = (
        registry.get_domain_randomizer_ood(cfg.task)
        if cfg.ood_setting
        else registry.get_domain_randomizer_eval(cfg.task)
    )
    if randomizer is None:
        raise ValueError(f"No evaluator randomizer is available for {cfg.task!r}.")
    dr_range_source = (
        env.ood_range
        if cfg.ood_setting
        else (env.dr_range_wide if cfg.dr_wide else env.dr_range)
    )
    dr_range_low, dr_range_high = dr_range_source

    v_randomization_fn = functools.partial(randomizer, dr_range=dr_range_source)

    eval_env = wrap_for_adv_training(
        env,
        episode_length=env_cfg.episode_length,
        action_repeat=env_cfg.action_repeat,
        randomization_fn=v_randomization_fn,
        param_size=len(dr_range_low),
        dr_range_low=dr_range_low,
        dr_range_high=dr_range_high,
    )

    ppo_params = load_ppo_config(cfg.task, cfg.work_dir)
    preprocess = (
        running_statistics.normalize
        if ppo_params.normalize_observations
        else types.identity_observation_preprocessor
    )
    network, _, _, _ = samplerppo_networks.make_samplerppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        dynamics_param_size=len(dr_range_low),
        num_envs=int(ppo_params.num_envs),
        batch_size=int(ppo_params.batch_size),
        init_key=jax.random.PRNGKey(0),
        preprocess_observations_fn=preprocess,
        bound_info=env.dr_range,
        **dict(ppo_params.network_factory),
    )
    make_policy_fn = samplerppo_networks.make_inference_fn(network)

    # Load saved parameters
    save_dir = os.path.join(cfg.work_dir, "models")
    param_file = os.path.join(save_dir, f"{cfg.policy}_params_latest.pkl")

    if not os.path.exists(param_file):
        print(f"Warning: Model file not found at {param_file}. Skipping seed.")
        return None, None

    with open(param_file, "rb") as f:
        params = pickle.load(f)

    accumulated_metrics = {}
    all_rewards_1d = []

    print(f"Starting {num_runs} evaluation runs...")

    for i in range(num_runs):
        eval_key, param_key, key = jax.random.split(key, 3)

        evaluator = AdvEvaluator(
            eval_env,
            functools.partial(make_policy_fn, deterministic=True),
            num_eval_envs=num_eval_envs,
            episode_length=env_cfg.episode_length,
            action_repeat=env_cfg.action_repeat,
            key=eval_key,
            non_stationary=cfg.non_stationary,
        )

        low = jnp.array(dr_range_low)
        high = jnp.array(dr_range_high)

        # --- DISTRIBUTION LOGIC ---
        if cfg.dist_type == "uniform":
            dynamics_params_grid = jax.random.uniform(
                param_key,
                shape=(
                    (env_cfg.episode_length, num_eval_envs, len(dr_range_low))
                    if cfg.non_stationary
                    else (num_eval_envs, len(dr_range_low))
                ),
                minval=low,
                maxval=high,
            )

        elif cfg.dist_type == "beta":
            beta_samples = jax.random.beta(
                param_key,
                cfg.beta_scale,
                cfg.beta_scale,
                shape=(
                    (env_cfg.episode_length, num_eval_envs, len(dr_range_low))
                    if cfg.non_stationary
                    else (num_eval_envs, len(dr_range_low))
                ),
            )
            dynamics_params_grid = low + (high - low) * beta_samples

        elif cfg.dist_type == "random_walk":
            param_key, init_key = jax.random.split(param_key)
            init_params = jax.random.uniform(
                init_key,
                shape=(num_eval_envs, len(dr_range_low)),
                minval=low,
                maxval=high,
            )

            def rw_step(carry, _):
                current_params, key = carry
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, shape=current_params.shape) * (
                    cfg.random_walk_scale * (high - low)
                )
                next_params = jnp.clip(current_params + noise, low, high)
                return (next_params, key), current_params

            if cfg.non_stationary:
                _, dynamics_params_grid = jax.lax.scan(
                    rw_step,
                    (init_params, param_key),
                    None,
                    length=env_cfg.episode_length,
                )
            else:
                dynamics_params_grid = init_params

        elif cfg.dist_type in ["jump_diffusion", "jump"]:
            param_key, init_key = jax.random.split(param_key)
            init_params = jax.random.uniform(
                init_key,
                shape=(num_eval_envs, len(dr_range_low)),
                minval=low,
                maxval=high,
            )

            def jump_step(carry, _):
                current_params, key = carry
                key, diff_key, jump_key, val_key = jax.random.split(key, 4)

                diff_scale = cfg.drift * (high - low)
                drifted = jnp.clip(
                    current_params
                    + jax.random.normal(diff_key, shape=current_params.shape)
                    * diff_scale,
                    low,
                    high,
                )

                jump_dest = jax.random.uniform(
                    val_key, shape=current_params.shape, minval=low, maxval=high
                )
                should_jump = jax.random.bernoulli(
                    jump_key, p=cfg.jump_prob, shape=current_params.shape
                )

                next_params = jnp.where(should_jump, jump_dest, drifted)
                return (next_params, key), current_params

            if cfg.non_stationary:
                _, dynamics_params_grid = jax.lax.scan(
                    jump_step,
                    (init_params, param_key),
                    None,
                    length=env_cfg.episode_length,
                )
            else:
                dynamics_params_grid = init_params

        elif cfg.dist_type == "langevin":
            if log_prob_fn is None:
                raise ValueError(
                    "Langevin requires a loaded GMM state (log_prob_fn). Check main() loader."
                )

            batch_score_fn = jax.vmap(jax.grad(log_prob_fn))
            param_key, init_key = jax.random.split(param_key)
            init_params = jax.random.uniform(
                init_key,
                shape=(num_eval_envs, len(dr_range_low)),
                minval=low,
                maxval=high,
            )

            def langevin_step(carry, _):
                current_params, key = carry
                key, subkey = jax.random.split(key)

                grads = batch_score_fn(current_params)
                grad_norm = jnp.linalg.norm(grads, axis=-1, keepdims=True) + 1e-6
                normalized_grads = grads / grad_norm

                drift_term = normalized_grads * cfg.drift * (high - low)
                noise_term = (
                    jax.random.normal(subkey, shape=current_params.shape)
                    * cfg.random_walk_scale
                    * (high - low)
                )

                next_params = jnp.clip(
                    current_params + drift_term + noise_term, low, high
                )
                return (next_params, key), current_params

            if cfg.non_stationary:
                _, dynamics_params_grid = jax.lax.scan(
                    langevin_step,
                    (init_params, param_key),
                    None,
                    length=env_cfg.episode_length,
                )
            else:
                dynamics_params_grid = init_params

        metrics, reward_1d, _ = evaluator.run_evaluation(
            params,
            dynamics_params=dynamics_params_grid,
            training_metrics={},
            num_eval_seeds=cfg.num_eval_seeds,
            success_threshold=cfg.success_threshold,
        )

        all_rewards_1d.append(reward_1d)
        for k, v in metrics.items():
            if k not in accumulated_metrics:
                accumulated_metrics[k] = []
            accumulated_metrics[k].append(v)

        print(
            f"  Run {i+1}/{num_runs}: Mean Reward = {float(metrics['eval/episode_reward_mean']):.2f}"
        )

    all_scores = np.array(jnp.concatenate(all_rewards_1d))
    sorted_scores = np.sort(all_scores)

    mean_val = np.mean(all_scores)
    min_val = np.min(all_scores)

    trim = int(len(sorted_scores) * 0.25)
    iqm_val = np.mean(sorted_scores[trim:-trim]) if trim > 0 else mean_val

    cvar10_val = np.mean(sorted_scores[: max(1, int(len(sorted_scores) * 0.10))])
    cvar20_val = np.mean(sorted_scores[: max(1, int(len(sorted_scores) * 0.20))])

    avg_metrics = {
        k: float(jnp.mean(jnp.stack(v))) for k, v in accumulated_metrics.items()
    }
    avg_metrics.update(
        {
            "reward_mean": float(mean_val),
            "reward_iqm": float(iqm_val),
            "reward_min": float(min_val),
            "reward_cvar10": float(cvar10_val),
            "reward_cvar20": float(cvar20_val),
        }
    )

    eval_output_dir = os.path.join(cfg.work_dir, "eval_results")
    os.makedirs(eval_output_dir, exist_ok=True)

    dist_label = distribution_label(cfg)

    # Dump using the robust NumpyEncoder
    with open(
        os.path.join(
            eval_output_dir, f"metrics_{cfg.task}_{dist_label}_seed{cfg.seed}.json"
        ),
        "w",
    ) as f:
        json.dump(avg_metrics, f, indent=4, cls=NumpyEncoder)

    np.save(
        os.path.join(
            eval_output_dir, f"rewards_{cfg.task}_{dist_label}_seed{cfg.seed}.npy"
        ),
        all_scores,
    )

    return avg_metrics, all_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="AcrobotSwingup")
    parser.add_argument(
        "--policy",
        default="ppo",
        choices=(
            "ppo",
            "ppo_nodr",
            "gmmppo",
            "gbsppo",
            "flowppo",
            "epoptppo",
            "adrppo",
            "doraemonppo",
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--log_root", default="logs")
    parser.add_argument(
        "--run_dir", help="Exact run folder; requires a single --seeds value."
    )
    parser.add_argument(
        "--sampler_checkpoint", help="Reference GMM checkpoint for Langevin dynamics."
    )
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    parser.add_argument("--num_eval_envs", type=int, default=4096)
    parser.add_argument("--num_eval_seeds", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--dr_wide", action="store_true")
    parser.add_argument("--ood", dest="ood_setting", action="store_true")
    parser.add_argument(
        "--non_stationary", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dist_type",
        default="random_walk",
        choices=(
            "uniform",
            "beta",
            "random_walk",
            "jump",
            "jump_diffusion",
            "langevin",
        ),
    )
    parser.add_argument("--beta", type=float, default=-10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--success_threshold", type=float, default=0.7)
    parser.add_argument("--success_rate_condition", type=float, default=0.7)
    parser.add_argument(
        "--rw_scale", dest="random_walk_scale", type=float, default=0.05
    )
    parser.add_argument("--beta_scale", type=float, default=0.2)
    parser.add_argument("--drift", type=float, default=0.05)
    parser.add_argument("--jump_prob", type=float, default=0.05)
    parser.add_argument(
        "--sweep_config",
        help="Optional YAML parameter sweep; see learning/sweep_config.yaml.",
    )
    args = parser.parse_args()
    if args.run_dir and len(args.seeds) != 1:
        parser.error("--run_dir requires exactly one --seeds value.")
    if min(args.num_eval_envs, args.num_eval_seeds, args.num_runs) <= 0:
        parser.error("Evaluation counts must be positive.")
    base_cfg = OmegaConf.create(vars(args))
    combinations = [{}]
    if args.sweep_config:
        sweep = OmegaConf.load(args.sweep_config)
        if args.dist_type == "random_walk":
            combinations = [
                {"random_walk_scale": scale} for scale in sweep.random_walk.scales
            ]
        elif args.dist_type in ("jump", "jump_diffusion"):
            combinations = [
                dict(drift=drift, jump_prob=prob)
                for drift, prob in itertools.product(
                    sweep.jump.drifts, sweep.jump.probs
                )
            ]
        elif args.dist_type == "langevin":
            combinations = [
                dict(drift=drift, random_walk_scale=noise)
                for drift, noise in itertools.product(
                    sweep.langevin.drifts, sweep.langevin.noises
                )
            ]

    for combination in combinations:
        results = {}
        for seed in args.seeds:
            cfg = OmegaConf.merge(base_cfg, combination, {"seed": seed})
            cfg.work_dir = get_work_dir(cfg)
            policy_path = Path(cfg.work_dir) / f"models/{cfg.policy}_params_latest.pkl"
            if not policy_path.exists():
                print(f"Skipping missing policy: {policy_path}")
                continue
            log_prob_fn = (
                load_gmm_log_prob(cfg) if cfg.dist_type == "langevin" else None
            )
            metrics, _ = evaluate(cfg, jax.random.PRNGKey(cfg.eval_seed), log_prob_fn)
            results[str(seed)] = metrics
        if not results:
            raise FileNotFoundError("No policies found for the requested runs.")
        aggregate = {
            "seeds_used": list(results),
            "per_seed": results,
            "config": OmegaConf.to_container(OmegaConf.merge(base_cfg, combination)),
        }
        for name in ("reward_mean", "reward_min", "reward_cvar10", "reward_cvar20"):
            values = [metrics[name] for metrics in results.values()]
            aggregate[f"{name}_avg"] = float(np.mean(values))
            aggregate[f"{name}_std"] = float(np.std(values))
        output_dir = (
            Path(args.run_dir) / "eval_results"
            if args.run_dir
            else Path(args.log_root) / args.task / "averaged_results"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        policy_label = args.policy
        if args.policy in ("gmmppo", "gbsppo", "flowppo"):
            policy_label += f"_beta={args.beta}"
        if args.policy == "flowppo":
            policy_label += f"_gamma={args.gamma}"
        elif args.policy == "epoptppo":
            policy_label += f"_eps={args.epsilon}"
        elif args.policy in ("adrppo", "doraemonppo"):
            policy_label += f"_threshold={args.success_threshold}"
            if args.policy == "doraemonppo":
                policy_label += f"_condition={args.success_rate_condition}"
        output_path = (
            output_dir / f"avg_metrics_{distribution_label(cfg)}_{policy_label}.json"
        )
        output_path.write_text(json.dumps(aggregate, indent=2, cls=NumpyEncoder) + "\n")
        print(f"Saved averaged evaluation: {output_path}")


if __name__ == "__main__":
    main()
