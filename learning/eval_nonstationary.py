import itertools
import os
import json
import pickle
import functools
import numpy as np
import jax
import jax.numpy as jnp
import hydra
from etils import epath
from omegaconf import OmegaConf, open_dict
from brax.training.acme import running_statistics
import argparse
import flax
from pathlib import Path  # Needed for checking Path objects

# --- Local Imports ---
from helper import parse_cfg
from custom_envs import registry, dm_control_suite, locomotion
from learning.agents.ppo import networks as ppo_networks
from learning.configs import dm_control_training_config, locomotion_training_config
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.evaluator import AdvEvaluator
from learning.agents.sampler_ppo import networks as samplerppo_networks
# --- HELPER: Robust JSON Encoder (Fixed for NumPy 2.0) ---
class NumpyEncoder(json.JSONEncoder):
    """ 
    Handles Numpy types, JAX arrays, and Path objects 
    to prevent JSON serialization errors.
    """
    def default(self, obj):
        # 1. Handle Numpy Integers
        # FIXED: Use np.integer to catch all numpy int types (safe for NumPy 2.0)
        if isinstance(obj, np.integer):
            return int(obj)
        
        # 2. Handle Numpy Floats
        # FIXED: Use np.floating to catch all numpy float types (safe for NumPy 2.0)
        elif isinstance(obj, np.floating):
            return float(obj)
        
        # 3. Handle Numpy/JAX Arrays
        elif isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        elif hasattr(obj, '__jax_array__'): # Catch-all for other JAX array types
            return np.array(obj).tolist()
            
        # 4. Handle Pathlib Objects
        elif isinstance(obj, Path):
            return str(obj)
            
        return json.JSONEncoder.default(self, obj)
# --- Default Configuration ---
CONFIG = {
    "task": "WalkerWalk",
    "policy": "ppo",
    "selected_seeds": [100, 101, 102, 103, 104, 105, 106],
    "eval_seed": 42,
    "dr_wide": False,
    "ood_setting": False,
    "dist_type": "beta",
    "beta_scale": 0.2,
    "non_stationary": True,
    "random_walk_scale": 0.05,
    "jump_prob": 0.05,
    "drift": 0.05,
    "beta": -20,
    "epsilon": 0.4,
}

# --- Environment Setup ---
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['XLA_FLAGS'] = xla_flags
os.environ['JAX_PLATFORM_NAME'] = 'gpu'

def get_work_dir(cfg):
    base_path = f"./logs/{cfg.task}/{cfg.seed}/{cfg.policy}"
    if cfg.policy == "gmmppo":
        return f"{base_path}/beta={int(cfg.beta)}"
    elif cfg.policy == 'epoptppo':
        return f"{base_path}/epsilon={cfg.epsilon}"
    return base_path

def load_sampler_state(path, sampler_choice, network_factory_args):
    """Loads the sampler state from a checkpoint."""
    dummy_key = jax.random.PRNGKey(0)
    samplerppo_network, _, _, (init_autodr, init_doraemon, init_flow, init_gmm) = \
        samplerppo_networks.make_samplerppo_networks(
            init_key=dummy_key,
            **network_factory_args
        )

    if sampler_choice == "AutoDR": template_state = init_autodr
    elif sampler_choice == "DORAEMON": template_state = init_doraemon
    elif "FLOW" in sampler_choice: template_state = init_flow
    elif sampler_choice == "GMM": template_state = init_gmm
    else: raise ValueError(f"Unknown sampler choice: {sampler_choice}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
        
    with open(path, "rb") as f:
        byte_data = f.read()

    restored_state = flax.serialization.from_bytes(template_state, byte_data)
    # print(f"Successfully loaded {sampler_choice} state from {path}")
    
    return samplerppo_network, restored_state

def evaluate(cfg, key, log_prob_fn=None):
    num_eval_envs = 4096
    num_runs = 1 
    
    # Load environment
    env = registry.load(cfg.task)
    env_cfg = registry.get_default_config(cfg.task)
    
    # Domain Randomization Setup
    randomizer = registry.get_domain_randomizer_ood(cfg.task) if cfg.ood_setting else registry.get_domain_randomizer_eval(cfg.task)
    dr_range_source = env.ood_range if cfg.ood_setting else (env.dr_range_wide if cfg.dr_wide else env.dr_range)
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

    # Policy Setup
    if "ppo" in cfg.policy:
        if cfg.task in dm_control_suite._envs:
            ppo_params = dm_control_training_config.brax_ppo_config(cfg.task)
        elif cfg.task in locomotion._envs:
            ppo_params = locomotion_training_config.locomotion_ppo_config(cfg.task)
            
        network_factory = ppo_networks.make_ppo_networks
        if "network_factory" in ppo_params:
            network_factory = functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory)
        
        ppo_network = network_factory(
            observation_size=env.observation_size,
            action_size=env.action_size,
            preprocess_observations_fn=running_statistics.normalize if ppo_params.normalize_observations else None,
        )
        make_policy_fn = ppo_networks.make_inference_fn(ppo_network)

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
                shape=(env_cfg.episode_length, num_eval_envs, len(dr_range_low)) if cfg.non_stationary else (num_eval_envs, len(dr_range_low)), 
                minval=low, maxval=high
            )
        
        elif cfg.dist_type == "beta":
            beta_samples = jax.random.beta(
                param_key, cfg.beta_scale, cfg.beta_scale, 
                shape=(env_cfg.episode_length, num_eval_envs, len(dr_range_low)) if cfg.non_stationary else (num_eval_envs, len(dr_range_low))
            )
            dynamics_params_grid = low + (high - low) * beta_samples

        elif cfg.dist_type == "random_walk":
            param_key, init_key = jax.random.split(param_key)
            init_params = jax.random.uniform(init_key, shape=(num_eval_envs, len(dr_range_low)), minval=low, maxval=high)

            def rw_step(carry, _):
                current_params, key = carry
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, shape=current_params.shape) * (cfg.random_walk_scale * (high - low))
                next_params = jnp.clip(current_params + noise, low, high)
                return (next_params, key), current_params

            if cfg.non_stationary:
                _, dynamics_params_grid = jax.lax.scan(rw_step, (init_params, param_key), None, length=env_cfg.episode_length)
            else:
                dynamics_params_grid = init_params
        
        elif cfg.dist_type in ["jump_diffusion", "jump"]:
            param_key, init_key = jax.random.split(param_key)
            init_params = jax.random.uniform(init_key, shape=(num_eval_envs, len(dr_range_low)), minval=low, maxval=high)

            def jump_step(carry, _):
                current_params, key = carry
                key, diff_key, jump_key, val_key = jax.random.split(key, 4)
                
                diff_scale = cfg.drift * (high - low)
                drifted = jnp.clip(current_params + jax.random.normal(diff_key, shape=current_params.shape)*diff_scale, low, high)
                
                jump_dest = jax.random.uniform(val_key, shape=current_params.shape, minval=low, maxval=high)
                should_jump = jax.random.bernoulli(jump_key, p=cfg.jump_prob, shape=current_params.shape)
                
                next_params = jnp.where(should_jump, jump_dest, drifted)
                return (next_params, key), current_params

            if cfg.non_stationary:
                _, dynamics_params_grid = jax.lax.scan(jump_step, (init_params, param_key), None, length=env_cfg.episode_length)
            else:
                dynamics_params_grid = init_params

        elif cfg.dist_type == "langevin":
            if log_prob_fn is None:
                raise ValueError("Langevin requires a loaded GMM state (log_prob_fn). Check main() loader.")

            batch_score_fn = jax.vmap(jax.grad(log_prob_fn))
            param_key, init_key = jax.random.split(param_key)
            init_params = jax.random.uniform(init_key, shape=(num_eval_envs, len(dr_range_low)), minval=low, maxval=high)

            def langevin_step(carry, _):
                current_params, key = carry
                key, subkey = jax.random.split(key)
                
                grads = batch_score_fn(current_params)
                grad_norm = jnp.linalg.norm(grads, axis=-1, keepdims=True) + 1e-6
                normalized_grads = grads / grad_norm
                
                drift_term = normalized_grads * cfg.drift * (high - low) 
                noise_term = jax.random.normal(subkey, shape=current_params.shape) * cfg.random_walk_scale * (high - low)
                
                next_params = jnp.clip(current_params + drift_term + noise_term, low, high)
                return (next_params, key), current_params

            if cfg.non_stationary:
                _, dynamics_params_grid = jax.lax.scan(langevin_step, (init_params, param_key), None, length=env_cfg.episode_length)
            else:
                dynamics_params_grid = init_params

        metrics, reward_1d, _ = evaluator.run_evaluation(
            params, dynamics_params=dynamics_params_grid,
            training_metrics={}, num_eval_seeds=50, success_threshold=0.7,
        )
        
        all_rewards_1d.append(reward_1d)
        for k, v in metrics.items():
            if k not in accumulated_metrics: accumulated_metrics[k] = []
            accumulated_metrics[k].append(v)
            
        print(f"  Run {i+1}/{num_runs}: Mean Reward = {float(metrics['eval/episode_reward_mean']):.2f}")

    all_scores = np.array(jnp.concatenate(all_rewards_1d))
    sorted_scores = np.sort(all_scores)
    
    mean_val = np.mean(all_scores)
    min_val = np.min(all_scores)
    
    trim = int(len(sorted_scores) * 0.25)
    iqm_val = np.mean(sorted_scores[trim:-trim]) if trim > 0 else mean_val
    
    cvar10_val = np.mean(sorted_scores[:int(len(sorted_scores)*0.10)])
    cvar20_val = np.mean(sorted_scores[:int(len(sorted_scores)*0.20)])

    avg_metrics = {k: float(jnp.mean(jnp.stack(v))) for k, v in accumulated_metrics.items()}
    avg_metrics.update({
        "reward_mean": float(mean_val),
        "reward_iqm": float(iqm_val),
        "reward_min": float(min_val),
        "reward_cvar10": float(cvar10_val),
        "reward_cvar20": float(cvar20_val),
    })

    eval_output_dir = os.path.join(cfg.work_dir, "eval_results")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    dist_label = cfg.dist_type
    if dist_label == 'langevin': 
        dist_label += f"_d={cfg.drift}_n={cfg.random_walk_scale}"
    elif dist_label == 'jump':
        dist_label += f"_jp={cfg.jump_prob}_drift={cfg.drift}"
    
    # Dump using the robust NumpyEncoder
    with open(os.path.join(eval_output_dir, f"metrics_{cfg.task}_{dist_label}_seed{cfg.seed}.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4, cls=NumpyEncoder)
    
    np.save(os.path.join(eval_output_dir, f"rewards_{cfg.task}_{dist_label}_seed{cfg.seed}.npy"), all_scores)

    return avg_metrics, all_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=CONFIG["task"])
    parser.add_argument("--policy", type=str, default=CONFIG["policy"])
    parser.add_argument("--seeds", type=int, nargs="+", default=CONFIG["selected_seeds"])
    parser.add_argument("--eval_seed", type=int, default=CONFIG["eval_seed"])
    parser.add_argument("--dr_wide", action="store_true", default=CONFIG["dr_wide"])
    parser.add_argument("--ood", action="store_true", default=CONFIG["ood_setting"])
    parser.add_argument("--non_stationary", action="store_true", default=CONFIG["non_stationary"])
    parser.add_argument("--dist_type", type=str, default=CONFIG["dist_type"])
    parser.add_argument("--beta", type=float, default=CONFIG["beta"])
    parser.add_argument("--epsilon", type=float, default=CONFIG["epsilon"])
    parser.add_argument("--rw_scale", type=float, default=CONFIG["random_walk_scale"])
    parser.add_argument("--beta_scale", type=float, default=CONFIG["beta_scale"])
    parser.add_argument("--drift", type=float, default=CONFIG["drift"])
    parser.add_argument("--jump_prob", type=float, default=CONFIG["jump_prob"])
    parser.add_argument("--sweep_config", type=str, default="sweep_config.yaml", help="Path to sweep config yaml")
    args = parser.parse_args()

    # --- 1. Load Base Config ---
    cfg_path = epath.Path(".").resolve()
    yaml_path = os.path.join(cfg_path, "config.yaml")
    base_cfg = OmegaConf.load(yaml_path)

    # --- 2. Update Base Config with CLI Args ---
    with open_dict(base_cfg):
        base_cfg.task = args.task
        base_cfg.policy = args.policy
        base_cfg.selected_seeds = args.seeds
        base_cfg.eval_seed = args.eval_seed
        base_cfg.dr_wide = args.dr_wide
        base_cfg.ood_setting = args.ood
        base_cfg.non_stationary = args.non_stationary
        base_cfg.dist_type = args.dist_type
        base_cfg.beta = args.beta
        base_cfg.epsilon = args.epsilon
        # Initial defaults
        base_cfg.random_walk_scale = args.rw_scale
        base_cfg.beta_scale = args.beta_scale
        base_cfg.drift = args.drift
        base_cfg.jump_prob = args.jump_prob

    base_cfg = parse_cfg(base_cfg)

    # --- 3. Pre-load GMM State (Langevin Only) ---
    log_prob_fn = None
    if base_cfg.dist_type == 'langevin':
        print("\n[LANGEVIN] Pre-loading GMM state...")
        try:
            temp_env = registry.load(base_cfg.task)
            dr_source = temp_env.ood_range if base_cfg.ood_setting else (temp_env.dr_range_wide if base_cfg.dr_wide else temp_env.dr_range)
            
            factory_args = {
                "observation_size": temp_env.observation_size,
                "action_size": temp_env.action_size,
                "dynamics_param_size": len(dr_source[0]),
                "batch_size": 1, "num_envs": 1,
                "bound_info": dr_source,
                "preprocess_observations_fn" : running_statistics.normalize,
                "success_threshold": 0.6, "success_rate_condition": 0.5,
                "sampler_choice": "GMM", 
            }
            
            file_path = f"./logs/{base_cfg.task}/sampler_state_latest.msgpack"
            sampler_net, gmm_state = load_sampler_state(file_path, "GMM", factory_args)
            actual_gmm_state = gmm_state.model_state.gmm_state
            
            @jax.jit
            def _gmm_log_prob(single_param):
                return sampler_net.gmm_network.model.log_density(actual_gmm_state, single_param)
            
            log_prob_fn = _gmm_log_prob
            print("[LANGEVIN] GMM State loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load GMM: {e}")
            print("Running without GMM (will fail if Langevin is actually used)...")

    # --- 4. PREPARE PARAMETER SWEEP ---
    sweep_cfg = OmegaConf.load(args.sweep_config)
    
    sweep_param_names = []
    sweep_combinations = []

    if args.dist_type in ['random_walk', 'walk']:
        scales = sweep_cfg.get('random_walk', {}).get('scales', [args.rw_scale])
        sweep_param_names = ['random_walk_scale']
        sweep_combinations = [(s,) for s in scales]

    elif args.dist_type in ['jump', 'jump_diffusion']:
        drifts = sweep_cfg.get('jump', {}).get('drifts', [args.drift])
        probs = sweep_cfg.get('jump', {}).get('probs', [args.jump_prob])
        sweep_param_names = ['drift', 'jump_prob']
        sweep_combinations = list(itertools.product(drifts, probs))

    elif args.dist_type == 'langevin':
        drifts = sweep_cfg.get('langevin', {}).get('drifts', [args.drift])
        noises = sweep_cfg.get('langevin', {}).get('noises', [args.rw_scale])
        sweep_param_names = ['drift', 'random_walk_scale']
        sweep_combinations = list(itertools.product(drifts, noises))
    
    else:
        sweep_combinations = [(None,)]

    print(f"\nStarting Sweep for {args.dist_type.upper()}")
    print(f"Sweeping over: {sweep_param_names}")
    print(f"Total combinations: {len(sweep_combinations)}")

    # --- 5. SWEEP LOOP ---
    for combo_idx, params in enumerate(sweep_combinations):
        
        # 5a. Create specific config for this iteration
        current_sweep_cfg = base_cfg.copy()
        
        sweep_info = {}
        if sweep_param_names:
            with open_dict(current_sweep_cfg):
                for name, val in zip(sweep_param_names, params):
                    current_sweep_cfg[name] = val
                    sweep_info[name] = val
        
        # --- NEW: CHECK IF FILE EXISTS BEFORE RUNNING ---
        
        # Construct the Policy Name
        p_name = current_sweep_cfg.policy
        if p_name == 'gmmppo': p_name += f"_beta={int(current_sweep_cfg.beta)}"
        elif p_name == 'epoptppo': p_name += f"_eps={current_sweep_cfg.epsilon}"
        
        # Construct the Dist Name
        d_name = current_sweep_cfg.dist_type
        if 'walk' in d_name: d_name += f"_rw={current_sweep_cfg.random_walk_scale}"
        elif 'jump' in d_name: d_name += f"_jp={current_sweep_cfg.jump_prob}_drift={current_sweep_cfg.drift}"
        elif 'langevin' in d_name: d_name += f"_drift={current_sweep_cfg.drift}_rw={current_sweep_cfg.random_walk_scale}"
        elif 'beta' in d_name: d_name += f"_scale={current_sweep_cfg.beta_scale}"

        avg_out = f"./logs/{current_sweep_cfg.task}/averaged_results"
        fname = f"avg_metrics_{current_sweep_cfg.task}_{d_name}_{p_name}_ns={current_sweep_cfg.non_stationary}.json"
        full_path = os.path.join(avg_out, fname)

        if os.path.exists(full_path):
            print(f"\n>>> [Sweep {combo_idx+1}/{len(sweep_combinations)}] SKIPPING - File exists: {fname}")
            continue
        
        # --- IF FILE DOES NOT EXIST, RUN EVALUATION ---
        print(f"\n>>> [Sweep {combo_idx+1}/{len(sweep_combinations)}] Running: {sweep_info}")
        
        results_summary = {}
        aggregated_metrics = {"reward_mean": [], "reward_min": [], "reward_cvar10": [], "reward_cvar20": []}

        for seed in current_sweep_cfg.selected_seeds:
            current_seed_cfg = current_sweep_cfg.copy()
            current_seed_cfg.seed = seed
            current_seed_cfg.work_dir = get_work_dir(current_seed_cfg)
            
            metrics, _ = evaluate(current_seed_cfg, jax.random.PRNGKey(current_seed_cfg.eval_seed), log_prob_fn=log_prob_fn)
            
            if metrics:
                results_summary[seed] = metrics
                for k in aggregated_metrics: aggregated_metrics[k].append(metrics[k])
            else:
                print(f"Skipping summary for seed {seed}")

        # --- SAVE RESULTS ---
        if results_summary:
            global_stats = {}
            for k, v in aggregated_metrics.items():
                global_stats[f"{k}_avg"] = np.mean(v)
                global_stats[f"{k}_std"] = np.std(v)

            os.makedirs(avg_out, exist_ok=True)
            
            config_dict = OmegaConf.to_container(current_sweep_cfg, resolve=True)
            save_data = {
                **global_stats, 
                "seeds_used": list(results_summary.keys()), 
                "config": config_dict
            }

            with open(full_path, "w") as f:
                json.dump(save_data, f, indent=4, cls=NumpyEncoder)
            
            print(f"Saved Average Metrics to: {fname}")
            print(f"Mean Reward: {global_stats['reward_mean_avg']:.2f} ± {global_stats['reward_mean_std']:.2f}")
if __name__ == "__main__":
    main()