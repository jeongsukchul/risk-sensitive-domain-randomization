import argparse
import os
import pathlib
import pickle
import sys
import time
from types import SimpleNamespace

if os.environ.get("DISPLAY") is None and os.environ.get("MUJOCO_GL") is None:
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LEARNING_ROOT = REPO_ROOT / "learning"
LEAP_API_ROOT = REPO_ROOT / "control" / "LEAP_Hand_API" / "python"

for path in (LEARNING_ROOT, LEAP_API_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agents.ppo import networks as ppo_networks  # noqa: E402
from configs.manipulation_training_config import manipulation_ppo_config  # noqa: E402
from custom_envs import manipulation  # noqa: E402


def load_policy(params_path: pathlib.Path, task: str = "LeapCubeRotateZAxis"):
    env = manipulation._envs[task]()
    ppo_cfg = manipulation_ppo_config(task)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_cfg:
        network_factory = lambda **kwargs: ppo_networks.make_ppo_networks(  # noqa: E731
            **ppo_cfg.network_factory, **kwargs
        )

    ppo_network = network_factory(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=(
            running_statistics.normalize if ppo_cfg.normalize_observations else None
        ),
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    policy = jax.jit(make_policy(params, deterministic=True))
    return env, policy


def import_leap_hardware():
    from main import LeapNode  # noqa: E402
    import leap_hand_utils.leap_hand_utils as lhu  # noqa: E402

    return LeapNode, lhu


def warm_start_hand(leap, lhu, target_real: np.ndarray, hz: float, seconds: float):
    current = np.asarray(leap.read_pos(), dtype=np.float32)
    steps = max(1, int(seconds * hz))
    for alpha in np.linspace(0.0, 1.0, steps):
        cmd = current * (1.0 - alpha) + target_real * alpha
        leap.set_leap(lhu.angle_safety_clip(cmd))
        time.sleep(1.0 / hz)


def build_observation(leap, lhu, last_action: np.ndarray, history: np.ndarray):
    joint_real = np.asarray(leap.read_pos(), dtype=np.float32)
    joint_sim = np.asarray(lhu.LEAPhand_to_LEAPsim(joint_real), dtype=np.float32)
    state = np.concatenate([joint_sim, last_action], dtype=np.float32)
    history = np.roll(history, state.size)
    history[: state.size] = state
    obs = {"state": jnp.asarray(history, dtype=jnp.float32)}
    return obs, history, joint_sim


def snapshot_render_state(state):
    data = state.data
    render_data = SimpleNamespace(
        qpos=np.asarray(data.qpos),
        qvel=np.asarray(data.qvel),
        mocap_pos=np.asarray(data.mocap_pos),
        mocap_quat=np.asarray(data.mocap_quat),
        xfrc_applied=np.asarray(data.xfrc_applied),
    )
    return SimpleNamespace(data=render_data)


def run_hardware_preflight(leap, lhu, default_real: np.ndarray, hz: float, seconds: float):
    print("Running hardware preflight...")
    current = np.asarray(leap.read_pos(), dtype=np.float32)
    current = lhu.angle_safety_clip(current)
    print(f"Current joint[0]: {current[0]:+.3f}")

    open_real = np.asarray(lhu.allegro_to_LEAPhand(np.zeros(16), zeros=False), dtype=np.float32)
    open_real = lhu.angle_safety_clip(open_real)

    print("Preflight: moving to open pose...")
    warm_start_hand(leap, lhu, open_real, hz, seconds)
    time.sleep(0.5)
    print("Preflight: moving to policy default pose...")
    warm_start_hand(leap, lhu, default_real, hz, seconds)
    time.sleep(0.5)

    verify = np.asarray(leap.read_pos(), dtype=np.float32)
    verify = lhu.angle_safety_clip(verify)
    err = np.abs(verify - default_real)
    print(
        "Preflight done. "
        f"mean_abs_err={err.mean():.4f} "
        f"max_abs_err={err.max():.4f}"
    )


def run_hardware(args, env, policy):
    params_path = pathlib.Path(args.params).resolve()
    LeapNode, lhu = import_leap_hardware()
    leap = LeapNode()

    default_sim = np.asarray(env._default_pose, dtype=np.float32)
    default_real = np.asarray(lhu.LEAPsim_to_LEAPhand(default_sim), dtype=np.float32)
    default_real = lhu.angle_safety_clip(default_real)

    print(f"Loaded policy from: {params_path}")
    print(f"Task: {args.task}")
    print(f"Loop rate: {args.hz:.1f} Hz")
    if not args.skip_preflight:
        run_hardware_preflight(leap, lhu, default_real, args.hz, args.preflight_sec)
    else:
        print("Skipping hardware preflight.")
        print("Moving hand to the policy default pose...")
        warm_start_hand(leap, lhu, default_real, args.hz, args.warmup_sec)

    history = np.zeros(env.observation_size["state"], dtype=np.float32)
    last_action = np.zeros(env.action_size, dtype=np.float32)
    key = jax.random.PRNGKey(args.seed)

    dt = 1.0 / args.hz
    start = time.time()
    step = 0

    print("Starting policy control. Press Ctrl+C to stop.")
    try:
        while True:
            if args.duration_sec is not None and (time.time() - start) >= args.duration_sec:
                break

            loop_start = time.time()
            obs, history, joint_sim = build_observation(leap, lhu, last_action, history)
            action, _ = policy(obs, key)
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)

            target_sim = default_sim + env._config.action_scale * action
            target_real = np.asarray(lhu.LEAPsim_to_LEAPhand(target_sim), dtype=np.float32)
            target_real = lhu.angle_safety_clip(target_real)
            leap.set_leap(target_real)

            last_action = action
            if step % args.print_every == 0:
                print(
                    f"step={step} "
                    f"joint_sim[0]={joint_sim[0]:+.3f} "
                    f"action[0]={action[0]:+.3f}"
                )

            step += 1
            sleep_time = dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        print("Returning to default pose...")
        warm_start_hand(leap, lhu, default_real, args.hz, 1.0)


def run_sim(args, env, policy):
    key = jax.random.PRNGKey(args.seed)
    reset_key, policy_key = jax.random.split(key)
    state = env.reset(reset_key)
    step_fn = jax.jit(env.step)

    print(f"Loaded policy from: {pathlib.Path(args.params).resolve()}")
    print(f"Task: {args.task}")
    print("Running in simulation dry-run mode.")

    max_steps = args.sim_steps if args.duration_sec is None else int(args.duration_sec * args.hz)
    max_steps = max(1, max_steps)
    rollout = [snapshot_render_state(state)]

    for step in range(max_steps):
        action, _ = policy(state.obs, policy_key)
        state = step_fn(state, action)
        rollout.append(snapshot_render_state(state))

        if step % args.print_every == 0:
            reward = float(np.asarray(state.reward))
            done = bool(np.asarray(state.done))
            action0 = float(np.asarray(action)[0])
            print(f"step={step} reward={reward:+.4f} done={done} action[0]={action0:+.3f}")

        if bool(np.asarray(state.done)):
            print(f"Episode terminated at step {step}. Resetting sim state.")
            reset_key, policy_key = jax.random.split(policy_key)
            state = env.reset(reset_key)
            rollout.append(snapshot_render_state(state))

    camera = args.camera
    frames = env.render(rollout, camera=camera, width=args.width, height=args.height)
    output_path = pathlib.Path(args.output).resolve()
    os.makedirs(output_path.parent, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=args.hz)
    print(f"Saved sim render to: {output_path}")


def run(args):
    params_path = pathlib.Path(args.params).resolve()
    env, policy = load_policy(params_path, task=args.task)
    if args.mode == "sim":
        run_sim(args, env, policy)
    else:
        run_hardware(args, env, policy)


def parse_args():
    default_params = (
        REPO_ROOT
        / "learning"
        / "logs"
        / "LeapCubeRotateZAxis"
        / "4"
        / "ppo"
        / "models"
        / "ppo_params_latest.pkl"
    )
    parser = argparse.ArgumentParser(
        description="Run the learned LeapCubeRotateZAxis PPO policy on the real LEAP hand."
    )
    parser.add_argument("--mode", choices=("hardware", "sim"), default="hardware")
    parser.add_argument("--params", type=str, default=str(default_params))
    parser.add_argument("--task", type=str, default="LeapCubeRotateZAxis")
    parser.add_argument("--hz", type=float, default=20.0, help="Use 20 Hz to match ctrl_dt=0.05.")
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--preflight-sec", type=float, default=1.0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--sim-steps", type=int, default=200)
    parser.add_argument("--camera", type=str, default="side")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "control" / "sim_leap_rotate_policy.mp4"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
