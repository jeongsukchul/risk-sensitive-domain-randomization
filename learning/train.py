import os
import sys
import imageio
import mediapy as media
from omegaconf import OmegaConf
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
from typing import Any, Callable, Dict, Sequence, Tuple, Union
from agents.ppo import networks as ppo_networks
from agents.ppo import train as ppo
from agents.sampler_ppo  import train as sampler_ppo
from agents.sampler_ppo import networks as samplerppo_networks
from agents.m2td3 import train as m2td3
from agents.m2td3 import networks as m2td3_networks
from etils import epath
from flax.training import orbax_utils
import jax
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp
import wandb
from learning.configs.dm_control_training_config import brax_ppo_config, brax_td3_config
from learning.configs.locomotion_training_config import locomotion_ppo_config, locomotion_td3_config
from learning.configs.manipulation_training_config import manipulation_ppo_config, manipulation_td3_config
import hydra
from custom_envs import registry, dm_control_suite, locomotion, manipulation
from helper import parse_cfg
from helper import make_dir
import pickle
import shutil
from learning.module.wrapper.wrapper import Wrapper
from custom_envs import mjx_env
from utils import save_configs_to_wandb_and_local
from learning.module.wrapper.wrapper import Wrapper
import scipy
import jax.numpy as jnp
# # Ignore the info logs from brax
# logging.set_verbosity(logging.WARNING)

# warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# # Suppress DeprecationWarnings from JAX
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# # Suppress UserWarnings from absl (used by JAX and TensorFlow)
# warnings.filterwarnings("ignore", category=UserWarning, module="absl")

env_name = "FishSwim"  # @param ["AcrobotSwingup", "AcrobotSwingupSparse", "BallInCup", "CartpoleBalance", "CartpoleBalanceSparse", "CartpoleSwingup", "CartpoleSwingupSparse", "CheetahRun", "FingerSpin", "FingerTurnEasy", "FingerTurnHard", "FishSwim", "HopperHop", "HopperStand", "HumanoidStand", "HumanoidWalk", "HumanoidRun", "PendulumSwingup", "PointMass", "ReacherEasy", "ReacherHard", "SwimmerSwimmer6", "WalkerRun", "WalkerStand", "WalkerWalk"]
CAMERAS = {
    "AcrobotSwingup": "fixed",
    "AcrobotSwingupSparse": "fixed",
    "BallInCup": "cam0",
    "CartpoleBalance": "fixed",
    "CartpoleBalanceSparse": "fixed",
    "CartpoleSwingup": "fixed",
    "CartpoleSwingupSparse": "fixed",
    "CheetahRun": "side",
    "FingerSpin": "cam0",
    "FingerTurnEasy": "cam0",
    "FingerTurnHard": "cam0",
    "FishSwim": "fixed_top",
    "HopperHop": "cam0",
    "HopperStand": "cam0",
    "HumanoidStand": "side",
    "HumanoidWalk": "side",
    "HumanoidRun": "side",
    "PendulumSwingup": "fixed",
    "PointMass": "cam0",
    "ReacherEasy": "fixed",
    "ReacherHard": "fixed",
    "SwimmerSwimmer6": "tracking1",
    "WalkerRun": "side",
    "WalkerWalk": "side",
    "WalkerStand": "side",
    "Go1Handstand": "side",
    "Go1JoystickRoughTerrain": "track",
    "G1InplaceGaitTracking" : "track",
    "G1JoystickGaitTracking" : "track",
    "T1JoystickFlatTerrain" :"track",
    "T1JoystickRoughTerrain" :"track",
    "LeapCubeRotateZAxis" :"side",
    "LeapCubeReorient" :"side",
}
camera_name = CAMERAS[env_name]

class BraxDomainRandomizationWrapper(Wrapper):
  """Brax wrapper for domain randomization."""
  def __init__(
      self,
      env: mjx_env.MjxEnv,
      randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
  ):
    super().__init__(env)
    self._mjx_model, self._in_axes = randomization_fn(self.env.mjx_model)
    self.env.unwrapped._mjx_model = self._mjx_model

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = self.env.reset(rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    res = self.env.step(state, action)
    return res

def policy_params_fn(current_step, make_policy, params, ckpt_path: epath.Path):
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  path = ckpt_path / f"{current_step}"
  orbax_checkpointer.save(path, params, force=True, save_args=save_args)
def progress_fn(num_steps, metrics, use_wandb=True):
    if use_wandb:
        wandb.log(metrics, step=num_steps)
    print("-------------------------------------------------------------------")
    print(f"num_steps: {num_steps}")
    print(f"num_update_steps: {num_steps//8}")
    
    for k,v in metrics.items():
        print(f" {k} :  {v}")
    print("-------------------------------------------------------------------")


def train_ppo(cfg:dict, randomization_fn, env, eval_env=None):

    print("training with ppo")
    if cfg.task in dm_control_suite._envs:
        ppo_params = brax_ppo_config(cfg.task)
    elif cfg.task in locomotion._envs:
        ppo_params = locomotion_ppo_config(cfg.task)
    elif cfg.task in manipulation._envs:
        ppo_params = manipulation_ppo_config(cfg.task)
    if cfg.randomization:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}"
        # if cfg.custom_wrapper and cfg.adv_wrapper:
        #     wandb_name+=f".adv_wrapper={cfg.adv_wrapper}"#dr_train_ratio={cfg.dr_train_ratio}"
    else:
        wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}.final_rand={cfg.final_randomization}"
    if cfg.custom_wrapper:
        randomizer = registry.get_domain_randomizer_eval(cfg.task)
    else:
        randomizer = randomization_fn
    if cfg.policy=='ppo_nodr':
        sampler_choice = 'NODR'
        group = sampler_choice
    elif cfg.policy=='ppo':
        sampler_choice = 'UDR'
        group = sampler_choice
    elif cfg.policy=='flowppo':
        sampler_choice = 'FLOW_NS'
        wandb_name+= f" [gamma={cfg.gamma}_beta={cfg.beta}_iters={cfg.n_sampler_iters}]"
        group = sampler_choice
        group+=f" [gamma={cfg.gamma}_beta={cfg.beta}_iters={cfg.n_sampler_iters}]"
    elif cfg.policy=='gmmppo':
        sampler_choice = 'GMM'
        wandb_name+= f" [beta={cfg.beta}]"
        group = sampler_choice
        if cfg.use_scheduling:
            wandb_name+= f" [lr={cfg.scheduler_lr}_window={cfg.scheduler_window_size}]_sampler_update_freq={cfg.sampler_update_freq}"
            group+=f" [beta={cfg.beta}]_sampler_update_freq={cfg.sampler_update_freq}"
        else:
            wandb_name+= f" [beta={cfg.beta}]_sampler_update_freq={cfg.sampler_update_freq}"
            group+=f" [beta={cfg.beta}]_sampler_update_freq={cfg.sampler_update_freq}"
    elif cfg.policy=='adrppo':
        sampler_choice = 'AutoDR'
        wandb_name+= f" [threshold={cfg.success_threshold}]"
        group = sampler_choice
        group += f" [threshold={cfg.success_threshold}]"
    elif cfg.policy=='doraemonppo':
        sampler_choice = 'DORAEMON'
        wandb_name += f" [threshold={cfg.success_threshold}_condition={cfg.success_rate_condition}]"
        group = sampler_choice
        group += f" [threshold={cfg.success_threshold}_condition={cfg.success_rate_condition}]"
    else:
        raise ValueError("No ppo variant!")
    wandb_name += cfg.comment
    cfg.group = group
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name,
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})
    ppo_params.num_evals = cfg.num_evals
    network_factory = samplerppo_networks.make_samplerppo_networks
    train_fn = sampler_ppo.train
    for param in ppo_params.keys():
        if param in cfg and getattr(cfg, param) is not None:
            ppo_params[param] = getattr(cfg, param)
    ppo_training_params = dict(ppo_params)
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            ppo_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            network_factory,
            **ppo_params.network_factory
        )
        
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)

    train_fn = functools.partial(
        train_fn, **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=functools.partial(policy_params_fn, ckpt_path=cfg.work_dir / "models" ),
        randomization_fn=randomizer,
        use_wandb=cfg.use_wandb,
        seed=cfg.seed,
        sampler_choice=sampler_choice,
        gamma = cfg.gamma,
        beta = cfg.beta,
        sampler_update_freq =cfg.sampler_update_freq,
        n_sampler_iters = cfg.n_sampler_iters,
        success_threshold = cfg.success_threshold,
        success_rate_condition = cfg.success_rate_condition,
        work_dir = cfg.work_dir,
        use_scheduling = cfg.use_scheduling,
        scheduler_lr =cfg.scheduler_lr,
        scheduler_window_size = cfg.scheduler_window_size,
    )
    
    make_inference_fn, params, metrics = train_fn(
        environment=env,
    )
    return make_inference_fn, params, metrics

def train_m2td3(cfg:dict, randomization_fn, env, eval_env=None):
    if cfg.task in dm_control_suite._envs:
        m2td3_params = brax_td3_config(cfg.task)
    elif cfg.task in locomotion._envs:
        m2td3_params = locomotion_td3_config(cfg.task)
    m2td3_params.omega_distance_threshold = 0.1
    for param in m2td3_params.keys():
        if param in cfg and getattr(cfg, param) is not None:
            m2td3_params[param] = getattr(cfg, param)
    print("omega_distance_threshold:", m2td3_params.omega_distance_threshold)
    m2td3_training_params = dict(m2td3_params)
    wandb_name = f"{cfg.task}.{cfg.policy}.{cfg.seed}.asym={cfg.asymmetric_critic}.dist={m2td3_params.omega_distance_threshold}"
    wandb_name += cfg.comment
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project, 
            entity=cfg.wandb_entity, 
            name=wandb_name, 
            dir=make_dir(cfg.work_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"env_name": cfg.task})

    network_factory = m2td3_networks.make_m2td3_networks
    if "network_factory" in m2td3_params:
        del m2td3_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            m2td3_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            m2td3_networks.make_m2td3_networks,
            **m2td3_params.network_factory
        )
    
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    train_fn = functools.partial(
        m2td3.train, **dict(m2td3_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn=randomization_fn,
        dr_train_ratio = cfg.dr_train_ratio,
        seed=cfg.seed,
    )
    make_inference_fn, params, metrics = train_fn(        
        environment=env,
    )
    return make_inference_fn, params, metrics

@hydra.main(config_name="config", config_path=".", version_base=None)
def train(cfg: dict):
    
    cfg = parse_cfg(cfg)
    print("cfg :", cfg)

    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    rng = jax.random.PRNGKey(cfg.seed)
    
    path = epath.Path(".").resolve()
    cfg_dir = make_dir(cfg.work_dir / "cfg")
    shutil.copy('config.yaml', os.path.join(cfg_dir, 'config.yaml'))
    env_cfg = registry.get_default_config(cfg.task)
    env_cfg['impl'] = cfg.impl
    if cfg.policy == "td3" :
        if "stand" in cfg.task:
            env_cfg.reward_config.scales.energy = -5e-5
            env_cfg.reward_config.scales.action_rate = -1e-1
            env_cfg.reward_config.scales.torques = -1e-3
        elif "T1" in cfg.task or "G1" in cfg.task:
            env_cfg.reward_config.scales.energy = -5e-5
            env_cfg.reward_config.scales.action_rate = -1e-1
            env_cfg.reward_config.scales.torques = -1e-3
            env_cfg.reward_config.scales.pose = -1.0
            env_cfg.reward_config.scales.tracking_ang_vel = 1.25
            env_cfg.reward_config.scales.tracking_lin_vel = 1.25
            env_cfg.reward_config.scales.feet_phase = 1.0
            env_cfg.reward_config.scales.ang_vel_xy = -0.3
            env_cfg.reward_config.scales.orientation = -5.0
    
    env = registry.load(cfg.task, config=env_cfg)

    if cfg.randomization:
        randomizer = registry.get_domain_randomizer(cfg.task)
        randomization_fn = randomizer
    else:
        randomization_fn = None 

    if "ppo" in cfg.policy:
        make_inference_fn, params, metrics = train_ppo(cfg, randomization_fn, env)
    else:
        print("no policy!")


    save_dir = make_dir(cfg.work_dir / "models")
    print(f"Saving parameters to {save_dir}")
    with open(os.path.join(save_dir, f"{cfg.policy}_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"), "wb") as f:
        pickle.dump(params, f)
    latest_path = os.path.join(save_dir, f"{cfg.policy}_params_latest.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump(params, f)

    # Save config.yaml and randomization config to wandb and local directory
    save_configs_to_wandb_and_local(cfg, cfg.work_dir)
    env_cfg['impl'] = 'jax'

    if cfg.final_randomization:
        eval_rng, rng = jax.random.split(rng)
        randomizer_eval = registry.get_domain_randomizer_eval(cfg.task)
        randomizer_eval = functools.partial(randomizer_eval, rng=eval_rng, dr_range=env.dr_range)
        eval_env = BraxDomainRandomizationWrapper(
            registry.load(cfg.task, config=env_cfg),
            randomization_fn=randomizer_eval,
        )
    else:
        eval_env = registry.load(cfg.task, config=env_cfg)
        
    if cfg.save_video and cfg.use_wandb:
        n_episodes = 10
        jit_inference_fn = jax.jit(make_inference_fn(params,deterministic=True))
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        reward_list = []
        rollout = []
        rng, eval_rng = jax.random.split(rng)
        rngs = jax.random.split(eval_rng, n_episodes)
        import gc  
        for i in range(n_episodes): #10 episodes
            state = jit_reset(rngs[i])
            if i==0:
                rollout = [state]
            rewards = 0
            for _ in range(env_cfg.episode_length):
                act_rng, rng = jax.random.split(rng)
                action, info = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, action)
                if i==0:
                    rollout.append(jax.device_get(state))
                rewards += state.reward
            reward_list.append(rewards)

        print("Starting video rendering...")
        frames = eval_env.render(rollout, camera=CAMERAS[cfg.task])
        fps = 1.0 / env.sim_dt
        video_path = f"video_{cfg.policy}_{cfg.task}.mp4"
        print(f"Encoding video to {video_path}...")
        media.write_video(video_path, frames, fps=fps)
        print("Video saved successfully.")
        # 2. Open the writer
        print(f"Video saved successfully to {video_path}")
        
        # ... (rest of your wandb logging) ...
        try:
            del frames
            gc.collect()
            wandb.log({'eval_video': wandb.Video(video_path, fps=fps, format='mp4')})
        except Exception as e:
            print(f"Could not upload to WandB due to memory limits: {e}")

        wandb.log({'final_eval_reward' : rewards.mean()}) 
        wandb.log({'final_eval_reward_iqm' : scipy.stats.trim_mean(rewards, proportiontocut=0.25, axis=None) })
        wandb.log({'final_eval_reward_std' :rewards.std() })

   
if __name__ == "__main__":
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    train()