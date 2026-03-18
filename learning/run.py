import os
import sys
import imageio
import mediapy as media
import copy
from omegaconf import OmegaConf
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
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
from agents.td3 import networks as td3_networks
from agents.td3 import train as td3
from etils import epath
from flax.training import orbax_utils
import jax
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp
import wandb
from learning.configs.dm_control_training_config import brax_ppo_config, brax_td3_config
from learning.configs.locomotion_training_config import locomotion_ppo_config#, locomotion_td3_config
from learning.configs.manipulation_training_config import manipulation_ppo_config#, manipulation_td3_config
import hydra
from custom_envs import registry, dm_control_suite, locomotion, manipulation
from helper import parse_cfg
from helper import make_dir
import pickle
import shutil
from learning.module.wrapper.wrapper import Wrapper
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from custom_envs import mjx_env
from utils import save_configs_to_wandb_and_local
from learning.module.wrapper.wrapper import Wrapper
import scipy
import jax.numpy as jnp
import math
from PIL import Image, ImageDraw
# # Ignore the info logs from brax
# logging.set_verbosity(logging.WARNING)

# warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# # Suppress DeprecationWarnings from JAX
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# # Suppress UserWarnings from absl (used by JAX and TensorFlow)
# warnings.filterwarnings("ignore", category=UserWarning, module="absl")

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
    "PandaPickCube" :None,
    "PandaPickCubeOrientation" :None,
    "PandaOpenCabinet" :None,
    "AlohaSinglePegInsertion" : "collaborator_pov",
    "PandaRobotiqPushCube" : None,

}

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

def _apply_fixed_dynamics_params(eval_env, randomizer_eval, dynamics_params):
    if randomizer_eval is None:
        return eval_env
    new_model, _ = randomizer_eval(
        model=eval_env.mjx_model,
        dr_range=eval_env.dr_range,
        params=jnp.asarray(dynamics_params),
        rng=None,
    )
    eval_env._mjx_model = new_model
    if hasattr(eval_env, "unwrapped"):
        eval_env.unwrapped._mjx_model = new_model
    return eval_env

def _extract_single_trajectory(trajectory, batch_index, batch_size):
    def _select_leaf(x):
        if not hasattr(x, "shape"):
            return x
        if len(x.shape) == 0:
            return x
        if x.shape[0] == batch_size:
            return x[batch_index]
        return x
    return [
        jax.tree_util.tree_map(_select_leaf, state)
        for state in trajectory
    ]

def _tile_frame_sequences(frame_sequences, grid_cols=4, bg_value=0, tile_labels=None):
    if not frame_sequences:
        return []
    max_t = max(len(seq) for seq in frame_sequences)
    n = len(frame_sequences)
    grid_rows = math.ceil(n / grid_cols)
    h = max(seq[0].shape[0] for seq in frame_sequences if len(seq) > 0)
    w = max(seq[0].shape[1] for seq in frame_sequences if len(seq) > 0)
    c = frame_sequences[0][0].shape[2]

    tiled = []
    for t in range(max_t):
        canvas = np.full((grid_rows * h, grid_cols * w, c), bg_value, dtype=np.uint8)
        for i, seq in enumerate(frame_sequences):
            frame = seq[min(t, len(seq) - 1)]
            fh, fw = frame.shape[:2]
            r, col = divmod(i, grid_cols)
            y0 = r * h + (h - fh) // 2
            x0 = col * w + (w - fw) // 2
            canvas[y0:y0+fh, x0:x0+fw] = frame
            if tile_labels is not None and i < len(tile_labels):
                pil_img = Image.fromarray(canvas)
                draw = ImageDraw.Draw(pil_img)
                label = str(tile_labels[i])
                text_x = col * w + 8
                text_y = r * h + 8
                # Draw a simple black box and white text for readability.
                text_box_w = 10 + 8 * max(len(label), 1)
                text_box_h = 24
                draw.rectangle(
                    [text_x - 4, text_y - 2, text_x - 4 + text_box_w, text_y - 2 + text_box_h],
                    fill=(0, 0, 0),
                )
                draw.text((text_x, text_y), label, fill=(255, 255, 255))
                canvas = np.array(pil_img, copy=True)
        tiled.append(canvas)
    return tiled


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
        group += f"_impl={cfg.impl}"
    elif cfg.policy=='epoptppo':
        sampler_choice = 'EPOpt'
        wandb_name+= f" [epsilon={cfg.epsilon}]"
        group = sampler_choice
        group+=f" [epsilon={cfg.epsilon}]"
    elif cfg.policy=='flowppo':
        sampler_choice = 'FLOW_NS'
        wandb_name+= f" [gamma={cfg.gamma}_beta={cfg.beta}_iters={cfg.n_sampler_iters}]"
        group = sampler_choice
        group+=f" [gamma={cfg.gamma}_beta={cfg.beta}_iters={cfg.n_sampler_iters}]"
    elif cfg.policy=='gbsppo':
        sampler_choice = 'GBS'
        wandb_name+= f" [beta={cfg.beta}_iters={cfg.n_sampler_iters}]"
        group = sampler_choice
        group+=f" [beta={cfg.beta}_iters={cfg.n_sampler_iters}]"
    elif cfg.policy=='gmmppo':
        sampler_choice = 'GMM'
        group = sampler_choice
        if cfg.use_scheduling:
            wandb_name+= f" {cfg.scheduler_mode} scheduling[{cfg.start_beta}, {cfg.end_beta}]"
            group+=f" {cfg.scheduler_mode} scheduling[{cfg.start_beta}, {cfg.end_beta}]"
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

    train_gamma = cfg.gamma if "FLOW" in sampler_choice else 0.0
    train_fn = functools.partial(
        train_fn, **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=functools.partial(policy_params_fn, ckpt_path=cfg.work_dir / "models" ),
        randomization_fn=randomizer,
        use_wandb=cfg.use_wandb,
        seed=cfg.seed,
        sampler_choice=sampler_choice,
        gamma = train_gamma,
        beta = cfg.beta,
        sampler_update_freq =cfg.sampler_update_freq,
        n_sampler_iters = cfg.n_sampler_iters,
        sampler_visualization=getattr(cfg, "sampler_visualization", False),
        success_threshold = cfg.success_threshold,
        success_rate_condition = cfg.success_rate_condition,
        work_dir = cfg.work_dir,
        use_scheduling = cfg.use_scheduling,
        scheduler_lr =cfg.scheduler_lr,
        scheduler_window_size = cfg.scheduler_window_size,
        epsilon = cfg.epsilon,
        start_beta = cfg.start_beta,
        end_beta = cfg.end_beta,
        scheduler_mode=     cfg.scheduler_mode,
        gbs_process_type=getattr(cfg, "gbs_process_type", "vp"),
        gbs_num_steps=getattr(cfg, "gbs_num_steps", 100),
        gbs_lr=getattr(cfg, "gbs_lr", 1e-3),
        gbs_clip_grad=getattr(cfg, "gbs_clip_grad", 1.0),
        gbs_init_std=getattr(cfg, "gbs_init_std", 1.0),
        gbs_max_rnd=getattr(cfg, "gbs_max_rnd", 1e8),
        gbs_sde_ctrl_noise=getattr(cfg, "gbs_sde_ctrl_noise", None),
        gbs_sde_ctrl_dropout=getattr(cfg, "gbs_sde_ctrl_dropout", None),
        gbs_use_tanh_bijection=getattr(cfg, "gbs_use_tanh_bijection", True),
        gbs_model_num_layers=getattr(cfg, "gbs_model_num_layers", 2),
        gbs_model_num_hid=getattr(cfg, "gbs_model_num_hid", 64),
        gbs_sigma_const=getattr(cfg, "gbs_sigma_const", 1.0),
        gbs_vp_diff_coeff_sq_min=getattr(cfg, "gbs_vp_diff_coeff_sq_min", 0.1),
        gbs_vp_diff_coeff_sq_max=getattr(cfg, "gbs_vp_diff_coeff_sq_max", 10.0),
        gbs_vp_scale_diff_coeff=getattr(cfg, "gbs_vp_scale_diff_coeff", 1.0),
        gbs_terminal_t=getattr(cfg, "gbs_terminal_t", 1.0),
        gbs_include_base_drift=getattr(cfg, "gbs_include_base_drift", True),
    )
    
    make_inference_fn, params, metrics = train_fn(
        environment=env,
    )
    return make_inference_fn, params, metrics
def train_td3(cfg:dict, randomization_fn, env, eval_env=None):
    if cfg.task in dm_control_suite._envs:
        td3_params = brax_td3_config(cfg.task)
    # elif cfg.task in locomotion._envs:
    #     td3_params = locomotion_td3_config(cfg.task)
    # elif cfg.task in manipulation._envs:
    #     td3_params = manipulation_td3_config(cfg.task)
    td3_training_params = dict(td3_params)
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
    if cfg.policy=='td3_nodr':
        sampler_choice = 'NODR'
        group = sampler_choice
    elif cfg.policy=='td3':
        sampler_choice = 'UDR'
        group = sampler_choice
        group += f"_impl={cfg.impl}"
    elif cfg.policy=='epopttd3':
        sampler_choice = 'EPOpt'
        wandb_name+= f" [epsilon={cfg.epsilon}]"
        group = sampler_choice
        group+=f" [epsilon={cfg.epsilon}]"
    elif cfg.policy=='flowtd3':
        sampler_choice = 'FLOW_NS'
        wandb_name+= f" [gamma={cfg.gamma}_beta={cfg.beta}_iters={cfg.n_sampler_iters}]"
        group = sampler_choice
        group+=f" [gamma={cfg.gamma}_beta={cfg.beta}_iters={cfg.n_sampler_iters}]"
    elif cfg.policy=='gmmtd3':
        sampler_choice = 'GMM'
        group = sampler_choice
        if cfg.use_scheduling:
            wandb_name+= f" {cfg.scheduler_mode} scheduling[{cfg.start_beta}, {cfg.end_beta}]"
            group+=f" {cfg.scheduler_mode} scheduling[{cfg.start_beta}, {cfg.end_beta}]"
        else:
            wandb_name+= f" [beta={cfg.beta}]_sampler_update_freq={cfg.sampler_update_freq}"
            group+=f" [beta={cfg.beta}]_sampler_update_freq={cfg.sampler_update_freq}"
    elif cfg.policy=='adrtd3':
        sampler_choice = 'AutoDR'
        wandb_name+= f" [threshold={cfg.success_threshold}]"
        group = sampler_choice
        group += f" [threshold={cfg.success_threshold}]"
    elif cfg.policy=='doraemontd3':
        sampler_choice = 'DORAEMON'
        wandb_name += f" [threshold={cfg.success_threshold}_condition={cfg.success_rate_condition}]"
        group = sampler_choice
        group += f" [threshold={cfg.success_threshold}_condition={cfg.success_rate_condition}]"
    else:
        raise ValueError("No td3 variant!")
    if cfg.nonstationary:
        wandb_name+='_nonstationary'
        group +=f"_nonstationary"
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

    network_factory = td3_networks.make_td3_networks
    if "network_factory" in td3_params:
        del td3_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            td3_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            td3_networks.make_td3_networks,
            **td3_params.network_factory
        )
    
    progress = functools.partial(progress_fn, use_wandb=cfg.use_wandb)
    if cfg.custom_wrapper:
        randomizer = registry.get_domain_randomizer_eval(cfg.task)
    else:
        randomizer = randomization_fn
    train_fn = functools.partial(
        td3.train, **dict(td3_training_params),
        network_factory=network_factory,
        progress_fn=progress,
        randomization_fn = randomizer,
        eval_randomization_fn=randomization_fn,
        dr_train_ratio = cfg.dr_train_ratio,
        seed=cfg.seed,
        sampler_choice=sampler_choice,
        gamma = cfg.gamma,
        beta = cfg.beta,
        sampler_update_freq =cfg.sampler_update_freq,
        n_sampler_iters = cfg.n_sampler_iters,
        success_threshold = cfg.success_threshold,
        success_rate_condition = cfg.success_rate_condition,
        work_dir = cfg.work_dir,
        use_wandb=cfg.use_wandb,
        nonstationary = cfg.nonstationary,
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
    from absl import logging
    logging.set_verbosity(logging.WARNING)

    cfg = parse_cfg(cfg)
    print("cfg :", cfg)
    if cfg.policy == "epoptppo":
        cfg.work_dir = cfg.work_dir / f"epsilon={cfg.epsilon}"
    elif cfg.policy == "flowppo":
        cfg.work_dir = cfg.work_dir / f"beta={cfg.beta}_gamma={cfg.gamma}"
    elif cfg.policy == "gbsppo":
        cfg.work_dir = cfg.work_dir / f"beta={cfg.beta}"
    elif cfg.policy == "gmmppo":
        cfg.work_dir = cfg.work_dir / f"beta={cfg.beta}"
    elif cfg.policy == "adrppo":
        cfg.work_dir = cfg.work_dir / f"threshold={cfg.success_threshold}"
    elif cfg.policy == "doraemonppo":
        cfg.work_dir = cfg.work_dir / f"threshold={cfg.success_threshold}_condition={cfg.success_rate_condition}"
    
    print("Working directory:", cfg.work_dir)

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
    elif "td3" in cfg.policy:
        make_inference_fn, params, metrics = train_td3(cfg, randomization_fn, env)

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
    eval_env = registry.load(cfg.task, config=env_cfg)
        
    if cfg.save_video and cfg.use_wandb:
        policy_fn = make_inference_fn(params, deterministic=True)
        jit_policy_fn = jax.jit(policy_fn)
        fps = 1.0 / env.sim_dt

        percentile_levels = metrics.get('final_eval/percentile_levels', None) if isinstance(metrics, dict) else None
        percentile_params = metrics.get('final_eval/dynamics_params_percentiles', None) if isinstance(metrics, dict) else None
        use_percentile_rollouts = (
            "ppo" in cfg.policy
            and percentile_levels is not None
            and percentile_params is not None
            and cfg.randomization
        )

        if use_percentile_rollouts:
            randomizer_eval = registry.get_domain_randomizer_eval(cfg.task)
            percentile_levels = [int(p) for p in percentile_levels]
            percentile_params = np.asarray(percentile_params, dtype=np.float32)
            percentile_params_jax = jnp.asarray(percentile_params)
            rollout_frames = []

            if randomizer_eval is not None:
                batched_eval_env = wrap_for_adv_training(
                    copy.deepcopy(eval_env),
                    param_size=percentile_params.shape[1],
                    episode_length=env_cfg.episode_length,
                    action_repeat=getattr(env_cfg, "action_repeat", 1),
                    randomization_fn=functools.partial(
                        randomizer_eval,
                        dr_range=eval_env.dr_range,
                    ),
                    dr_range_low=jnp.asarray(eval_env.dr_range[0]),
                    dr_range_high=jnp.asarray(eval_env.dr_range[1]),
                )
                jit_batched_reset = jax.jit(batched_eval_env.reset)
                jit_batched_step = jax.jit(
                    lambda state, action, dynamics_params_batch: batched_eval_env.step(
                        state,
                        action,
                        dynamics_params_batch,
                    )
                )

                reset_rng, rng = jax.random.split(rng)
                reset_keys = jax.random.split(reset_rng, len(percentile_levels))
                state = jit_batched_reset(reset_keys)
                batched_trajectory = [jax.device_get(state)]
                reward_batch = np.zeros(len(percentile_levels), dtype=np.float32)

                for _ in range(env_cfg.episode_length):
                    act_rng, rng = jax.random.split(rng)
                    action, _ = jit_policy_fn(state.obs, act_rng)
                    state = jit_batched_step(state, action, percentile_params_jax)
                    state_cpu = jax.device_get(state)
                    batched_trajectory.append(state_cpu)
                    reward_batch += np.asarray(state_cpu.reward, dtype=np.float32)

                reward_list = reward_batch.tolist()

                for batch_index in range(len(percentile_levels)):
                    rollout = _extract_single_trajectory(
                        batched_trajectory,
                        batch_index,
                        len(percentile_levels),
                    )
                    frames_i = eval_env.render(rollout, camera=CAMERAS[cfg.task])
                    rollout_frames.append(frames_i)
            else:
                reward_list = []
                for dyn_params in percentile_params:
                    percentile_env = registry.load(cfg.task, config=env_cfg)
                    percentile_env = _apply_fixed_dynamics_params(
                        percentile_env,
                        randomizer_eval,
                        dyn_params,
                    )
                    jit_reset = jax.jit(percentile_env.reset)
                    jit_step = jax.jit(percentile_env.step)
                    reset_rng, rng = jax.random.split(rng)
                    state = jit_reset(reset_rng)
                    rollout = [jax.device_get(state)]
                    episode_reward = 0.0

                    for _ in range(env_cfg.episode_length):
                        act_rng, rng = jax.random.split(rng)
                        action, _ = jit_policy_fn(state.obs, act_rng)
                        state = jit_step(state, action)
                        rollout.append(jax.device_get(state))
                        episode_reward += float(state.reward)

                    frames_i = eval_env.render(rollout, camera=CAMERAS[cfg.task])
                    rollout_frames.append(frames_i)
                    reward_list.append(episode_reward)

            tile_labels = [
                f"p{p} | R={r:.2f}"
                for p, r in zip(percentile_levels, reward_list)
            ]
            tiled_frames = _tile_frame_sequences(
                rollout_frames,
                grid_cols=4,
                tile_labels=tile_labels,
            )
            reward_array = np.array(reward_list)
            video_path = cfg.work_dir / f"video_{cfg.policy}_{cfg.task}_p0_to_p100.mp4"
            os.makedirs(video_path.parent, exist_ok=True)
            imageio.mimsave(video_path, tiled_frames, fps=fps)

            percentile_reward_log = {
                f'final_eval_reward_p{p}': float(r)
                for p, r in zip(percentile_levels, reward_list)
            }
            wandb.log({
                'final_eval_reward': reward_array.mean(),
                'final_eval_reward_iqm': scipy.stats.trim_mean(reward_array, 0.25),
                'final_eval_reward_std': reward_array.std(),
                'eval_video_percentiles': wandb.Video(video_path, fps=fps, format='mp4'),
                **percentile_reward_log,
            })
        else:
            n_episodes = 10
            jit_reset = jax.jit(eval_env.reset)
            jit_step = jax.jit(eval_env.step)
            reward_list = []
            rollout = []
            rng, eval_rng = jax.random.split(rng)
            rngs = jax.random.split(eval_rng, n_episodes)

            for i in range(n_episodes):
                state = jit_reset(rngs[i])
                if i == 0:
                    rollout = [jax.device_get(state)]
                current_episode_reward = 0
                for _ in range(env_cfg.episode_length):
                    act_rng, rng = jax.random.split(rng)
                    action, _ = jit_policy_fn(state.obs, act_rng)
                    state = jit_step(state, action)
                    if i == 0:
                        rollout.append(jax.device_get(state))
                    current_episode_reward += state.reward
                reward_list.append(float(current_episode_reward))

            reward_array = np.array(reward_list)
            frames = eval_env.render(rollout, camera=CAMERAS[cfg.task])
            video_path = cfg.work_dir / f"video_{cfg.policy}_{cfg.task}.mp4"
            os.makedirs(video_path.parent, exist_ok=True)
            imageio.mimsave(video_path, frames, fps=fps)
            wandb.log({
                'final_eval_reward': reward_array.mean(),
                'final_eval_reward_video': reward_array[0],
                'final_eval_reward_iqm': scipy.stats.trim_mean(reward_array, 0.25),
                'final_eval_reward_std': reward_array.std(),
                'eval_video': wandb.Video(video_path, fps=fps, format='mp4')
            })
if __name__ == "__main__":
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    train()
