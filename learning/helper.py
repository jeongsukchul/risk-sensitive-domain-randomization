import re
import json
import math
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import os
import numpy as np
import pandas as pd
import datetime
import jax
import jax.numpy as jnp
from termcolor import colored
from PIL import Image, ImageDraw

CONSOLE_FORMAT = [
    ("iteration", "I", "int"),
    ("episode", "E", "int"),
    ("step", "I", "int"),
    ("episode_reward", "R", "float"),
    ("episode_success", "S", "float"),
    ("total_time", "T", "time"),
]

CAT_TO_COLOR = {
    "pretrain": "yellow",
    "train": "blue",
    "eval": "green",
    "results": "magenta",
}

def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """
    Parses a Hydra config. Mostly for convenience.
    """

    # Logic
    for k in cfg.keys():
        try:
            v = cfg[k]
            if v == None:
                v = True
        except:
            pass

    # Algebraic expressions
    for k in cfg.keys():
        try:
            v = cfg[k]
            if isinstance(v, str):
                match = re.match(r"(\d+)([+\-*/])(\d+)", v)
                if match:
                    cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
                    if isinstance(cfg[k], float) and cfg[k].is_integer():
                        cfg[k] = int(cfg[k])
        except:
            pass

    # Convenience
    learning_root = Path(__file__).resolve().parent
    try:
        original_cwd = Path(hydra.utils.get_original_cwd()).resolve()
        if learning_root.is_relative_to(original_cwd):
            base_dir = learning_root
        else:
            base_dir = original_cwd / "learning"
    except Exception:
        base_dir = learning_root

    cfg.work_dir = (
        base_dir
        / "logs"
        / cfg.task
        / str(cfg.seed)
        / cfg.policy
    )
    return cfg


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def load_percentile_dynamics_params(cfg=None, policy=None, work_dir=None, path=None):
    """Load saved percentile dynamics params from a run's models directory."""
    if path is None:
        resolved_policy = policy or getattr(cfg, "policy", None)
        resolved_work_dir = Path(work_dir or getattr(cfg, "work_dir", ""))
        if resolved_policy is None or not str(resolved_work_dir):
            raise ValueError("Either `path` or both `policy` and `work_dir`/`cfg.work_dir` must be provided.")
        path = resolved_work_dir / "models" / f"{resolved_policy}_percentile_dynamics_params_latest.json"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Percentile dynamics params file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    payload["percentile_levels"] = np.asarray(payload["percentile_levels"])
    payload["dynamics_params_percentiles"] = np.asarray(payload["dynamics_params_percentiles"], dtype=np.float32)
    if payload.get("reward_percentiles") is not None:
        payload["reward_percentiles"] = np.asarray(payload["reward_percentiles"], dtype=np.float32)

    return payload


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
            canvas[y0:y0 + fh, x0:x0 + fw] = frame
            if tile_labels is not None and i < len(tile_labels):
                pil_img = Image.fromarray(canvas)
                draw = ImageDraw.Draw(pil_img)
                label = str(tile_labels[i])
                text_x = col * w + 8
                text_y = r * h + 8
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


def _save_percentile_dynamics_params(metrics, work_dir, policy, use_wandb):
    if not isinstance(metrics, dict):
        return

    percentile_levels = metrics.get("final_eval/percentile_levels")
    percentile_params = metrics.get("final_eval/dynamics_params_percentiles")
    reward_percentiles = metrics.get("final_eval/reward_percentiles")

    if percentile_levels is None or percentile_params is None:
        return

    save_dir = make_dir(work_dir / "models")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "policy": policy,
        "percentile_levels": np.asarray(percentile_levels).tolist(),
        "dynamics_params_percentiles": np.asarray(percentile_params).tolist(),
        "reward_percentiles": None if reward_percentiles is None else np.asarray(reward_percentiles).tolist(),
    }

    timestamped_path = os.path.join(
        save_dir,
        f"{policy}_percentile_dynamics_params_{timestamp}.json",
    )
    latest_path = os.path.join(
        save_dir,
        f"{policy}_percentile_dynamics_params_latest.json",
    )

    for path in (timestamped_path, latest_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if use_wandb:
        import wandb

        wandb.save(timestamped_path)
        wandb.save(latest_path)


def cfg_to_group(cfg, return_list=False):
    """
    Return a wandb-safe group name for logging.
    Optionally returns group name as list.
    """
    lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)

def print_run(cfg):
    """
    Pretty-printing of current run information.
    Logger calls this method at initialization.
    """
    prefix, color, attrs = "  ", "green", ["bold"]

    def _limstr(s, maxlen=36):
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def _pprint(k, v):
        print(
            prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs),
            _limstr(v),
        )

    observations = ", ".join([str(v) for v in cfg.obs_shape.values()])
    kvs = [
        ("task", cfg.task_title),
        ("steps", f"{int(cfg.steps):,}"),
        ("observations", observations),
        ("actions", cfg.action_dim),
        ("experiment", cfg.exp_name),
    ]
    w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
    div = "-" * w
    print(div)
    for k, v in kvs:
        _pprint(k, v)
    print(div)

class VideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, cfg, wandb, fps=15):
        self.cfg = cfg
        self._save_dir = make_dir(cfg.work_dir / "eval_video")
        self._wandb = wandb
        self.fps = fps
        self.frames = []
        self.enabled = self._save_dir and self._wandb 

    def record(self, env):
        if self.enabled:
            self.frames.append(env.render())

    def save(self, step, key="videos/eval_video"):
        if self.enabled and len(self.frames) > 0:
            frames = np.stack(self.frames)
            return self._wandb.log(
                {
                    key: self._wandb.Video(
                        frames.transpose(0, 3, 1, 2), fps=self.fps, format="mp4"
                    )
                },
                step=step,
            )


class Logger:
    """Primary logging object. Logs either locally or using wandb."""

    def __init__(self, cfg):
        self._log_dir = make_dir(cfg.work_dir)
        self._model_dir = make_dir(self._log_dir / "models")
        self._save_csv = cfg.save_csv
        self._save_agent = cfg.save_agent
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._eval = []
        print_run(cfg)
        self.project = cfg.get("wandb_project", "none")
        self.entity = cfg.get("wandb_entity", "none")
        if cfg.disable_wandb or self.project == "none" or self.entity == "none":
            print(colored("Wandb disabled.", "blue", attrs=["bold"]))
            cfg.save_agent = False
            cfg.save_video = False
            self._wandb = None
            self._video = None
            return
        os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
        import wandb

        wandb.init(
            project=self.project,
            entity=self.entity,
            name=f"{cfg.task}.tdmpc.{cfg.exp_name}.{cfg.seed}",
            #group=self._group,
            tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
            dir=self._log_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        self._wandb = wandb
        self._video = (
            VideoRecorder(cfg, self._wandb) if self._wandb and cfg.save_video else None
        )

    @property
    def video(self):
        return self._video

    @property
    def model_dir(self):
        return self._model_dir

    def save_agent(self, agent=None, identifier="final"):
        if self._save_agent and agent:
            fp = self._model_dir / f"{str(identifier)}.pt"
            agent.save(fp)
            if self._wandb:
                artifact = self._wandb.Artifact(
                    self._group + "-" + str(self._seed) + "-" + str(identifier),
                    type="model",
                )
                artifact.add_file(fp)
                self._wandb.log_artifact(artifact)

    def finish(self, agent=None):
        try:
            self.save_agent(agent)
        except Exception as e:
            print(colored(f"Failed to save model: {e}", "red"))
        if self._wandb:
            self._wandb.finish()

    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key+":", "blue")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key+":", "blue")} {value:.01f}'
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "blue")} {value}'
        else:
            raise f"invalid log format type: {ty}"

    def _print(self, d, category):
        category = colored(category, CAT_TO_COLOR[category])
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            if k in d:
                pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
        print("   ".join(pieces))

    def log(self, d, category="train"):
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        if self._wandb:
            if category in {"train", "eval", "results"}:
                xkey = "step"
            elif category == "pretrain":
                xkey = "iteration"
            for k, v in d.items():
                if category == "results" and k == "step":
                    continue
                self._wandb.log({category + "/" + k: v}, step=d[xkey])
        if category == "eval" and self._save_csv:
            keys = ["step", "episode_reward"]
            self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
            pd.DataFrame(np.array(self._eval)).to_csv(
                self._log_dir / "eval.csv", header=keys, index=None
            )
        if category != "results":
            self._print(d, category)
