"""Persist the configuration actually used for a training run."""

from pathlib import Path
import shutil

from omegaconf import OmegaConf
import wandb

from learning.helper import make_dir


def save_configs_to_wandb_and_local(cfg, work_dir, upload=True):
    """Save defaults and resolved overrides, optionally uploading both to W&B."""
    cfg_dir = Path(make_dir(Path(work_dir) / "cfg"))
    default_path = Path(__file__).with_name("config.yaml")
    shutil.copyfile(default_path, cfg_dir / "config.yaml")
    resolved_path = cfg_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, resolved_path, resolve=True)

    if upload and cfg.use_wandb:
        artifact = wandb.Artifact("config", type="config")
        artifact.add_file(str(cfg_dir / "config.yaml"))
        artifact.add_file(str(resolved_path))
        wandb.log_artifact(artifact)
