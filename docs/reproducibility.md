# Reproducibility and implementation notes

This guide records the behavior of the submitted code and its relationship to the local final manuscript, `RS-DR_final.tex`. The TeX source is a local review input and is not tracked by this repository. The README contains the installation and main training commands.

## Manuscript alignment

The manuscript title is **Distributionally Robust Domain Randomization with Learned Risk-Sensitive Dynamics Samplers**. The main method uses a fixed inverse temperature `beta` and a reverse-KL GMMVI sampler. This is an entropic-risk/KL-penalized surrogate; the fixed-temperature implementation does not enforce a prescribed KL radius as the policy changes.

The following distinctions matter when comparing newly generated results with the manuscript:

| Item | Manuscript | Current implementation / action |
| --- | --- | --- |
| Cheetah floor friction | `[0.01, 3.5]` | `cheetah.py::Run.dr_range` uses `[0.1, 3.0]` |
| Cheetah torso mass | `[0.1, 23.0]` | `cheetah.py::Run.dr_range` uses `[0.8, 20.0]` |
| 2D evaluation resolution | 64×64 = 4,096 settings | Default 128×128; sweep scripts explicitly select `eval_grid_size_2d=64` |
| Higher-dimensional evaluation | 16,384 uniform settings | Four batches of 4,096 after each training epoch; initial evaluation uses one batch |
| Training seeds | 10 independent seeds | New sweep scripts default to 0–9; older exploratory scripts use other subsets |
| AutoDR/DORAEMON success threshold | 0.7 for DM Control, 0.5 for Panda | Shared default is 0.6; `baselines.sh` explicitly selects task-specific manuscript thresholds |
| Fixed-radius dual update | Additional experiment / extension | Current shared default is `dual_update_mode=beta`; select `lambda` explicitly for the standard dual parameterization |

The code cleanup preserves the existing dynamics ranges and training objectives. The Cheetah range mismatch needs reconciliation against the original experiment configuration before the code and manuscript can be described as an exact match. Changing those ranges would change the experiment, so the cleanup does not silently substitute manuscript values.

The other five main tasks' domain ranges can be inspected in their `dr_range` properties. Mass, damping, and actuator entries in Panda tasks include multiplicative scales; consult the randomizer functions for the physical interpretation of each coordinate.

## Runtime environment

The final validation used the existing Conda environment `jax_flow`, without upgrading or replacing its packages. Its principal versions are:

| Package | Installed version |
| --- | --- |
| Python | 3.12.12 |
| JAX / jaxlib | 0.9.0.1 / 0.9.0.1 |
| Flax | 0.12.2 |
| Brax | 0.14.0 |
| MuJoCo / mujoco-mjx | 3.4.0 / 3.4.0 |
| MuJoCo Playground (`playground`) | 0.0.5 |
| Warp | 1.11.1 |
| Optax | 0.2.6 |
| Orbax Checkpoint | 0.11.31 |
| Hydra | 1.3.2 |
| NumPy | 2.4.0 |
| bijx | 1.1.3 |

Earlier structural equivalence checks also passed in the existing Python 3.11 `rsdr-env` environment. The current requirements pin `warp-lang==1.10.1`, which differs from `jax_flow`'s installed 1.11.1. The dependency list remains a runtime specification rather than a lockfile of either validation environment or the original paper. Archive `python -m pip freeze` alongside final benchmark runs to record the full environment.

## Validation

The final checks in `jax_flow` completed before the release test-file cleanup:

- 41 regression checks passed: 37 agent checks and 4 configuration-persistence/domain-consistency checks.
- All eight methods completed a GPU Acrobot training update, evaluation, checkpoint/result export, and configuration/cost recording: GMMVI, DIS-LV, UDR, nominal PPO, EPOpt, AutoDR, DORAEMON, and GOFLOW.
- Fixed-radius GMMVI with the lambda update and UDR with the Warp backend also completed their smoke runs.
- JIT reset/step checks passed for AcrobotSwingup, CartpoleSwingup, CheetahRun, WalkerWalk, PandaPickCubeOrientation, and PandaOpenCabinet. State/reward finiteness, observation shapes, dynamics dimensions, and PPO configuration serialization were checked.
- Restored-policy random-walk evaluation and Langevin evaluation against a reference GMM sampler completed.
- Saved policy parameters and evaluation returns were finite. Transition/update counters, sampler checkpoints, and effective configuration artifacts were checked.

The training smoke configuration used 32 training environments, batch size 32, 256 requested transitions, one PPO minibatch/update, unroll length 8, a 4×4 evaluation grid, and two empirical return estimates. EPOpt collected additional transitions before filtering. DIS-LV used four diffusion steps for this execution check. These checks do not establish full-budget benchmark performance, full-budget manipulation training, fresh-install compatibility, W&B upload, or camera video rendering.

Validation identified and corrected two execution issues: Acrobot's nominal dynamics vector had three entries for a two-dimensional randomization domain; per-epoch AutoDR/DORAEMON/GOFLOW/DIS-LV plots relied on an initial-evaluation key that was undefined when `num_evals=1`. The plotting key is now derived from the current evaluation key without consuming the training RNG stream.

After validation, all 16 project source paths ending in `test.py` were removed. Fourteen were tests or legacy drivers. Two modules containing runtime functions were preserved as `learning/module/gbs/experimental_losses.py` and `experimental_trainer.py`, and the toy comparison imports were updated. The normal training/smoke entry points remain available in the README. Experimental notebooks and upstream target examples may require additional research dependencies and local path selection.

## Evaluation protocol and saved results

For 2D tasks, `eval_grid_size_2d=N` selects an equal-area, cell-centered N×N grid. The full-episode evaluator uses a deterministic policy, averages each setting over 10 evaluation seeds, and reports aggregate and lower-tail performance. For higher-dimensional tasks, evaluation after a training epoch combines four independently sampled uniform batches, each containing 4,096 dynamics settings.

`results/<sampler>/rewards_<step>.npy` contains evaluation returns. The final return array is also saved as `reward_1d_<step>.npy`. Adaptive methods save samples as `samples_in_sampler_<step>.npy`; density-capable methods also save `logq_in_sampler_<step>.npy`. The latest available sampler state is written under `results/<sampler>/checkpoints/sampler_state_latest.msgpack` for methods supported by the trainer's serialization path. Step-specific sampler checkpoints preserve each evaluated step; the previous trainer overwrote the step-zero sampler file. Flow state saving is not implemented by that path.

`learning/evaluation.ipynb`, `learning/heatmap_drawing.ipynb`, and `learning/compare_eval_results.py` provide analysis workflows and may require selecting local run paths.

### Nonstationary dynamics

The standalone evaluator restores saved policy parameters and uses saved PPO/environment settings when available. It supports uniform, beta, random-walk, jump-diffusion, and GMM-score-based Langevin dynamics. It defaults to 4,096 parameter trajectories and 10 evaluation seeds; evaluation counts can be reduced for smoke checks.

```bash
# Evaluate a trained UDR policy under a random walk.
python -m learning.eval_nonstationary --task AcrobotSwingup --policy ppo \
  --seeds 0 --dist_type random_walk --rw_scale 0.01

# Apply the same reference RSDR sampler as a Langevin attack on a UDR policy.
python -m learning.eval_nonstationary --task AcrobotSwingup --policy ppo \
  --seeds 0 --dist_type langevin --drift 0.05 --rw_scale 0.001 \
  --sampler_checkpoint logs/AcrobotSwingup/0/gmmppo/beta=-10/results/GMM/checkpoints/sampler_state_latest.msgpack
```

For GMMVI policies, Langevin evaluation defaults to that run's own sampler checkpoint. To compare methods against the same attack, supply the same `--sampler_checkpoint`. Parameter drift and noise are scaled coordinate-wise by the domain width and clipped to the selected bounds. Langevin drift uses the normalized score of the learned GMM. These procedures are evaluation attacks; they do not train another adversary.

Use `--run_dir /absolute/path/to/run --seeds 0` to select an exact run, including fixed-radius runs or other custom output locations. The run directory must contain `models/<policy>_params_latest.pkl`. Paths inferred from `--log_root` default to `logs/<task>/<seed>/<policy>/<sampler settings>`.

Optional `--sweep_config learning/sweep_config.yaml` selects the archived drift/noise sweep; without it the CLI evaluates only the supplied values. `--no-non_stationary` holds the generated initial dynamics fixed. Individual rewards and metrics are saved to the run's `eval_results/`; per-seed results and mean/std summaries are saved under `logs/<task>/averaged_results/`, or the selected run's `eval_results/` when `--run_dir` is used. The stationary sweep scripts do not automatically run this additional evaluation.

### Empirical target and sampler diagnostics

GMMVI evaluation also estimates how closely the learned sampler tracks its policy-induced tilted target. These diagnostics use sampler-matched rollout estimates rather than the separate full-episode deterministic evaluation:

- `empirical_num_rollouts=10` independent estimates start from fresh resets.
- Each estimate collects `batch_size * num_minibatches / num_envs` blocks of `unroll_length` transitions using the stochastic policy.
- Rewards are averaged and multiplied by the environment's `reward_scale_for_sampler`.
- `empirical_clip_negative_rewards=false` retains signed rewards; set it to `true` to match the training sampler's per-step nonnegative clipping.
- The target mass is `softmax(beta * estimated_return)`. Centered log-sum-exp is computed separately for the normalizer diagnostic. The learned GMM log density is normalized over the same grid.

Logged comparisons include reverse and forward KL, Jensen–Shannon divergence, total variation, Hellinger distance, overlap, entropies, ESS fractions, and target/sampler expected returns. Split-half disagreement and logit standard errors are logged under `eval/empirical_target_split_*` and `eval/empirical_target_logit_se_*`.

The target's reverse KL to uniform is `log(N*N) - H(target_mass)`. Both `eval/empirical_target_reverse_kl_to_uniform` and its explicit alias `eval/empirical_weighted_target_reverse_kl_to_uniform` report it. `eval/empirical_sampler_reverse_kl_to_uniform` reports the grid-based sampler diagnostic. These finite-grid values differ from the sample-based KL estimate used for dual updates.

`results/GMM/empirical_target_sampler_<step>.npz` stores the grid, estimated returns, normalized target/sampler masses and log masses, beta, and KL/radius diagnostics. A comparison PNG is saved alongside it. Fixed-radius runs additionally report signed radius residuals (`KL - radius`) and nonnegative violations.

## Fixed-KL-radius extension

Enable the adaptive-temperature extension explicitly:

```bash
python -m learning.run task=AcrobotSwingup policy=gmmppo \
  fixed_radius=true dual_update_mode=lambda kl_radius=0.1 beta=-30 \
  dual_lr=0.001 use_wandb=false save_video=false eval_grid_size_2d=64
```

Here `beta` is the initial negative inverse temperature. `dual_update_mode=lambda` optimizes `lambda=-1/beta` with projected dual ascent. The update uses an EMA of the measured KL, initialized at the requested radius:

```text
ema_kl = dual_ema_decay * ema_kl + (1 - dual_ema_decay) * estimated_kl
clip_bound = kl_violation_clip * kl_radius
violation = clip(ema_kl - kl_radius, -clip_bound, clip_bound)
```

`kl_violation_clip` is a fraction of the requested radius. For example, `kl_violation_clip=0.25` permits a clipped dual signal of at most 25% of the radius. `dual_lambda_min` and `dual_lambda_max` set projection bounds; the initial `-1/beta` must lie inside them.

The alternative `dual_update_mode=beta` performs the deliberately naive direct update `beta = clip(beta + dual_lr * violation)`, with negative projection bounds `dual_beta_min` and `dual_beta_max`. It uses the same EMA and clipping and omits the chain-rule factor `1 / beta**2`.

```bash
python -m learning.run task=AcrobotSwingup policy=gmmppo \
  fixed_radius=true dual_update_mode=beta kl_radius=0.1 beta=-1 \
  dual_lr=0.1 dual_beta_min=-100 dual_beta_max=-0.001 \
  use_wandb=false save_video=false eval_grid_size_2d=64
```

Fixed-radius mode requires `policy=gmmppo`, negative initial beta, and `use_scheduling=false`. It logs `dual_lambda`, `dual_beta`, `resulting_beta`, `beta_update_delta`, `dual_kl_ema`, and raw, EMA, and clipped KL violations. `beta_used` identifies the coefficient used by the sampler update. In fixed-beta mode, the logged KL diagnostics do not update beta.

## Training logs and computational cost

`training_log_freq` is measured in policy-batch updates. A value of `0` disables extra training records; `5` records every fifth update. Records are buffered during the compiled epoch and emitted before evaluation, avoiding a synchronization at every logging point. This setting is independent of `sampler_update_freq` and `num_evals`.

Training always records actual environment transitions, outer PPO-batch updates, minibatch optimizer steps, sampler update events, and total wall-clock time as `training/*` metrics and in `computational_cost.csv`:

- EPOpt transitions include oversampling before worst-return filtering.
- `training/policy_updates` counts complete PPO batches.
- `training/policy_optimizer_steps` counts individual minibatch gradient steps.
- Sampler update counts reflect the method, update frequency, and inner iterations.

For dedicated timing comparisons, set `profile_training_cost=true` for every compared method. It inserts synchronized rollout/update timing boundaries and supports one local device. Ordinary runs retain counters and total wall-clock time but report `NaN` for separate phase timings. Phase times exclude JIT compilation; total wall-clock time includes compilation, evaluation, and checkpoint work inside the trainer, and excludes the final post-training video rendering in the entry point.

```bash
TASKS="CheetahRun" SEEDS="0 1 2" BETAS="-10" \
  bash learning/commands/reproduce.sh profile_training_cost=true \
  use_wandb=false save_video=false
TASKS="CheetahRun" SEEDS="0 1 2" POLICIES="ppo epoptppo" \
  bash learning/commands/baselines.sh profile_training_cost=true epsilon=0.4 \
  use_wandb=false save_video=false
```

Use identical PPO hyperparameters, hardware, backend, evaluation resolution, and profiling settings for comparisons. Keep timing runs in separate `work_dir` locations from benchmark runs so output files do not overwrite previous results.
