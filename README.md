**"This repository is for Neurips 2026 submission and will be de-anonymized upon acceptance."**

# Risk-Sensitive Domain Randomization (RSDR)

This repository contains the official implementation of **Risk-Sensitive Domain Randomization (RSDR)**, a framework for robust Sim2Real transfer in reinforcement learning.

RSDR utilizes **Gaussian Mixture Model Variational Inference (GMMVI)** to learn a curriculum of dynamics parameters. By adjusting a risk-sensitivity coefficient ($\beta$), the sampler can prioritize "worst-case" scenarios (risk-averse) to improve robustness, or "best-case" scenarios (risk-seeking) for exploration.

We also provide highly optimized, JAX-based reimplementations of several state-of-the-art Domain Randomization baselines:
* **GOFLOW** (Flow-based DR via Neural Spline Flows)
* **EPOpt** (CVaR optimization).
* **AutoDR / DORAEMON** (Reference implementations)

All code is built on **JAX**, **Flax**, and **Brax** for high-performance GPU-accelerated training. We use **Mujoco Playground** for simulation of our algorithms.

---

## Installation

We recommend using a Conda environment with **Python 3.11**.

### 1. Create and Activate Environment
```bash
conda create -n rsdr-env python=3.11
conda activate rsdr-env
```
### 2. Install Jax (CUDA 12 Support)
You must install the CUDA-compatible version of JAX first.
```bash
pip install --upgrade "jax[cuda12_pip]" -f [https://storage.googleapis.com/jax-releases/jax_cuda_releases.html](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html)
```
### 3. Install Dependencies & Package
```bash
pip install -r requirements.txt
pip install -e .
```

---
## Usage & Experiments
The training script run.py (or train.py) uses Hydra/OmegaConf for configuration. Below are the commands to reproduce the experiments from the paper.

### 1. Risk-Sensitive Domain Randomization (Ours)
RSDR is implemented as policy=gmmppo. The behavior is controlled by the beta ($\beta$) parameter:
- $\beta<0$ : **Risk-Averse** (Robustness)
- $\beta=0$ : **Risk-Neutral** (Uniform Sampling)
- $\beta>0$ : **Risk-Seeking** (Curriculum Learning)

### Run RSDR Example

```bash
# Beta = -30 (High Robustness)
python train.py policy=gmmppo beta=-30 wandb_project="rsdr-cheetah" task=CheetahRun seed=0

# Beta = 0 (Uniform Baseline)
python train.py policy=gmmppo beta=0 wandb_project="rsdr-cheetah" task=CheetahRun seed=0
```

All GMMPPO modes estimate and log `gmm_kl_to_uniform`. In fixed-beta mode,
`beta` and `beta_used` remain the configured inverse temperature. It also logs
the signed diagnostic `kl_violation = gmm_kl_to_uniform - kl_radius` and the
positive-only `kl_radius_violation`; these metrics do not update fixed beta.
GMMPPO also supports a fixed KL radius relative to the uniform
domain-randomization distribution. In this mode, `beta` is the negative initial
value $\beta_0$.
The default `dual_update_mode=lambda` optimizes $\lambda=-1/\beta$ by projected
dual ascent. The dual signal uses an EMA of the measured KL, initialized at the
requested radius, and then clips the EMA violation symmetrically:
`ema_kl <- dual_ema_decay * ema_kl + (1 - dual_ema_decay) * estimated_KL`;
`clip_value <- kl_violation_clip * kl_radius`;
`dual_violation <- clip(ema_kl - kl_radius, -clip_value, clip_value)`.
With `kl_violation_clip=0.25`, the actual bound is 25 percent of the requested
radius. It additionally logs `dual_lambda`, `dual_beta`,
`resulting_beta`, `beta_update_delta`, `dual_update_mode_beta`, `kl_radius`,
`dual_kl_ema`, `kl_violation_raw`, `kl_violation_ema`, and the clipped
`kl_violation` used by the update. The `kl_violation_clip` metric is the
effective bound, while `kl_violation_clip_ratio` is the configured fraction.

```bash
cd learning
python run.py policy=gmmppo fixed_radius=true kl_radius=0.1 beta=-30 \
  dual_lr=0.001 wandb_project="rsdr-cheetah" task=CheetahRun seed=0
```

The alternative `dual_update_mode=beta` applies the deliberately naive direct
update
`beta <- clip(beta + dual_lr * dual_violation)`. It uses the same EMA and
violation clipping and does not use the chain-rule factor `1 / beta**2`.

```bash
python run.py policy=gmmppo fixed_radius=true dual_update_mode=beta \
  kl_radius=0.1 beta=-1 dual_lr=0.1 dual_beta_min=-100 \
  dual_beta_max=-0.001 task=AcrobotSwingup seed=0
```

`fixed_radius=true` cannot be combined with `use_scheduling=true`. The optional
projection bounds are `dual_lambda_min`/`dual_lambda_max` in lambda mode and
`dual_beta_min`/`dual_beta_max` in direct-beta mode. Both modes continue to log
lambda using the diagnostic conversion $\lambda=-1/\beta$.

For two-dimensional tasks, evaluation uses an equal-area, cell-centered
128-by-128 grid (`eval_grid_size_2d=128`, or 16,384 parameter settings). GMMPPO
constructs the empirical target with
`log_softmax(beta * (estimated_return - center))` and compares it with the
grid-normalized GMM density. Evaluation logs:

The empirical target return estimator mirrors one training sampler update.
Each of `empirical_num_rollouts=10` independent estimates starts from a fresh
reset and collects
`batch_size * num_minibatches / num_envs` blocks of `unroll_length`
transitions. By default it uses the original signed rewards
(`empirical_clip_negative_rewards=false`) and applies
`reward_scale_for_sampler`. Set the option to `true` to reproduce the training
sampler's per-step nonnegative clipping. The estimates are averaged to
construct the target; split-half disagreement and logit standard errors are
logged as `eval/empirical_target_split_*` and
`eval/empirical_target_logit_se_*`. Full-episode deterministic policy
evaluation remains separate and unchanged.

- `eval/empirical_reverse_kl_q_to_target`
- `eval/empirical_forward_kl_target_to_q`
- `eval/empirical_js_divergence`
- `eval/empirical_total_variation`
- `eval/empirical_hellinger_distance`
- `eval/empirical_overlap`
- `eval/empirical_target_reverse_kl_to_uniform`
- `eval/empirical_weighted_target_reverse_kl_to_uniform` (explicit alias)
- `eval/empirical_sampler_reverse_kl_to_uniform`
- centered log-sum-exp, target log normalizer, entropies
- target and sampler ESS fractions and expected returns

Because all grid cells have equal area, the uniform reference has mass
$1/16384$ per cell. Thus the target KL-radius diagnostic is
$D_{\mathrm{KL}}(\hat p_{\mathrm{target}}\|U)
=\log(16384)-H(\hat p_{\mathrm{target}})$. In fixed-radius mode, evaluation
also logs `eval/empirical_kl_radius` and the signed `*_kl_radius_residual`
(`KL - radius`) plus the nonnegative `*_kl_radius_violation` for both the
weighted target and learned sampler.

The grid, estimated returns, normalized target/sampler masses, log masses, and
beta are saved in `results/GMM/empirical_target_sampler_<step>.npz`. The
artifact also stores both KL-to-uniform estimates and the configured radius,
together with a three-panel comparison PNG.

Training metrics can be recorded more frequently than evaluation with
`training_log_freq`. The value is measured in training updates, is independent
of `sampler_update_freq` and `num_evals`, and `0` disables the extra records.
For example, `training_log_freq=5` records every fifth update. These scalar
records are buffered during the compiled training epoch and emitted before the
next evaluation, avoiding a device-to-host synchronization at every logging
point.

#### If you want to use diffusion based parameterization use (DIS-LV)
```bash
python train.py policy=gbsppo beta=-30 wandb_project="rsdr-cheetah" task=CheetahRun seed=0
```
### Run GOFLOW Example

```bash
python train.py policy=flowppo alpha=1 gamma=0.5 wandb_project="rsdr-cheetah" task=CheetahRun seed=0
```
### Run DORAEMON Example

```bash
python train.py policy=doraemonppo success_threshold=.8 success_rate_condition=.8 wandb_project="rsdr-cheetah" task=CheetahRun seed=0
```
### Run AutoDR Example

```bash
python train.py policy=adrppo success_threshold=.8  wandb_project="rsdr-cheetah" task=CheetahRun seed=0
```
### Run EPOpt Example

```bash
python train.py policy=epopt epsilon=.4  wandb_project="rsdr-cheetah" task=CheetahRun seed=0
```
---
## Acknowledgements & Credits
This codebase leverages several open-source libraries. We explicitly thank the authors of:
- **Mujoco Playground**:
Simulation Environment. https://github.com/google-deepmind/mujoco_playground

- **Google Brax**:
Our PPO implementation is based on Brax. We modified it to support Asymmetric Actor-Critic  (conditioning value functions on latent parameters $\xi$) to support privileged information during training. https://github.com/google/brax

- **GMMVI Code**:
  Our GMMVI implementation is based on the code in https://github.com/DenisBless/variational_sampling_methods (with few modification)
- **DIS-LV Code**:
  Our DIS-LV implementation is based on the code in https://github.com/juliusberner/sde_sampler (with jax based reimplementation)
- **BIJX**:
Used for implementing Rational Quadratic Splines in GOFLOW baseline. https://github.com/mathisgerdes/bijx
