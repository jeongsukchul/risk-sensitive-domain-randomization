
**"This repository is for ICML 2026 submission and will be de-anonymized upon acceptance."**
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
  Our GMMVI implementation is based on the code in https://github.com/DenisBless/variational_sampling_methods.
- **BIJX**:
Used for implementing Rational Quadratic Splines in GOFLOW baseline. https://github.com/mathisgerdes/bijx