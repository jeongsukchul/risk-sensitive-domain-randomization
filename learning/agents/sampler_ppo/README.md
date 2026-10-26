# Sampler PPO internals

`train.py::train` is the public training entry point. It constructs environments and networks, defines compiled rollout/update functions, and runs the training/evaluation loop. `TrainingState` retains its existing fields for checkpoint compatibility.

| Module | Responsibility |
| --- | --- |
| `train.py` | Environment rollout, PPO/sampler update ordering, device replication, training orchestration |
| `networks.py` | Actor/critic networks, action distribution, sampler initialization, inference |
| `losses.py` | PPO objective and generalized advantage estimation |
| `sampler_losses.py` | GMMVI MORE update and flow sampler objective |
| `adr.py` | AutoDR boundary sampling and adaptive domain bounds |
| `doraemon.py` | Factorized beta sampler and constrained distribution updates |
| `distributions.py` | Generic distributions and compatibility exports for the original imports |
| `dual_optimization.py` | Fixed-radius temperature updates and KL estimation |
| `scheduling.py` | Risk-temperature scheduler state and bandit helpers |
| `transforms.py` | Bounded diffusion sampler transforms and Jacobian |
| `evaluation.py` | Evaluation grids, return percentile selection, empirical return estimation, target/sampler artifacts |
| `plotting.py`, `distribution_plots.py` | Diagnostic figures and animations; figure capture owns the frame and closes its figure |
| `checkpoints.py` | Method-specific sampler state selection and msgpack serialization |
| `training_logging.py`, `training_cost.py` | Buffered logging offsets, transition/update counts, phase timing |

A training update samples dynamics, collects trajectories, updates the observation normalizer and PPO parameters, and then performs the selected sampler update when due. The sampler reuses the collected returns. EPOpt selects the lowest-return trajectory subset before PPO optimization. Fixed-radius GMMVI additionally updates the dual temperature; fixed-beta runs only log the KL diagnostic.

Random key splitting, reward scaling/clipping, optimizer order, and update frequencies are part of the reproducibility contract. The structural cleanup preserves them. GMMVI component adaptation remains disabled, and its state is retained in the serialized schema. Flow sampler checkpoint serialization remains unsupported.

Sampler checkpoints use `results/<sampler>/checkpoints/sampler_state_<step>.msgpack` and `sampler_state_latest.msgpack`. Each evaluated step now writes its own file; earlier training epochs overwrote the step-zero path. Policy checkpoints are managed separately by the entry point's callback.

Validation in the existing `jax_flow` environment completed 37 agent checks and 4 configuration checks, all eight training methods on Acrobot, the fixed-radius extension, the Warp backend, and all six manuscript environments' JIT reset/step checks. Test source files were removed from the release checkout after validation. The root README retains a smoke command for installation checks.

The validation also corrected Acrobot's nominal parameter vector to match its two-dimensional domain and made per-epoch diagnostic plots derive their own evaluation key when the initial evaluation is skipped. Runtime functions previously placed in test-named modules are retained as `learning.module.gbs.experimental_losses` and `experimental_trainer` for the toy comparison workflow.

During cleanup, 37 agent tests passed. Fixed-seed Acrobot smoke runs for GMMVI, UDR, and fixed-radius GMMVI produced identical policy parameters to the pre-cleanup runs (maximum absolute difference zero), with matching evaluation returns and empirical target artifacts. Sampler checkpoint schemas were unchanged; the largest observed sampler-state difference was 3.81e-6, within the float32 comparison tolerance. AutoDR sampling/updates and DORAEMON restoration/optimization/skip paths were compared directly against the previous implementations. These checks establish small-run equivalence; they do not replace full-budget benchmark evaluation.
