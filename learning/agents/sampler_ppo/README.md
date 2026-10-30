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
| `scheduling.py` | Risk-temperature scheduler state and bandit helpers |
| `transforms.py` | Bounded diffusion sampler transforms and Jacobian |
| `evaluation.py` | Evaluation grids, return percentile selection, and empirical return estimation |
| `plotting.py`, `distribution_plots.py` | Training figures and animations; figure capture owns the frame and closes its figure |
| `checkpoints.py` | Method-specific sampler state selection and msgpack serialization |
| `training_logging.py`, `training_cost.py` | Buffered logging offsets, transition/update counts, phase timing |

A training update samples dynamics, collects trajectories, updates the observation normalizer and PPO parameters, and then performs the selected sampler update when due. The sampler reuses the collected returns. EPOpt selects the lowest-return trajectory subset before PPO optimization.

Random key splitting, reward scaling/clipping, optimizer order, and update frequencies are part of the reproducibility contract. The structural cleanup preserves them. GMMVI component adaptation remains disabled, and its state is retained in the serialized schema. Flow sampler checkpoint serialization remains unsupported.

Sampler checkpoints use `results/<sampler>/checkpoints/sampler_state_<step>.msgpack` and `sampler_state_latest.msgpack`. Each evaluated step now writes its own file; earlier training epochs overwrote the step-zero path. Policy checkpoints are managed separately by the entry point's callback.

The root README provides a smoke command for installation checks. Runtime
functions used by the toy comparison workflow are retained in
`learning.module.gbs.experimental_losses` and `experimental_trainer`.
