wandb_project="highdimtest2"
task=HumanoidWalk

use_scheduling=true
start_beta=10
end_beta=-20
scheduler_mode="linear"

for seed in 0 1 2 3
do
python run.py policy=gmmppo use_scheduling=true start_beta=$start_beta end_beta=$end_beta scheduler_mode=$scheduler_mode wandb_project=$wandb_project task=$task seed=$seed sampler_update_freq=1
done