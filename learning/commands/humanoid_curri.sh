wandb_project="highdimtest2"
task=HumanoidWalk

success_threshold=.6
success_rate_condition=.6
trust_region=0.005
for seed in 3 4 5 
do
    python run.py policy=adrppo wandb_project=$wandb_project  task=$task success_threshold=$success_threshold seed=$seed
    # python run.py policy=doraemonppo wandb_project=$wandb_project success_threshold=$success_threshold task=$task trust_region=$trust_region success_rate_condition=$success_rate_condition seed=$seed
    # pytho nrun.py policy=flowppo wandb_project=$wandb_project  task=$task beta=1. seed=$seed gamma=.5
done

