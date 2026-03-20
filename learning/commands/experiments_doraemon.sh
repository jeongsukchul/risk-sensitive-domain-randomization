wandb_project=cheetah-sampler2
task=CheetahRun
success_threshold=.7

# for trust_region in 0.005 0.01 0.05
# do
# for success_rate_condition in .5 0.75 0.8
# do
#     for seed in 0 1 2 3 4 5
#     do
#        python run.py policy=doraemonppo task=$task wandb_project=$wandb_project success_threshold=$success_threshold success_rate_condition=$success_rate_condition seed=$seed trust_region=$trust_region
#     done
# done
# done

# wandb_project=cartpole-sampler6
# task=CartpoleSwingupSparse
# success_threshold=.7

# for trust_region in 0.005 0.01 0.05
# do
# for success_rate_condition in .5 0.75 0.8
# do
#     for seed in 0 1 2 3 4 5
#     do
#        python run.py policy=doraemonppo task=$task wandb_project=$wandb_project success_threshold=$success_threshold success_rate_condition=$success_rate_condition seed=$seed trust_region=$trust_region
#     done
# done
# done
wandb_project=cartpole-sampler10
task=CartpoleSwingup
success_threshold=.7

for trust_region in 0.01 0.005
do
for success_rate_condition in 0.5 0.7 0.8
do
    for seed in 1 2 3 4
    do
       python run.py policy=doraemonppo task=$task wandb_project=$wandb_project success_threshold=$success_threshold success_rate_condition=$success_rate_condition seed=$seed trust_region=$trust_region
    done
done
done

