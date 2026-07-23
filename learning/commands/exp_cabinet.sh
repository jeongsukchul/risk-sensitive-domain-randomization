wandb_project="cabinet-sampler5"
for seed in 10 11 12 13 14 15
    do
    # python run.py policy=flowppo beta=1 gamma=1 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=20
    # python run.py policy=gmmppo beta=-20 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gmmppo beta=-10 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gmmppo beta=-30 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=ppo wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=ppo wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gmmppo start_beta=5 end_beta=-30 use_scheduling=true wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gbsppo start_beta=5 end_beta=-30 use_scheduling=true wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # for gamma in 0 1 2
    # do
    python run.py policy=flowppo beta=1 gamma=1 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=flowppo beta=2 gamma=$gamma wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=flowppo beta=5 gamma=$gamma wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # done
    # python run.py policy=gmmppo beta=-5 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gbsppo beta=-5 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gmmppo beta=-10 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gbsppo beta=-10 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gmmppo beta=-20 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gbsppo beta=-20 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gmmppo beta=-30 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gbsppo beta=-30 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gmmppo beta=0 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gbsppo beta=0 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gmmppo beta=5 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1
    # python run.py policy=gbsppo beta=5 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=1

    # python run.py policy=doraemonppo success_threshold=0.5 success_rate_condition=0.5 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=20
    # python run.py policy=doraemonppo success_threshold=0.5 success_rate_condition=0.8 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=20
    # python run.py policy=doraemonppo success_threshold=0.75 success_rate_condition=0.5 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=20
    python run.py policy=doraemonppo success_threshold=0.75 success_rate_condition=0.8 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=20
    # python run.py policy=adrppo success_threshold=0.5 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=20
    # python run.py policy=adrppo success_threshold=0.75 wandb_project=$wandb_project task=PandaOpenCabinet seed=$seed sampler_update_freq=20
    
    done
# wandb_project="franka-sampler"
# for seed in 0 1 2 3 4 5 6 7 8
#     do
#     python run.py policy=ppo wandb_project=$wandb_project task=PandaPickCubeOrientation seed=$seed impl=warp sampler_update_freq=1
#     # python run.py policy=gmmppo beta=-30 wandb_project=$wandb_project task=PandaPickCubeOrientation seed=$seed impl=warp sampler_update_freq=1
#     # python run.py policy=gmmppo beta=-20 wandb_project=$wandb_project task=PandaPickCubeOrientation seed=$seed impl=warp sampler_update_freq=1
#     # python run.py policy=gmmppo beta=-10 wandb_project=$wandb_project task=PandaPickCubeOrientation seed=$seed impl=warp sampler_update_freq=1
#     done