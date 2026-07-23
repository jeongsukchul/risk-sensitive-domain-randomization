task=AcrobotSwingup

# python eval_nonstationary.py \
#   --task $task \
#   --policy ppo \
#   --beta -10 \
#   --dist_type random_walk \
#   --seeds 0 3 4 5 7 12 13 100 101 
python eval_nonstationary.py \
  --task $task \
  --policy ppo \
  --beta -10 \
  --dist_type langevin \
  --seeds 0 3 4 5 7 12 13 100 101 
# python eval_nonstationary.py \
#   --task $task \
#   --policy gmmppo \
#   --beta -10 \
#   --dist_type random_walk \
#   --seeds 2 10 20 21 22 23 100 101
# python eval_nonstationary.py \
#   --task $task \
#   --policy gmmppo \
#   --beta -10 \
#   --dist_type langevin \
#   --seeds 2 10 20 21 22 23 100 101

# python eval_nonstationary.py \
#   --task $task \
#   --policy gmmppo \
#   --beta -20 \
#   --dist_type random_walk \
#   --seeds 1 4 5 10 12 100 101 20 22
# python eval_nonstationary.py \
#   --task $task \
#   --policy gmmppo \
#   --beta -20 \
#   --dist_type langevin \
#   --seeds 1 4 5 10 12 100 101 20 22


# python eval_nonstationary.py \
#   --task $task \
#   --policy gmmppo \
#   --beta -30 \
#   --dist_type random_walk \
#   --seeds 0 4 10 20 22 100 101
# python eval_nonstationary.py \
#   --task $task \
#   --policy gmmppo \
#   --beta -30 \
#   --dist_type langevin \
#   --seeds 0 4 10 20 22 100 101
