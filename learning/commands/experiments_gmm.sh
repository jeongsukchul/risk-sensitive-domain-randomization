
for beta in -10 -5 -2 -1 0 1 2 5 10
    do
    python train.py policy=gmmppo beta=$beta
    done