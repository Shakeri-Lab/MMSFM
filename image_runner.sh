#!/bin/bash

CMDS="train eval plot"

TRAINARGS=\
"\
    --dataname cifar10 \
    --size 32 \
    --window_size 2 \
    --spline cubic \
    --monotonic \
    --score_matching \
    --method exact \
    --zt 0 1 2 3 \
    --progression 2 4 6 8 \
    --batch_size 16 \
    --accum_steps 2 \
    --n_steps 20 \
    --n_epochs 10 \
    --sigma 0.15 \
    --t_sampler stratified \
    --diff_ref miniflow \
    --lr 1e-8 1e-4 \
    --total_iters_inc 0.5 \
    --save_interval 2 \
    --ckpt_interval 2 \
    --checkpoint ckpt
"

INFERENCEARGS=\
"\
    --n_infer 10 \
    --t_infer 9 \
"
# --load_models

PLOTARGS=\
"\
    --scale 4 \
"

WANDBARGS=\
"\
    --entity <YOUR ENTITY HERE> \
    --project <YOUR PROJECT HERE> \
    --run_name cifar10
"
# --resume
# --group
# --no_wandb

MISCARGS=\
"\
    --outdir cifar10
"
# --nogpu

ARGS="$CMDS $TRAINARGS $INFERENCEARGS $PLOTARGS $WANDBARGS $MISCARGS"

echo $(which python)
echo python scripts/images/images_main.py $ARGS

python scripts/images/images_main.py $ARGS

