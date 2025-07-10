#!/bin/bash

TRAINARGS=\
"\
    --dataname sg \
    --scaler_type minmax \
    --flowmatcher sb \
    --agent_type triplet \
    --t_sampler stratified \
    --diff_ref miniflow \
    --window_size 2 \
    --spline cubic \
    --monotonic \
    --modelname mlp \
    --modeldepth 2 \
    --method exact \
    --batch_size 64 \
    --n_steps 200 \
    --n_epochs 5 \
    --sigma 0.15 \
    --lr 1e-4 \
    --w_len 16 \
    --zt 0 1 2 3 4 5 6 \
    --hold_one_out 5
"

INFERENCEARGS=\
"\
    --n_infer 100 \
    --t_infer 100
"

EVALARGS=\
"\
    --eval_method nearest \
    --eval_zt_idx 5 \
    --reg 0.1
"

PLOTARGS=\
"\
    --plot_d1 0 \
    --plot_d2 1 \
    --plot_n_background 100 \
    --plot_n_highlight 15 \
    --plot_n_pairs 10 \
    --plot_n_trajs 5 \
    --plot_n_snaps 11 \
    --plot_interval 200 \
    --plot_fps 5
"

WANDBARGS=\
"\
    --entity <YOUR ENTITY HERE> \
    --project <YOUR PROJECT HERE> \
    --run_name sg
"
    #--no_wandb

MISCARGS=\
"\
    --outdir sg
"

ARGS="$TRAINARGS $INFERENCEARGS $EVALARGS $PLOTARGS $WANDBARGS $MISCARGS"

echo $(which python)
echo python scripts/main.py $ARGS

python scripts/main.py $ARGS
