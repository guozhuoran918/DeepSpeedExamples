#! /bin/bash


RANK=0
WORLD_SIZE=1

python pretrain_t5.py \
       --batch-size 2 \
       --train-iters 10 \
       --save checkpoints/t5_large \
       --load checkpoints/t5_large \
       --resume-dataloader \
       --lazy-loader \
       --max_len 512 \
       --summary_len 150 \
       --train-data-path t5_news_summary/t5_train.csv \
       --val-data-path t5_news_summary/t5_val.csv \
       --test-data-path t5_news_summary/t5_test.csv\
       --cache-dir cache \
       --distributed-backend gloo \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16


set +x
