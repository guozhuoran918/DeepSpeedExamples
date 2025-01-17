
#! /bin/bash


RANK=0
WORLD_SIZE=1

python pretrain_t5.py \
       --batch-size 4 \
       --train-iters 50 \
       --eval-iters 1 \
       --eval-interval 50 \
       --eval-batch-size 4 \
       --log-interval 1 \
       --save checkpoints/t5_large \
       --load checkpoints/t5_large \
       --lazy-loader \
       --max_len 512 \
       --summary_len 50 \
       --train-data-path t5_news_summary/t5_train.csv \
       --val-data-path t5_news_summary/t5_val.csv \
       --test-data-path t5_news_summary/t5_test.csv\
       --cache-dir cache \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --seed 42 \
       --checkpoint-activations \
       --fp16


set +x
