#! /bin/bash


RANK=0
WORLD_SIZE=1

python pretrain_tf5.py \
 
       --batch-size 2\
       --train-iters 10 \
       --save checkpoints/tf5_large \
       --load checkpoints/tf5_large \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --max_len 512 \
       --summary_len 150 \
       --train_data_path tf5_news_summary/tf5_train.csv \
       --val_data_path tf5_news_summary/tf5_val.csv \
       --test_data_path tf5_news_summary/tf5_test.csv \
       --cache-dir cache \      
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16


set +x
