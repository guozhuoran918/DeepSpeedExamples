

import json
import os
import pandas as pd
import numpy as np
import torch
from torch.multiprocessing import Lock
from torch.utils.data import Dataset

import mpu
from data_utils.samplers import DistributedBatchSampler
from transformers import T5Tokenizer

def make_tf5_dataloaders(args):

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    def make_data_loader_(dataset_path):
        # Build the dataset.
        dataframe= pd.read_csv(dataset_path)
        dataset = T5Dataset(dataframe=dataframe, tokenizer=tokenizer,
                            source_len =args.max_len, 
                            summ_len=args.summary_len)
        # Use a simple sampler with distributed batch sampler.
        sampler = torch.utils.data.SequentialSampler(dataset)
        batch_sampler = DistributedBatchSampler(sampler=sampler,
                                                batch_size=global_batch_size,
                                                drop_last=True,
                                                rank=rank,
                                                world_size=world_size)
        # Torch dataloader.
        return torch.utils.data.DataLoader(dataset,
                                           batch_sampler=batch_sampler,
                                           num_workers=num_workers,
                                           pin_memory=True)

    train = make_data_loader_(args.train_data_path)
    valid = make_data_loader_(args.val_data_path)
    test = make_data_loader_(args.test_data_path)

    args.do_train = False
    args.do_valid = False
    args.do_test = False

    if train is not None:
        args.do_train = True
    if valid is not None:
        args.do_valid = True
    if test is not None:
        args.do_test = True

    # Tokenizer.
    eod_token = tokenizer.encoder['<|endoftext|>']
    num_tokens = eod_token + 1

    return (train, valid, test), num_tokens, eod_token

class T5Dataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }