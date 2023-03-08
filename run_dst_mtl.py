# coding=utf-8
#
# Copyright 2020-2022 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import random
import glob
import json
import math
import re
import gc

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from torch.optim import (AdamW)
from tqdm import tqdm, trange
from collections import deque

from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME,
                          BertConfig, BertTokenizer, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          RobertaConfig, RobertaTokenizer, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          ElectraConfig, ElectraTokenizer, ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          get_linear_schedule_with_warmup)

from modeling_dst import (TransformerForDST)
from data_processors import PROCESSORS
from run_dst import (evaluate, load_and_cache_examples, set_seed, batch_to_device)
from utils_dst import (print_header, convert_examples_to_features, convert_aux_examples_to_features)
from tensorlistdataset import (TensorListDataset)

logger = logging.getLogger(__name__)

ALL_MODELS = tuple(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP)
ALL_MODELS += tuple(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
ALL_MODELS += tuple(ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP)

MODEL_CLASSES = {
    'bert': (BertConfig, TransformerForDST('bert'), BertTokenizer),
    'roberta': (RobertaConfig, TransformerForDST('roberta'), RobertaTokenizer),
    'electra': (ElectraConfig, TransformerForDST('electra'), ElectraTokenizer),
}


def train_mtl(args, train_dataset, aux_dataset, aux_task_def, features, model, tokenizer, processor, continue_from_global_step=0):
    assert not args.mtl_use or args.gradient_accumulation_steps == 1

    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True)
    aux_sampler = RandomSampler(aux_dataset) if args.local_rank == -1 else DistributedSampler(aux_dataset)
    aux_dataloader = DataLoader(aux_dataset, sampler=aux_sampler, batch_size=args.train_batch_size, pin_memory=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.save_epochs > 0:
        args.save_steps = t_total // args.num_train_epochs * args.save_epochs

    num_warmup_steps = int(t_total * args.warmup_proportion)
    mtl_last_step = int(t_total * args.mtl_ratio)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    scaler = torch.cuda.amp.GradScaler()
    if 'cuda' in args.device.type:
        autocast = torch.cuda.amp.autocast(enabled=args.fp16)
    else:
        autocast = torch.cpu.amp.autocast(enabled=args.fp16)

    # multi-gpu training
    model_single_gpu = model
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model_single_gpu)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...", continue_from_global_step)
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    # Iterators for aux tasks
    aux_loss_diff = 0.0
    aux_logged_steps = 0
    tr_aux_loss = 0.0
    logging_aux_loss = 0.0
    loss_diff_queue = deque([], args.mtl_diff_window)
    aux_iterator_dict = iter(aux_dataloader)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            # If training is continued from a checkpoint, fast forward
            # to the state of that checkpoint.
            if global_step < continue_from_global_step:
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    global_step += 1
                continue

            #model.train()
            batch = batch_to_device(batch, args.device)

            # This is what is forwarded to the "forward" def.
            inputs = {'input_ids':       batch[0],
                      'input_mask':      batch[1], 
                      'segment_ids':     batch[2],
                      'start_pos':       batch[3],
                      'end_pos':         batch[4],
                      'inform_slot_id':  batch[5],
                      'refer_id':        batch[6],
                      'diag_state':      batch[7],
                      'class_label_id':  batch[8]}

            # MTL (optional)
            if args.mtl_use and global_step < mtl_last_step:
                if args.mtl_print_loss_diff:
                    model.eval()
                    with torch.no_grad():
                        pre_aux_loss = model(**inputs)[0]

                try:
                    aux_batch = batch_to_device(next(aux_iterator_dict), args.device)
                except StopIteration:
                    logger.info("Resetting iterator for aux task")
                    aux_iterator_dict = iter(aux_dataloader)
                    aux_batch = batch_to_device(next(aux_iterator_dict), args.device)

                aux_inputs = {'input_ids':       aux_batch[0],
                              'input_mask':      aux_batch[1],
                              'segment_ids':     aux_batch[2],
                              'start_pos':       aux_batch[3],
                              'end_pos':         aux_batch[4],
                              'class_label_id':  aux_batch[5],
                              'aux_task_def':    aux_task_def}
                model.train()
                with autocast:
                    aux_outputs = model(**aux_inputs)
                aux_loss = aux_outputs[0]

                if args.n_gpu > 1:
                    aux_loss = aux_loss.mean() # mean() to average on multi-gpu parallel (not distributed) training

                tr_aux_loss += aux_loss.item()
                aux_logged_steps += 1

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                model.zero_grad()
                if args.mtl_print_loss_diff:
                    model.eval()
                    with torch.no_grad():
                        post_aux_loss = model(**inputs)[0]
                    aux_loss_diff = pre_aux_loss - post_aux_loss
                else:
                    post_aux_loss = 0.0 # TODO: move somewhere else...

                pre_aux_loss = post_aux_loss

                loss_diff_queue.append(aux_loss_diff)

            # Normal training
            model.train()

            with autocast:
                outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Aux losses
                if args.mtl_use and args.local_rank in [-1, 0] and args.logging_steps > 0:
                    tb_writer.add_scalar('loss_diff', aux_loss_diff, global_step)
                    # TODO: make nicer
                    if len(loss_diff_queue) > 0:
                        tb_writer.add_scalar('loss_diff_mean', sum(loss_diff_queue) / len(loss_diff_queue), global_step)
                    if aux_logged_steps > 0 and tr_aux_loss != logging_aux_loss:
                        tb_writer.add_scalar('aux_loss', tr_aux_loss - logging_aux_loss, global_step)
                        logging_aux_loss = tr_aux_loss

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            results = evaluate(args, model_single_gpu, tokenizer, processor, prefix=global_step)
            for key, value in results.items():
                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        # To prevent GPU memory to overflow
        gc.collect()
        torch.cuda.empty_cache()

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    tr_aux_loss /= aux_logged_steps

    return global_step, tr_loss / global_step, tr_aux_loss


def load_and_cache_aux_examples(args, model, tokenizer, aux_task_def=None):
    if aux_task_def is None:
        return None, None

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_file = os.path.join(os.path.dirname(args.output_dir), 'cached_aux_features')
    if os.path.exists(cached_file) and not args.overwrite_cache: # and not output_examples:
        logger.info("Loading features from cached file %s", cached_file)
        features = torch.load(cached_file)
    else:
        processor = PROCESSORS['aux_task']()
        logger.info("Creating features from aux task dataset file at %s", args.mtl_data_dir)
        examples = processor.get_aux_task_examples(data_dir=args.mtl_data_dir,
                                                   data_name=args.mtl_train_dataset,
                                                   max_seq_length=args.max_seq_length)

        features = convert_aux_examples_to_features(examples=examples, aux_task_def=aux_task_def, max_seq_length=args.max_seq_length)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_file)
            torch.save(features, cached_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_pos for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_pos for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorListDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_label)

    return dataset, features


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="Name of the task (e.g., multiwoz21).")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Task database.")
    parser.add_argument("--dataset_config", default=None, type=str, required=True,
                        help="Dataset configuration file.")
    parser.add_argument("--predict_type", default=None, type=str, required=True,
                        help="Portion of the data to perform prediction on (e.g., dev, test).")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the <predict_type> set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--dropout_rate", default=0.3, type=float,
                        help="Dropout rate for BERT representations.")
    parser.add_argument("--heads_dropout", default=0.0, type=float,
                        help="Dropout rate for classification heads.")
    parser.add_argument("--class_loss_ratio", default=0.8, type=float,
                        help="The ratio applied on class loss in total loss calculation. "
                             "Should be a value in [0.0, 1.0]. "
                             "The ratio applied on token loss is (1-class_loss_ratio)/2. "
                             "The ratio applied on refer loss is (1-class_loss_ratio)/2.")
    parser.add_argument("--token_loss_for_nonpointable", action='store_true',
                        help="Whether the token loss for classes other than copy_value contribute towards total loss.")
    parser.add_argument("--refer_loss_for_nonpointable", action='store_true',
                        help="Whether the refer loss for classes other than refer contribute towards total loss.")

    parser.add_argument("--no_append_history", action='store_true',
                        help="Whether or not to append the dialog history to each turn.")
    parser.add_argument("--no_use_history_labels", action='store_true',
                        help="Whether or not to label the history as well.")
    parser.add_argument("--no_label_value_repetitions", action='store_true',
                        help="Whether or not to label values that have been mentioned before.")
    parser.add_argument("--swap_utterances", action='store_true',
                        help="Whether or not to swap the turn utterances (default: usr|sys, swapped: sys|usr).")
    parser.add_argument("--delexicalize_sys_utts", action='store_true',
                        help="Whether or not to delexicalize the system utterances.")
    parser.add_argument("--class_aux_feats_inform", action='store_true',
                        help="Whether or not to use the identity of informed slots as auxiliary featurs for class prediction.")
    parser.add_argument("--class_aux_feats_ds", action='store_true',
                        help="Whether or not to use the identity of slots in the current dialog state as auxiliary featurs for class prediction.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float,
                        help="Linear warmup over warmup_proportion * steps.")
    parser.add_argument("--svd", default=0.0, type=float,
                        help="Slot value dropout ratio (default: 0.0)")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=0,
                        help="Save checkpoint every X updates steps. Overwritten by --save_epochs.")
    parser.add_argument('--save_epochs', type=int, default=0,
                        help="Save checkpoint every X epochs. Overrides --save_steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument('--local_files_only', action='store_true',
                        help="Whether to only load local model files (useful when working offline).")

    parser.add_argument('--mtl_use', action='store_true', help="")
    parser.add_argument('--mtl_task_def', type=str, default="aux_task_def.json", help="")
    parser.add_argument('--mtl_train_dataset', type=str, default="", help="cola|mnli|mrpc|qnli|qqp|rte|sst|wnli|squad|squad-v2")
    parser.add_argument("--mtl_data_dir", type=str, default="data/aux/bert_base_uncased_lower", help="")
    parser.add_argument("--mtl_ratio", type=float, default=1.0, help="")
    parser.add_argument("--mtl_diff_window", type=int, default=10)
    parser.add_argument('--mtl_print_loss_diff', action='store_true', help="")

    args = parser.parse_args()

    assert args.warmup_proportion >= 0.0 and args.warmup_proportion <= 1.0
    assert args.svd >= 0.0 and args.svd <= 1.0
    assert args.class_aux_feats_ds is False or args.per_gpu_eval_batch_size == 1
    assert not args.class_aux_feats_inform or args.per_gpu_eval_batch_size == 1
    assert not args.class_aux_feats_ds or args.per_gpu_eval_batch_size == 1

    assert not args.mtl_use or args.gradient_accumulation_steps == 1
    assert args.mtl_ratio >= 0.0 and args.mtl_ratio <= 1.0

    task_name = args.task_name.lower()
    if task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (task_name))

    # Load the MTL task definitions
    if args.mtl_use:
        with open(args.mtl_task_def, "r", encoding='utf-8') as reader:
            aux_task_defs = json.load(reader)
        aux_task_def = aux_task_defs[args.mtl_train_dataset]
    else:
        aux_task_def = None

    processor = PROCESSORS[task_name](args.dataset_config)
    dst_slot_list = processor.slot_list
    dst_class_types = processor.class_types
    dst_class_labels = len(dst_class_types)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging, print header
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print_header()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, local_files_only=args.local_files_only)

    # Add DST specific parameters to config
    config.dst_max_seq_length = args.max_seq_length
    config.dst_dropout_rate = args.dropout_rate
    config.dst_heads_dropout_rate = args.heads_dropout
    config.dst_class_loss_ratio = args.class_loss_ratio
    config.dst_token_loss_for_nonpointable = args.token_loss_for_nonpointable
    config.dst_refer_loss_for_nonpointable = args.refer_loss_for_nonpointable
    config.dst_class_aux_feats_inform = args.class_aux_feats_inform
    config.dst_class_aux_feats_ds = args.class_aux_feats_ds
    config.dst_slot_list = dst_slot_list
    config.dst_class_types = dst_class_types
    config.dst_class_labels = dst_class_labels
    if aux_task_def is not None:
        config.aux_task_def = aux_task_def

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case, local_files_only=args.local_files_only)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, local_files_only=args.local_files_only)

    logger.info("Updated model config: %s" % config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # If output files already exists, assume to continue training from latest checkpoint (unless overwrite_output_dir is set)
        continue_from_global_step = 0 # If set to 0, start training from the beginning
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/*/' + WEIGHTS_NAME, recursive=True)))
            if len(checkpoints) > 0:
                checkpoint = checkpoints[-1]
                logger.info("Resuming training from the latest checkpoint: %s", checkpoint)
                continue_from_global_step = int(checkpoint.split('-')[-1])
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
        
        train_dataset, features = load_and_cache_examples(args, model, tokenizer, processor, evaluate=False)
        aux_dataset, _ = load_and_cache_aux_examples(args, model, tokenizer, aux_task_def)
        global_step, tr_loss, aux_loss = train_mtl(args, train_dataset, aux_dataset, aux_task_def, features, model, tokenizer, processor, continue_from_global_step)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = []
    if args.do_eval and args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(args.output_dir, "eval_res.%s.json" % (args.predict_type))
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for cItr, checkpoint in enumerate(checkpoints):
            # Reload the model
            global_step = checkpoint.split('-')[-1]
            if cItr == len(checkpoints) - 1:
                global_step = "final"
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, processor, prefix=global_step)
            result_dict = {k: float(v) for k, v in result.items()}
            result_dict["global_step"] = global_step
            results.append(result_dict)

            for key in sorted(result_dict.keys()):
                logger.info("%s = %s", key, str(result_dict[key]))

        with open(output_eval_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
