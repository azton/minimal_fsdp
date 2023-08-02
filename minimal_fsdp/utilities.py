"""
    To keep basic_training a bit cleaner, 
    I put the utilities here; model loading,
    setting up FSDP, etc.
"""
import torch
import numpy as np
from argparse import Namespace, ArgumentParser
import torch.distributed as td
from torch.utils.data import DistributedSampler
import os
import h5py
from typing import Union
from pathlib import Path
import torch.nn as nn
from minimal_fsdp.minGPT import (
        GPT, 
        GPTConfig, 
        Block, 
        MLP, 
        CausalSelfAttention
        )
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp.wrap import (
   always_wrap_policy as wrap_policy,
   transformer_auto_wrap_policy,
   wrap
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
if hasattr(torch.distributed.algorithms._checkpoint.checkpoint_wrapper, "apply_activation_checkpointing"):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
    )
    fsdp_ckpt = True
else:
    print("WARNING: torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing not found")
    print("FSDP will not apply activation checkpointing")
    fsdp_ckpt = False
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullyShardedDataParallel as FSDP,
                                                                CPUOffload,
                                                                MixedPrecision,
                                                                ShardingStrategy,
                                                                FullStateDictConfig,
                                                                BackwardPrefetch,
                                                                StateDictType
)
from functools import partial
rank = int(os.environ.get('PMI_RANK', '0'))
zero = rank == 0


def init_distributed(args: Namespace) -> Namespace:
    print(f"{args.rank} init_distributed...")
    torch.cuda.set_device(args.local_rank)
    global_pg = td.init_process_group(backend=args.backend, init_method="env://")
    tot = torch.tensor(1).to(torch.cuda.current_device())
    # Use all-reduce to verify success here
    td.all_reduce(tot, op=td.ReduceOp.SUM)

    print(f"ARANK {args.rank}: Global rank {td.get_rank()} info: WORLD={td.get_world_size()}, has devices {torch.cuda.device_count()}, on device = {torch.cuda.current_device()}")
    print(f"ARANK {args.rank}: Local rank {args.local_rank}: {torch.cuda.current_device()}")
    print(f"ARANK {args.rank}: Total number of processes: {tot}")
    return args

def setup_environment(args: Namespace) -> Namespace:
    """
        Torch distributed init using env:// requires the correct environment variables to be set.
        Map from default environment variables to the ones used on Polaris. Or wherever.
        Easily extensible by mapping, eg, slurm environment variables to the ones used here.

    """
    if not torch.cuda.is_available():
        raise NotImplementedError("No CUDA? FSDP needs CUDA or accelerators")
    if args.environment == 'pbs':
        os.environ['RANK'] = os.environ['PMI_RANK']# global 
        os.environ['LOCAL_RANK'] = os.environ['PMI_LOCAL_RANK'] # local
        os.environ['WORLD_SIZE'] = os.environ['PMI_SIZE']
        args.world_size = int(os.environ['PMI_SIZE'])
        args.global_rank = int(os.environ['PMI_RANK'])
        args.rank = args.global_rank
        args.local_rank = int(os.environ['PMI_LOCAL_RANK']) # wraparound since LOCAL_RANK is actually global?? WRT/ 
        args.local_size = int(os.environ['PMI_LOCAL_SIZE'])
        args.backend = 'nccl'
        args.num_nodes = args.world_size // args.local_size
    elif args.environment == 'local':
        # torch.backends.cuda.flash_sdp_enabled()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        args.rank = 0
        args.world_size = 1
        args.global_rank = 0
        args.local_rank = 0
        args.local_size = 1
        args.backend = 'nccl' 
    return args

def setup_model(args: Namespace) -> torch.nn.Module:
    """
        Initializing a model in FSDP is a bit esoteric and has a lot more 
        intricacy than a general model initialization. 
        Adding another model is straightforward and follows the same pattern as used here for GPT.
        The only intricacy is to import and correctly identify the transformer blocks for FSDP.
    """
    # Setup up configurations.  These are very similar to HF interface
    if args.model == 'mingpt':
        config = GPTConfig(
            block_size = args.seq_length, # configured in tokenizer to match GPT-3
            vocab_size = args.vocab_size,
            n_layer = args.num_layers,
            n_head = args.num_heads,
            n_embd = args.embed_size,
            dropout = args.dropout,
        )
        arch = GPT
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    sharding = ShardingStrategy.NO_SHARD
    if args.sharding == 'full-shard':
        sharding = ShardingStrategy.FULL_SHARD
    elif args.sharding == 'grad-shard':
        sharding = ShardingStrategy.SHARD_GRAD_OP
    elif args.sharding == 'hybrid-full-shard':
        sharding = ShardingStrategy.HYBRID_SHARD
    elif args.sharding == 'hybrid-grad-shard':
        sharding = ShardingStrategy._HYBRID_SHARD_ZERO2

    # set up mixed precision
    prec = torch.float32 if args.precision == 'float32' else torch.bfloat16
    mixed_precision = MixedPrecision(
        param_dtype=prec,
        reduce_dtype=prec,
        buffer_dtype=prec
    )
    # set up auto wrapping policy
    twrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block,}
        )
    # set up checkpointing, if supported.
    if fsdp_ckpt:
            ckpt_fn = lambda submod: isinstance(submod, Block)
            non_reent_wrapper = partial(
                checkpoint_wrapper,
                offload_to_cpu=args.cpu_offload,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )



    # large models need to be materialized as they  are
    # created to prevent OOM errors
    if args.meta_init:
        with torch.device('meta'):
            model_arch = arch(config)
    else:
        model_arch = arch(config)
    model = FSDP(model_arch,
            auto_wrap_policy=twrap_policy,
            mixed_precision=mixed_precision,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding, #FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
            cpu_offload=CPUOffload(offload_params=args.cpu_offload),
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE, # bit faster async comms, bit higher memory
            limit_all_gathers=False,
            use_orig_params=True,
            forward_prefetch=True,

            )
    if args.activation_checkpointing and fsdp_ckpt:
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reent_wrapper, check_fn=ckpt_fn
        )

    return model


"""
    Checkpointing and restarting...
    Loading is done in steps, since for large models the model+optimizer states 
    might be too large for memory.
"""

def load_fsdp_model_checkpoint(model, args):
    ckpt = torch.load(args.restart_checkpoint_file, map_location = torch.device('cpu'))
    model.load_state_dict(ckpt['model_state']) 
    ckpt = None
    return model

def load_fsdp_optimizer_checkpoint(optimizer, model, args):
    full_osd = None
    # if args.rank == 0:
    ckpt = torch.load(args.restart_checkpoint_file, map_location = torch.device('cpu'))
    full_osd = ckpt['optimizer_state']
    if args.sharding != 'no-shard':
        sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, model, optimizer)
    else:
        sharded_osd = full_osd.to(torch.cuda.current_device())
    optimizer.load_state_dict(sharded_osd)
    ckpt = None
    return optimizer

def load_restart_metrics(trainer, inference_only: bool=False):
     # how?
    ckpt = torch.load(trainer.args.restart_checkpoint_file, map_location=torch.device('cpu'))
    if not inference_only:
        trainer.scheduler.load_state_dict(ckpt['scheduler_state'])
    trainer.val_record_epochloss = ckpt['record_loss']
    trainer.completed_train_steps = ckpt['completed_train_steps']
    trainer.completed_val_steps = ckpt['completed_val_steps']
    trainer.start_epoch = ckpt['epoch_number'] + 1
    newargs = ckpt['args']

    # with some exceptions, we want to 
    # carry the hparams of prior run forward
    for k, v in newargs.items():
        if (
            k != 'restart_checkpoint_file' # dont overwrite where the checkpoint came from
            and k != 'max_epochs' # allow extending the run
            and 'steps_per_epoch' not in k # changing iteration/epoch is also ok
            and k != 'batch_size' # good to change batch size if needed
            and ('data' not in k  and 'path' not in k)# allow updating data paths
            
        ):
            setattr(trainer.args, k, v)
    rfilepath = trainer.args.restart_checkpoint_file
    if not inference_only:
        rfilepath = rfilepath.split('/')
        vnum = int(rfilepath[-2].split('_')[-1])
        # delete old logger and recreate in the old versions directory
    ckpt = None
    return trainer

def save_restart_checkpoint(save_directory: Union[str, Path],
                            STR:str,
                            model: nn.Module,
                            optimizer: torch.optim.Optimizer,
                            scheduler: torch.optim.lr_scheduler._LRScheduler,
                            args: Namespace,
                            epochnum: int, 
                            ):
    """
        special checkpoint where we'll save all relevant items for a restart, including optimizer states.  I hope.
    """
    save_policy = FullStateDictConfig(offload_to_cpu=True if args.sharding != 'no-shard' else False, rank0_only=True)
    with FSDP.state_dict_type(
                model, 
                StateDictType.FULL_STATE_DICT, 
                save_policy
            ):
                cpu_state = model.state_dict()
    # call on every rank, only actually exists on rank 0
    if args.sharding != 'no-shard':
        opt_state = FSDP.optim_state_dict(model, optimizer) # specify group if using HYBRID_SHARD...
    else:
        opt_state = optimizer.state_dict()
    if args.rank == 0:
        if not os.path.exists(f"{save_directory}"):
            os.makedirs(f"{save_directory}")
        

        save_name = f"{save_directory}/{args.run_name}_epoch{epochnum:02d}.pt"
        saveargs = vars(args)
        
        if args.rank == 0:
            save_obj = dict(
                        model_state=cpu_state,
                        optimizer_state=opt_state,
                        scheduler_state=scheduler.state_dict(),
                        epoch_number=epochnum,
                        args=saveargs
            )
            torch.save(save_obj, save_name)

class BasicLoader(Dataset):
    def __init__(self, 
                args:Namespace, 
                split:str='train') -> None:
        super().__init__()
        self.args = args
        self.split = split
        self.data = None
        if split == 'train':
            self.data = h5py.File(args.train_data, 'r')
        elif split == 'val':
            self.data = h5py.File(args.val_data, 'r')
        
    def __len__(self):
        return self.data['input_ids'].shape[0]

    def __getitem__(self, idx: int):
        ids = torch.from_numpy(self.data['input_ids'][idx].astype('int32')).int()
        Input = ids[:-1]
        target = ids[1:]

        return Input, target.long()


def get_dataloaders(args:Namespace):
    train_loader = BasicLoader(args, split='train')
    val_loader = BasicLoader(args, split='val')
    train_sampler = DistributedSampler(train_loader, shuffle=True, num_replicas = args.world_size, rank = args.rank)
    val_sampler = DistributedSampler(val_loader, shuffle=False, num_replicas = args.world_size, rank = args.rank)
    train_loader = DataLoader(train_loader, batch_size=args.batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_loader, batch_size=args.batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)
    return train_loader, val_loader

def get_args()-> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--environment', type=str, default='local')
    parser.add_argument('--model', type=str, default='mingpt')
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--train_data', type=str, default='data/shakespeare/train.h5')
    parser.add_argument('--val_data', type=str, default='data/shakespeare/validation.h5')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--restart_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_length', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--embed_size', type=int, default=768)
    parser.add_argument('--ffn_hidden_size', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--restart_checkpoint_file', type=str, default=None)
    parser.add_argument('--cpu_offload', action='store_true')
    parser.add_argument('--sharding', type=str, default='no-shard', choices=['no-shard', 'hybrid-shard', 'full-shard', 'grad-shard','hybrid-full-shard'])
    parser.add_argument('--meta_init', action='store_true')
    parser.add_argument('--activation_checkpointing', action='store_true')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--num_tokens', type=int, default=100)
    args = parser.parse_args()
    if zero:
        print('Running with parameters:')
        for k, v in vars(args).items():
            print(f'{k}: {v}')
            
    return args


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor