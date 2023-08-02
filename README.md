# minimal_fsdp

The goal of this repository is to give a functional example of scalable LLM training using Pytorch FSDP.  While other frameworks might work, there is a lot of complexity hiding under the hood.  This has less of that than, e.g., pytorch-lightning or transformers accelerate.  

The example dataset in `/data` is a sampling of Shakespeares works; its not large enough to train full LLMs at scale, so we shouldnt expect *good* results.  However, it is large enough to demonstrate the functionality of FSDP.

# Setup
You need an environment where we can install pytorch and other dependencies.  FSDP requires at least `pytorch>= 1.13`, however, activation checkpointing isnt functioning with FSDP until `pytorch >= 2.0`.
Get`minimal_fsdp` using a conda environment:
```
conda create -n fsdp python=3.10
conda activate fsdp
```
This was developed on pytorch 2.0.0; it may work on later versions, but to ensure it functions on Polaris, get the correct version using:
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
```
git clone git@github.com:azton/minimal_fsdp.git
cd minimal_fsdp
pip install -e .
pip install -r requirements.txt
```
This could be similarly accomplished with pythons virtual environments if you are so inclined. 

# Running:

## Training
This package can run both training and evaluation.  To train from the `minimal_fsdp` folder, run:
```
mpiexec -n $NUM_GPUS python basic_training.py <command line args>
```
where `$NUM_GPUS` is the number of GPUs you want to use and needs to be configured in your environment in this example.  The `mpiexec` command is indicating whatever command your computer might use to launch multiple tasks. If you are using it on a single task, you can omit 'mpiexec' and the '-n $NUM_GPUS' portions.
## Arguments
### Model and environment args:
```
--environment: The environment to use.  Currently, only 'pbs' or 'local' are supported.
--model: The model to use.  Currently, only 'mingpt' is supported.
--run_name: The name of the run; will be used to label checkpoints.
--vocab_size: The size of the vocabulary to use. Tokenizer dependent.
--seq_len: The sequence length to use.  
        Data in /data is already tokenized to 256, so anything <256 will work with it.
--num_heads: Number of attention heads in the multi-head attention of the model
--num_layers: number of layers
--embed_size: embedding dimension of the model
--restart_checkpoint_file: If you want to restart from a checkpoint, 
        provide the path to it here.

```
### FSDP args:
```
--sharding: The sharding strategy to use.  
        Choose from full-shard, grad-shard, no-shard, hybrid-full-shard, hybrid-grad-shard
--precision: The precision to use.  choose from 'float32', 'bf16'.  
        bf16 is highly recommended.
--cpu_offload: Whether to offload the optimizer and model parameters to CPU.  
--activation_checkpoints: Whether to use activation checkpointing.  
        Gradients won't be stored, but recalculated in .backward() call.
--meta_init: For large models that wont fit on CPU/GPU memory, 
        will use meta initialization to reduce memory footprint.

```
### Training args:
```
--train_data/--val_data: path to the training/validation data respectively.
--max_epochs: The maximum number of epochs to train for.  
        If set to `<0`, will skip training and only evaluate.
--lr: The learning rate to use.
--dropout: The dropout to use in FFNs.
--batch_size: The batch size to use.  This is the batch size per GPU.
```

## Evaluation
This isn't really evaluation, but more like loading a model and putting a prompt through it.  For now, you need to supply the correct arguments to build the model.  Ideally, the model structure would be read from the checkpoint to be agnostic of the command line args.

### Arguments
```
--max_epochs: Set to -1 to skip training and only evaluate.  
        Else, the model will be trained first.
--prompt: The prompt to put through the model.
--num_tokens: The number of tokens to generate.

```
