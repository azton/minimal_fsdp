"""
    This is a minimal example that can get you up and running with
    GPT models at scale.

    We'll be using FSDP parallelism to shard the model across GPUs, 
    and using hybrid sharding to cope with larger models that don't 
    fit on a single GPU.

    We don't use huggingface (although that could be a drop-in replacement).
    For large models (>20B parameters), we need to use special initialization,
    and that is easier to manage without hiding the models in HF.

    This is significantly plugged from https://github.com/karpathy/nanoGPT
    with modification to work with FSDP.
"""

from argparse import ArgumentParser
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from minimal_fsdp.utilities import *
from tokenizers import Tokenizer
rank = int(os.environ.get('PMI_RANK', '0'))
zero = rank == 0


def run_the_thing():
    args = get_args()
    args = setup_environment(args)
    args = init_distributed(args)
    model = setup_model(args)

    train_dl, val_ds = get_dataloaders(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineWarmupScheduler(optimizer, 10*len(train_dl), 50*len(train_dl))
    if args.max_epochs > 0:
        estart = 0
        if args.restart_checkpoint_file is not None:
            if zero: print('Reloading model from checkpoint...')
            model = load_fsdp_model_checkpoint(model, args)
            optimizer = load_fsdp_optimizer_checkpoint(optimizer, model, args)
            estart = args.restart_epoch

        epoch_pbar = None
        if zero:
            epoch_pbar = tqdm(range(args.max_epochs), desc='Epochs', position=0)

        for epoch in range(estart, args.max_epochs):

            model.train()
            train_loss = []
            if zero:
                batchbar = tqdm(total=len(train_dl), desc='Train', position=1)
            for i, batch in enumerate(train_dl):
                optimizer.zero_grad()
                output = model(input_ids = batch[0], labels = batch[1])
                output.loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss.append(output.loss.item())
                if zero:
                    batchbar.set_postfix_str(f'Loss: {train_loss[-1]:.3f} LR: {scheduler.get_last_lr()[0]:.3e}')
                    batchbar.update()
            if zero: batchbar.close()
            train_loss = np.mean(train_loss)


            model.eval()
            val_loss = []
            if zero:
                batchbar = tqdm(total=len(val_ds), desc='Val', position=1)
            for i, batch in enumerate(val_ds):
                output = model(input_ids = batch[0], labels = batch[1])
                val_loss.append(output.loss.item())
                if zero:
                    batchbar.set_postfix_str(f'Loss: {val_loss[-1]:.3f}')
                    batchbar.update()
            if zero: batchbar.close()
            val_loss = np.mean(val_loss)

            if zero:   
                epoch_pbar.set_description(f'Epoch {epoch} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')
                epoch_pbar.update()
        if zero:
            epoch_pbar.close()
        print('Saving checkpoint for restart/inference ...')
        save_restart_checkpoint('./checkpoints',
                                    'FINAL',
                                    model,
                                    optimizer,
                                    scheduler,
                                    args,
                                    epoch)
        print('\n\n Training complete! \n\n')
    else:
        # loading prior model state for inference
        if zero: print("Skipping training and loading checkpoint for inference...")
        model = load_fsdp_model_checkpoint(model, args)
    # now that the training is "complete", we can do a pass of inference and see what it'll do
    if zero:
        tokenizer = Tokenizer.from_pretrained('gpt2')
        model.eval()

        context = "Romeo, Romeo, wherefore art thou " if args.prompt is None else args.prompt
        print(f"Starting with context: {context}")
        context_tokens = tokenizer.encode(context)
        out_ids = context_tokens.ids
        inp_ids = context_tokens.ids
        for tok_gen in tqdm(range(150)):
            if len(inp_ids) > args.seq_length:
                inp_ids = inp_ids[-seq_length:]
            context = torch.tensor(inp_ids, 
                                dtype=torch.long, 
                                device=torch.cuda.current_device()).unsqueeze(0)
            # model.generate in mingpt crashes here with FSDP.  
            # workaround using forward to get the same result.
            generated = model.forward(context)
            logits = generated.logits
            logits = torch.nn.functional.softmax(logits, dim=-1)
            # just snagging the max leads to mode collapse.  multinomial 
            # keeps some entropy in the output.  You can use argmax if you really
            # want to.
            generated = torch.multinomial(logits[:, -1], num_samples=1).squeeze(1)
            out_ids.append(generated.item())
            inp_ids.append(generated.item())
        print(f"Input context + generated token ids: {out_ids}")
        text = tokenizer.decode(out_ids)
        print(f"Decoded generated text:\n{text}")

if __name__ == '__main__':
    run_the_thing()