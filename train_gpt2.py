### Learn from Karp  base on nanoGPT
import os
import math
import glob
import struct
import inspect
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
import yaml
from torch.utils.tensorboard import SummaryWriter
from shampoo import Shampoo, ShampooHyperParams, LayerwiseGrafting # Import Shampoo

#------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
   
            # manual implementation of attention
        # this materializes the large (T,T) matrix for all the queries and keys

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


# A direct alias, as CausalSelfAttention already uses the optimized F.scaled_dot_product_attention
# This provides an explicit name for configuration purposes.
FlashAttention = CausalSelfAttention


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x






class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # Determine attention implementation based on config
        if config.attention_type == "flash":
            self.attn = FlashAttention(config)
            # print("Using FlashAttention") # Optional: uncomment for debugging
        elif config.attention_type == "causal":
            self.attn = CausalSelfAttention(config)
            # print("Using CausalSelfAttention") # Optional: uncomment for debugging
        else:
            raise ValueError(f"Unknown attention type: {config.attention_type}")
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x






@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # Number of tokens
    n_layer: int = 12   # number of layers
    n_head: int = 12    # number of heads
    n_embd: int = 768   # embedding dimension
    attention_type: str = "causal" # Type of attention mechanism ('causal' or 'flash')

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        #init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer)  ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)       
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def configure_optimizers(self, optimizer_type, weight_decay, learning_rate, device):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process: # Only print on master process
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create the optimizer based on the type specified in the config
        optimizer = None
        if optimizer_type == 'AdamW':
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and 'cuda' in device
            if master_process: print(f"Using AdamW optimizer (fused: {use_fused})")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        elif optimizer_type == 'Shampoo':
            if master_process: print(f"Using Shampoo optimizer")
            # Note: Shampoo's weight decay is handled by its hyperparams
            # We need to separate params for Shampoo's internal handling if needed,
            # but the original paper applies it globally. Here, we pass it via hyperparams.
            shampoo_hps = ShampooHyperParams(
                weight_decay=weight_decay,
                # Keep other Shampoo defaults or allow configuration via yaml later
                block_size=128, # Default from shampoo.py
                beta2=0.999, # A common default, adjust if needed
                graft_type=LayerwiseGrafting.SGD, # Default from shampoo.py
            )
            # Shampoo doesn't use optim_groups in the same way for weight decay.
            # It applies WD internally based on its hyperparameter.
            # We pass all parameters requiring grad.
            all_params = [p for p in param_dict.values() if p.requires_grad]
            optimizer = Shampoo(all_params, lr=learning_rate, momentum=0.9, hyperparams=shampoo_hps)
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

        return optimizer




    def forward(self, idx, targets = None ):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (t)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model    
#--------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        #get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) >0, f"no shards found for split {split}"
        print (f"found {len(shards)}  shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)

        self.current_position += B*T*self.num_processes

        if self.current_position + (B*T*self.num_processes+1) > len(self.tokens):
            self.current_shard = (self.current_shard +1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T *self.process_rank
        return x, y












#------------------
if __name__ == "__main__":
    import time
    import argparse
    import tiktoken
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments= True

    # --- Configuration Loading ---
    parser = argparse.ArgumentParser(description='Train a GPT-2 model.')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration YAML file.')
    args = parser.parse_args()

    # Load config from YAML file
    try:
        with open(args.config, 'r') as f:
            config_yaml = yaml.safe_load(f)
            print(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        print(f"Warning: Configuration file {args.config} not found. Using default settings.")
        config_yaml = {} # Use defaults if file not found
    except Exception as e:
        print(f"Error loading configuration file {args.config}: {e}")
        sys.exit(f"Exiting due to configuration error: {e}") # Exit if config is invalid


    # --- Device Setup ---
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print (f"using device: {device}")

    # --- DDP Setup (Basic Placeholder) ---
    # --- DDP Setup ---
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # DDP run: need to initialize process group
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        print(f"DDP enabled: rank={ddp_rank}, local_rank={ddp_local_rank}, world_size={ddp_world_size}, device={device}")
    else:
        # vanilla non-DDP run
        # assert torch.cuda.is_available(), "DDP requires CUDA"
        # init_process_group(backend='nccl')
        # ddp_rank = int(os.environ['RANK'])
        # ddp_local_rank = int(os.environ['LOCAL_RANK'])
        # ddp_world_size = int(os.environ['WORLD_SIZE'])
        # device = f'cuda:{ddp_local_rank}'
        # torch.cuda.set_device(device)
        ddp_rank = 0
        ddp_local_rank = 0 # Can be different from rank if using multiple nodes
        ddp_world_size = 1
        master_process = True

    # --- Model Configuration ---
    # --- Model Configuration ---
    # Update GPTConfig defaults with values from YAML if they exist
    gpt_config_args = {
        'block_size': config_yaml.get('block_size', 1024),
        'vocab_size': config_yaml.get('vocab_size', 50304), # 50257 for GPT-2, 50304 for tiktoken gpt2
        'n_layer': config_yaml.get('n_layer', 12),
        'n_head': config_yaml.get('n_head', 12),
        'n_embd': config_yaml.get('n_embd', 768),
        'attention_type': config_yaml.get('attention_type', 'causal') # Default to causal
    }
    config = GPTConfig(**gpt_config_args)
    if master_process:
        print("GPT Configuration:")
        for key, value in gpt_config_args.items():
            print(f"  {key}: {value}")

    # --- Training Hyperparameters ---
    # --- Training Hyperparameters ---
    # Get training params from config or use defaults
    optimizer_type = config_yaml.get('optimizer_type', 'AdamW') # Get optimizer type
    num_return_sequences = config_yaml.get('num_return_sequences', 5)
    max_length = config_yaml.get('max_length', 30)
    total_batch_size = config_yaml.get('total_batch_size', 524288) # 2**19, ~0.5M tokens
    B = config_yaml.get('batch_size', 16) # micro batch size
    T = config.block_size # sequence length, derived from model config

    # Get max_lr and ensure it's a float
    max_lr_config = config_yaml.get('max_lr', 6e-4)
    try:
        max_lr = float(max_lr_config)
    except (ValueError, TypeError):
        if master_process: print(f"Warning: Invalid value '{max_lr_config}' for max_lr in config. Using default 6e-4.")
        max_lr = 6e-4

    # Get min_lr and calculate default if necessary
    min_lr_config = config_yaml.get('min_lr') # Get value or None if missing
    if min_lr_config is None:
        min_lr = max_lr * 0.1 # Calculate default using validated max_lr
        if master_process: print(f"min_lr not found in config, calculating default: {min_lr:.4e}")
    else:
        try:
            min_lr = float(min_lr_config)
        except (ValueError, TypeError):
            default_min_lr = max_lr * 0.1
            if master_process: print(f"Warning: Invalid value '{min_lr_config}' for min_lr in config. Calculating default {default_min_lr:.4e}.")
            min_lr = default_min_lr

    warmup_steps = int(config_yaml.get('warmup_steps', 10)) # Ensure int
    max_steps = int(config_yaml.get('max_steps', 50)) # Ensure int
    weight_decay = float(config_yaml.get('weight_decay', 0.1)) # Ensure float
    val_interval = int(config_yaml.get('val_interval', 25)) # Ensure int
    val_loss_steps = int(config_yaml.get('val_loss_steps', 20)) # Ensure int
    log_interval = int(config_yaml.get('log_interval', 10)) # Ensure int
    checkpoint_interval = int(config_yaml.get('checkpoint_interval', 50)) # Ensure int
    grad_clip = float(config_yaml.get('grad_clip', 1.0)) # Ensure float
    compile_model = config_yaml.get('compile_model', True) # Boolean
    seed = int(config_yaml.get('seed', 1337)) # Ensure int

    # Set seed (moved slightly down to ensure seed is loaded first)
    torch.manual_seed(seed + ddp_rank) # Add rank for different seeds in DDP
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + ddp_rank)

    # Gradient Accumulation
    assert total_batch_size % (B * T * ddp_world_size) == 0, f"total_batch_size ({total_batch_size}) must be divisible by B*T*world_size ({B*T*ddp_world_size})"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"Total desired batch size: {total_batch_size}")
        print(f"Micro batch size B: {B}, Sequence length T: {T}")
        print(f"World size: {ddp_world_size}")
        print(f"Calculated gradient accumulation steps: {grad_accum_steps}")

    # DataLoader setup
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    # Set precision
    torch.set_float32_matmul_precision('medium')

    # Model Initialization
    model = GPT(config) # Use the config object created from YAML/defaults
    model.to(device)
    if compile_model:
        if master_process: print("Compiling model...")
        model = torch.compile(model)
        if master_process: print("Model compiled.")
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # Learning Rate Schedule
    def get_lr(it):
        # 1) linear warmup for warmup_steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > max_steps, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    # Logging setup (Log file and TensorBoard)
    log_dir = config_yaml.get('log_dir', "log")
    writer = None # Initialize writer to None
    if master_process:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log.txt")
        with open(log_file, "w") as f: # open for writing to clear the file
            pass
        # Initialize TensorBoard writer only on the master process
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")

    # Optimizer
    # Pass optimizer_type from config to the method
    optimizer = raw_model.configure_optimizers(
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        learning_rate=max_lr,
        device=device
    )
    # TODO: Add ZeroRedundancyOptimizer if using DDP and Shampoo supports it well

    # Tokenizer
    enc = tiktoken.get_encoding('gpt2')

    # Training loop
    for step in range(max_steps):
        t0 = time.time()

        # Validation step
        if step % val_interval == 0 and master_process: # Only master process runs validation
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
            # Autocast for mixed precision
            # Note: MPS autocast is still limited, might need adjustments
            ptdtype = torch.bfloat16 if device == 'cuda' else torch.float32 # Use bfloat16 on CUDA
            ctx = torch.amp.autocast(device_type=device.split(':')[0], dtype=ptdtype) if device != 'cpu' else nullcontext()
            with ctx:
                logits, loss = model(x, y)
            loss = loss / val_loss_steps # Corrected indentation
            val_loss_accum += loss.detach() # Corrected indentation
            print(f"Validation loss at step {step}: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            # Log validation loss to TensorBoard
            if writer:
                writer.add_scalar('Loss/val', val_loss_accum.item(), step)

        # Generate samples
        # Generate samples
        if step > 0 and step % checkpoint_interval == 0 and master_process: # Only master generates samples
            model.eval() # Use the potentially DDP-wrapped model for eval
            # num_return_sequences defined above from config
            # max_length defined above from config
            prompt = config_yaml.get('sample_prompt', "Hello, I'm a language model,")
            tokens = enc.encode(prompt)
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(seed + ddp_rank + step) # Ensure sample seed changes
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    # take the logits at the last position
                    # Use the potentially DDP-wrapped model here too
                    ptdtype = torch.bfloat16 if device == 'cuda' else torch.float32
                    ctx = torch.amp.autocast(device_type=device.split(':')[0], dtype=ptdtype) if device != 'cpu' else nullcontext()
                    with ctx:
                        logits, loss = model(xgen) # model forward pass
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (B, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not require generator argument when sampling on CPU
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"Rank {ddp_rank} sample {i}: {decoded}")
                with open(log_file, "a") as f:
                    f.write(f"step {step} sample {i}: {decoded}\n")

        # Checkpointing
        if step > 0 and step % checkpoint_interval == 0 and master_process: # Only master saves checkpoints
            # Use the raw_model (unwrapped) for saving state_dict
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': raw_model.config, # Save the config used for this model
                'step': step,
                'val_loss': val_loss_accum.item() if 'val_loss_accum' in locals() else float('inf'), # Save last val loss if available
                'optimizer': optimizer.state_dict(),
            }
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)
            print("Checkpoint saved.")

        # Training step
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        model.train() # Use the potentially DDP-wrapped model for training
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # DDP gradient accumulation context manager
            sync_context = model.no_sync if ddp and micro_step < grad_accum_steps - 1 else nullcontext()
            # Autocast for mixed precision
            ptdtype = torch.bfloat16 if device == 'cuda' else torch.float32
            ctx = torch.amp.autocast(device_type=device.split(':')[0], dtype=ptdtype) if device != 'cpu' else nullcontext()
            with sync_context:
                with ctx:
                    logits, loss = model(x, y)
            # Scale the loss for gradient accumulation
            # Integer division is required to avoid issues with floating point precision.
            loss = loss / grad_accum_steps # Scale loss
            loss_accum += loss.detach() # Accumulate detached loss for logging
            loss.backward() # Backward pass accumulates gradients

        # Synchronize accumulated loss across ranks for logging (if DDP)
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Clip gradients globally across all parameters
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # Use wrapped model's parameters

        # Determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # Step the optimizer
        optimizer.step()

        # Timing and logging
        if device == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        elif device == "mps":
            torch.mps.synchronize() # wait for the MPS device to finish work
        t1 = time.time()
        dt = (t1 - t0) * 1000 # time difference in milliseconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / (t1 - t0)

        if master_process and step % log_interval == 0:
            log_msg = f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            print(log_msg)
            with open(log_file, "a") as f:
                # Log train loss to file (keep simple format)
                f.write(f"{step} train {loss_accum.item():.6f}\n")
            # Log metrics to TensorBoard
            if writer:
                writer.add_scalar('Loss/train', loss_accum.item(), step)
                writer.add_scalar('LR', lr, step)
                writer.add_scalar('GradNorm', norm, step)
                writer.add_scalar('Perf/tokens_per_sec', tokens_per_sec, step)
                writer.add_scalar('Perf/ms_per_step', dt, step) # Log step time

    # Clean up DDP and TensorBoard writer
    if master_process and writer:
        writer.close()
        print("TensorBoard writer closed.")
    if ddp:
        destroy_process_group()

    # Final exit (optional, helps ensure script terminates cleanly)
    # import sys; sys.exit(0) # Commented out to allow potential post-training actions






"""     enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device) 


    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
    
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)  """
