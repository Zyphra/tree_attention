import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from llama_flash_sp import *
import numpy as np  # For computing mean and std deviation

os.environ["HF_TOKEN"] = <hf_token>

def main(seq_len, num_new_tokens, ring):
    model, _ = Transformer.from_pretrained("meta-llama/Llama-3.2-1B", ring_only=ring)
    model = model.to(torch.bfloat16)

    x_in = torch.randint(0, model.model_args.vocab_size, (seq_len,)).view(1, -1)

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    ws = dist.get_world_size()

    model = model.to(rank)
    model.eval()
    chunk_size = seq_len // ws
    x = x_in[:, rank * chunk_size : (rank + 1) * chunk_size].to(rank)

    with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model.generate(x, num_new_tokens=1)
    torch.cuda.empty_cache()
    ttft_times = []
    deltat_times = []

    for _ in range(10):  # Perform 10 timing runs
        # Time to first token
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model.generate(x, num_new_tokens=1)
        end.record()
        torch.cuda.synchronize()
        ttft_times.append(start.elapsed_time(end))  # Record in milliseconds
        torch.cuda.empty_cache()

        # Time to generate all new tokens
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        start2.record()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model.generate(x, num_new_tokens=num_new_tokens)
        end2.record()
        torch.cuda.synchronize()
        deltat_times.append(start2.elapsed_time(end2))  # Record in milliseconds
        torch.cuda.empty_cache()

    if rank == ws - 1:
        ttft_mean = np.mean(ttft_times) / 1000  # Convert to seconds
        ttft_std = np.std(ttft_times) / 1000   # Convert to seconds
        deltat_mean = np.mean(deltat_times) / 1000  # Convert to seconds
        deltat_std = np.std(deltat_times) / 1000   # Convert to seconds

        if ring:
            print(f"With Ring decoding at seq_len {seq_len}:")
        else:
            print(f"With Tree decoding at seq_len {seq_len}:")
        print(f"Time to first token: {ttft_mean:.4f}s ± {ttft_std:.4f}s")
        print(f"Time to generate {num_new_tokens} new tokens: {deltat_mean:.4f}s ± {deltat_std:.4f}s")

    dist.destroy_process_group()

if __name__ == "__main__":
    #main(seq_len=20000, num_new_tokens=10, ring=True)
    main(seq_len=20000, num_new_tokens=10, ring=False)
    
