import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional, Tuple
import torch.distributed as dist
from dataclasses import dataclass
import math
import inspect
from functools import cache
from flash_attn import flash_attn_func, flash_attn_with_kvcache
# from flash_attn.flash_attn_interface import _flash_attn_forward

#from: https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/utils.py

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    process_group=None,
    causal=True,
):
    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None
    k = k.contiguous()
    v = v.contiguous()
    b,t,nh,hs = q.shape
    if t==1:
        # Determine max size along t-axis across ranks
        local_t_size = k.size(1)  # Assuming shape is [batch, t, head_dim, ...]
        max_t_size = torch.tensor(local_t_size, device=k.device)
        dist.all_reduce(max_t_size, op=dist.ReduceOp.MAX, group=process_group)
        max_t_size = max_t_size.item()

        # Pad k and v to have the same t size
        pad_size = max_t_size - local_t_size

        if pad_size > 0:
            pad_tens = torch.zeros(b,pad_size,nh,hs).to(k.device)
            k = torch.cat((k, pad_tens),dim=1).to(q.dtype)
            v = torch.cat((v, pad_tens),dim=1).to(q.dtype)

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()
        
        if not causal or step <= comm.rank:
            
            if t!=1:

                block_out, block_lse,_ = flash_attn_func(q, k, v, 
                                                        softmax_scale= softmax_scale,
                                                        causal = causal and step==0, 
                                                        return_attn_probs=True)

                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            if t==1:

                # Clip k and v back to their original sizes before attention
                clipped_k = k[:, :local_t_size] if pad_size > 0 else k
                clipped_v = v[:, :local_t_size] if pad_size > 0 else v
                block_out, block_lse = flash_attn_with_kvcache(
                    q, clipped_k, clipped_v,
                    softmax_scale=softmax_scale,
                    #causal=causal and step == 0,
                    return_softmax_lse=True
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)


              
        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v
 
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out#, lse

def tree_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    eps=1e-8):

    if softmax_scale is None:
        scale = q.shape[-1] ** -0.5
    else:
        scale=softmax_scale


    local_out, lse = flash_attn_with_kvcache(q,k,v,  
                                        softmax_scale=scale, 
                                        #causal=True, 
                                        return_softmax_lse=True)
    max_lse = lse.clone()
    dist.all_reduce(max_lse, dist.ReduceOp.MAX)

    # derive numerator and denominator
    den = (lse - max_lse).exp()
    num = local_out * den

    # second and third all reduce (sum)

    dist.all_reduce(den)
    dist.all_reduce(num)

    out = num.div_(den.clamp(min = eps))

    return out

#Need: repeat kv, rope, cache utils and then finally llama atenttion
# from https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/llama2_model.py#L348

#matches: https://github.com/meta-llama/llama/blob/main/llama/model.py
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    intermediate_size: int = 4*dim


    norm_eps: float = 1e-5
    allgather = False

    max_batch_size: int = 32
    max_seq_len: int = 32768
    rope_kwargs: Optional = None
    ring_only:bool = False
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config = None,
    ):
        super().__init__()

        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            self.max_seq_len_cached = config['original_max_position_embeddings']
            self.original_max_seq_len = config['original_max_position_embeddings']

        self.config = config

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        factor = config["factor"]  # `8` in the original implementation
        low_freq_factor = config["low_freq_factor"]  # `1` in the original implementation
        high_freq_factor = config["high_freq_factor"]  # `4` in the original implementation
        old_context_len = config["original_max_position_embeddings"]  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        self.register_buffer("inv_freq", inv_freq_llama, persistent=False)
        self.original_inv_freq = self.inv_freq

   
    @torch.no_grad()
    def forward(self, x, position_ids):
        
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore

class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.

     

    Attributes:
        up_proj (Linear): Linear transformation for the first layer.
        gate_proj (Linear): Linear transformation for the second layer.
        down_proj (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,


    ):
        super().__init__()
    
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_local_kv_heads (int): Number of local key and value heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        q_proj (Linear): Linear transformation for queries.
        k_proj (Linear): Linear transformation for keys.
        v_proj (Linear): Linear transformation for values.
        o_proj (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.ring_only = model_args.ring_only
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.q_proj = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )



    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor],
        kv_cache: None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape # seq_len can be 1 during decoding 
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)

        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim) 
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim) 

        
        
        prefill = False

        if is_distributed():
            is_final_shard = (torch.distributed.get_rank() == torch.distributed.get_world_size() - 1)

            if kv_cache is not None:
                cached_k = kv_cache.get("k", None)
                cached_v = kv_cache.get("v", None)
                
                if cached_k is not None and cached_v is not None:

                    if is_final_shard:
                        xk = torch.cat([cached_k, xk], dim=1)
                        xv = torch.cat([cached_v, xv], dim=1)

                    else:
                        xk = cached_k 
                        xv = cached_v
            else:
                prefill=True
                kv_cache = {"k": xk, "v": xv}

        else:
        # Concatenate the cached keys/values if available
            if kv_cache is not None:
                cached_k = kv_cache.get("k", None)
                cached_v = kv_cache.get("v", None)

                if cached_k is not None and cached_v is not None:
                    xk = torch.cat([cached_k, xk], dim=1)
                    xv = torch.cat([cached_v, xv], dim=1)

            # Update the KV cache with current keys and values
            kv_cache = {"k": xk, "v": xv}
        
        xq, xk, xv = tuple(map(lambda t : t.transpose(1,2), (xq, xk, xv)))

        xq, xk = apply_rotary_emb(xq, xk, *freqs_cis)
        # Repeat the keys and values for local heads
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq, xk, xv = tuple(map(lambda t : t.transpose(1,2), (xq, xk, xv)))

        

        if is_distributed():

            if prefill:
                output = ring_attention(xq, xk, xv, softmax_scale=None, process_group=None)
            
            else: #decoding!
                
                if is_final_shard:
                    xq_last_step = xq[:, -1, ...].unsqueeze(1)
                    xq_last_step = xq_last_step.contiguous()
                else:
                    xq_last_step = torch.zeros_like(xq[:, -1, ...].unsqueeze(1)) 
                    xq_last_step = xq_last_step.contiguous()
                
                dist.broadcast(xq_last_step, src = torch.distributed.get_world_size() - 1)
                dist.barrier()
     
                if self.ring_only:
                    output = ring_attention(xq_last_step, xk, xv, softmax_scale=None, process_group=None)
                else:
                    output = tree_decode(xq_last_step, xk, xv, softmax_scale=None)
                

        else:
            output = flash_attn_func(xq, xk, xv, causal=True)

        output = output.contiguous()

        output = output.view(bsz, seqlen, -1)

        return self.o_proj(output), kv_cache



class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.self_attn = Attention(model_args)
        self.mlp = FeedForward(
            dim=model_args.dim,
            hidden_dim= model_args.intermediate_size,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.input_layernorm = RMSNorm(
            dim=model_args.dim, eps=model_args.norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            dim=model_args.dim, eps=model_args.norm_eps
        )



    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: dict = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
         # Normalize the input to attention layer
        normed_x = self.input_layernorm(x)

        # Pass through attention with KV caching
        attention_out, updated_kv_cache = self.self_attn(normed_x, freqs_cis, kv_cache=kv_cache)

        # Residual connection after attention
        h = x + attention_out

        # Normalize and apply feedforward layer
        out = h + self.mlp(self.post_attention_layernorm(h))

        return out, updated_kv_cache


class Transformer(nn.Module):
    """
    Transformer Module with KV caching support.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.model_dim = model_args.dim


        self.embed_tokens = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.rotary_emb = LlamaRotaryEmbedding(model_args.dim//model_args.n_heads,config=model_args.rope_kwargs)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(model_args.n_layers):
            self.layers.append(TransformerBlock(layer_id, model_args))

        self.norm = RMSNorm(
            dim=model_args.dim, eps=model_args.norm_eps
        )

        self.lm_head = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, kv_cache: dict = None):
        """
        Perform a forward pass through the Transformer model with KV caching.

        Args:
            tokens (torch.Tensor): Input token indices.
            kv_cache (dict, optional): Dictionary containing cached keys and values for each layer.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.
            dict: Updated KV cache.
        """
        _, seqlen = tokens.shape

        if kv_cache is not None:

            k_layer_cache= kv_cache.get(1,None)

            if k_layer_cache is not None:

                k_cache = k_layer_cache.get('k',None)

                if k_cache is not None:

                    _,c_seq_len,_,_ = k_cache.size()
                    seqlen += c_seq_len

        
        h = self.embed_tokens(tokens)


        if is_distributed():

            rank = dist.get_rank()

            if kv_cache is None:
                start = seqlen*rank #assumes sequence length is evenly divided by the world size
                end = seqlen*(rank+1)
                freqs_cis = self.rotary_emb(h, 
                                            position_ids = torch.arange(start,end).to(h.device).view(1,-1))

            else:
                new_toks = seqlen-kv_cache['prompt_len']
                chunk_size = kv_cache['prompt_len']
                start = chunk_size*rank #assumes sequence length is evenly divided by the world size
                end = chunk_size*(rank+1)

                if rank!=dist.get_world_size()-1:
                    freqs_cis = self.rotary_emb(h, 
                                                position_ids = torch.arange(start,end).to(h.device).view(1,-1))
                else:
                    end+=new_toks
                    freqs_cis = self.rotary_emb(h, 
                                                position_ids = torch.arange(start,end).to(h.device).view(1,-1))
                
        else:
            freqs_cis = self.rotary_emb(h, 
                                        position_ids = torch.arange(seqlen).to(h.device).view(1,-1))

            

        if kv_cache is None:
            kv_cache = {}
            kv_cache['prompt_len'] = seqlen




        # Forward pass through each layer with KV caching
        for layer_id, layer in enumerate(self.layers):
            layer_kv_cache = kv_cache.get(layer_id, None)
            h, updated_kv_cache = layer(h, freqs_cis, kv_cache=layer_kv_cache)
            kv_cache[layer_id] = updated_kv_cache  # Update the cache for this layer

        h = self.norm(h)
        output = self.lm_head(h).float()
        return output, kv_cache
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, num_new_tokens: int, kv_cache: dict = None):
        """
        Generate text using the Transformer model with greedy decoding.

        Args:
            input_ids (torch.Tensor): Input token indices of shape (batch_size, seq_len).
            num_new_tokens (int): Number of new tokens to generate.
            kv_cache (dict, optional): Dictionary containing cached keys and values for each layer.

        Returns:
            torch.Tensor: Generated token indices of shape (batch_size, seq_len + num_new_tokens).
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()

        def one_new_tok(logits,generated):
            # Get the next token (greedy decoding: take argmax)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if is_distributed():
                # append the next token on final rank only
                if (torch.distributed.get_rank() == torch.distributed.get_world_size() - 1):

                    generated = torch.cat([generated, next_token], dim=-1)
            else:
                generated = torch.cat([generated, next_token], dim=-1)

            return generated
        
         
        # Prefill for first token:
        if kv_cache is None:
            logits, kv_cache = self.forward(generated, kv_cache=kv_cache)
            generated = one_new_tok(logits,generated) #1 new token

            for _ in range(num_new_tokens-1):
                # Forward pass for the current sequence
                logits, kv_cache = self.forward(generated[:, -1:], kv_cache=kv_cache)
                generated = one_new_tok(logits,generated)
        else:
            for _ in range(num_new_tokens):
                # Forward pass for the current sequence
                logits, kv_cache = self.forward(generated[:, -1:], kv_cache=kv_cache)
                generated = one_new_tok(logits,generated)
       
        return generated

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "Transformer":
        """
        Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)

    @classmethod
    def from_pretrained(cls,model_name:str,ring_only=False) -> "Transformer":

        from transformers import AutoModelForCausalLM

        model_hf = AutoModelForCausalLM.from_pretrained(model_name,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)

        with torch.no_grad():

            m_st_dct = model_hf.model.state_dict()
            lmh_st_dct = model_hf.lm_head.state_dict()

            config = model_hf.config

            #del model_hf

            model_args = ModelArgs(
                dim = config.hidden_size,
                intermediate_size = config.intermediate_size,
                n_layers = config.num_hidden_layers,
                n_heads = config.num_attention_heads, 
                n_kv_heads = config.num_key_value_heads,
                vocab_size = config.vocab_size,
                max_seq_len = config.max_position_embeddings,
                rope_kwargs = config.rope_scaling,
                ring_only=ring_only,
            )
        

            model = Transformer.from_model_args(model_args)
            model.rotary_emb = model_hf.model.rotary_emb
            model.rotary_emb.inv_freq = model_hf.model.rotary_emb.inv_freq

            model.load_state_dict(m_st_dct,strict=False)
            model.lm_head.load_state_dict(lmh_st_dct,strict=False)

        return model, model_hf #model
        
