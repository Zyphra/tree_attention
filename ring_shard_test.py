import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from flash_attn_jax.flash import _flash_mha_fwd
from flash_attn_jax.ring_attention import ring_fwd
import jax.random as rand
import os
import time
import pickle

# slurm auto initializes this (see https://jax.readthedocs.io/en/latest/multi_process.html#initializing-the-cluster if not using slurm)
#jax.distributed.initialize()

mesh = Mesh(mesh_utils.create_device_mesh(jax.device_count()), axis_names=('i',))
seq_spec = P(None, 'i', None, None)



@partial(jax.jit,static_argnums=0)
def make_data(shape):
    B, nh, T, C = shape
    k1, k2, k3 = rand.split(rand.PRNGKey(0), 3)
    Q = rand.normal(k1, (B, 1, nh, C)).astype(jnp.float16)
    K = lax.with_sharding_constraint(
            rand.normal(k2, (B, T, nh, C)).astype(jnp.float16), NamedSharding(mesh, seq_spec)
    )
    V = lax.with_sharding_constraint(
            rand.normal(k3, (B, T, nh, C)).astype(jnp.float16), NamedSharding(mesh, seq_spec)
    )
    return Q, K, V

in_specs=(P(None, None, None, None), seq_spec, seq_spec)
out_specs=P(None, None, None,None)

@jax.jit
@partial(shard_map,mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
def ring_decode(q, k, v):
    out,_ = ring_fwd(
            q, k, v, axis_name='i',
            axis_size=jax.device_count(),
            mha_fwd=_flash_mha_fwd,
            softmax_scale=1.0
    )
    return out

if __name__ == '__main__':
    
    print(f"device count: {jax.device_count()}")
    seq_len = 64000
    num_heads = 16
    head_dim = 128
    qfl, kfl, vfl = make_data((1, num_heads, seq_len, head_dim))
    print(f"seq_len: {seq_len}, hid_dim: {num_heads*head_dim}")
    start_time = time.time()
    output = ring_decode(qfl, kfl, vfl).block_until_ready()
    end_time = time.time()
    print(f"time: {end_time-start_time}s")
   
