import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.sharding import Mesh,NamedSharding,PositionalSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from flash_attn_jax.flash import _flash_mha_vjp
import jax.random as rand
import os
import time
import pickle

# slurm auto initializes this (see https://jax.readthedocs.io/en/latest/multi_process.html#initializing-the-cluster if not using slurm)
#jax.distributed.initialize()

mesh = Mesh(mesh_utils.create_device_mesh(jax.device_count(),), axis_names=('i',))
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
@partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
def tree_decode(q, k, v):

    def flash_res_lse(q, k, v, config=dict(softmax_scale=1.0, is_causal=False, window_size=(-1, -1))):
        tup = _flash_mha_vjp.fwd(q, k, v, config)
        res, lse = tup[1][3], tup[1][4]
        return res, lse
    
    loc_res, loc_lse = flash_res_lse(q, k, v)
    a_max_global = lax.pmax(loc_lse, axis_name='i')
    num_global = lax.psum(loc_res * jnp.exp(loc_lse - a_max_global), axis_name='i')
    den_global = lax.psum(jnp.exp(loc_lse - a_max_global), axis_name='i')
    return num_global / den_global

if __name__ == '__main__':

    print(f"device count: {jax.device_count()}")
    seq_len = 64000
    num_heads = 16
    head_dim = 128
    qfl, kfl, vfl = make_data((1, num_heads, seq_len, head_dim))
    print(f"seq_len: {seq_len}, hid_dim: {num_heads*head_dim}")
    start_time = time.time()
    output = tree_decode(qfl,kfl,vfl).block_until_ready()
    end_time = time.time()
    print(f"time: {end_time-start_time}s")


    
