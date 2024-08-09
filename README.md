<h1 align="center">
<p>Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters</p>
</h1>

<p align="center">
    <a>
        <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue">
    </a>
    <a>
        <img alt="JAX" src="https://img.shields.io/badge/JAX-0.4.31-blue">
    </a>
    <a>
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
</p>

Our paper can be found here: [https://arxiv.org/pdf/2408.04093].

## Requirements

- Python > 3.7
- JAX-0.4.31


### Install using Conda and pip

```bash
# Create a virtual environment
conda install cuda -c nvidia
pip install -U "jax[cuda12]"
pip install flash-attn-jax
```

### Running experiments

