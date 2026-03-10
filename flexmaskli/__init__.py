"""Flexmaskli: flex-attention BlockMask creation library.

Two mask types:
- **docmask**: Multi-document packed sequences. Multiple documents in one sequence,
  with per-document causal attention and optional dense (bidirectional) regions.
  CPU (numpy/numba) and GPU (superblock) implementations.
- **batchmask**: Batched single-document sequences. One document per batch element,
  with causal + dense regions. CPU (numba) and GPU implementations, designed for decoding.
"""

# --- docmask: multi-document packed sequences ---
from .docmask_cpu import (
    make_docmask_numpy, make_docmask_numba, make_docmask_cpu, HAS_NUMBA,
)
from .docmask_gpu import (
    make_docmask_gpu_v3 as make_docmask_gpu,
)

# --- batchmask: batched single-document sequences ---
from .batchmask_cpu import (
    make_batchmask_numpy, make_batchmask_numba, make_batchmask_cpu,
)
from .batchmask_gpu import make_batchmask_gpu

# --- utilities ---
from .to_gpu import blockmask_to_gpu
