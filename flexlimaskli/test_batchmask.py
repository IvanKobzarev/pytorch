import pytest
import torch
import numpy as np

import flexlimaskli.docmask_gpu as uf
from flexlimaskli.test_utils import blockmask_to_dense


def _batched_element_dense(bm, b):
    """Extract dense block mask for batch element b from a batched BlockMask."""
    NB = bm.kv_num_blocks.shape[-1]
    dense = torch.zeros(1, 1, NB, NB, dtype=torch.bool)
    for name in ['kv', 'full_kv']:
        n = getattr(bm, f'{name}_num_blocks')[b, 0]
        idx = getattr(bm, f'{name}_indices')[b, 0]
        for qb in range(NB):
            for j in range(n[qb]):
                dense[0, 0, qb, idx[qb, j]] = True
    return dense


def test_batched_mini():
    """Batched with BLOCK_SIZE=2, single doc per element."""
    import flexlimaskli.docmask_cpu as ufn
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu, make_batchmask_numpy

    BS = 2
    ntoks = 12
    B = 3
    ar = np.zeros((B, ntoks), dtype=np.int64)
    ar[1, :6] = 1  # dense prefix on element 1
    ar[2, :4] = 2  # different dense prefix on element 2

    for make_fn in [make_batchmask_cpu, make_batchmask_numpy]:
        batched = make_fn(ntoks, ar, BLOCK_SIZE=BS)
        assert batched.kv_num_blocks.shape[0] == B

        for b in range(B):
            di = np.zeros(ntoks, dtype=np.int64)
            ref = ufn.make_docmask_numba(ntoks, ar[b], di, BLOCK_SIZE=BS)
            assert torch.equal(_batched_element_dense(batched, b), blockmask_to_dense(ref)), \
                f"Element {b}: dense masks differ ({make_fn.__name__})"


def test_batched_dense_variations():
    """Batched with varied dense region patterns per element."""
    import flexlimaskli.docmask_cpu as ufn
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu, make_batchmask_numpy

    BS = 64
    ntoks = 512
    B = 4
    ar = np.zeros((B, ntoks), dtype=np.int64)

    # Element 0: purely causal (decode-like, no dense)
    # Element 1: dense prefix (half)
    ar[1, :256] = 1
    # Element 2: dense prefix (short)
    ar[2, :100] = 1
    # Element 3: two distinct dense regions
    ar[3, :100] = 1
    ar[3, 300:400] = 2

    for make_fn in [make_batchmask_cpu, make_batchmask_numpy]:
        batched = make_fn(ntoks, ar, BLOCK_SIZE=BS)

        for b in range(B):
            di = np.zeros(ntoks, dtype=np.int64)
            ref = ufn.make_docmask_numba(ntoks, ar[b], di, BLOCK_SIZE=BS)
            assert torch.equal(_batched_element_dense(batched, b), blockmask_to_dense(ref)), \
                f"Element {b}: dense masks differ ({make_fn.__name__})"


def test_batched_against_gpu_reference():
    """Compare batched numba against GPU create_block_mask reference."""
    import flexlimaskli.docmask_cpu as ufn
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu

    BS = 128
    ntoks = 1024
    B = 2
    ar = np.zeros((B, ntoks), dtype=np.int64)
    ar[0, :200] = 1
    ar[1, :500] = 1

    batched = make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=BS)

    for b in range(B):
        di = torch.zeros(ntoks, dtype=torch.int64)
        ref = uf.make_docmask_gpu(ntoks, torch.tensor(ar[b]), di, BLOCK_SIZE=BS)
        assert torch.equal(_batched_element_dense(batched, b), blockmask_to_dense(ref)), \
            f"Element {b}: batched disagrees with GPU reference"


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_batchmask_gpu_vs_cpu(variant):
    """Compare make_batchmask_gpu against make_batchmask_cpu element-by-element."""
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu
    from flexlimaskli.batchmask_gpu import make_batchmask_gpu

    BS = 64
    ntoks = 512
    B = 4
    ar = np.zeros((B, ntoks), dtype=np.int64)
    ar[1, :256] = 1
    ar[2, :100] = 1
    ar[3, :100] = 1
    ar[3, 300:400] = 2

    cpu_mask = make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=BS)
    gpu_mask = make_batchmask_gpu(ntoks, torch.tensor(ar), BLOCK_SIZE=BS, compile=False)

    for b in range(B):
        assert torch.equal(_batched_element_dense(cpu_mask, b),
                           _batched_element_dense(gpu_mask, b)), \
            f"Element {b}: CPU vs GPU batchmask differ"


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_batchmask_gpu_vs_docmask(variant):
    """Compare make_batchmask_gpu against per-element make_docmask_gpu."""
    from flexlimaskli.batchmask_gpu import make_batchmask_gpu

    BS = 128
    ntoks = 1024
    B = 2
    ar = np.zeros((B, ntoks), dtype=np.int64)
    ar[0, :200] = 1
    ar[1, :500] = 1

    batched = make_batchmask_gpu(ntoks, torch.tensor(ar), BLOCK_SIZE=BS, compile=False)

    for b in range(B):
        di = torch.zeros(ntoks, dtype=torch.int64)
        ref = uf.make_docmask_gpu(ntoks, torch.tensor(ar[b]), di, BLOCK_SIZE=BS)
        assert torch.equal(_batched_element_dense(batched, b), blockmask_to_dense(ref)), \
            f"Element {b}: batchmask_gpu disagrees with per-element docmask_gpu"


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_batchmask_gpu_mini(variant):
    """Batchmask GPU with BLOCK_SIZE=2, matching the CPU mini test."""
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu
    from flexlimaskli.batchmask_gpu import make_batchmask_gpu

    BS = 2
    ntoks = 12
    B = 3
    ar = np.zeros((B, ntoks), dtype=np.int64)
    ar[1, :6] = 1
    ar[2, :4] = 2

    cpu_mask = make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=BS)
    gpu_mask = make_batchmask_gpu(ntoks, torch.tensor(ar), BLOCK_SIZE=BS, compile=False)

    for b in range(B):
        assert torch.equal(_batched_element_dense(cpu_mask, b),
                           _batched_element_dense(gpu_mask, b)), \
            f"Element {b}: CPU vs GPU batchmask differ"


def test_batched_compact_indices():
    """Compact index arrays: partial indices are narrower than NB."""
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu, make_batchmask_numpy

    BS = 64
    ntoks = 4096
    NB = ntoks // BS

    # Purely causal → partial arrays should be narrow (just diagonals)
    ar_causal = np.zeros((2, ntoks), dtype=np.int64)

    # Large dense prefix → different widths but still compact
    ar_dense = np.zeros((2, ntoks), dtype=np.int64)
    ar_dense[:, :3000] = 1

    for make_fn in [make_batchmask_cpu, make_batchmask_numpy]:
        mask_c = make_fn(ntoks, ar_causal, BLOCK_SIZE=BS)
        mask_d = make_fn(ntoks, ar_dense, BLOCK_SIZE=BS)

        # Partial kv_indices should be much narrower than NB
        assert mask_c.kv_indices.shape[-1] < NB, \
            f"kv_indices not compact: {mask_c.kv_indices.shape[-1]} vs NB={NB}"

        # All index tensors should have mark_dynamic set
        for m in [mask_c, mask_d]:
            for attr in ["kv_indices", "full_kv_indices", "q_indices", "full_q_indices"]:
                t = getattr(m, attr)
                assert hasattr(t, '_dynamo_dynamic_indices'), \
                    f"{attr} missing mark_dynamic ({make_fn.__name__})"


def test_batched_b1():
    """Batch size 1 matches unbatched."""
    import flexlimaskli.docmask_cpu as ufn
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu, make_batchmask_numpy

    BS = 128
    ntoks = 512
    ar = np.zeros((1, ntoks), dtype=np.int64)
    ar[0, :200] = 1

    di = np.zeros(ntoks, dtype=np.int64)
    ref = ufn.make_docmask_numba(ntoks, ar[0], di, BLOCK_SIZE=BS)
    for make_fn in [make_batchmask_cpu, make_batchmask_numpy]:
        batched = make_fn(ntoks, ar, BLOCK_SIZE=BS)
        assert torch.equal(_batched_element_dense(batched, 0), blockmask_to_dense(ref))


def test_batched_unaligned():
    """Non-block-aligned ntoks are handled internally (no caller padding needed)."""
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu, make_batchmask_numpy

    BS = 128
    for make_fn in [make_batchmask_cpu, make_batchmask_numpy]:
        for ntoks in [100, 200, 300, 500, 1000]:
            NB = (ntoks + BS - 1) // BS
            ar = np.zeros((2, ntoks), dtype=np.int64)
            ar[0, :ntoks // 3] = 1
            ar[1, :ntoks // 2] = 1
            mask = make_fn(ntoks, ar, BLOCK_SIZE=BS)
            assert mask.kv_num_blocks.shape == (2, 1, NB)

    # Also test make_docmask_numba (unbatched)
    import flexlimaskli.docmask_cpu as ufn
    for ntoks in [100, 200, 300, 500, 1000]:
        NB = (ntoks + BS - 1) // BS
        ar = np.zeros(ntoks, dtype=np.int64)
        ar[:ntoks // 3] = 1
        di = np.zeros(ntoks, dtype=np.int64)
        mask = ufn.make_docmask_numba(ntoks, ar, di, BLOCK_SIZE=BS)
        assert mask.kv_num_blocks.shape == (1, 1, NB)


@pytest.mark.gpu
def test_batched_compiled_flex_attention_no_recompile():
    """mark_dynamic on compact index tensors prevents recompilation under
    torch.compile(dynamic=False) when index widths vary between calls.

    Mirrors docmask's test_v3_mark_dynamic_recompile: creates two masks with
    genuinely different compact widths (causal vs fully-dense), verifies shapes
    differ, then verifies compiled flex_attention doesn't recompile."""
    from torch.nn.attention.flex_attention import flex_attention
    from flexlimaskli.to_gpu import blockmask_to_gpu
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu

    BS = 128
    ntoks = 4096
    B = 2
    NB = ntoks // BS

    # Mask A: purely causal → full_kv_indices width = NB-1
    ar_causal = np.zeros((B, ntoks), dtype=np.int64)

    # Mask B: fully dense → full_kv_indices width = NB
    ar_dense = np.ones((B, ntoks), dtype=np.int64)

    mask_a = blockmask_to_gpu(make_batchmask_cpu(ntoks, ar_causal, BLOCK_SIZE=BS), "cuda")
    mask_b = blockmask_to_gpu(make_batchmask_cpu(ntoks, ar_dense, BLOCK_SIZE=BS), "cuda")

    # Verify shapes genuinely differ (compact arrays are data-dependent)
    assert mask_a.full_kv_indices.shape[-1] != mask_b.full_kv_indices.shape[-1], \
        f"Test setup error: full_kv shapes should differ, got " \
        f"{mask_a.full_kv_indices.shape[-1]} and {mask_b.full_kv_indices.shape[-1]}"

    # Compile flex_attention with strict settings
    torch._dynamo.config.recompile_limit = 1
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    cflex = torch.compile(flex_attention, dynamic=False, fullgraph=True)

    q = torch.randn(B, 1, ntoks, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, 1, ntoks, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, 1, ntoks, 64, device="cuda", dtype=torch.bfloat16)

    # First call compiles
    with torch.no_grad():
        cflex(q, k, v, block_mask=mask_a)

    # Second call with different compact widths — must NOT recompile (mark_dynamic)
    with torch.no_grad():
        cflex(q, k, v, block_mask=mask_b)


def test_batched_edge_case_sizes():
    """Test with various error-prone sizes, including very small sequences.

    Mirrors docmask's test_edge_case_sizes."""
    import flexlimaskli.docmask_cpu as ufn
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu, make_batchmask_numpy

    BS = 128
    test_cases = [1, 2, 3, 63, 127, 128, 129, 255, 256, 257]

    for ntoks in test_cases:
        NB = (ntoks + BS - 1) // BS
        ar = np.zeros((2, ntoks), dtype=np.int64)
        ar[0, :max(1, ntoks // 3)] = 1

        for make_fn in [make_batchmask_cpu, make_batchmask_numpy]:
            mask = make_fn(ntoks, ar, BLOCK_SIZE=BS)
            assert mask.kv_num_blocks.shape == (2, 1, NB), \
                f"ntoks={ntoks}: wrong shape {mask.kv_num_blocks.shape}"

            # Verify against per-element docmask reference
            for b in range(2):
                di = np.zeros(ntoks, dtype=np.int64)
                ref = ufn.make_docmask_numba(ntoks, ar[b], di, BLOCK_SIZE=BS)
                assert torch.equal(_batched_element_dense(mask, b), blockmask_to_dense(ref)), \
                    f"ntoks={ntoks}, b={b}: dense masks differ ({make_fn.__name__})"


def test_batched_dense_at_block_boundary():
    """Dense regions starting/ending exactly at or straddling block boundaries.

    Mirrors docmask's test_block_boundary_straddle: ensures partial/full
    classification is correct when dense regions align with block edges."""
    import flexlimaskli.docmask_cpu as ufn
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu, make_batchmask_numpy

    BS = 64
    ntoks = 512
    B = 4
    NB = ntoks // BS

    ar = np.zeros((B, ntoks), dtype=np.int64)

    # Element 0: dense region exactly aligned to 2 block boundaries
    ar[0, :128] = 1
    # Element 1: dense region ending mid-block
    ar[1, :100] = 1
    # Element 2: dense region starting mid-block
    ar[2, 50:200] = 1
    # Element 3: two adjacent dense regions with different IDs at a block boundary
    ar[3, :64] = 1   # exactly one block
    ar[3, 64:200] = 2  # starts at next block boundary

    for make_fn in [make_batchmask_cpu, make_batchmask_numpy]:
        mask = make_fn(ntoks, ar, BLOCK_SIZE=BS)

        for b in range(B):
            di = np.zeros(ntoks, dtype=np.int64)
            ref = ufn.make_docmask_numba(ntoks, ar[b], di, BLOCK_SIZE=BS)
            assert torch.equal(_batched_element_dense(mask, b), blockmask_to_dense(ref)), \
                f"b={b}: dense masks differ ({make_fn.__name__})"


def test_batched_mark_dynamic_preserved_by_to_gpu():
    """blockmask_to_gpu preserves _dynamo_dynamic_indices set by make_batchmask_cpu.

    This is critical for torch.compile compatibility: the compact index arrays
    have data-dependent widths, and mark_dynamic must survive the CPU→GPU transfer."""
    from flexlimaskli.batchmask_cpu import make_batchmask_cpu
    from flexlimaskli.to_gpu import blockmask_to_gpu

    BS = 128
    ntoks = 1024
    ar = np.zeros((2, ntoks), dtype=np.int64)
    ar[0, :200] = 1

    cpu_mask = make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=BS)

    attrs = ["kv_indices", "full_kv_indices", "q_indices", "full_q_indices"]

    # Verify mark_dynamic is set on CPU mask
    for attr in attrs:
        t = getattr(cpu_mask, attr)
        assert hasattr(t, '_dynamo_dynamic_indices'), \
            f"CPU {attr} missing mark_dynamic"

    # Transfer (to CPU device to avoid needing GPU for this check)
    transferred = blockmask_to_gpu(cpu_mask, "cpu")

    # Verify mark_dynamic preserved after transfer
    for attr in attrs:
        t = getattr(transferred, attr)
        assert hasattr(t, '_dynamo_dynamic_indices'), \
            f"After blockmask_to_gpu: {attr} lost mark_dynamic"
