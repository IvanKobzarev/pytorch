import pytest
import torch
import numpy as np

import flexmaskli.docmask_gpu as uf
from flexmaskli.test_utils import blockmask_to_dense, compare_block_masks


def _batched_element_dense(bm, b):
    """Extract dense mask for batch element b, mirroring flex_attention behavior."""
    return blockmask_to_dense(bm, b=b)


def test_batched_mini():
    """Batched with BLOCK_SIZE=2, single doc per element."""
    import flexmaskli.docmask_cpu as ufn
    from flexmaskli.batchmask_cpu import make_batchmask_cpu, make_batchmask_numpy

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



def test_batched_against_gpu_reference():
    """Compare batched numba against GPU create_block_mask reference."""
    import flexmaskli.docmask_cpu as ufn
    from flexmaskli.batchmask_cpu import make_batchmask_cpu

    BS = 128
    ntoks = 1024
    B = 5
    ar = np.zeros((B, ntoks), dtype=np.int64)
    ar[0, :200] = 1
    ar[1, :500] = 1
    # Same-value non-contiguous regions (zero gap)
    ar[2, :200] = 1
    ar[2, 400:800] = 1
    # Same value with different-value gap
    ar[3, :200] = 1
    ar[3, 200:400] = 2
    ar[3, 400:700] = 1
    # attn_regions2 pattern
    ar[4, :100] = 1
    ar[4, 100:500] = -1
    ar[4, 500:600] = 1

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
    from flexmaskli.batchmask_cpu import make_batchmask_cpu
    from flexmaskli.batchmask_gpu import make_batchmask_gpu

    BS = 64
    ntoks = 512
    B = 7
    ar = np.zeros((B, ntoks), dtype=np.int64)
    ar[1, :256] = 1
    ar[2, :100] = 1
    ar[3, :100] = 1
    ar[3, 300:400] = 2
    # Same-value non-contiguous regions (zero gap)
    ar[4, :100] = 1
    ar[4, 200:400] = 1
    # Same value with different-value gap (val=1, val=2, val=1)
    ar[5, :100] = 1
    ar[5, 100:200] = 2
    ar[5, 200:350] = 1
    # attn_regions2 pattern (question=1, image=-1, regs=1)
    ar[6, :52] = 1
    ar[6, 52:248] = -1
    ar[6, 248:260] = 1

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
    from flexmaskli.batchmask_gpu import make_batchmask_gpu

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
    from flexmaskli.batchmask_cpu import make_batchmask_cpu
    from flexmaskli.batchmask_gpu import make_batchmask_gpu

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


def test_batched_b1():
    """Batch size 1 matches unbatched."""
    import flexmaskli.docmask_cpu as ufn
    from flexmaskli.batchmask_cpu import make_batchmask_cpu, make_batchmask_numpy

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
    from flexmaskli.batchmask_cpu import make_batchmask_cpu, make_batchmask_numpy

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
    import flexmaskli.docmask_cpu as ufn
    for ntoks in [100, 200, 300, 500, 1000]:
        NB = (ntoks + BS - 1) // BS
        ar = np.zeros(ntoks, dtype=np.int64)
        ar[:ntoks // 3] = 1
        di = np.zeros(ntoks, dtype=np.int64)
        mask = ufn.make_docmask_numba(ntoks, ar, di, BLOCK_SIZE=BS)
        assert mask.kv_num_blocks.shape == (1, 1, NB)


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_batchmask_decode_patterns(variant):
    """Decode-like patterns where prompt is followed by padding then a decode token.

    Mirrors real eval/decode_lib.py layouts: prompt (dense+AR) is padded to
    max_prefix+max_decode with 0s, then tokens are placed one-at-a-time."""
    from flexmaskli.batchmask_cpu import make_batchmask_cpu
    from flexmaskli.batchmask_gpu import make_batchmask_gpu

    BS = 64
    ntoks = 512  # max_prefix + max_decode
    B = 5
    ar = np.zeros((B, ntoks), dtype=np.int64)

    # Element 0: simple decode — [img(1)] [AR(0)] [pad(0)] [decode(0)]
    ar[0, :100] = 1

    # Element 1: attn_regions2 decode — [Q(1)] [img(-1)] [reg(1)] [AR(0)] [pad(0)]
    ar[1, :30] = 1         # BOS + question
    ar[1, 30:220] = -1     # image patches
    ar[1, 220:230] = 1     # registers
    # rest is 0 = AR + padding + decode positions

    # Element 2: longer prompt, short decode area
    ar[2, :50] = 1         # question
    ar[2, 50:350] = -1     # large image
    ar[2, 350:370] = 1     # registers
    # 370..512 = AR + pad + decode

    # Element 3: very short prompt with attn_regions2 pattern
    ar[3, :10] = 1         # question
    ar[3, 10:80] = -1      # image
    ar[3, 80:90] = 1       # registers
    # 90..512 = AR + pad + decode

    # Element 4: two images with registers between (multi-image decode)
    ar[4, :20] = 1         # question
    ar[4, 20:120] = -1     # image 1
    ar[4, 120:130] = 1     # registers
    ar[4, 130:230] = -1    # image 2
    ar[4, 230:240] = 1     # more registers
    # 240..512 = AR + pad + decode

    cpu_mask = make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=BS)
    gpu_mask = make_batchmask_gpu(ntoks, torch.tensor(ar), BLOCK_SIZE=BS, compile=False)

    for b in range(B):
        assert torch.equal(_batched_element_dense(cpu_mask, b),
                           _batched_element_dense(gpu_mask, b)), \
            f"Element {b}: CPU vs GPU batchmask differ"


@pytest.mark.gpu
def test_compiled_flex_attention_cpu_vs_gpu_mask():
    """Diagnose: does compiled flex_attention produce correct results when
    called multiple times with different CPU-built batchmasks?

    Compares outputs of compiled flex_attention using:
    1. Non-compiled flex_attention as ground truth
    2. CPU-built masks vs GPU-built masks
    3. Two different attn_regions patterns with the SAME tensor shapes

    If compiled CPU masks produce wrong results but GPU masks are correct,
    the issue is in how torch.compile handles the CPU mask_mod's closure tensor.
    """
    from torch.nn.attention.flex_attention import flex_attention
    from flexmaskli.batchmask_cpu import make_batchmask_cpu
    from flexmaskli.batchmask_gpu import make_batchmask_gpu
    from flexmaskli.to_gpu import blockmask_to_gpu

    BS = 128
    ntoks = 512
    B = 4
    head_dim = 64

    # Two VERY different attn_regions patterns (same shape → no recompile trigger)
    ar1 = np.zeros((B, ntoks), dtype=np.int64)
    ar1[0, :200] = 1   # large dense prefix
    ar1[1, :100] = 1
    ar1[2, :50] = 1; ar1[2, 50:250] = -1; ar1[2, 250:270] = 1  # attn_regions2
    ar1[3, :300] = 1

    ar2 = np.zeros((B, ntoks), dtype=np.int64)
    # Purely causal (no dense) — maximally different from ar1

    # Fixed Q, K, V
    q = torch.randn(B, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)

    # --- Reference: non-compiled flex_attention ---
    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
    cpu1 = blockmask_to_gpu(make_batchmask_cpu(ntoks, ar1, BLOCK_SIZE=BS), "cuda")
    cpu2 = blockmask_to_gpu(make_batchmask_cpu(ntoks, ar2, BLOCK_SIZE=BS), "cuda")
    gpu1 = blockmask_to_gpu(make_batchmask_gpu(ntoks, torch.tensor(ar1), BLOCK_SIZE=BS, compile=False), "cuda")
    gpu2 = blockmask_to_gpu(make_batchmask_gpu(ntoks, torch.tensor(ar2), BLOCK_SIZE=BS, compile=False), "cuda")

    with torch.no_grad():
        ref1 = flex_attention(q, k, v, block_mask=gpu1)
        ref2 = flex_attention(q, k, v, block_mask=gpu2)
    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = False

    # Sanity: different masks should produce different outputs
    assert not torch.allclose(ref1, ref2, atol=1e-2), \
        "Test setup error: different masks should give different outputs"

    # --- Test: compiled flex_attention with CPU masks ---
    torch._dynamo.reset()
    cflex = torch.compile(flex_attention, dynamic=False, fullgraph=True)
    with torch.no_grad():
        out_cpu1 = cflex(q, k, v, block_mask=cpu1)
        out_cpu2 = cflex(q, k, v, block_mask=cpu2)

    # Check 1: CPU mask batch 1 matches reference
    diff1 = (out_cpu1 - ref1).abs().max().item()

    # Check 2: CPU mask batch 2 matches reference (detects stale closure tensor)
    diff2 = (out_cpu2 - ref2).abs().max().item()
    stale = torch.allclose(out_cpu1, out_cpu2, atol=1e-3)

    # Check which reference out_cpu2 is closer to
    diff2_vs_ref1 = (out_cpu2 - ref1).abs().max().item()
    print(f"\n=== CPU MASK DIAGNOSTICS ===")
    print(f"  Batch 1 vs ref1: max_diff={diff1:.6f}  (should be ~0)")
    print(f"  Batch 2 vs ref2: max_diff={diff2:.6f}  (should be ~0)")
    print(f"  Batch 2 vs ref1: max_diff={diff2_vs_ref1:.6f}  (closeness to wrong ref)")
    print(f"  Batch 1 == Batch 2: {stale}")
    print(f"  CPU mask1 compact widths: kv={cpu1.kv_indices.shape[-1]} fkv={cpu1.full_kv_indices.shape[-1]}")
    print(f"  CPU mask2 compact widths: kv={cpu2.kv_indices.shape[-1]} fkv={cpu2.full_kv_indices.shape[-1]}")
    print(f"  GPU mask1 compact widths: kv={gpu1.kv_indices.shape[-1]} fkv={gpu1.full_kv_indices.shape[-1]}")
    print(f"  GPU mask2 compact widths: kv={gpu2.kv_indices.shape[-1]} fkv={gpu2.full_kv_indices.shape[-1]}")

    # --- Test: compiled flex_attention with GPU masks ---
    torch._dynamo.reset()
    cflex_g = torch.compile(flex_attention, dynamic=False, fullgraph=True)
    with torch.no_grad():
        out_gpu1 = cflex_g(q, k, v, block_mask=gpu1)
        out_gpu2 = cflex_g(q, k, v, block_mask=gpu2)

    diff_g1 = (out_gpu1 - ref1).abs().max().item()
    diff_g2 = (out_gpu2 - ref2).abs().max().item()
    print(f"\n=== GPU MASK DIAGNOSTICS ===")
    print(f"  Batch 1 vs ref1: max_diff={diff_g1:.6f}")
    print(f"  Batch 2 vs ref2: max_diff={diff_g2:.6f}")

    # --- Assertions ---
    assert diff1 < 0.02, f"CPU mask batch 1 wrong! {diff1:.6f}"
    assert diff2 < 0.02, \
        f"CPU mask batch 2 wrong! Max diff vs ref: {diff2:.6f}. " \
        f"Same as batch 1: {stale}. vs ref1: {diff2_vs_ref1:.6f}"
    assert diff_g1 < 0.02, f"GPU mask batch 1 wrong! {diff_g1:.6f}"
    assert diff_g2 < 0.02, f"GPU mask batch 2 wrong! {diff_g2:.6f}"


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_batchmask_complex_decode_patterns(variant):
    """Complex decode patterns with multiple non-contiguous same-value regions.

    Tests scenarios like:
    [img] [hole(-1)] [reg] [hole] [AR] [hole] [reg] [hole] [decode_tok]
    """
    from flexmaskli.batchmask_cpu import make_batchmask_cpu
    from flexmaskli.batchmask_gpu import make_batchmask_gpu

    BS = 64
    ntoks = 1024
    B = 6
    ar = np.zeros((B, ntoks), dtype=np.int64)

    # Element 0: [img(1)] [hole(-1)] [reg(1)] [hole(-1)] [AR(0)] [hole(-1)] [reg(1)] [hole(-1)] [decode(0)]
    ar[0, :100] = 1        # image
    ar[0, 100:150] = -1    # hole
    ar[0, 150:170] = 1     # registers
    ar[0, 170:200] = -1    # hole
    ar[0, 200:400] = 0     # AR tokens
    ar[0, 400:450] = -1    # hole
    ar[0, 450:470] = 1     # registers (second set)
    ar[0, 470:500] = -1    # hole
    # 500..1024 = decode area (0)

    # Element 1: three separate value=1 regions with -1 gaps
    ar[1, :80] = 1         # region A
    ar[1, 80:200] = -1     # gap
    ar[1, 200:250] = 1     # region B
    ar[1, 250:400] = -1    # gap
    ar[1, 400:430] = 1     # region C

    # Element 2: value=2 regions mixed with value=1 regions
    ar[2, :60] = 1         # question (dense group 1)
    ar[2, 60:200] = 2      # image (dense group 2)
    ar[2, 200:220] = 1     # registers (dense group 1 again)
    ar[2, 220:400] = -1    # invisible gap
    ar[2, 400:420] = 2     # second image (dense group 2 again)

    # Element 3: alternating value=1 and value=-1, many small regions
    for i in range(0, 640, 64):
        ar[3, i:i+32] = 1
        ar[3, i+32:i+64] = -1

    # Element 4: realistic attn_regions2 with block-aligned boundaries
    ar[4, :64] = 1         # question (exactly 1 block)
    ar[4, 64:448] = -1     # image (exactly 6 blocks)
    ar[4, 448:512] = 1     # registers (exactly 1 block)

    # Element 5: realistic attn_regions2 with non-aligned boundaries
    ar[5, :52] = 1         # question (partial block)
    ar[5, 52:248] = -1     # image (straddles blocks)
    ar[5, 248:260] = 1     # registers (small, within one block)

    cpu_mask = make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=BS)
    gpu_mask = make_batchmask_gpu(ntoks, torch.tensor(ar), BLOCK_SIZE=BS, compile=False)

    for b in range(B):
        assert torch.equal(_batched_element_dense(cpu_mask, b),
                           _batched_element_dense(gpu_mask, b)), \
            f"Element {b}: CPU vs GPU batchmask differ"


@pytest.mark.gpu
def test_batched_full_width_indices():
    """CPU batchmask uses full NB-width index arrays (no compaction)."""
    from flexmaskli.batchmask_cpu import make_batchmask_cpu
    BS = 128
    ntoks = 512
    NB = ntoks // BS
    ar = np.zeros((2, ntoks), dtype=np.int64)
    ar[0, :200] = 1
    mask = make_batchmask_cpu(ntoks, ar, BLOCK_SIZE=BS)
    assert mask.kv_indices.shape[-1] == NB
    assert mask.full_kv_indices.shape[-1] == NB
