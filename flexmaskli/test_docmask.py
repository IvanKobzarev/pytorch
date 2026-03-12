from functools import partial

import pytest
import torch

import flexmaskli.docmask_gpu as uf
from flexmaskli.test_utils import blockmask_to_dense, compare_block_masks


def _dense_docmask(make_fn, ntoks, attn_regions, document_ids, BS):
    return blockmask_to_dense(
        make_fn(ntoks, attn_regions, document_ids, BLOCK_SIZE=BS)
    )[0, 0, :ntoks, :ntoks]


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
    pytest.param("gpu-compile", marks=pytest.mark.gpu)
])
def test_mini(variant):
    """Test basic functionality with simple document structure"""
    device = "cuda" if variant.startswith("gpu") else "cpu"

    document_ids = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4], device=device)
    attn_regions = torch.zeros(len(document_ids), dtype=torch.int32, device=device)

    # Expected full mask (putting doc id for 1's for readability):
    # 1 0 0 0  0 0 0 0  0 0 0 0
    # 0 2 0 0  0 0 0 0  0 0 0 0
    # 0 2 2 0  0 0 0 0  0 0 0 0
    # 0 0 0 3  0 0 0 0  0 0 0 0
    #
    # 0 0 0 3  3 0 0 0  0 0 0 0
    # 0 0 0 3  3 3 0 0  0 0 0 0
    # 0 0 0 0  0 0 4 0  0 0 0 0
    # 0 0 0 0  0 0 4 4  0 0 0 0
    #
    # 0 0 0 0  0 0 4 4  4 0 0 0
    # 0 0 0 0  0 0 4 4  4 4 0 0
    # 0 0 0 0  0 0 4 4  4 4 4 0
    # 0 0 0 0  0 0 4 4  4 4 4 4

    kwargs = dict(ntoks=len(document_ids), document_ids=document_ids, attn_regions=attn_regions, compile="compile" in variant)

    ref_mask = uf.make_docmask_gpu(BLOCK_SIZE=2, **kwargs)
    super_mask_v2 = uf.make_docmask_gpu_v2(BLOCK_SIZE=2, SUPERBLOCK_SIZE=4, **kwargs)
    compare_block_masks(ref_mask, super_mask_v2)
    super_mask_v3 = uf.make_docmask_gpu_v3(BLOCK_SIZE=2, SUPERBLOCK_SIZE=4, **kwargs)
    compare_block_masks(ref_mask, super_mask_v3)

    import flexmaskli.docmask_cpu as ufn
    cpu_kwargs = dict(ntoks=len(document_ids), document_ids=document_ids.cpu(), attn_regions=attn_regions.cpu())
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        compare_block_masks(ref_mask, fn(BLOCK_SIZE=2, **cpu_kwargs))


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
    pytest.param("gpu-compile", marks=pytest.mark.gpu)
])
def test_simple_case(variant):
    """Test basic functionality with simple document structure"""
    device = "cuda" if variant.startswith("gpu") else "cpu"

    ntoks = 256
    document_ids = torch.tensor([0] * 128 + [1] * 128, device=device)
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)

    ref_mask = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, compile="compile" in variant)

    super_mask_v2 = uf.make_docmask_gpu_v2(ntoks, attn_regions, document_ids, SUPERBLOCK_SIZE=256, compile="compile" in variant)
    compare_block_masks(ref_mask, super_mask_v2)

    import flexmaskli.docmask_cpu as ufn
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        compare_block_masks(ref_mask, fn(ntoks, attn_regions.cpu(), document_ids.cpu()))


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
    pytest.param("gpu-compile", marks=pytest.mark.gpu)
])
def test_edge_case_sizes(variant):
    """Test with various error-prone sizes"""
    device = "cuda" if variant.startswith("gpu") else "cpu"
    compile_flag = "compile" in variant
    test_cases = [127, 128, 129, 255, 256, 257, 1, 63]

    for ntoks in test_cases:
        if compile_flag:
            # This loop intentionally changes ntoks and closures every iteration.
            # Reset/clear compiled create_block_mask cache to avoid stale compiled state.
            torch._dynamo.reset()
            uf.maybe_compiled_fn.cache_clear()
        mid = ntoks // 2
        document_ids = torch.tensor([0] * mid + [1] * (ntoks - mid), device=device)
        attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)

        ref_mask = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, compile=compile_flag)

        import flexmaskli.docmask_cpu as ufn
        for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
            compare_block_masks(ref_mask, fn(ntoks, attn_regions.cpu(), document_ids.cpu()))


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
    pytest.param("gpu-compile", marks=pytest.mark.gpu)
])
def test_complex_document_structure(variant):
    """Test with more complex document and attention region patterns"""
    device = "cuda" if variant.startswith("gpu") else "cpu"

    ntoks = 384
    document_ids = torch.tensor([i // 64 for i in range(ntoks)], device=device)
    attn_regions = torch.tensor([0 if i < 192 else 1 for i in range(ntoks)], device=device)

    ref_mask = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, compile="compile" in variant)

    import flexmaskli.docmask_cpu as ufn
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        compare_block_masks(ref_mask, fn(ntoks, attn_regions.cpu(), document_ids.cpu()))


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
    pytest.param("gpu-compile", marks=pytest.mark.gpu)
])
def test_with_padding(variant):
    """Test with padding tokens (attn_regions = -1)"""
    device = "cuda" if variant.startswith("gpu") else "cpu"

    ntoks = 256
    document_ids = torch.tensor([0] * 128 + [1] * 128, device=device)
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)
    attn_regions[-32:] = -1

    ref_mask = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, compile="compile" in variant)

    import flexmaskli.docmask_cpu as ufn
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        compare_block_masks(ref_mask, fn(ntoks, attn_regions.cpu(), document_ids.cpu()))



# Test functions that require SUPERBLOCK_SIZE alignment
def _test_super_functions_helper(ntoks, document_ids, attn_regions, device, compile_flag, BLOCK_SIZE=128, SUPERBLOCK_SIZE=1024):
    """Helper to test super functions with proper size constraints"""
    import flexmaskli.docmask_cpu as ufn

    # Ensure ntoks is aligned to SUPERBLOCK_SIZE
    aligned_ntoks = ((ntoks + SUPERBLOCK_SIZE - 1) // SUPERBLOCK_SIZE) * SUPERBLOCK_SIZE

    # Pad inputs if needed
    if aligned_ntoks > ntoks:
        pad_size = aligned_ntoks - ntoks
        # Pad with -1 for both (padding tokens, not real documents)
        document_ids = torch.cat([document_ids, torch.full((pad_size,), -1, device=device)])
        attn_regions = torch.cat([attn_regions, torch.full((pad_size,), -1, dtype=torch.int32, device=device)])

    # Get reference (only test up to original ntoks to avoid padding effects)
    ref_mask = uf.make_docmask_gpu(aligned_ntoks, attn_regions, document_ids, BLOCK_SIZE=BLOCK_SIZE, compile=compile_flag)

    # Test super function v2
    super_mask_v2 = uf.make_docmask_gpu_v2(aligned_ntoks, attn_regions, document_ids,
                                             BLOCK_SIZE=BLOCK_SIZE, SUPERBLOCK_SIZE=SUPERBLOCK_SIZE, compile=False)

    # Compare to reference
    compare_block_masks(ref_mask, super_mask_v2)

    # Test super function v3
    super_mask_v3 = uf.make_docmask_gpu_v3(aligned_ntoks, attn_regions, document_ids,
                                             BLOCK_SIZE=BLOCK_SIZE, SUPERBLOCK_SIZE=SUPERBLOCK_SIZE, compile=False)
    compare_block_masks(ref_mask, super_mask_v3)

    # Test numpy and numba implementations against exact GPU reference
    cpu_ar = attn_regions.cpu()
    cpu_di = document_ids.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        compare_block_masks(ref_mask, fn(aligned_ntoks, cpu_ar, cpu_di, BLOCK_SIZE=BLOCK_SIZE))


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
    pytest.param("gpu-compile", marks=pytest.mark.gpu)
])
def test_super_simple_case(variant):
    """Test super functions with simple document structure"""
    device = "cuda" if variant.startswith("gpu") else "cpu"
    compile_flag = "compile" in variant

    # Test case that aligns well with superblocks
    ntoks = 1024  # Exactly 1 superblock
    document_ids = torch.tensor([0] * 512 + [1] * 512, device=device)
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)

    _test_super_functions_helper(ntoks, document_ids, attn_regions, device, compile_flag, SUPERBLOCK_SIZE=1024)


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
    pytest.param("gpu-compile", marks=pytest.mark.gpu)
])
def test_super_edge_cases(variant):
    """Test super functions with tricky sizes around block/superblock boundaries"""
    device = "cuda" if variant.startswith("gpu") else "cpu"
    compile_flag = "compile" in variant

    BLOCK_SIZE = 128
    SUPERBLOCK_SIZE = 512  # Smaller superblock for more interesting boundaries

    # Test cases that create interesting boundary conditions
    test_cases = [
        # Exactly aligned cases
        512,   # 1 superblock
        1024,  # 2 superblocks

        # Cases that need padding
        500,   # Just under 1 superblock
        600,   # Just over 1 superblock
        1000,  # Just under 2 superblocks
        1100,  # Just over 2 superblocks

        # Block boundary cases
        384,   # 3 blocks, will pad to 512 (4 blocks)
        640,   # 5 blocks, will pad to 1024 (8 blocks)
    ]

    for ntoks in test_cases:
        # Create document structure with boundaries that don't align with blocks
        doc_boundary1 = ntoks // 3
        doc_boundary2 = 2 * ntoks // 3
        document_ids = torch.cat([
            torch.zeros(doc_boundary1, device=device),
            torch.ones(doc_boundary2 - doc_boundary1, device=device),
            torch.full((ntoks - doc_boundary2,), 2, device=device)
        ])

        # Mix of attention regions
        attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)
        attn_regions[ntoks//4:3*ntoks//4] = 1  # Dense region in middle

        _test_super_functions_helper(ntoks, document_ids, attn_regions, device, compile_flag,
                                   BLOCK_SIZE=BLOCK_SIZE, SUPERBLOCK_SIZE=SUPERBLOCK_SIZE)


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
    pytest.param("gpu-compile", marks=pytest.mark.gpu)
])
def test_super_complex_documents(variant):
    """Test super functions with complex document patterns across superblock boundaries"""
    device = "cuda" if variant.startswith("gpu") else "cpu"
    compile_flag = "compile" in variant

    BLOCK_SIZE = 64   # Smaller blocks for more granular testing
    SUPERBLOCK_SIZE = 512

    ntoks = 1536  # 3 superblocks

    # Create documents that span superblock boundaries
    document_ids = torch.tensor([
        # Doc 0: spans first 1.5 superblocks
        *([0] * 768),
        # Doc 1: middle of superblock 2 to middle of superblock 3
        *([1] * 512),
        # Doc 2: rest of superblock 3
        *([2] * 256)
    ], device=device)

    # Attention regions that create interesting patterns
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)
    attn_regions[200:400] = 1    # Dense region in superblock 1
    attn_regions[800:1000] = 1   # Dense region spanning superblocks 2-3
    attn_regions[1400:] = 2      # Different dense region in superblock 3

    _test_super_functions_helper(ntoks, document_ids, attn_regions, device, compile_flag,
                               BLOCK_SIZE=BLOCK_SIZE, SUPERBLOCK_SIZE=SUPERBLOCK_SIZE)


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
    pytest.param("gpu-compile", marks=pytest.mark.gpu)
])
def test_super_with_padding(variant):
    """Test super functions with padding tokens across superblock boundaries"""
    device = "cuda" if variant.startswith("gpu") else "cpu"
    compile_flag = "compile" in variant

    BLOCK_SIZE = 128
    SUPERBLOCK_SIZE = 1024

    ntoks = 1500  # Will pad to 2048 (2 superblocks)

    # Documents with padding at various positions
    document_ids = torch.tensor([0] * 600 + [1] * 500 + [2] * 400, device=device)
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)

    # Add padding in different superblocks
    attn_regions[400:450] = -1   # Padding in first superblock
    attn_regions[1200:] = -1     # Padding at end (will extend into second superblock)
    attn_regions[800:850] = 1    # Dense region

    _test_super_functions_helper(ntoks, document_ids, attn_regions, device, compile_flag,
                               BLOCK_SIZE=BLOCK_SIZE, SUPERBLOCK_SIZE=SUPERBLOCK_SIZE)


@pytest.mark.parametrize("make_fn,kw_extra", [
    ("make_docmask_gpu", {}),
    ("make_docmask_gpu_v2", {}),
    ("make_docmask_gpu_v3", {"max_per_row": 64}),
    ("make_docmask_gpu_v3", {"max_per_row": "dynamic"}),
])
def test_stable_shapes_for_compile(make_fn, kw_extra):
    """BlockMask tensor shapes must not change across batches with different
    document structures, otherwise torch.compile(dynamic=False) recompiles.
    This mirrors production: train.py sets recompile_limit=1."""
    fn = getattr(uf, make_fn)
    BS = 64
    ntoks = 4096
    NB = ntoks // BS  # 64 blocks — large enough that max_per_row can differ

    # Batch A: one huge document spanning all 64 blocks
    doc_a = torch.zeros(ntoks, dtype=torch.int64)
    ar_a = torch.zeros(ntoks, dtype=torch.int32)

    # Batch B: many tiny documents (64 tokens = 1 block each)
    doc_b = torch.tensor([i for i in range(ntoks // 64) for _ in range(64)])
    ar_b = torch.zeros(ntoks, dtype=torch.int32)

    kw = dict(BLOCK_SIZE=BS, **kw_extra)
    if make_fn != "make_docmask_gpu":
        kw["SUPERBLOCK_SIZE"] = 512
        kw["compile"] = False

    mask_a = fn(ntoks, ar_a, doc_a, **kw)
    mask_b = fn(ntoks, ar_b, doc_b, **kw)

    if kw_extra.get("max_per_row") == "dynamic":
        # With dynamic, shapes may differ but dims are marked dynamic.
        # Just verify the function ran without error.
        return

    # All BlockMask tensors must have identical shapes across batches
    for attr in ["kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
                 "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices"]:
        shape_a = getattr(mask_a, attr).shape
        shape_b = getattr(mask_b, attr).shape
        assert shape_a == shape_b, (
            f"{make_fn}: {attr} shape changed: {shape_a} vs {shape_b} "
            f"(batch A has 1 huge doc, batch B has {ntoks//64} tiny docs)"
        )


def test_stable_shapes_numpy():
    """Test that numpy implementations support max_per_row for stable shapes."""
    import flexmaskli.docmask_cpu as ufn

    BS = 64
    ntoks = 4096
    NB = ntoks // BS

    doc_a = torch.zeros(ntoks, dtype=torch.int64)
    ar_a = torch.zeros(ntoks, dtype=torch.int32)
    doc_b = torch.tensor([i for i in range(ntoks // 64) for _ in range(64)])
    ar_b = torch.zeros(ntoks, dtype=torch.int32)

    for mpr in [NB, "dynamic"]:
        mask_a = ufn.make_docmask_numpy(ntoks, ar_a, doc_a, BLOCK_SIZE=BS, max_per_row=mpr)
        mask_b = ufn.make_docmask_numpy(ntoks, ar_b, doc_b, BLOCK_SIZE=BS, max_per_row=mpr)
        if mpr == "dynamic":
            continue  # shapes may differ, just verify no crash
        for attr in ["kv_indices", "full_kv_indices", "q_indices", "full_q_indices"]:
            assert getattr(mask_a, attr).shape == getattr(mask_b, attr).shape, \
                f"numpy mpr={mpr}: {attr} shape differs"


def test_max_per_row_too_small_raises():
    import flexmaskli.docmask_cpu as ufn
    BS = 128
    ntoks = 1024
    di = torch.zeros(ntoks, dtype=torch.int64)
    ar = torch.zeros(ntoks, dtype=torch.int32)
    with pytest.raises(AssertionError, match="max_per_row=.*too small"):
        ufn.make_docmask_numpy(ntoks, ar, di, BLOCK_SIZE=BS, max_per_row=1)
    with pytest.raises(AssertionError, match="max_per_row=.*too small"):
        ufn.make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=BS, max_per_row=1)


@pytest.mark.gpu
def test_v3_cross_superblock_max_per_row():
    """v3 must handle documents spanning multiple superblocks correctly.

    Regression test: the old max_per_row formula used the single longest doc
    length, but a q_superblock can contain parts of multiple documents that
    collectively reach more kv_superblocks than any single doc.

    Minimal repro: 5 superblocks (40960 tokens, BS=128, SB=8192), 3 docs of
    ~10-13k tokens each (all spanning 2+ superblocks), with dense prefixes."""
    BS, SB = 128, 8192
    ntoks = 5 * SB  # 40960
    NB = ntoks // BS  # 320
    ids = torch.tensor([0]*10016 + [1]*12698 + [2]*10478 + [-1]*7768, device='cuda')
    att = torch.tensor([1]*5008 + [0]*5008 + [1]*6349 + [0]*6349 + [1]*5239 + [0]*5239 + [-1]*7768, device='cuda')

    ref = uf.make_docmask_gpu_v2(ntoks, att, ids, BLOCK_SIZE=BS, SUPERBLOCK_SIZE=SB, compile=False)

    # v3 with max_per_row=NB
    v3_nb = uf.make_docmask_gpu_v3(ntoks, att, ids, BLOCK_SIZE=BS, SUPERBLOCK_SIZE=SB, max_per_row=NB, compile=False)
    compare_block_masks(ref, v3_nb)

    # v3 with auto-computed max_per_row (no mark_dynamic)
    v3_auto = uf.make_docmask_gpu_v3(
        ntoks, att, ids, BLOCK_SIZE=BS, SUPERBLOCK_SIZE=SB, max_per_row=None, compile=False
    )
    compare_block_masks(ref, v3_auto)


@pytest.mark.gpu
def test_v3_mark_dynamic_recompile():
    """mark_dynamic on BlockMask index tensors prevents recompilation
    under torch.compile(dynamic=False) when the index dimension changes
    size between calls.

    This was broken when we used mark_dynamic(t, -1) because negative
    indices aren't normalized (PyTorch bug). Fixed by using positive
    indices in make_docmask_gpu_v3."""
    from torch.nn.attention.flex_attention import flex_attention

    BS = 128
    ntoks = 4096
    NB = ntoks // BS  # 32

    doc_all = torch.zeros(ntoks, dtype=torch.int64, device='cuda')  # 1 huge doc
    ar_all = torch.zeros(ntoks, dtype=torch.int32, device='cuda')
    doc_many = torch.tensor([i // 64 for i in range(ntoks)], device='cuda')  # 64 tiny docs
    ar_many = torch.zeros(ntoks, dtype=torch.int32, device='cuda')

    # Create two masks with genuinely different index widths
    mask_wide = uf.make_docmask_gpu_v3(ntoks, ar_all, doc_all, BLOCK_SIZE=BS,
        SUPERBLOCK_SIZE=512, max_per_row="dynamic", compile=False)
    mask_narrow = uf.make_docmask_gpu_v3(ntoks, ar_many, doc_many, BLOCK_SIZE=BS,
        SUPERBLOCK_SIZE=512, max_per_row="dynamic", compile=False)

    # Verify shapes actually differ
    assert mask_wide.kv_indices.shape[-1] != mask_narrow.kv_indices.shape[-1], \
        f"Test setup error: shapes should differ, got {mask_wide.kv_indices.shape[-1]} and {mask_narrow.kv_indices.shape[-1]}"

    # Compile flex_attention with strict settings
    torch._dynamo.config.recompile_limit = 1
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    cflex = torch.compile(flex_attention, dynamic=False, fullgraph=True)

    q = torch.randn(1, 1, ntoks, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, ntoks, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1, ntoks, 64, device="cuda", dtype=torch.bfloat16)

    # First call compiles OK
    with torch.no_grad():
        cflex(q, k, v, block_mask=mask_wide)

    # Second call with different shape should NOT recompile (mark_dynamic)
    with torch.no_grad():
        cflex(q, k, v, block_mask=mask_narrow)


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
    pytest.param("gpu-compile", marks=pytest.mark.gpu)
])
def test_super_minimal_overlap(variant):
    """Test super functions with minimal document overlap between superblocks"""
    device = "cuda" if variant.startswith("gpu") else "cpu"
    compile_flag = "compile" in variant

    BLOCK_SIZE = 128
    SUPERBLOCK_SIZE = 512

    ntoks = 1024  # 2 superblocks

    # Each superblock gets mostly its own documents, with tiny overlaps
    document_ids = torch.cat([
        torch.zeros(500, device=device),     # Doc 0 mostly in superblock 1
        torch.ones(12, device=device),       # Doc 1 tiny bit in superblock 1
        torch.ones(500, device=device),      # Doc 1 mostly in superblock 2
        torch.full((12,), 2, device=device)  # Doc 2 tiny bit in superblock 2
    ])

    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)
    attn_regions[200:300] = 1    # Dense region in superblock 1
    attn_regions[700:800] = 1    # Dense region in superblock 2

    _test_super_functions_helper(ntoks, document_ids, attn_regions, device, compile_flag,
                               BLOCK_SIZE=BLOCK_SIZE, SUPERBLOCK_SIZE=SUPERBLOCK_SIZE)


# ---------------------------------------------------------------------------
# Regression test: block-boundary straddling segments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_block_boundary_straddle(variant):
    """Segments straddling block boundaries must not cause ValueError.

    Regression: _compute_segments_and_mpr used (end - start + BS - 1) // BS
    to estimate per-block counts, but that computes the minimum blocks needed
    to *fit* the segment's tokens -- not the actual blocks *spanned*.  When a
    segment starts mid-block and continues into the next, the span is 2 but
    the old formula returned 1, underestimating max_per_row.

    This test uses BS=2 with many 2-token documents straddling every block
    boundary plus dense attn_regions so above-diagonal blocks are included,
    which pushes actual per-row counts above the wrongly-computed max."""
    device = "cuda" if variant == "gpu" else "cpu"

    # Doc0:tok0  Doc1:tok1-2  Doc2:tok3-4  Doc3:tok5-6  Doc4:tok7-8  Doc5:tok9-10  Doc6:tok11
    # With BS=2, block boundaries at 0,2,4,6,8,10 -- every Doc1..Doc5 straddles.
    document_ids = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6], device=device)
    attn_regions = torch.ones(len(document_ids), dtype=torch.int32, device=device)
    ntoks = len(document_ids)

    ref = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, BLOCK_SIZE=2)

    import flexmaskli.docmask_cpu as ufn
    cpu_di, cpu_ar = document_ids.cpu(), attn_regions.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba, ufn.make_docmask_cpu]:
        mask = fn(ntoks, cpu_ar, cpu_di, BLOCK_SIZE=2)
        compare_block_masks(ref, mask)


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_block_boundary_straddle_large(variant):
    """Same idea as test_block_boundary_straddle but with BS=128 and
    realistic-sized sequences to catch overflow in larger index arrays."""
    device = "cuda" if variant == "gpu" else "cpu"

    BS = 128
    # Create alternating 1-token and 2-token documents at block boundaries.
    # E.g. with BS=128: doc boundary at token 127 makes a 2-token segment
    # straddling blocks 0-1.
    ids = []
    doc = 0
    tok = 0
    while tok < 1024:
        boundary = ((tok // BS) + 1) * BS  # next block boundary
        gap = boundary - tok
        if gap > 2:
            ids.extend([doc] * (gap - 1))
            tok += gap - 1
            doc += 1
        # 2-token segment straddling the boundary
        ids.extend([doc, doc])
        tok += 2
        doc += 1
    ntoks = len(ids)

    document_ids = torch.tensor(ids, device=device)
    # Dense prefix covering half the sequence
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)
    attn_regions[:ntoks // 2] = 1

    ref = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, BLOCK_SIZE=BS)

    import flexmaskli.docmask_cpu as ufn
    cpu_di, cpu_ar = document_ids.cpu(), attn_regions.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba, ufn.make_docmask_cpu]:
        mask = fn(ntoks, cpu_ar, cpu_di, BLOCK_SIZE=BS)
        compare_block_masks(ref, mask)


# ---------------------------------------------------------------------------
# Non-contiguous same-value dense regions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_noncontiguous_same_value_regions(variant):
    """CPU must not miss blocks when two dense regions share the same value
    but are separated by a gap (0 or -1).

    This is the docmask equivalent of the batchmask bug: the above-diagonal
    loop breaks after the first dense region that overlaps both q and kv blocks,
    missing cross-region pairs when a later region has the same value."""
    device = "cuda" if variant == "gpu" else "cpu"
    BS = 64
    ntoks = 512

    document_ids = torch.zeros(ntoks, dtype=torch.int64, device=device)
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)

    # Two value=1 regions separated by a zero gap
    attn_regions[:100] = 1
    attn_regions[200:400] = 1

    ref = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, BLOCK_SIZE=BS)

    import flexmaskli.docmask_cpu as ufn
    cpu_di, cpu_ar = document_ids.cpu(), attn_regions.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba, ufn.make_docmask_cpu]:
        mask = fn(ntoks, cpu_ar, cpu_di, BLOCK_SIZE=BS)
        compare_block_masks(ref, mask)


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_noncontiguous_same_value_with_negative_gap(variant):
    """Same-value regions separated by -1 (invisible) tokens, like attn_regions2."""
    device = "cuda" if variant == "gpu" else "cpu"
    BS = 64
    ntoks = 512

    document_ids = torch.zeros(ntoks, dtype=torch.int64, device=device)
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)

    # attn_regions2 pattern: question=1, image=-1, registers=1
    attn_regions[:52] = 1
    attn_regions[52:248] = -1
    attn_regions[248:260] = 1

    ref = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, BLOCK_SIZE=BS)

    import flexmaskli.docmask_cpu as ufn
    cpu_di, cpu_ar = document_ids.cpu(), attn_regions.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba, ufn.make_docmask_cpu]:
        mask = fn(ntoks, cpu_ar, cpu_di, BLOCK_SIZE=BS)
        compare_block_masks(ref, mask)


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_prefix_lm_large_hole_exact(variant):
    """Prefix-LM style masks keep large full-block area even with big -1 holes."""
    device = "cuda" if variant == "gpu" else "cpu"
    BS = 128
    ntoks = 4096

    document_ids = torch.zeros(ntoks, dtype=torch.int64, device=device)
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)
    attn_regions[:3000] = 1
    attn_regions[1700:2200] = -1
    attn_regions[2600:2660] = -1

    ref = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, BLOCK_SIZE=BS)
    assert ref.full_kv_num_blocks.sum().item() > ref.kv_num_blocks.sum().item()

    import flexmaskli.docmask_cpu as ufn
    cpu_di, cpu_ar = document_ids.cpu(), attn_regions.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba, ufn.make_docmask_cpu]:
        mask = fn(ntoks, cpu_ar, cpu_di, BLOCK_SIZE=BS)
        compare_block_masks(ref, mask)


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_decode_like_docmask_patterns(variant):
    """Decode-like layouts within a single document:
    [image(1)] [AR(0)] [padding gap(0)] [decode token(0)]
    and attn_regions2 variants with -1 holes."""
    device = "cuda" if variant == "gpu" else "cpu"
    BS = 64
    ntoks = 512

    import flexmaskli.docmask_cpu as ufn

    patterns = [
        # Simple: [img(1)] [AR+pad(0)]
        {"ar": [(0, 100, 1)]},
        # attn_regions2: [Q(1)] [img(-1)] [reg(1)] [AR+pad(0)]
        {"ar": [(0, 30, 1), (30, 220, -1), (220, 230, 1)]},
        # Larger image
        {"ar": [(0, 50, 1), (50, 350, -1), (350, 370, 1)]},
        # Multi-image: [Q(1)] [img1(-1)] [reg(1)] [img2(-1)] [reg(1)]
        {"ar": [(0, 20, 1), (20, 120, -1), (120, 130, 1),
                (130, 230, -1), (230, 240, 1)]},
    ]

    for i, pat in enumerate(patterns):
        document_ids = torch.zeros(ntoks, dtype=torch.int64, device=device)
        attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)
        for s, e, v in pat["ar"]:
            attn_regions[s:e] = v

        ref = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, BLOCK_SIZE=BS)

        cpu_di, cpu_ar = document_ids.cpu(), attn_regions.cpu()
        for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba, ufn.make_docmask_cpu]:
            mask = fn(ntoks, cpu_ar, cpu_di, BLOCK_SIZE=BS)
            compare_block_masks(ref, mask)


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_complex_noncontiguous_docmask(variant):
    """Complex patterns: multiple non-contiguous same-value regions,
    mixed values, multi-document with holes."""
    device = "cuda" if variant == "gpu" else "cpu"
    BS = 64
    ntoks = 1024

    import flexmaskli.docmask_cpu as ufn

    # --- Pattern 0: [img(1)] [hole(-1)] [reg(1)] [hole(-1)] [AR(0)] [hole(-1)] [reg(1)] ---
    document_ids = torch.zeros(ntoks, dtype=torch.int64, device=device)
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)
    attn_regions[:100] = 1
    attn_regions[100:150] = -1
    attn_regions[150:170] = 1
    attn_regions[170:200] = -1
    attn_regions[200:400] = 0  # AR
    attn_regions[400:450] = -1
    attn_regions[450:470] = 1

    ref = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, BLOCK_SIZE=BS)
    cpu_di, cpu_ar = document_ids.cpu(), attn_regions.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        mask = fn(ntoks, cpu_ar, cpu_di, BLOCK_SIZE=BS)
        compare_block_masks(ref, mask)

    # --- Pattern 1: three value=1 regions with -1 gaps ---
    attn_regions2 = torch.zeros(ntoks, dtype=torch.int32, device=device)
    attn_regions2[:80] = 1
    attn_regions2[80:200] = -1
    attn_regions2[200:250] = 1
    attn_regions2[250:400] = -1
    attn_regions2[400:430] = 1

    ref2 = uf.make_docmask_gpu(ntoks, attn_regions2, document_ids, BLOCK_SIZE=BS)
    cpu_ar2 = attn_regions2.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        mask = fn(ntoks, cpu_ar2, cpu_di, BLOCK_SIZE=BS)
        compare_block_masks(ref2, mask)

    # --- Pattern 2: mixed value=1 and value=2, both non-contiguous ---
    attn_regions3 = torch.zeros(ntoks, dtype=torch.int32, device=device)
    attn_regions3[:60] = 1
    attn_regions3[60:200] = 2
    attn_regions3[200:220] = 1   # same as first region
    attn_regions3[220:400] = -1
    attn_regions3[400:420] = 2   # same as second region

    ref3 = uf.make_docmask_gpu(ntoks, attn_regions3, document_ids, BLOCK_SIZE=BS)
    cpu_ar3 = attn_regions3.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        mask = fn(ntoks, cpu_ar3, cpu_di, BLOCK_SIZE=BS)
        compare_block_masks(ref3, mask)

    # --- Pattern 3: multi-doc, each with non-contiguous same-value regions ---
    document_ids4 = torch.zeros(ntoks, dtype=torch.int64, device=device)
    document_ids4[512:] = 1
    attn_regions4 = torch.zeros(ntoks, dtype=torch.int32, device=device)
    # Doc 0: value=1 at [0:100], value=-1 at [100:300], value=1 at [300:350]
    attn_regions4[:100] = 1
    attn_regions4[100:300] = -1
    attn_regions4[300:350] = 1
    # Doc 1: value=1 at [512:600], value=-1 at [600:800], value=1 at [800:850]
    attn_regions4[512:600] = 1
    attn_regions4[600:800] = -1
    attn_regions4[800:850] = 1

    ref4 = uf.make_docmask_gpu(ntoks, attn_regions4, document_ids4, BLOCK_SIZE=BS)
    cpu_di4, cpu_ar4 = document_ids4.cpu(), attn_regions4.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        mask = fn(ntoks, cpu_ar4, cpu_di4, BLOCK_SIZE=BS)
        compare_block_masks(ref4, mask)


@pytest.mark.gpu
def test_compiled_flex_attention_docmask_cpu_dynamic():
    """Diagnose: does compiled flex_attention produce correct results when
    called with docmask_cpu masks using max_per_row="dynamic"?

    This mirrors the exact production usage in simple_data.py:
        make_docmask_cpu(maxtok, v, seq["iseq"], ..., max_per_row="dynamic")

    Two different document packings (same ntoks) produce masks with different
    compact index widths. If mark_dynamic doesn't properly communicate the
    dynamic dimension to the triton kernel, the second call will produce
    wrong outputs (stale kernel reads indices at the first call's width).
    """
    from torch.nn.attention.flex_attention import flex_attention
    from flexmaskli.docmask_cpu import make_docmask_cpu
    from flexmaskli.to_gpu import blockmask_to_gpu

    BS = 128
    ntoks = 4096
    head_dim = 64

    # Packing 1: one huge document → max_per_row is large (many blocks per row)
    di1 = torch.zeros(ntoks, dtype=torch.int64)
    ar1 = torch.zeros(ntoks, dtype=torch.int32)
    ar1[:3000] = 1  # large dense prefix

    # Packing 2: many tiny documents (64 tokens each) → max_per_row is small
    di2 = torch.tensor([i // 64 for i in range(ntoks)], dtype=torch.int64)
    ar2 = torch.zeros(ntoks, dtype=torch.int32)

    # Build CPU masks with dynamic max_per_row (production config)
    cpu1 = make_docmask_cpu(ntoks, ar1, di1, BLOCK_SIZE=BS, max_per_row="dynamic")
    cpu2 = make_docmask_cpu(ntoks, ar2, di2, BLOCK_SIZE=BS, max_per_row="dynamic")

    # Verify shapes actually differ (otherwise test is vacuous)
    w1 = cpu1.kv_indices.shape[-1]
    w2 = cpu2.kv_indices.shape[-1]
    assert w1 != w2, \
        f"Test setup error: compact widths should differ, got {w1} and {w2}"

    # GPU reference masks (full NB width, no compact issues)
    gpu1 = uf.make_docmask_gpu(ntoks, ar1.cuda(), di1.cuda(), BLOCK_SIZE=BS)
    gpu2 = uf.make_docmask_gpu(ntoks, ar2.cuda(), di2.cuda(), BLOCK_SIZE=BS)

    # Move all masks to GPU
    cpu1 = blockmask_to_gpu(cpu1, "cuda")
    cpu2 = blockmask_to_gpu(cpu2, "cuda")

    # Fixed Q, K, V
    q = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)

    # --- Reference: non-compiled flex_attention ---
    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
    with torch.no_grad():
        ref1 = flex_attention(q, k, v, block_mask=gpu1)
        ref2 = flex_attention(q, k, v, block_mask=gpu2)
    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = False

    # Sanity: different masks should produce different outputs
    assert not torch.allclose(ref1, ref2, atol=1e-2), \
        "Test setup error: different masks should give different outputs"

    # --- Test: compiled flex_attention with CPU dynamic masks ---
    torch._dynamo.reset()
    cflex = torch.compile(flex_attention, dynamic=False, fullgraph=True)
    with torch.no_grad():
        out1 = cflex(q, k, v, block_mask=cpu1)
        out2 = cflex(q, k, v, block_mask=cpu2)

    diff1 = (out1 - ref1).abs().max().item()
    diff2 = (out2 - ref2).abs().max().item()
    diff2_vs_ref1 = (out2 - ref1).abs().max().item()
    stale = torch.allclose(out1, out2, atol=1e-3)

    print(f"\n=== DOCMASK CPU DYNAMIC DIAGNOSTICS ===")
    print(f"  Compact widths: mask1={w1}, mask2={w2}")
    print(f"  Call 1 vs ref1: max_diff={diff1:.6f}  (should be ~0)")
    print(f"  Call 2 vs ref2: max_diff={diff2:.6f}  (should be ~0)")
    print(f"  Call 2 vs ref1: max_diff={diff2_vs_ref1:.6f}  (closeness to wrong ref)")
    print(f"  Call 1 == Call 2: {stale}")

    assert diff1 < 0.02, f"Docmask CPU call 1 wrong! {diff1:.6f}"
    assert diff2 < 0.02, \
        f"Docmask CPU call 2 wrong! Max diff vs ref: {diff2:.6f}. " \
        f"Same as call 1: {stale}. vs ref1: {diff2_vs_ref1:.6f}"


@pytest.mark.gpu
def test_compiled_flex_attention_docmask_dynamic():
    """Verify max_per_row=\"dynamic\" works under compiled flex_attention."""
    from torch.nn.attention.flex_attention import flex_attention
    from flexmaskli.docmask_cpu import make_docmask_cpu
    from flexmaskli.to_gpu import blockmask_to_gpu

    BS = 128
    ntoks = 4096
    head_dim = 64

    di1 = torch.zeros(ntoks, dtype=torch.int64)
    ar1 = torch.zeros(ntoks, dtype=torch.int32)
    ar1[:3000] = 1

    di2 = torch.tensor([i // 64 for i in range(ntoks)], dtype=torch.int64)
    ar2 = torch.zeros(ntoks, dtype=torch.int32)

    cpu1 = make_docmask_cpu(ntoks, ar1, di1, BLOCK_SIZE=BS, max_per_row="dynamic")
    cpu2 = make_docmask_cpu(ntoks, ar2, di2, BLOCK_SIZE=BS, max_per_row="dynamic")

    # Verify dynamic uses smaller width and has mark_dynamic
    assert cpu1.kv_indices.shape[-1] <= ntoks // BS
    assert hasattr(cpu1.kv_indices, '_dynamo_dynamic_indices')

    # GPU reference
    gpu1 = uf.make_docmask_gpu(ntoks, ar1.cuda(), di1.cuda(), BLOCK_SIZE=BS)
    gpu2 = uf.make_docmask_gpu(ntoks, ar2.cuda(), di2.cuda(), BLOCK_SIZE=BS)

    cpu1 = blockmask_to_gpu(cpu1, "cuda")
    cpu2 = blockmask_to_gpu(cpu2, "cuda")

    q = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)

    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
    with torch.no_grad():
        ref1 = flex_attention(q, k, v, block_mask=gpu1)
        ref2 = flex_attention(q, k, v, block_mask=gpu2)
    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = False

    torch._dynamo.reset()
    cflex = torch.compile(flex_attention, dynamic=False, fullgraph=True)
    with torch.no_grad():
        out1 = cflex(q, k, v, block_mask=cpu1)
        out2 = cflex(q, k, v, block_mask=cpu2)

    diff1 = (out1 - ref1).abs().max().item()
    diff2 = (out2 - ref2).abs().max().item()
    assert diff1 < 0.02, f"max_per_row=dynamic call 1 wrong: {diff1:.6f}"
    assert diff2 < 0.02, f"max_per_row=dynamic call 2 wrong: {diff2:.6f}"


@pytest.mark.gpu
def test_compiled_flex_attention_docmask_holes_exact():
    """Hole-heavy patterns must match GPU reference exactly under compile."""
    from torch.nn.attention.flex_attention import flex_attention
    from flexmaskli.docmask_cpu import make_docmask_cpu
    from flexmaskli.to_gpu import blockmask_to_gpu

    BS = 128
    ntoks = 1024
    head_dim = 64

    di1 = torch.zeros(ntoks, dtype=torch.int64)
    ar1 = torch.zeros(ntoks, dtype=torch.int32)
    ar1[:52] = 1
    ar1[52:248] = -1
    ar1[248:260] = 1

    di2 = torch.zeros(ntoks, dtype=torch.int64)
    ar2 = torch.zeros(ntoks, dtype=torch.int32)
    ar2[:20] = 1
    ar2[20:120] = -1
    ar2[120:130] = 1
    ar2[130:230] = -1
    ar2[230:240] = 1

    cpu1 = make_docmask_cpu(ntoks, ar1, di1, BLOCK_SIZE=BS, max_per_row="dynamic")
    cpu2 = make_docmask_cpu(ntoks, ar2, di2, BLOCK_SIZE=BS, max_per_row="dynamic")
    gpu1 = uf.make_docmask_gpu(ntoks, ar1.cuda(), di1.cuda(), BLOCK_SIZE=BS)
    gpu2 = uf.make_docmask_gpu(ntoks, ar2.cuda(), di2.cuda(), BLOCK_SIZE=BS)

    compare_block_masks(gpu1, cpu1)
    compare_block_masks(gpu2, cpu2)

    cpu1 = blockmask_to_gpu(cpu1, "cuda")
    cpu2 = blockmask_to_gpu(cpu2, "cuda")

    q = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1, ntoks, head_dim, device="cuda", dtype=torch.bfloat16)

    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
    with torch.no_grad():
        ref1 = flex_attention(q, k, v, block_mask=gpu1)
        ref2 = flex_attention(q, k, v, block_mask=gpu2)
    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = False

    torch._dynamo.reset()
    cflex = torch.compile(flex_attention, dynamic=False, fullgraph=True)
    with torch.no_grad():
        out1 = cflex(q, k, v, block_mask=cpu1)
        out2 = cflex(q, k, v, block_mask=cpu2)

    diff1 = (out1 - ref1).abs().max().item()
    diff2 = (out2 - ref2).abs().max().item()
    assert diff1 < 0.02, f"holes case 1 wrong under compile: {diff1:.6f}"
    assert diff2 < 0.02, f"holes case 2 wrong under compile: {diff2:.6f}"


def test_docmask_max_per_row_shapes():
    """dynamic/auto/fixed max_per_row modes have expected shape+marking behavior."""
    import flexmaskli.docmask_cpu as ufn
    BS = 128
    ntoks = 4096
    NB = ntoks // BS

    di = torch.tensor([i // 64 for i in range(ntoks)], dtype=torch.int64)
    ar = torch.zeros(ntoks, dtype=torch.int32)

    full = ufn.make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=BS, max_per_row=NB)
    auto = ufn.make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=BS, max_per_row=None)
    comp = ufn.make_docmask_cpu(ntoks, ar, di, BLOCK_SIZE=BS)  # default: "dynamic"

    assert full.kv_indices.shape[-1] == NB
    assert auto.kv_indices.shape[-1] <= NB
    assert comp.kv_indices.shape[-1] <= NB
    assert not hasattr(auto.kv_indices, '_dynamo_dynamic_indices')
    assert hasattr(comp.kv_indices, '_dynamo_dynamic_indices')
    assert not hasattr(full.kv_indices, '_dynamo_dynamic_indices')


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_ar2_full_partial_correctness(variant):
    """Blocks containing ar=-1 must never be marked full."""
    device = "cuda" if variant.startswith("gpu") else "cpu"
    BS = 64
    ntoks = 1024

    import flexmaskli.docmask_cpu as ufn
    from flexmaskli.docmask_cpu import _mask_fn

    document_ids = torch.zeros(ntoks, dtype=torch.int64, device=device)
    document_ids[512:] = 1
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)
    attn_regions[:50] = 1
    attn_regions[50:300] = -1
    attn_regions[300:320] = 1
    attn_regions[512:560] = 1
    attn_regions[560:800] = -1
    attn_regions[800:820] = 1

    mask_mod = partial(_mask_fn, attn_regions=attn_regions.cpu(), document_ids=document_ids.cpu())
    qi = torch.arange(ntoks, dtype=torch.int32)[:, None]
    ki = torch.arange(ntoks, dtype=torch.int32)[None, :]
    ground_truth = mask_mod(0, 0, qi, ki)

    cpu_di, cpu_ar = document_ids.cpu(), attn_regions.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba, ufn.make_docmask_cpu]:
        dense = _dense_docmask(fn, ntoks, cpu_ar, cpu_di, BS)
        assert torch.equal(dense, ground_truth), f"{fn.__name__}: dense mask differs from ground truth"

    if variant.startswith("gpu"):
        ref = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, BLOCK_SIZE=BS)
        ref_dense = blockmask_to_dense(ref)[0, 0, :ntoks, :ntoks]
        assert torch.equal(ref_dense, ground_truth)


@pytest.mark.parametrize("variant", [
    "cpu",
    pytest.param("gpu", marks=pytest.mark.gpu),
])
def test_ar2_packed_multidoc(variant):
    """Packed multi-document attn_regions2 case should match dense GPU reference."""
    device = "cuda" if variant.startswith("gpu") else "cpu"
    BS = 128
    ntoks = 4096

    import flexmaskli.docmask_cpu as ufn

    document_ids = torch.zeros(ntoks, dtype=torch.int64, device=device)
    attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)
    pos = 0
    for doc_id in range(4):
        dlen = ntoks // 4
        document_ids[pos:pos+dlen] = doc_id
        nq = int(0.10 * dlen)
        nimg = int(0.40 * dlen)
        nreg = int(0.05 * dlen)
        attn_regions[pos:pos+nq] = 1
        attn_regions[pos+nq:pos+nq+nimg] = -1
        attn_regions[pos+nq+nimg:pos+nq+nimg+nreg] = 1
        pos += dlen

    ref_dense = blockmask_to_dense(uf.make_docmask_gpu(ntoks, attn_regions, document_ids, BLOCK_SIZE=BS))
    cpu_di, cpu_ar = document_ids.cpu(), attn_regions.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba, ufn.make_docmask_cpu]:
        dense = _dense_docmask(fn, ntoks, cpu_ar, cpu_di, BS)
        assert torch.equal(ref_dense[0, 0, :ntoks, :ntoks], dense), \
            f"{fn.__name__}: dense mask differs from GPU reference"
