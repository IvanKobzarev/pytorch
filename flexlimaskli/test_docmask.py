import pytest
import torch

import flexlimaskli.docmask_gpu as uf
from flexlimaskli.test_utils import blockmask_to_dense


def compare_block_masks(mask1, mask2, structural=True):
    # Move to CPU for comparison so mixed-device masks work.
    def cpu(t): return t.cpu() if t.is_cuda else t

    if structural:
        # Check block-level structure matches exactly
        assert torch.equal(cpu(mask1.kv_num_blocks), cpu(mask2.kv_num_blocks)), f"kv_num_blocks mismatch: {mask1.kv_num_blocks} vs {mask2.kv_num_blocks}"
        assert mask1.seq_lengths == mask2.seq_lengths, f"seq_lengths mismatch: {mask1.seq_lengths} vs {mask2.seq_lengths}"
        assert mask1.BLOCK_SIZE == mask2.BLOCK_SIZE, f"BLOCK_SIZE mismatch: {mask1.BLOCK_SIZE} vs {mask2.BLOCK_SIZE}"

        # Compare the sparse representation (valid entries in kv_indices)
        B, H, num_q_blocks, _ = mask1.kv_indices.shape
        for b in range(B):
            for h in range(H):
                for q_block in range(num_q_blocks):
                    num_valid = mask1.kv_num_blocks[b, h, q_block].item()
                    valid_indices1 = cpu(mask1.kv_indices[b, h, q_block, :num_valid])
                    valid_indices2 = cpu(mask2.kv_indices[b, h, q_block, :num_valid])
                    assert torch.equal(valid_indices1, valid_indices2), f"kv_indices mismatch at [{b},{h},{q_block}]: {valid_indices1} vs {valid_indices2}"

    # Dense comparison using helper that works with compact indices
    dense1 = blockmask_to_dense(mask1)
    dense2 = blockmask_to_dense(mask2)
    # CPU impl may conservatively include extra blocks (e.g. for intra-document
    # padding that the GPU can detect as empty). So for non-structural comparison,
    # only check that mask2 is a superset of mask1 (all blocks in ref are present).
    if structural:
        assert torch.equal(dense1, dense2), f"Dense masks differ! Shape: {dense1.shape}, diff locations: {(dense1 != dense2).sum().item()}"
    else:
        missing = dense1 & ~dense2
        assert not missing.any(), f"CPU mask is missing blocks present in GPU ref! Missing: {missing.sum().item()}"


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

    import flexlimaskli.docmask_cpu as ufn
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

    import flexlimaskli.docmask_cpu as ufn
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
    test_cases = [127, 128, 129, 255, 256, 257, 1, 63]

    for ntoks in test_cases:
        mid = ntoks // 2
        document_ids = torch.tensor([0] * mid + [1] * (ntoks - mid), device=device)
        attn_regions = torch.zeros(ntoks, dtype=torch.int32, device=device)

        ref_mask = uf.make_docmask_gpu(ntoks, attn_regions, document_ids, compile="compile" in variant)

        import flexlimaskli.docmask_cpu as ufn
        for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
            compare_block_masks(ref_mask, fn(ntoks, attn_regions.cpu(), document_ids.cpu()), structural=False)


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

    import flexlimaskli.docmask_cpu as ufn
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        compare_block_masks(ref_mask, fn(ntoks, attn_regions.cpu(), document_ids.cpu()), structural=False)


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

    import flexlimaskli.docmask_cpu as ufn
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        compare_block_masks(ref_mask, fn(ntoks, attn_regions.cpu(), document_ids.cpu()), structural=False)



# Test functions that require SUPERBLOCK_SIZE alignment
def _test_super_functions_helper(ntoks, document_ids, attn_regions, device, compile_flag, BLOCK_SIZE=128, SUPERBLOCK_SIZE=1024):
    """Helper to test super functions with proper size constraints"""
    import flexlimaskli.docmask_cpu as ufn

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

    # Test numpy and numba implementations (structural=False: CPU may conservatively
    # include extra blocks for intra-document padding, which is correct but imprecise)
    cpu_ar = attn_regions.cpu()
    cpu_di = document_ids.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba]:
        compare_block_masks(ref_mask, fn(aligned_ntoks, cpu_ar, cpu_di, BLOCK_SIZE=BLOCK_SIZE), structural=False)


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
    import flexlimaskli.docmask_cpu as ufn

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

    # v3 with auto-computed max_per_row
    v3_auto = uf.make_docmask_gpu_v3(ntoks, att, ids, BLOCK_SIZE=BS, SUPERBLOCK_SIZE=SB, compile=False)
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

    import flexlimaskli.docmask_cpu as ufn
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

    import flexlimaskli.docmask_cpu as ufn
    cpu_di, cpu_ar = document_ids.cpu(), attn_regions.cpu()
    for fn in [ufn.make_docmask_numpy, ufn.make_docmask_numba, ufn.make_docmask_cpu]:
        mask = fn(ntoks, cpu_ar, cpu_di, BLOCK_SIZE=BS)
        compare_block_masks(ref, mask)
