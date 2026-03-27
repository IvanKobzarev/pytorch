"""
"<|bos|><|sep|>{Image}<|sep|>alt_text<|eos|>"
Image captioning: model sees image and generates alt_text.

Bagz-indexed v2: lightweight make_exids (yields plain indices),
all I/O happens in make_example via pmap's 16 threads.

Requires: pre-dumped sample IDs via mango_sm_alt_2b_dump_sids.py
"""

import asyncio
import copy
import time

import cv2
import numpy as np
from airstore.client.airstore_tabular import AIRStorePathHandler
from fairstore.data.airstore_random_access_by_sample_id_dataset import AIRStoreRandomAccessBySampleIdDataset
from fairstore.data.settings.fairstore_dataset import AccessMode
from iopath.common.file_io import PathManager

import bv2.data.dpack as d
import bv2.utils as u
from bv2.data import pp
from bv2.data.common import get_sackli_reader, shuffled_iota_exids
from bv2.data.tokenizer import get_tiktoken

SID_BAGZ_SPLITS = {
    "train": "/checkpoint/rigi/data/mango_2b04_sids/train@64.bag",
    "val": "/checkpoint/rigi/data/mango_2b04_sids/val@4.bag",
    "val_mini": "/checkpoint/rigi/data/mango_2b04_sids/val_mini@1.bag",
}


def _open_sample_id_loader():
    pm = PathManager()
    pm.register_handler(AIRStorePathHandler())
    ds = pm.opent("airstore://rigi_mango_2b04", access_mode=AccessMode.kRandomBySampleID)
    loader = ds.__enter__()
    settings = copy.copy(loader._data_set._settings)
    # not tuned, vibe coded numbers that work well.
    settings.num_of_threads = 1
    settings.max_holding_bundles = 50
    settings.bundle_download_parallel = 1
    settings.bundle_download_timeout_ms = 60_000
    settings.max_retries = 20
    loader._data_set = AIRStoreRandomAccessBySampleIdDataset(settings)
    loader._random_access_mode = True
    return ds, loader


class Dataset:
    def __init__(self, ps=16, max_patches=3136, rand_max_patches=None, nreg=0,
                 greyout_frac=0.0, tokenizer=None, seed=0, split="train", epochs=None):
        self.ps = dict(ph=ps, pw=ps)
        self.max_patches = max_patches
        self.rand_max_patches = rand_max_patches or {}
        self.nreg = nreg
        self.tt = get_tiktoken(**tokenizer or {})
        self.greyout_frac = greyout_frac
        self.data_seed = seed
        self._ds, self._loader = _open_sample_id_loader()
        self.reader = get_sackli_reader(SID_BAGZ_SPLITS[split])
        self.epochs = epochs

    def __del__(self):
        self._ds.__exit__(None, None, None)

    def _fetch(self, sid):
        """Fetch one row by sample ID."""
        # it happens occasionally, we could optionally skip those examples
        for attempt in range(5):
            try:
                return asyncio.run(self._loader.get_sample_by_id_async(sid))
            except RuntimeError as e:
                if attempt == 4:
                    raise
                print(f"WARNING: _fetch attempt {attempt+1}/5 failed for sid={sid!r}: {e}", flush=True)
                time.sleep(2 ** attempt)

    def make_example(self, exid, **_kw):
        row = self._fetch(self.reader[exid].decode())

        alt_text = row["alt_text"]
        img = cv2.imdecode(np.frombuffer(row["storage_handle"], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        suffix = self.tt.encode(alt_text)

        key = (self.data_seed, exid)
        img = pp.reasonable_resize(img, pp.rand_max_patches(
            img.shape[:2], self.max_patches, key=(key, "resize"), **self.rand_max_patches, **self.ps))
        if u.rng(key, "greyout").random() < self.greyout_frac:
            img[...] = 128
        patches, positions = pp.patchify(img, **self.ps)

        nsuf = len(suffix)
        nimg = patches.shape[0] * patches.shape[1]
        nreg = self.nreg

        nbytes = max(d.nbytes_text(), d.nbytes_image(**self.ps), d.nbytes_reg())
        tokens = np.zeros((1 + 1 + nimg + nreg + 1 + nsuf + 1, nbytes), np.uint8)

        txtpos = np.arange(1 + 1 + 1 + nsuf + 1)
        d.pack_text([self.tt.bos, self.tt.sep], positions=txtpos[:2], out=tokens[:2])
        d.pack_image(patches, positions, out=tokens[2 : 2 + nimg])
        d.pack_regs(nreg, out=tokens[2 + nimg : -(1 + nsuf + 1)])
        d.pack_text([self.tt.sep, suffix, self.tt.eos], positions=txtpos[-(1 + nsuf + 1):], out=tokens[-(1 + nsuf + 1):])  # fmt: skip

        example = {
            "toki": tokens[..., :-1, :],
            "toko": tokens[..., 1:, :],
            "lowe": np.r_[0, [0] * nimg, [0] * nreg, 0, [1] * nsuf, 1].astype(np.float32),
            "attn_regions": np.r_[1, 1, [1] * nimg, [1] * nreg, 1, [0] * nsuf].astype(np.int64),
            "ndatatoks": nimg + nsuf,
            "id": exid,
        }
        if nreg:
            example["attn_regions2"] = np.r_[1, 1, [-1] * nimg, [1] * nreg, 1, [0] * nsuf].astype(np.int64)

        return pp.sanity_check(example)

    def make_exids(self, **kw):
        yield from shuffled_iota_exids(len(self.reader), epochs=self.epochs, **kw)

    def vocab_size(self):
        return self.tt.n_vocab
