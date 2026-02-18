from functools import cache

import cv2
import numpy as np
from einops import rearrange

import bv2.utils as u


def sanity_check(example):
    attn_regions_keys = [k for k in example if k.startswith("attn_regions")]

    keys_with_same_size = ("toki", "toko", "lowe") + tuple(attn_regions_keys)
    if len(set(len(example[k]) for k in keys_with_same_size)) != 1:
        raise ValueError(
            "Keys with unexpectedly different sizes:\n"
            + "\n".join(f"{k}: {len(example[k])}" for k in keys_with_same_size)
        )

    assert example["toki"].dtype == np.uint8, f"{example['toki'].dtype=}, expected uint8"
    assert example["toko"].dtype == np.uint8, f"{example['toko'].dtype=}, expected uint8"
    assert example["lowe"].dtype == np.float32, f"{example['lowe'].dtype=}, expected float32"
    for k in attn_regions_keys:
        assert example[k].dtype == np.int64, f"{k}: {example[k].dtype=}, expected int64"
    assert isinstance(example["ndatatoks"], int), f"{example['ndatatoks']=}, expected POD int"
    return example


def patchify(img, *, ph, pw):
    img = np.asarray(img)
    h, w, c = img.shape

    # Assume the image is already properly sizes as multiple from a previous step.
    assert w % pw == 0 and h % ph == 0 and c == 3, f"Need proper shape {w=} {h=} {c=} {pw=} {ph=}; forgot pp?"

    patches = rearrange(img, "(ny ph) (nx pw) c -> ny nx ph pw c", ph=ph, pw=pw)

    # ij = y/x-index of patch ; mn = number of patches in height/width
    m, n = h // ph, w // pw
    i, j = np.mgrid[:m, :n].astype(np.int16)
    positions = np.stack([i, j, np.full((m, n), m, np.int16), np.full((m, n), n, np.int16)], axis=-1)  # fmt: skip

    return patches, positions


def unpatchify(patches, positions):
    patches = np.asarray(patches)
    pos = np.asarray(positions)

    num_patches, ph, pw, c = patches.shape

    # Channels 2 and 3 are the number of patches in height and width.
    img = np.zeros((pos[0, 2] * ph, pos[0, 3] * pw, c), dtype=patches.dtype)
    for idx in range(num_patches):
        i, j = positions[idx, 0], positions[idx, 1]
        y_start, y_end = i * ph, (i + 1) * ph
        x_start, x_end = j * pw, (j + 1) * pw
        img[y_start:y_end, x_start:x_end, :] = patches[idx]

    return img


def max_patches(hw, max_patches, *, ph=16, pw=16):
    orig_h, orig_w = target_h, target_w = hw

    # First, get w/h below target pixel area if needed:
    orig_pixels = orig_w * orig_h
    max_pixels = max_patches * ph * pw
    if orig_pixels >= max_pixels:
        scale = (max_pixels / orig_pixels) ** 0.5
        target_w = int(orig_w * scale)
        target_h = int(orig_h * scale)

    # Then, round w/h down to the next multiple of pw/ph, but avoid 0 h/w.
    if target_w % pw != 0:
        target_w = max((target_w // pw) * pw, pw)
    if target_h % ph != 0:
        target_h = max((target_h // ph) * ph, ph)

    return (target_h, target_w)


@cache
def _patch_vals(nmin, nmax, exp, mode=None):
    vals = np.arange(nmin, nmax + 1)
    if mode:
        probs = np.float_power(abs(vals - mode) + 1.0, -exp)
    else:
        probs = np.float_power(vals, -exp)  # To avoid int neg power issues.
    probs /= probs.sum()
    return probs, vals


def rand_max_patches(hw, nmax, exp=None, key=None, nmin=64, mode=None, *, ph=16, pw=16):
    if exp is not None:
        probs, vals = _patch_vals(nmin, nmax, exp, mode)
        nmax = u.rng(key, "choice").choice(vals, p=probs)

    return max_patches(hw, nmax, ph=ph, pw=pw)


def reasonable_resize(img, hw, warning_exid=None):
    if hw[0] < img.shape[0] or hw[1] < img.shape[1]:
        # AREA is the only reasonable downscale: https://lucasb.eyer.be/a/vit_cnn_speed.html
        return cv2.resize(img, hw[::-1], interpolation=cv2.INTER_AREA)  # Takes (w, h) for (h, w) imgs!
        # TODO: For big downscales (>2x) this can be sped-up by doing halvings first.
    elif hw == img.shape[:2]:
        return img
    else:
        print(f"Warning: upscaling image from {img.shape=} to {hw=}. Exid: {warning_exid}")
        return cv2.resize(img, hw[::-1], interpolation=cv2.INTER_LINEAR)
