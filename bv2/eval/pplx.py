import torch

import bv2.simple_data as simple_data
import bv2.utils as u


def run(predict_fn, ds, iter, **comms):
    # These are all things we collect PER PROCESS/GPU in the loop.
    # We'll summarize across processes once at the end.
    tokens_seen, examples_seen = 0, 0
    total_pplx, total_correct = 0, 0
    total_loss_w, total_loss_toks = 0, 0

    for step, data in enumerate(simple_data.data_iter(ds, max_ep=1, **iter, **comms)):
        # Before making any step, figure out if all ranks are done.
        # Due to example packing, there simply is no way without global comms.
        im_done = (data["iseq"] == -1).all()
        whos_done = u.all_gather(im_done)
        if all(whos_done):
            break

        _, extras = predict_fn(
            data["tokens"], data["flex_masks"], data["loss_weights"], data["iseq"])

        num_tokens = sum(data["lens"])
        num_examples = len(data["lens"])
        tokens_seen += num_tokens
        examples_seen += num_examples

        total_loss_w += data["loss_weights"].sum().item()
        total_loss_toks += (data["loss_weights"] > 0).sum().item()
        total_pplx += extras["pplx"].item()
        total_correct += extras["ncorrect"].item()

    # Get all sum/info to rank0. `g` stands for `globally`.
    if g := u.sum_to(
        rank=0,
        tokens_seen=tokens_seen,
        examples_seen=examples_seen,
        total_loss_w=total_loss_w,
        total_loss_toks=total_loss_toks,
        total_pplx=total_pplx,
        total_correct=total_correct,
    ):
        return {
            "pplx": g["total_pplx"] / g["examples_seen"],
            "tacc": g["total_correct"] / g["total_loss_toks"],
        }


# Everything below is to run a test that verifies the pipeline on multi-process when
# things don't evenly divide, to be run manually on 2-process machine.


def test_predict_fn(tokens, flex_masks, loss_weights, iseq):
    lsum = sum(tokens * loss_weights)
    pplx = sum(tokens * (loss_weights > 0))
    loss = lsum / max(1, sum(loss_weights))
    return loss, {"pplx": pplx, "lsum": lsum}


class TestDataset:
    def make_exids(self, seed, epoch=0, rank=0, world_size=1):
        return {
            0: [3, 5, 6],
            1: [4, 5],
        }[rank]

    def make_example(self, exid, epoch):
        import numpy as np
        return {
            "tokens": np.array([exid] * exid),
            "loss_weights": np.array([1 / exid] * exid),
            "id": exid,
        }


def test(rank, local_rank, world_size):
    assert world_size == 2, f"This test is made for world-size 2, not {world_size=}"

    # In theory we only need `init_device_mesh`, but in practice, we need this
    # whole verbose `init_process_group` or else the `barrier` will throw a warning.
    import torch.distributed as distr
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    distr.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device
    )
    import bv2.pdb_distr
    bv2.pdb_distr.enable_as_default()
    distr.device_mesh.init_device_mesh(
        "cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),  # Add "tp" for 2d parallel
    )

    from functools import partial

    import bv2.simple_data
    data_iter = bv2.simple_data.get_iter(TestDataset(), eagerness=0)
    data_iter = partial(data_iter, maxtok=10, device=device, rank=rank, world_size=world_size)
    if results := run(test_predict_fn, data_iter):
        print(results)
        import numpy as np
        np.testing.assert_allclose(results["loss"], np.mean([3,5,6,4,5]))
        np.testing.assert_allclose(results["pplx"], np.mean(np.square([3,5,6,4,5])))


if __name__ == "__main__":
    import os
    test(
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ.get("LOCAL_RANK", os.environ["RANK"])),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    print("SUCCESS")
