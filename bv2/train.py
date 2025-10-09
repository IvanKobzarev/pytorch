"""
pip install -U -r bv2/requirements.txt
torchrun --nproc_per_node=gpu -m bv2.train
"""

import json
import os
import re
from datetime import datetime
from functools import partial
from getpass import getuser
from importlib import import_module
from itertools import chain
from os.path import join as pjoin
from time import perf_counter

import numpy as np
import sws
import torch
import torch.distributed as distr
from torch.profiler import profile, ProfilerActivity, record_function

import bv2.pdb_distr  # isort: skip
import bv2.simple_data  # isort: skip
import bv2.simple_fsdp  # isort: skip
import bv2.utils as u  # isort: skip
from bv2.metrics import WandbLogger  # isort: skip
from bv2.model import SimpleTransformer  # isort: skip


# Allow using the (lower-precision) tensorcores for all fp32 matmuls.
# See https://docs.pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
torch.backends.fp32_precision = "tf32"

# Reduce limit to make the issue appear faster
# torch._dynamo.config.recompile_limit = 1


# This section configures pytorch to be fully deterministic.
# No noticeable perf impact for our toy model so far.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_deterministic_debug_mode("error")  # raises error on non-determinism


def main(c, rank, local_rank, world_size):  # noqa: C901
    name = c.get("name", f"{getuser()}-{datetime.now():%y%m%d-%H%M%S}")

    prints = partial(print_stamped, rank=rank)
    prints0 = prints if rank == 0 else lambda *args, **kwargs: None

    prints0(f"Running with arguments:\n{c}")

    # start from the beginning to track every gpu memory allocation
    # otherwise we lost cpp tracestack for model initialization
    torch.cuda.memory._record_memory_history(max_entries=10000000)

    # In theory we only need `init_device_mesh`, but in practice, we need this
    # whole verbose `init_process_group` or else the `barrier` will throw a warning.
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    distr.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device
    )
    bv2.pdb_distr.enable_as_default()
    mesh = distr.device_mesh.init_device_mesh(
        "cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),  # Add "tp" for 2d parallel
    )

    # Get workdir from rank0 to make sure it's consistent across hosts (timestamp)
    workdir = f"/checkpoint/rigi/bv2/workdirs/{name}"
    workdir = u.broadcast_object_from(rank=0, obj=workdir)
    prints0(f"Workdir: {workdir}")

    # Now that we know the final workdir, dump some info in it and start wandb with it.
    if rank == 0:
        os.makedirs(workdir, exist_ok=True)
        with open(pjoin(workdir, "config.json"), "w+") as f:
            f.write(c.to_flat_json(indent=0))
    wlogger = WandbLogger(
        c.to_dict(), rank, name, workdir, project="bv2" if c.nsteps > 50 else "bv2-dev"
    )

    # Import and get data source. We need it early on to know vocab size.
    ds, data_iter = bv2.simple_data.from_config(c)

    # Create the model on "meta" device, this avoids materializing param buffers.
    with torch.device("meta"):
        model = SimpleTransformer(
            **{"vocab": ds.vocab_size(), **c.model.to_dict()},
        )

    # This wraps all param properties with a shard/gather code.
    model = bv2.simple_fsdp.data_parallel(
        model,
        mesh,
        mode="fully_shard",  # Or: "replicate"
        # Only do FSDP's "remat" (of params/comms) if we don't already cover that ourselves.
        need_ac=not c.model.get("remat", True),
        mp_policy=bv2.simple_fsdp.MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        ),
        min_bytes=1024 * 1024,  # Don't shard params that are less than 1MiB
    )
    prints0(model)

    # Allocate buffers and sharded parameters on GPU
    model.to_empty(device=device)

    # And then run initializers on them, one by one.
    rng = np.random.default_rng(c.seed)
    rng, rng_param = rng.spawn(2)
    with torch.no_grad():
        with bv2.simple_fsdp.disable_data_parallel():  # super important, or nothing happens.
            rng_param = torch.Generator(device=device).manual_seed(
                int(rng_param.integers(0, 2**32, world_size)[rank])
            )
            model.init_weights(rng_param)
    log_pg(model, wlogger)
    if rank == 0:
        summary_table(model, stats=c.get("param_stats", False))

    # NOTE: Optimizer doesn't alloc here, only allocs on `.step()`.
    optim = torch.optim.AdamW(model.parameters(), lr=0.0, fused=True)
    decay_params = [p for n, p in model.named_parameters() if is_decay(n)]

    @record_function("fwd_and_bwd")
    @torch.compile
    def _fwd_and_bwd_step(weight_decay, *a, **kw):
        loss, extras = model(*a, mode="loss and bwd", **kw)
        optim.step()
        if weight_decay:
            with torch.no_grad():
                for param in decay_params:
                    param.mul_(1.0 - weight_decay)
        return loss, extras

    @torch.compile
    def _fwd(*a, **kw):
        return model(*a, mode="loss", **kw)

    # Make sure each hosts generates different data.
    data_seed = rng.integers(2**32, size=world_size)[rank].item()

    # Potentially resume from a checkpoint, if not, init stuff.
    first_step, tokens_seen, examples_seen = 0, 0, 0
    resumed_epoch, resumed_i = 0, 0
    if extras := maybe_load_ckpt(c.get("resume") or pjoin(workdir, "latest"), model, optim, extras={
        "step": first_step,
        "tokens_seen": tokens_seen,
        "examples_seen": examples_seen,
        "data": {"seed": data_seed, "ep": 0, "i": 0},
    }):  # fmt: skip
        data_seed, resumed_epoch, resumed_i = (
            extras["data"]["seed"], extras["data"]["ep"], extras["data"]["i"])  # fmt: skip
        first_step, tokens_seen, examples_seen = (
            extras["step"], extras["tokens_seen"], extras["examples_seen"])  # fmt: skip
        wlogger.step = first_step
    ckpt_future = None

    peak_mems = []
    train_times = []
    t0 = t_prev_step_end = perf_counter()
    prof = c.nsteps > 50 and profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,  # Done with torch.cuda functions instead.
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )

    # Eval loop is reused, so we wrap it as a function
    def run_evals(step):
        """Run evaluations for a given step if conditions are met."""
        for ev in c.get("evals", {}):
            is_every_n_steps = step % c.evals[ev].steps == 0
            is_final = step == c.nsteps
            if not(is_every_n_steps) and not(is_final):
                continue
            em = import_module(f"bv2.eval.{c.evals[ev].type}")
            _, ev_data_iter = bv2.simple_data.from_config(c.evals[ev])
            ev_data_iter = partial(ev_data_iter, c.maxtok, device, rank, world_size)
            with torch.no_grad():
                if results := em.run(_fwd, ev_data_iter):
                    wlogger.log({f"{ev}/{k}": v for k, v in results.items()})
            # TODO: Check how switching train/eval mode (dropout) interacts with compile

    if not c.get("skip_initial_eval", False):
        run_evals(first_step)

    # NOTE: this way of timing misses waits for data.
    for step, data in zip(
        range(first_step, c.nsteps),
        data_iter(
            c.maxtok, device, rank, world_size, data_seed, resumed_epoch, resumed_i
        ),
    ):
        tprev, t0 = t0, perf_counter()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        if prof and step == 3:
            torch.cuda.cudart().cudaProfilerStart()
            prof.start()

        sched = global_schedule(
            step=step,
            total_steps=c.nsteps,
            warmup_steps=c.warmup_nsteps,
        )

        lr = sched * c.lr
        set_lr_(optim, lr)
        wlogger.log({"chrono/lr": lr, "chrono/sched": sched})

        model.zero_grad(set_to_none=True)

        all_lens = u.all_gather_object(data["lens"])
        tokens_seen += sum(sum(l) for l in all_lens)
        num_examples = sum(len(l) for l in all_lens)
        examples_seen += num_examples
        wlogger.log({"chrono/tokens_seen": tokens_seen})
        wlogger.log({"chrono/examples_seen": examples_seen})
        wlogger.log({"chrono/num_examples": num_examples})
        wlogger.log({"chrono/percent": (step + 1) / c.nsteps})
        all_max_epoch = u.all_gather_object(max(s["ep"] for s in data["state_after"]))
        wlogger.log({"chrono/epoch": max(all_max_epoch)})

        loss, extras = _fwd_and_bwd_step(
            c.wd * sched,
            data["tokens"],
            data["flex_masks"],
            data["loss_weights"],
            data["iseq"],
        )
        if "pplx" in extras:
            all_pplx = u.all_gather_object(extras["pplx"].cpu())
            extras["pplx"] = sum(all_pplx) / num_examples
            extras["pplx/bits"] = extras["pplx"] / np.log(2)
        loss = loss.item()  # Causes a transfer.
        prints0(f"step {step}: loss {loss:.8f}")
        wlogger.log({"loss/train": loss})
        wlogger.log({"loss/train/bits": loss / np.log(2)})
        # log only scalar values
        wlogger.log(
            {f"loss/{k}": v.item() for k, v in extras.items() if v.numel() == 1}
        )


        # After the update is done, we are at the step+1
        torch.cuda.synchronize()
        wlogger.end_step()
        step += 1

        train_times.append((perf_counter() - t0) * 1000)  # ms
        peak_mems.append(torch.cuda.max_memory_allocated() / 1024**2)  # MiB
        wlogger.log({"chrono/peakmem": peak_mems[-1]})
        wlogger.log({"chrono/traintime": train_times[-1]})
        wlogger.log({"chrono/steptime": t0 - tprev})
        wlogger.log({"chrono/datawait": t0 - t_prev_step_end})

        # Checkpoint, but note this is *after* `step`'s update, so +1.
        ckpt_future = maybe_save_ckpt(
            step, model, optim, workdir, last_future=ckpt_future, extras={
                "data": {"seed": data_seed, **data["state_after"][-1]},
                "tokens_seen": tokens_seen,
                "examples_seen": examples_seen,
            })  # fmt: skip

        if prof and step == 2:
            # dumping first 3 iterations from init are enough to include optim states.
            # Otherwise the .pkl becomes too big and freezes chrome.
            # Drag .pkl file to https://docs.pytorch.org/memory_viz
            torch.cuda.memory._dump_snapshot(pjoin(workdir, f"prof_memsnap_r{rank}.pkl"))  # fmt: skip
        if prof and step == 6:  # Open in about://tracing or ui.perfetto.dev
            torch.cuda.cudart().cudaProfilerStop()
            prof.stop()  # TODO: speedup gz
            prof.export_chrome_trace(pjoin(workdir, f"prof_trace_r{rank}.json.gz"))
            prof.export_stacks(pjoin(workdir, f"prof_stacks_cpu_r{rank}.txt"))

        run_evals(step)

        # visualize input tokens
        if c.nsteps >= 50 and step == 8:
            with open(pjoin(workdir, "data.pt"), "wb") as f:
                torch.save({k: v for k, v in data.items() if k != "flex_masks"}, f)
            if hasattr(ds, "vis_data_wandb"):
                wlogger.log({f"vis/data{step}": ds.vis_data_wandb(data)})

        if step % 1000 == 0 and hasattr(ds, "vis_output_wandb"):
            pred = extras["predictions"].detach().cpu()
            wlogger.log({f"vis/output{step}": ds.vis_output_wandb(data, pred)})

        distr.barrier()  # Just for simplicity for now.

        log_pg(model, wlogger)
        t_prev_step_end = perf_counter()

    prints(f"Peak mems (med: {np.median(peak_mems):.1f}MiB): {' '.join(f'{t:.0f}' for t in peak_mems)}")  # fmt: skip
    prints(f"Step times (med: {np.median(train_times):.1f}ms): {' '.join(f'{t:.0f}' for t in train_times)}")  # fmt: skip
    torch._dynamo.reset()  # Avoid hang: https://x.com/main_horse/status/1937900381574717940
    if ckpt_future:
        ckpt_future.result()
    distr.destroy_process_group()
    wlogger.finish()
    prints("Destroyed group")


###############
# UTILS BELOW #
###############


def global_schedule(*, step, total_steps, warmup_steps=1):
    """Implements constant schedule with warmup."""
    return min(1.0, step / warmup_steps)


def set_lr_(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def print_stamped(s, rank, **kw):
    t = datetime.now().time().isoformat(timespec="milliseconds")
    print(f"[{rank} {t}] {s}", **kw)


def global_reduce(x, method):
    """Call `method` such as norm, mean, std, ... and return global scalar."""
    if x is None:
        return None

    n = getattr(x, method)()

    # For DTensor, this returns a _NormPartial object which only holds the local "norm piece".
    # Only calling `full_tensor` on it syncs the output and gives each rank the same whole norm!
    if hasattr(n, "full_tensor"):
        n = n.full_tensor()

    return n.item()


def log_pg(model, wlogger=None):
    # TODO: these are only here because Lucas is unsure.
    torch.cuda.synchronize()
    distr.barrier()
    for name, param in model.named_parameters():
        wlogger.log(
            {
                f"pnorm/{name}": global_reduce(param, "norm"),
                f"gnorm/{name}": global_reduce(param.grad, "norm"),
            },
        )


def swissnum(x):
    return f"{x:_}".replace("_", "'")


def is_decay(name):
    decays = [
        r"blocks.*\.att\..*weight",
        r"blocks.*\.mlp\..*weight",
        r"txt_unemb\.head\.weight",
        r"img_emb.proj.weight",
    ]
    return any(re.match(d, name) for d in decays)


def summary_table(model, stats=True):
    import rich
    from rich.table import Table

    tbl = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=True,
        footer_style="bold magenta",
        box=rich.box.HORIZONTALS,
    )
    tbl.add_column("name", justify="left")
    tbl.add_column("shape", justify="right")
    tbl.add_column("dtype", justify="right")
    tbl.add_column("params", justify="right")
    tbl.add_column("placement", justify="right")
    tbl.add_column("local shape", justify="right")
    tbl.add_column("weight decay", justify="right")
    if stats:
        tbl.add_column("mean", justify="right")
        tbl.add_column("std", justify="right")

    total_num, total_bytes, local_bytes = 0, 0, 0
    for name, x in chain(model.named_parameters(), model.named_buffers()):
        total_num += x.numel()
        total_bytes += x.nbytes
        cols = [name]
        cols += [str(tuple(x.shape))]
        cols += [str(x.dtype)[len("torch."):]]  # fmt: skip
        cols += [swissnum(x.numel())]
        if hasattr(x, "placements"):
            cols += [str(x.placements), str(tuple(x.to_local().shape))]
            local_bytes += x.to_local().nbytes
        else:
            cols += ["-", "shape"]
        cols += [str(is_decay(name))]
        if stats:
            cols += [global_reduce(x, "mean"), global_reduce(x, "std")]
        tbl.add_row(*cols)

    tbl.columns[0].footer = f"Total: {swissnum(total_num)}"
    tbl.columns[1].footer = f"({total_bytes/1024/1024:.0f}MiB)"
    tbl.columns[2].footer = f"Local: {local_bytes/1024/1024:.0f}MiB"
    rich.print(tbl)


# ---- CHECKPOINTING ----
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcpsd


def suppress_warnings(message, category=Warning):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=message, category=category)
                return func(*args, **kwargs)

        return wrapper

    return decorator


class ModelState(dcp.stateful.Stateful):
    def __init__(self, model):
        self.model = model
        self.opts = dcpsd.StateDictOptions(cpu_offload=True)

    def state_dict(self):  # "Saving"
        return dcpsd.get_model_state_dict(self.model, options=self.opts)

    def load_state_dict(self, sd):
        dcpsd.set_model_state_dict(self.model, sd)


class OptimState(dcp.stateful.Stateful):
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.opts = dcpsd.StateDictOptions(cpu_offload=True)

    def state_dict(self):  # "Saving"
        return dcpsd.get_optimizer_state_dict(self.model, self.optim, options=self.opts)

    def load_state_dict(self, sd):
        dcpsd.set_optimizer_state_dict(self.model, self.optim, sd)


@suppress_warnings(".*version 2.5 of PyTorch, `overwrite` will default to False.*")
@suppress_warnings(".*TypedStorage is deprecated", UserWarning)
def maybe_save_ckpt(step, model, optim, workdir, extras=None, last_future=None):
    # TODO: lots of logic about which step, last step, etc. and configurable.
    if step % 1000 != 0:
        return last_future

    if last_future is not None:  # Wait for last one to finish.
        last_future.result()

    path = pjoin(workdir, "latest")
    print(f"Checkpointing to {path}")

    # This is funny, but the `async_save` below interacts with `distr` in some way such
    # that if we do the `gather_object` after it, it would deadlock. Unless we barrier,
    # which defeats the point of async. So, gather_object first.
    all_extras = u.gather_object_to(rank=0, obj={"step": step, **extras})

    last_future = dcp.async_save(
        state_dict={
            "model": ModelState(model),
            "optim": OptimState(model, optim),
        },
        storage_writer=dcp.FileSystemWriter(path, overwrite=True),
    )

    # TODO: For some reason async checkpoint cases non-deterministic failures.
    #
    # Error: terminate called after throwing an instance of 'gloo::EnforceNotMet'
    #         what():  [enforce fail at /pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:456]
    #         op.preamble.length <= op.nbytes. 6232 vs 4
    #
    # We should fix it, for now just make code synchronous.
    last_future.result()

    if all_extras:  # means we have extras *and* we are rank0
        # TODO: Networked filesystems (probably blobfile?)
        with open(pjoin(path, "extras.json"), "w+") as f:
            json.dump(all_extras, f)

    return last_future


def maybe_load_ckpt(path, model, optim, extras):
    if not os.path.exists(path or ""):
        return

    print(f"Resuming from {path}")

    # We `allow_partial_load` because...
    dcp.load(
        {"model": ModelState(model), "optim": OptimState(model, optim)},
        checkpoint_id=path,
        planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
    )

    with open(pjoin(path, "extras.json"), "r") as f:
        extras = json.load(f)
        if len(extras) != (w := distr.get_world_size()):
            raise RuntimeError(f"World size changed: {len(extras)} != {w}")
        return extras[distr.get_rank()]


# ---- END CHECKPOINTING ----


def get_config():
    c = sws.Config()
    c.seed = 0

    c.data_name = "random_nouns"

    c.maxtok = 8 * 4096

    c.nsteps = 16
    c.warmup_nsteps = 3
    c.lr = 3e-4
    c.wd = lambda: c.lr * 0.1

    c.model.dim = 4096
    c.model.depth = 4
    c.model.txt_unemb.chunks = 8

    c.resume = None  # Optional path to checkpoint.

    c.evals.pplx_val.type = "pplx"
    c.evals.pplx_val.steps = 10
    c.evals.pplx_val.data_name = lambda: c.data_name
    c.evals.pplx_val.data.seed = 31337  # "val split"
    c.evals.pplx_val.data.eagerness = 2

    return c


if __name__ == "__main__":
    sws.run(
        partial(
            main,
            rank=int(os.environ["RANK"]),
            local_rank=int(os.environ.get("LOCAL_RANK", os.environ["RANK"])),
            world_size=int(os.environ["WORLD_SIZE"]),
        ),
    )
