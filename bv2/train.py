"""
pip install -U -r bv2/requirements.txt
torchrun --nproc_per_node=gpu -m bv2.train
"""

import gc
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from functools import cache, partial
from getpass import getuser
from importlib import import_module
from itertools import chain
from os.path import join as pjoin
from time import perf_counter

import numpy as np
import rich
import sws
import torch
import torch.distributed as distr
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcpsd
from torch.profiler import ProfilerActivity, profile, record_function

import bv2.metrics
import bv2.pdb_distr
import bv2.simple_data
import bv2.simple_fsdp
import bv2.utils as u
from bv2.model import SimpleTransformer
from bv2.muon import Muon

# Allow using the (lower-precision) tensorcores for all fp32 matmuls.
# See https://docs.pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
torch.backends.fp32_precision = "tf32"
torch.backends.cuda.matmul.allow_tf32 = True

# Reduce limit to make the issue appear faster
torch._dynamo.config.recompile_limit = 1
torch._dynamo.config.fail_on_recompile_limit_hit = True
torch._dynamo.config.accumulated_recompile_limit = 10_000_000  # Basically inf.


# This section configures pytorch to be fully deterministic.
# No noticeable perf impact for our toy model so far.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_deterministic_debug_mode("error")  # raises error on non-determinism


def main(c, rank, local_rank, world_size):  # noqa: C901
    prints0(f"Running with arguments:\n{c}")

    # We want to control GC collection, exactly once per step.
    # Otherwise, different processes pause the world for collection at different times,
    # which introduces a "spike" in timing each time one process does a big (300+ms) collection.
    gc.disable()

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

    # Get xid/name from rank0 to make sure it's consistent across hosts (if it has timestamp)
    xid = c.get("xid", f"{datetime.now():%y%m%d_%H%M%S}")
    name = c.get("name", f"{getuser()}-{xid}") + (f"-{c.wid}" if "wid" in c else "")
    xid, name = u.broadcast_object_from(rank=0, obj=(xid, name))
    workdir = "workdirs" if c.nsteps >= 50 else "workdirs-dbg"
    workdir = pjoin("/checkpoint/rigi/bv2/", workdir, xid, name)
    prints0(f"Workdir: {u.BLUE}{workdir}{u.RESET}")

    if rank == 0:
        os.makedirs(workdir, exist_ok=True)
        with open(pjoin(workdir, "config.json"), "w+") as f:
            f.write(c.to_flat_json(indent=0))

    # Import and get data source. We need it early on to know vocab size.
    ds = bv2.simple_data.from_config({'seed': (c.seed, "dataset"), **c.data.to_dict()})

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

    # For initialization, it's important to disable FSDP, otherwise init would only
    # be applied to an all-gathered copy of each param, so have no effect.
    # Also, in latest PyTorch (after this? https://github.com/pytorch/pytorch/pull/159933)
    # we should pass the same RNG instance to DTensor on all processes.
    with torch.no_grad():
        with bv2.simple_fsdp.disable_data_parallel():  # super important, or nothing happens.
            model.init_weights(u.rng_torch(c.seed, "param_init", device=device))
    muon_args = c.muon.to_dict()
    param_modes = muon_args.pop("param_modes")

    def get_muon_param_mode(name):
        for mode, regexps in param_modes.items():
            if any(re.fullmatch(r, name) for r in regexps):
                return mode
        raise ValueError(f"Every param should be matched to an optimizer mode. `{name}` was not matched")

    if rank == 0:
        summary_table(model, stats=c.get("param_stats", False), param_mode=get_muon_param_mode)

    param_groups = defaultdict(list)
    for n, p in model.named_parameters():
        param_groups[get_muon_param_mode(n)].append(p)
    params = [{"params": params, "mode": mode} for mode, params in param_groups.items()]

    optim = Muon(params, lr=torch.tensor(0.0), **muon_args)
    optim.init_state() # we init state to avoid recompiles
    decay_params = [p for n, p in model.named_parameters() if is_decay(n)]

    @record_function("fwd_and_bwd_step")
    @u.suppress_warnings("`isinstance(treespec, LeafSpec)` is deprecated", FutureWarning)
    @u.suppress_warnings("`isinstance(treespec, TreeSpec)` is deprecated", FutureWarning)
    @torch.compile(dynamic=False)
    def _fwd_and_bwd_step(weight_decay, *a, **kw):
        loss, extras = model(*a, mode="loss and bwd", **kw)
        optim.step()
        if weight_decay is not None:
            with torch.no_grad():
                for param in decay_params:
                    param.mul_(1.0 - weight_decay)
        return loss, extras

    # Potentially resume/fork from a checkpoint, if not, init stuff.
    first_step, tokens_seen, examples_seen = 0, 0, 0
    resume_data = {}

    # Checkpoint loading priority: resume > fork > init
    ckpt_path = c.get("fork") or c.get("init")
    if os.path.exists(pjoin(workdir, "ckpt-latest")):  # := is_resuming
        ckpt_path = pjoin(workdir, "ckpt-latest")

    if ckpt_path:
        if extras := load_ckpt(ckpt_path, model, optim, weights_only=bool(c.get("init"))):
            first_step, resume_data = extras["step"], extras["data"]
            tokens_seen, examples_seen = extras["tokens_seen"], extras["examples_seen"]

    mw = bv2.metrics.MultiWriter(
        bytes=bv2.metrics.BytesWriter(rank, workdir, first_step),
        plattli=bv2.metrics.PlattliWriter(rank, workdir, first_step),
    )
    if rank == 0:  # Log once more after ckpt resume.
        summary_table(model, stats=c.get("param_stats", False), param_mode=get_muon_param_mode)
    prints0(model)

    peak_mems, model_times, step_times = [], [], []
    t0 = t_step_start = t_prev_step_end = perf_counter()
    prof = c.nsteps >= 50 and profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,  # Done with torch.cuda functions instead.
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )

    # We have a factory here, so that we get independent compiles and compile-limit
    # counters for individual evals. For example, we run different evals at varying
    # resolutions, batch-sizes, max-tokens, and don't want them to affect each other.
    @cache
    def get_fwd(eval_key):
        def _fwd(*a, **kw):
            return model(*a, **kw)

        fn = torch.compile(u.clone_function(_fwd, name_suffix=eval_key), dynamic=False)
        fn = u.suppress_warnings("`isinstance(treespec, LeafSpec)` is deprecated", FutureWarning)(fn)
        fn = u.suppress_warnings("`isinstance(treespec, TreeSpec)` is deprecated", FutureWarning)(fn)
        fn = u.suppress_warnings("remat_using_tags_for_fwd_loss_bwd_graph: Graph has recomputable ops but no backward region.", UserWarning)(fn)  # Fixed in https://github.com/pytorch/pytorch/pull/173528
        fn = record_function(f"eval_fwd_{eval_key}")(fn)
        return fn

    # Eval loop is reused, so we wrap it as a function
    def run_evals(step):
        """Run evaluations for a given step if conditions are met."""
        teval0, ran_eval = perf_counter(), False
        for ev_name in c.get("evals") or {}:
            ev = c.evals[ev_name]
            is_step = (step == 2 or (step > 2 and step % ev.steps == 0)) if isinstance(ev.steps, int) else step in ev.steps
            if u.about_to_get_killed() or not (is_step or step == c.nsteps):  # Always run on last step.
                continue
            tev0, ran_eval = perf_counter(), True
            prints0(f"Running evaluator {u.BLUE}{ev_name}{u.RESET}", end="", flush=True)
            em = import_module(f"bv2.eval.{ev.type}")
            ds_ev = bv2.simple_data.from_config(ev.data.to_dict())
            args = {k: v for k, v in ev.to_dict().items() if k not in {"type", "data", "steps"}}
            with torch.no_grad():
                if results := em.run(get_fwd(ev_name), ds_ev, **args, rank=rank, world_size=world_size, device=device):
                    mw.log({f"{ev_name}/{k}": v for k, v in results.items()}, flush=True)
                    print0("")  # End the line we did not end above.
                    for k, v in results.items():
                        prints0(f"Eval results: {ev_name}/{k}: {v}")
            u.global_gpu_barrier(device)  # For accurate timing and avoiding u.about_to_get_killed-related divergence.
            mw.log({f"chrono/evals/{ev_name}": perf_counter() - tev0})
        if ran_eval:
            mw.log({"chrono/evaltime": perf_counter() - teval0})
            gc.collect(2)  # Let's also use eval as opportunity to run a full GC collection.

    per_src_examples_seen, per_src_tokens_seen = Counter(), Counter()
    for step, data in zip(
        range(first_step, c.nsteps),
        bv2.simple_data.data_iter(
            ds, seed=(c.seed, "data_iter"),
            device=device, rank=rank, world_size=world_size, resume=resume_data,
            **c.iter.to_dict(),
        ),
    ):
        torch.cuda.reset_peak_memory_stats()
        u.global_gpu_barrier(device)  # For accurate global datawait timing.
        t_prev_step_start, t_step_start = t_step_start, perf_counter()

        if prof and (step - first_step) == 50:
            torch.cuda.cudart().cudaProfilerStart()
            prof.start()

        sched = global_schedule(
            step=step,
            total_steps=c.nsteps,
            warmup_steps=c.warmup_nsteps,
        )

        lr = sched * c.lr
        set_lr_(optim, lr)
        mw.log({"chrono/lr": lr, "chrono/sched": sched})

        model.zero_grad(set_to_none=True)

        all_lens = u.all_gather_object(data["lens"])
        num_tokens = sum(sum(l) for l in all_lens)
        num_examples = sum(len(l) for l in all_lens)
        tokens_seen += num_tokens
        examples_seen += num_examples
        mw.log({"chrono/tokens_seen": tokens_seen})
        mw.log({"chrono/examples_seen": examples_seen})
        mw.log({"chrono/num_tokens": num_tokens})
        mw.log({"chrono/num_examples": num_examples})
        mw.log({"chrono/percent": (step + 1) / c.nsteps})

        # Need to log param norms at this step before the update
        if step < 50 or step % 10 == 0:  # Interesting frequently early, sparsely later.
            mw.log({f"pnorm/{n}": global_reduce(p, "norm") for n, p in model.named_parameters()})

        t_before_model = perf_counter()  # Let's not sync/barrier, FSDP does that anyways.
        local_loss, extras = _fwd_and_bwd_step(
            torch.tensor(c.wd * sched) if c.wd else None,
            data["tokens"],
            data["flex_masks"],
            data["loss_weights"],
            data["iseq"],
        )

        u.global_gpu_barrier(device)  # For accurate "global" timings
        mw.log({"chrono/modeltime": (model_time := perf_counter() - t_before_model)})
        mw.log({"chrono/steptime": (step_time := t_step_start - t_prev_step_start)})
        mw.log({"chrono/proctime": perf_counter() - t0})
        mw.log({"chrono/datawait": t_step_start - t_prev_step_end})
        mw.log({"sys/gpu_peak_mem_gb": (peak_mem := torch.cuda.max_memory_allocated() / 1024**3)})
        if step % 10 == 0 and rank == 0:
            bv2.metrics.log_system_metrics(mw, gpu_index=0, prefix="sys")
        if c.nsteps < 50:
            model_times.append(model_time)
            step_times.append(step_time)
            peak_mems.append(peak_mem * 1024)  # MiB

        global_loss, global_pplx, global_ncorrect = u.all_reduce_scalars(
            local_loss, extras["pplx"], extras["ncorrect"])

        prints0(f"step {step}: loss {global_loss:.8f}")
        mw.log({"train/pplx": global_pplx / num_examples})
        mw.log({"train/loss": global_loss})  # loss used for bwd, so already normalized by a global weight
        mw.log({"train/tokacc": global_ncorrect / extras["global_total_loss_toks"].item()})
        mw.log({"train/n_loss_toks": extras["global_total_loss_toks"].item()})
        max_logits = u.all_reduce_scalars(*(blk["attn"]["max_logit"] for blk in extras["blk"].values()), op=distr.ReduceOp.MAX)
        mw.log({f"attn_max_logit/blk{i}": max_logits[i] for i in extras["blk"]})

        # For dataset mixtures, collect and report per-component stats and loss.
        # TODO: Update this to be global, or at least check!
        if "src" in data:
            # Count the number of examples of each subset source:
            all_counts = u.all_gather_object(Counter(data["src"]))
            per_src_examples_seen = sum(all_counts, per_src_examples_seen)
            mw.log({f"mix_examples_seen/{n}": c for n, c in per_src_examples_seen.items()})

            per_src_toks = Counter()
            per_src_pplx = defaultdict(list)
            for iseq, src in enumerate(data["src"]):  # This is basically for each example.
                iseq_mask = (data["iseq"] == iseq)  # Which token is from this example?
                per_src_toks[src] += iseq_mask.sum().cpu()
                per_src_pplx[src].append((extras["tok_losses"] * iseq_mask[:-1]).sum().cpu())

            per_src_tokens_seen = sum(u.all_gather_object(per_src_toks), per_src_tokens_seen)
            mw.log({f"mix_tokens_seen/{n}": v.item() for n, v in per_src_tokens_seen.items()})

            all_pplx = u.all_gather_object(per_src_pplx)  # List of dict of list
            for src in {k for d in all_pplx for k in d}:  # Union of all seen src
                pplx = np.concat([d.get(src, []) for d in all_pplx]).mean()  # In nats
                mw.log({f"mix_pplx/{src}": pplx / np.log(2)})  # In bits

        # Do controlled garbage collection to control for lag spikes.
        # gen0 cost about 3-6ms per step, gen2 about 300-500. gen0 every 10 steps 10x its cost => not useful.
        gc_t0 = perf_counter()
        gc_n  = gc.collect(0)
        mw.log({  # Adding a timing barrier would add a few ms, so we time rank0 only.
            "chrono/gctime": perf_counter() - gc_t0,
            "sys/rank0/gc_ncollected": gc_n,
            # **{f"sys/gc_nobj_{i}": len(gc.get_objects(i)) for i in (0, 1, 2)},  # Expensive
        })

        # And grad-norms are for this step, but we only get them after the update ran, i.e. here.
        if step < 50 or step % 10 == 0:  # Interesting frequently early, sparsely later.
            mw.log({f"gnorm/{n}": global_reduce(p.grad, "norm") for n, p in model.named_parameters()})

        # After the update is done, we are at the step+1
        mw.end_step()
        step += 1

        # Checkpoint, but note this is *after* `step`'s update, so +1.
        maybe_save_ckpt(
            step, save_steps=c.get("ckpt_steps", 1000), keep_steps=c.get('ckpt_keep_steps', ()),
            model=model, optim=optim, workdir=workdir, extras={
                "data": data["state_after"][-1],  # NOTE: This differs per process(!)
                "tokens_seen": tokens_seen,
                "examples_seen": examples_seen,
                "metrics": mw.save_ckpt(),
                "jid": c.get("jid", "n/a"),  # Just for future archeologs.
            })  # fmt: skip

        if u.about_to_get_killed():  # We checkpointed, yay, quick, byebye.
            break

        if prof and (step - first_step) == 2:
            # dumping first 3 iterations from init are enough to include optim states.
            # Otherwise the .pkl becomes too big and freezes chrome.
            # Drag .pkl file to https://docs.pytorch.org/memory_viz
            torch.cuda.memory._dump_snapshot(pjoin(workdir, f"prof_memsnap_s{step}_r{rank}.pkl"))  # fmt: skip
        if prof and (step - first_step) == 53:  # Open in about://tracing or ui.perfetto.dev
            torch.cuda.cudart().cudaProfilerStop()
            prof.stop()  # TODO: speedup gz
            prof.export_chrome_trace(pjoin(workdir, f"prof_trace_s{step}_r{rank}.json.gz"))
            prof.export_stacks(pjoin(workdir, f"prof_stacks_cpu_s{step}_r{rank}.txt"))

        run_evals(step)

        # visualize input tokens
        if c.nsteps >= 50 and step == 8:
            with open(pjoin(workdir, f"data_r{rank}.pt"), "wb") as f:
                torch.save({k: v for k, v in data.items() if k != "flex_masks"}, f)

        u.global_gpu_barrier(device)  # Sync to get accurate datawait timing.
        t_prev_step_end = perf_counter()

    if c.nsteps < 50:
        prints(f"Peak mems (med: {np.median(peak_mems):.1f}MiB): {' '.join(f'{t:.0f}' for t in peak_mems)}")  # fmt: skip
        prints(f"Model times (med: {np.median(model_times)*1000:.1f}ms): {' '.join(f'{t*1000:.0f}' for t in model_times)}")  # fmt: skip
        prints(f"Step times (med: {np.median(step_times)*1000:.1f}ms): {' '.join(f'{t*1000:.0f}' for t in step_times)}")  # fmt: skip

    if u.about_to_get_killed():
        mw.finish(training_done=False)
        prints(f"Finished {perf_counter() - u.about_to_get_killed()}s after getting the pre-emption call!")
    else:
        mw.finish(training_done=True)
        if rank == 0:
            with open(pjoin(workdir, "DONE"), "w+") as f:
                f.write("All good!")
            prints(f"Done. Workdir: {u.BLUE}{workdir}{u.RESET}")

    torch._dynamo.reset()  # Avoid hang: https://x.com/main_horse/status/1937900381574717940
    distr.destroy_process_group()
    prints(f"Destroyed group on rank {rank}. All done for real.")


###############
# UTILS BELOW #
###############


def global_schedule(*, step, total_steps, warmup_steps=1):
    """Implements constant schedule with warmup."""
    return min(1.0, step / warmup_steps)


def set_lr_(optimizer, lr):
    lr = torch.tensor(lr)  # For torch.compile, else it's a compile-time constant!
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


def summary_table(model, stats=True, param_mode=None):
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
    tbl.add_column("param mode", justify="right")
    if stats:
        tbl.add_column("mean", justify="right")
        tbl.add_column("std", justify="right")

    total_num, total_bytes, local_bytes = 0, 0, 0
    for name, x, mode in chain(
        ((n, x, param_mode(n)) for n, x in model.named_parameters()),
        ((n, x, "-") for n, x in model.named_buffers()),
    ):
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
        cols += [mode]
        if stats:
            cols += [global_reduce(x, "mean"), global_reduce(x, "std")]
        tbl.add_row(*cols)

    tbl.columns[0].footer = f"Total: {swissnum(total_num)}"
    tbl.columns[1].footer = f"({total_bytes/1024/1024:.0f}MiB)"
    tbl.columns[2].footer = f"Local: {local_bytes/1024/1024:.0f}MiB"
    rich.print(tbl)


# ---- CHECKPOINTING ----


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


@u.suppress_warnings("version 2.5 of PyTorch, `overwrite` will default to False")
@u.suppress_warnings("TypedStorage is deprecated", UserWarning)
def maybe_save_ckpt(step, save_steps, keep_steps, model, optim, workdir, extras=None):

    should_save = (step % save_steps == 0) if isinstance(save_steps, int) else step in save_steps
    if not (u.about_to_get_killed() or should_save):
        return

    gc.collect(2)  # A good opportunity to run a full GC collection.

    path = pjoin(workdir, f"ckpt-{step:06d}")
    prints0(f"Checkpointing to {path}")

    # This is funny, but the `async_save` below interacts with `distr` in some way such
    # that if we do the `gather_object` after it, it would deadlock. Unless we barrier,
    # which defeats the point of async. So, gather_object first.
    all_extras = u.gather_object_to(rank=0, obj={"step": step, **extras})

    # TODO: For some reason async checkpoint cases non-deterministic failures.
    #
    # Error: terminate called after throwing an instance of 'gloo::EnforceNotMet'
    #         what():  [enforce fail at /pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:456]
    #         op.preamble.length <= op.nbytes. 6232 vs 4
    #
    # We should fix it, for now just make code synchronous.
    dcp.save(
        state_dict={
            "model": ModelState(model),
            "optim": OptimState(model, optim),
        },
        storage_writer=dcp.FileSystemWriter(path, overwrite=True),
    )

    if distr.get_rank() == 0:
        with open(pjoin(path, "extras.json"), "w+") as f:
            json.dump(all_extras, f)

        # Now do a rename/link/delete dance, so that `ckpt-latest` always points to
        # the latest one, and we either keep, or delete, the previous one, while not
        # losing any data if we get killed/pre-empted in the middle of this dance.
        try:
            prev_ckpt = os.readlink(pjoin(workdir, "ckpt-latest"))
        except FileNotFoundError:
            prev_ckpt = None

        os.symlink(path, pjoin(workdir, "ckpt-tmp"))
        os.replace(pjoin(workdir, "ckpt-tmp"), pjoin(workdir, "ckpt-latest"))  # Atomic

        if prev_ckpt is not None:
            prev_step = int(prev_ckpt.rsplit("-")[-1])
            if (prev_step % keep_steps != 0) if isinstance(keep_steps, int) else prev_step not in keep_steps:
                shutil.rmtree(prev_ckpt)


def load_ckpt(path, model, optim, weights_only=False):
    if not os.path.exists(path):
        raise ValueError(f"Checkpoint path was not found: {path}")

    prints0(f"Resuming from {path}")

    # We `allow_partial_load` because...
    dcp.load(
        {"model": ModelState(model)} | ({} if weights_only else {"optim": OptimState(model, optim)}),
        checkpoint_id=path,
        planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
    )

    if weights_only:
        return {}

    with open(pjoin(path, "extras.json"), "r") as f:
        extras = json.load(f)
        if len(extras) != (w := distr.get_world_size()):
            raise RuntimeError(f"World size changed: {len(extras)} != {w}")
        return extras[distr.get_rank()]


# ---- END CHECKPOINTING ----


def get_config():
    c = sws.Config()
    c.seed = 0

    c.maxtok = 8 * 4096 + 1

    c.data.name = "random_nouns"
    c.data.min_nouns = 128
    c.data.max_nouns = 256
    c.data.tokenizer.first_N = 10_000

    c.iter.eagerness = 16
    c.iter.maxtok = lambda: c.maxtok

    c.nsteps = 16
    c.warmup_nsteps = 3
    c.lr = 3e-4
    c.wd = lambda: c.lr * 0.1

    c.muon.param_modes = {"muon_h": [r".*mlp.l[12].weight", r".*att.[qkvo].weight", r".*img_emb.proj.weight", r".*txt_unemb.head.weight"],
                          "adam": [r".*"]}

    c.model.dim = 4096
    c.model.depth = 4
    c.model.txt_unemb.chunksz = 4096

    c.evals.pplx_val.type = "pplx"
    c.evals.pplx_val.steps = 10
    c.evals.pplx_val.data.name = lambda: c.data.name
    c.evals.pplx_val.data.seed = 31337  # "val split" content
    c.evals.pplx_val.data.n = 150  # "val split" size
    c.evals.pplx_val.data.tokenizer.first_N = lambda: c.data.tokenizer.first_N
    c.evals.pplx_val.iter.maxtok = lambda: c.maxtok
    c.evals.pplx_val.iter.eagerness = 16

    return c


if __name__ == "__main__":
    if "RANK" in os.environ:  # Launched via bv2/tools/local_run or torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ:  # Launched via srun directly.
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        print("Local run on single-gpu")
        rank = local_rank = 0
        world_size = 1

    # Add rank to cache dir to avoid race-condition on the lock.
    # It means ranks don't share the compile cache, but it also
    # means we don't get the following startup crash randomly anymore:
    # torch._inductor.exc.InductorError: Timeout: The file lock [...] could not be acquired.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{getuser()}_rank{local_rank}"

    if not sys.stdout.isatty():  # Don't squeeze tables or use colors in logs!
        rich.reconfigure(width=500, color_system=None)

    # We need to "warmup" the einops backend cache; if we don't, then einops
    # has a multi-threading race-condition that makes it fail in our input pipeline.
    import einops
    einops._backends.get_backend(np.empty((1,1), np.uint8))
    einops.rearrange(np.empty((1,1), np.uint8), 'a b -> a b')

    prints = partial(print_stamped, rank=rank)
    print0 = print if rank == 0 else lambda *args, **kwargs: None
    prints0 = prints if rank == 0 else lambda *args, **kwargs: None

    u.install_preemption_handler()
    sws.run(partial(main, rank=rank, local_rank=local_rank, world_size=world_size))
