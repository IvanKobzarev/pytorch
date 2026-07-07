"""
pip install -U -r bv2/requirements-gpu-stable.txt (or -nightly)
bv2/tools/launch_local bv2.train
"""

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
from time import perf_counter, time

import numpy as np
import rich
import sws
import torch
import torch.distributed as distr
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcpsd
import zstandard as zstd
from torch.profiler import ProfilerActivity, profile, record_function

import bv2.graph_trainer_utils.adapter as gt_adapter
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

# We want to be intentional about recompiles and shape dynamism:
torch._dynamo.config.recompile_limit = 1  # Misleading name: 1 means 0 recompiles allowed.
torch._dynamo.config.fail_on_recompile_limit_hit = True
torch._dynamo.config.accumulated_recompile_limit = 10_000_000  # Basically inf.


# This section configures pytorch to be fully deterministic.
# No noticeable perf impact for our toy model so far.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_deterministic_debug_mode("error")  # raises error on non-determinism

# Use AutoHeuristics for pad_mm to automatically pad matmuls for better perf.
# Keep force_shape_pad disabled by default; forced padding regresses peak memory.
DEFAULT_INDUCTOR_CONFIG = {
    "shape_padding": True,
    "force_shape_pad": False,
    "autoheuristic_use.pad_mm": True,
}

try:
    torch._inductor.config.autoheuristic_use.pad_mm = True
except Exception:
    pass


def _dict_config(value):
    if value is None:
        return {}
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        obj = json.loads(value)
        if not isinstance(obj, dict):
            raise TypeError(f"Expected a JSON object config, got {type(obj).__name__}")
        return obj
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Expected a dict config, got {type(value).__name__}")


def _env_truthy(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _has_inductor_config_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_dict_config(value))
    if isinstance(value, dict):
        return bool(value)
    return True


def _inductor_config(c):
    config_value = c.get("inductor_configs", None)
    env_value = os.environ.get("RIGI_INDUCTOR_CONFIGS_JSON", None)
    override_enabled = c.get("allow_inductor_config_override", False) or _env_truthy(
        "RIGI_ENABLE_INDUCTOR_CONFIG_OVERRIDE"
    )
    cfg = dict(DEFAULT_INDUCTOR_CONFIG)
    if not override_enabled:
        if _has_inductor_config_value(config_value) or _has_inductor_config_value(env_value):
            raise RuntimeError(
                "Inductor config overrides require "
                "RIGI_ENABLE_INDUCTOR_CONFIG_OVERRIDE=1 or "
                "allow_inductor_config_override=True"
            )
        return cfg

    cfg.update(_dict_config(c.get("inductor_configs", None)))
    cfg.update(_dict_config(env_value))
    return cfg


def main(c, rank, local_rank, world_size):
    inductor_config = _inductor_config(c)
    with torch._inductor.config.patch(inductor_config):
        return _main(c, rank, local_rank, world_size)


def _main(c, rank, local_rank, world_size):  # noqa: C901
    prints0(f"Running with arguments:\n{c}")
    trainsched = u.schedule(c, prefix="n", name="training schedule")
    is_short_run = trainsched["unit"] == "nsteps" and trainsched["spec"] < 50
    prints0(f"Training target: {u.BLUE}{trainsched['unit']}={trainsched['spec']}{u.RESET}")

    # start from the beginning to track every gpu memory allocation
    # otherwise we lost cpp tracestack for model initialization
    if not is_short_run:
        torch.cuda.memory._record_memory_history(max_entries=10000000)

    # In theory we only need `init_device_mesh`, but in practice, we need this
    # whole verbose `init_process_group` or else the `barrier` will throw a warning.
    # Also, because optimal assignment of device to process depends on the launch environment,
    # we force the launcher to assign a single GPU for each process, and use that:
    assert (cvd := os.environ.get("CUDA_VISIBLE_DEVICES")) and "," not in cvd, (
        f"This codebase assumes you make a single GPU visible per process. {os.environ.get("CUDA_VISIBLE_DEVICES")=}")
    device = torch.device("cuda:0")
    prints(f"{os.environ.get("CUDA_VISIBLE_DEVICES")=} ; {device=}")

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
    workdir = "workdirs-dbg" if is_short_run else "workdirs"
    workdir = pjoin(c.get("workdir_base", "/checkpoint/rigi/bv2/"), workdir, xid, name)
    prints0(f"Workdir: {u.BLUE}{workdir}{u.RESET}")

    if rank == 0:
        os.makedirs(workdir, exist_ok=True)
        config_path = pjoin(workdir, "config.json")
        u.nfs_safe_overwrite(config_path, c.to_flat_json(indent=0))
    else:
        config_path = None
    write_exit_status = u.install_exit_handler(config_path)

    u.install_torch_trace(rank, workdir)

    # Import and get data source. We need it early on to know vocab size.
    ds = bv2.simple_data.from_config({'seed': (c.seed, "dataset"), **c.data.to_dict()})

    # Create the model on "meta" device, this avoids materializing param buffers.
    with torch.device("meta"):
        model = SimpleTransformer(
            **{"vocab": ds.vocab_size(), **c.model.to_dict()},
        )

    prints0(f"Mesh: {u.BLUE}{mesh}{u.RESET}")
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
        min_bytes=c.get("fsdp.min_bytes", 32 * 1024 * 1024),
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


    decay_patterns = c.get("decay_patterns", [r".*\.gamma", r".*\.gamma_head"])
    if rank == 0:
        summary_table(model, stats=c.get("param_stats", False), param_mode=get_muon_param_mode, decay_patterns=decay_patterns)

    param_groups = defaultdict(list)
    for n, p in model.named_parameters():
        param_groups[get_muon_param_mode(n)].append(p)
    params = [{"params": ps, "mode": mode} for mode, ps in param_groups.items()]

    optim = Muon(params, lr_adam=torch.tensor(0.0), lr_muon=torch.tensor(0.0), **muon_args)
    optim.init_state() # we init state to avoid recompiles
    decay_params = [p for n, p in model.named_parameters() if is_decay(n, decay_patterns)]

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

    _run_fwd_bwd_step = _fwd_and_bwd_step
    if c.use_graph_trainer:
        decay_param_names = [n for n, _ in model.named_parameters() if is_decay(n, decay_patterns)]
        _run_fwd_bwd_step = gt_adapter.make_train_step_dispatcher(
            model, optim, decay_param_names,
        )

    # Potentially resume/fork from a checkpoint, if not, init stuff.
    progress = u.TrainingProgress()
    past_proctime, resume_data = 0, {}

    # Checkpoint loading priority: resume > fork > init
    ckpt_path = c.get("fork") or c.get("init")
    if os.path.exists(pjoin(workdir, "ckpt-latest")):  # := is_resuming
        ckpt_path = pjoin(workdir, "ckpt-latest")

    if ckpt_path:
        if extras := load_ckpt(ckpt_path, model, optim, weights_only=bool(c.get("init"))):
            resume_data, past_proctime = extras["data"], extras.get("proctime", 0)
            progress = u.TrainingProgress(
                extras["step"], extras["examples_seen"], extras["data_tokens_seen"],
                extras["model_tokens_seen"], extras["loss_tokens_seen"])

    first_step = progress["nsteps"]
    mw = bv2.metrics.MultiWriter(
        bytes=bv2.metrics.BytesWriter(rank, workdir, first_step),
        plattli=bv2.metrics.PlattliWriter(rank, workdir, first_step),
    )
    if rank == 0:  # Log once more after ckpt resume.
        summary_table(model, stats=c.get("param_stats", False), param_mode=get_muon_param_mode, decay_patterns=decay_patterns)
    prints0(model)

    peak_mems, model_times, step_times = [], [], []
    t0 = t_step_start = t_prev_step_end = perf_counter()
    prof = not is_short_run and first_step == 0 and profile(
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

        fn = torch.compile(u.clone_function(_fwd, name_suffix=eval_key), dynamic=False, fullgraph=True)
        fn = u.suppress_warnings("`isinstance(treespec, LeafSpec)` is deprecated", FutureWarning)(fn)
        fn = u.suppress_warnings("`isinstance(treespec, TreeSpec)` is deprecated", FutureWarning)(fn)
        fn = u.suppress_warnings("remat_using_tags_for_fwd_loss_bwd_graph: Graph has recomputable ops but no backward region.", UserWarning)(fn)  # Fixed in https://github.com/pytorch/pytorch/pull/173528
        fn = record_function(f"eval_fwd_{eval_key}")(fn)
        return fn

    # Eval loop is reused, so we wrap it as a function
    def run_evals(step, progress, last_step=False):
        """Run evaluations for a given step if conditions are met."""
        teval0, ran_eval = perf_counter(), False
        for ev_name in c.get("evals") or {}:
            if (ev := c.evals[ev_name]) is None or not ev.type:
                continue
            schedule = u.schedule(ev, prefix="at_", name=f"Evaluator {ev_name} schedule", delay_first=True)
            should_run = progress.crossed(schedule)
            if u.about_to_get_killed() or not (should_run or last_step):  # Always run on last step.
                continue
            tev0, ran_eval = perf_counter(), True
            prints0(f"Running evaluator {u.BLUE}{ev_name}{u.RESET}", end="", flush=True)
            em = import_module(f"bv2.eval.{ev.type}")
            ds_ev = bv2.simple_data.from_config(ev.data.to_dict())
            args = {k: v for k, v in ev.to_dict().items() if k not in {"type", "data", "steps"} and not k.startswith("at_")}
            with torch.inference_mode():
                if results := em.run(get_fwd(ev_name), ds_ev, **args, rank=rank, world_size=world_size, device=device):
                    mw.log({f"{ev_name}/{k}": v for k, v in results.items()}, flush=True)
                    print0("")  # End the line we did not end above.
                    for k, v in results.items():
                        prints0(f"Eval results: {ev_name}/{k}: {v}")
            u.global_gpu_barrier(device)  # For accurate timing and avoiding u.about_to_get_killed-related divergence.
            mw.log({f"chrono/evals/{ev_name}": perf_counter() - tev0})
        if ran_eval:
            mw.log({"chrono/evaltime": perf_counter() - teval0})

    ckpt_schedule = u.schedule(c, prefix="ckpt_at_", name="checkpoint schedule", default=1000, required=False, none_disables=True)
    ckpt_keep_schedule = u.schedule(c, prefix="ckpt_keep_at_", name="checkpoint keep schedule", required=False, none_disables=True)

    # Print status of GIL late, because any lazy import can flip it back on.
    if hasattr(sys, "_is_gil_enabled"):
        if sys._is_gil_enabled():
            prints0(f"{u.BLUE}GIL{u.RESET} is {u.RED}ENABLED{u.RESET} (possibly re-enabled by an extension module)")
        else:
            prints0(f"{u.BLUE}GIL{u.RESET} is {u.GREEN}DISABLED{u.RESET} (free-threading is active)")
    else:
        prints0(f"{u.BLUE}Not a free-threaded build{u.RESET}")

    per_src_examples_seen, per_src_tokens_seen = Counter(), Counter()
    step = first_step
    train_iter = bv2.simple_data.data_iter(
        ds, seed=(c.seed, "data_iter"),
        device=device, rank=rank, world_size=world_size, resume=resume_data,
        **c.iter.to_dict(),
    )
    while not progress.reached(trainsched):
        if (data := next(train_iter, None)) is None:
            break

        torch.cuda.reset_peak_memory_stats()
        u.global_gpu_barrier(device)  # For accurate global datawait timing.
        t_prev_step_start, t_step_start = t_step_start, perf_counter()

        if prof and (step - first_step) == 50:
            torch.cuda.cudart().cudaProfilerStart()
            prof.start()

        num_data_tokens, num_examples, num_model_tokens, num_loss_tokens = u.all_reduce_scalars(
            sum(data["ndatatoks"]), len(data["ndatatoks"]), sum(data["ntok"]), (data["lowe"] > 0).sum())

        progress.advance(
            examples=num_examples, data_tokens=num_data_tokens,
            model_tokens=num_model_tokens, loss_tokens=num_loss_tokens)
        training_done = progress.crossed(trainsched)

        mw.log({"chrono/examples_seen": progress["nexamples"]})
        mw.log({"chrono/data_tokens_seen": progress["ndatatokens"]})
        mw.log({"chrono/model_tokens_seen": progress["nmodeltokens"]})
        mw.log({"chrono/num_data_tokens": num_data_tokens})
        mw.log({"chrono/num_model_tokens": num_model_tokens})
        mw.log({"chrono/num_examples": num_examples})

        # To avoid wasteful lr=0 steps, warmup uses after and cooldown uses before.
        sched = global_schedule(
            warmup_progress=progress[trainsched["unit"]],
            cooldown_progress=progress.before[trainsched["unit"]],
            total=trainsched["spec"],
            warmup=c.get(f"warmup_{trainsched['unit']}") or 1,
            cooldown=c.get(f"cooldown_{trainsched['unit']}") or 0,
        )

        set_lr_(optim, sched * c.lr_adam, "lr_adam")
        set_lr_(optim, sched * c.lr_muon, "lr_muon")
        mw.log({"chrono/lr_adam": sched * c.lr_adam, "chrono/lr_muon": sched * c.lr_muon, "chrono/sched": sched})

        model.zero_grad(set_to_none=True)

        # Need to log param norms at this step before the update
        if step < 50 or step % 10 == 0:  # Interesting frequently early, sparsely later.
            mw.log({f"pnorm/{k}": v for k, v in global_norms(model.named_parameters()).items()})

        t_before_model = perf_counter()  # Let's not sync/barrier, FSDP does that anyways.
        local_loss, extras = _run_fwd_bwd_step(
            torch.tensor(c.wd * sched) if c.wd else None,
            data["toki"],
            data["toko"],
            data["flex_masks"],
            data["lowe"],
            data["iseq"],
        )

        u.global_gpu_barrier(device)  # For accurate "global" timings
        mw.log({"chrono/modeltime": (model_time := perf_counter() - t_before_model)})
        mw.log({"chrono/steptime": (step_time := t_step_start - t_prev_step_start)})
        mw.log({"chrono/proctime": perf_counter() - t0 + past_proctime})
        mw.log({"chrono/axltime": np.float64(time() - 1751320800.0)})
        mw.log({"chrono/datawait": t_step_start - t_prev_step_end})
        mw.log({"sys/gpu_peak_mem_gb": (peak_mem := torch.cuda.max_memory_allocated() / 1024**3)})
        if step % 10 == 0 and rank == 0:
            bv2.metrics.log_system_metrics(mw, gpu_index=0, prefix="sys")
        if is_short_run:
            model_times.append(model_time)
            step_times.append(step_time)
            peak_mems.append(peak_mem * 1024)  # MiB

        global_loss, global_pplx, global_ncorrect = u.all_reduce_scalars(
            local_loss, extras["pplx"], extras["ncorrect"])

        prints0(f"step {step}: loss {global_loss:.8f}")

        mw.log({"chrono/num_loss_tokens": num_loss_tokens})
        mw.log({"chrono/loss_tokens_seen": progress["nlosstokens"]})

        mw.log({"train/pplx": global_pplx / num_examples})
        mw.log({"train/loss": global_loss})  # loss used for bwd, so already normalized by a global weight
        mw.log({"train/tacc": global_ncorrect / max(num_loss_tokens, 1)})
        max_logits = u.all_reduce_scalars(*(blk["attn"]["max_logit"] for blk in extras["blk"].values()), op=distr.ReduceOp.MAX)
        mw.log({f"attn_max_logit/blk{i}": max_logits[i] for i in extras["blk"]})

        # For dataset mixtures, collect and report per-component stats and loss.
        if "src" in data:
            # Sync source names across ranks (cheap: just ~200 strings, not tensors)
            local_srcs = sorted(set(data["src"]))
            all_srcs = sorted({s for ss in u.all_gather_object(local_srcs) for s in ss})
            s2i = {s: i for i, s in enumerate(all_srcs)}

            # Map each real token to its source index via iseq.
            src_per_ex = torch.tensor([s2i[s] for s in data["src"]], device=device)
            valid_toks = data["iseq"] >= 0
            src_per_tok = src_per_ex[data["iseq"][valid_toks]]

            # Vectorized per-source stats on GPU, then one all_reduce
            stats = torch.zeros(3, len(all_srcs), device=device)
            stats[0].scatter_add_(0, src_per_ex, torch.ones(len(data["src"]), device=device))
            stats[1].scatter_add_(0, src_per_tok, torch.ones_like(src_per_tok, dtype=stats.dtype))
            stats[2].scatter_add_(0, src_per_tok, extras["tok_losses"][valid_toks].float())
            distr.all_reduce(stats)

            for i, src in enumerate(all_srcs):
                per_src_examples_seen[src] += int(stats[0, i])
                per_src_tokens_seen[src] += int(stats[1, i])
            mw.log({f"mix_examples_seen/{s}": c for s, c in per_src_examples_seen.items()})
            mw.log({f"mix_tokens_seen/{s}": v for s, v in per_src_tokens_seen.items()})

            for i, src in enumerate(all_srcs):
                if stats[0, i] > 0:
                    mw.log({f"mix_pplx/{src}": (stats[2, i] / stats[0, i] / np.log(2)).item()})

        # And grad-norms are for this step, but we only get them after the update ran, i.e. here.
        if not c.use_graph_trainer and (step < 50 or step % 10 == 0):  # Interesting frequently early, sparsely later.
            mw.log({f"gnorm/{k}": v for k, v in global_norms((n, p.grad) for n, p in model.named_parameters()).items()})

        # After the update is done, we are at the step+1
        mw.log({"chrono/percent": min(1.0, progress[trainsched["unit"]] / trainsched["spec"])})
        step = progress["nsteps"]

        # Checkpoint, but note this is *after* `step`'s update, so +1.
        maybe_save_ckpt(
            step, ckpt_schedule, ckpt_keep_schedule, progress, force=training_done,
            model=model, optim=optim, workdir=workdir, extras={
                "data": data["state_after"],  # NOTE: This differs per process(!)
                "data_tokens_seen": progress["ndatatokens"],
                "model_tokens_seen": progress["nmodeltokens"],
                "loss_tokens_seen": progress["nlosstokens"],
                "examples_seen": progress["nexamples"],
                "proctime": perf_counter() - t0 + past_proctime,
                "metrics": mw.save_ckpt(),
                "jid": c.get("jid", "n/a"),  # Just for future archeologs.
            })  # fmt: skip

        if u.about_to_get_killed():  # We checkpointed, yay, quick, byebye.
            mw.end_step()
            break

        if prof and (step - first_step) == 2:
            # dumping first 3 iterations from init are enough to include optim states.
            # Otherwise the .pkl becomes too big and freezes chrome.
            # Drag .pkl file to https://docs.pytorch.org/memory_viz
            torch.cuda.memory._dump_snapshot(pjoin(workdir, f"prof_memsnap_s{step}_r{rank}.pkl"))  # fmt: skip
            torch.cuda.memory._record_memory_history(enabled=None)
        if prof and (step - first_step) == 54:  # Open in about://tracing or ui.perfetto.dev
            torch.cuda.cudart().cudaProfilerStop()
            prof.stop()  # TODO: speedup gz
            prof.export_chrome_trace(pjoin(workdir, f"prof_trace_s{step}_r{rank}.json.gz"))
            prof.export_stacks(pjoin(workdir, f"prof_stacks_cpu_s{step}_r{rank}.txt"))

        run_evals(step, progress, last_step=training_done)

        # visualize input tokens
        if not is_short_run and step == 8:
            with zstd.open(pjoin(workdir, f"data_r{rank}.pt.zst"), "wb") as f:
                torch.save({k: v for k, v in data.items() if k != "flex_masks"}, f)

        u.global_gpu_barrier(device)  # Sync to get accurate datawait timing.
        t_prev_step_end = perf_counter()
        mw.end_step()

        if training_done:
            break

    if is_short_run:
        u.printR(f"Peak mems (med: {np.median(peak_mems):.1f}MiB): {' '.join(f'{t:.0f}' for t in peak_mems)}")  # fmt: skip
        u.printR(f"Model times (med: {np.median(model_times)*1000:.1f}ms): {' '.join(f'{t*1000:.0f}' for t in model_times)}")  # fmt: skip
        u.printR(f"Step times (med: {np.median(step_times)*1000:.1f}ms): {' '.join(f'{t*1000:.0f}' for t in step_times)}")  # fmt: skip

    if u.about_to_get_killed():
        mw.finish(training_done=False)
        u.printR(f"Finished {perf_counter() - u.about_to_get_killed()}s after getting the shutdown signal!")
    else:
        write_exit_status("done")
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

def global_schedule(*, warmup_progress, cooldown_progress, total, warmup=1, cooldown=0):
    """Implements constant schedule with warmup and cooldown. Tested."""
    return min(1.0, warmup_progress / warmup, 1.0 if cooldown == 0 else (total - cooldown_progress) / cooldown)


def set_lr_(optimizer, lr, name):
    lr = torch.tensor(lr)  # For torch.compile, else it's a compile-time constant!
    for param_group in optimizer.param_groups:
        param_group[name] = lr


def print_stamped(s, rank, **kw):
    t = datetime.now().time().isoformat(timespec="milliseconds")
    print(f"[{rank} {t}] {s}", **kw)


def global_norms(named_tensors):
    """Batched global L2 norms via single all-reduce. Works with DTensors."""
    items = [(n, x) for n, x in named_tensors if x is not None]
    loc = lambda x: x.to_local() if hasattr(x, 'to_local') else x
    sums = u.all_reduce_scalars(*[loc(x).float().square().sum() for _, x in items])
    ws = distr.get_world_size()
    sh = lambda x: hasattr(x, 'placements') and any(pl.is_shard() for pl in x.placements)
    return {n: (s if sh(x) else s / ws) ** 0.5 for (n, x), s in zip(items, sums)}


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


def is_decay(name, patterns):
    return any(re.fullmatch(d, name) for d in patterns)


def summary_table(model, stats=True, param_mode=None, decay_patterns=()):
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
        cols += [str(is_decay(name, decay_patterns))]
        cols += [mode]
        if stats:
            cols += [global_reduce(x, "mean"), global_reduce(x, "std")]
        tbl.add_row(*cols)

    tbl.columns[0].footer = f"Total: {swissnum(total_num)}"
    tbl.columns[1].footer = f"({total_bytes/1024/1024:.0f}MiB)"
    tbl.columns[2].footer = f"Local: {local_bytes/1024/1024:.0f}MiB"
    rich.get_console().print(tbl)  # get_console so we do use the reconfigure from main.


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
def maybe_save_ckpt(step, save_schedule, keep_schedule, progress, model, optim, workdir, extras=None, force=False):
    if not save_schedule or workdir is None:
        return

    should_save = progress.crossed(save_schedule)
    if not (force or u.about_to_get_killed() or should_save):
        return

    path = pjoin(workdir, f"ckpt-{step:06d}")
    prints0(f"Checkpointing to {path}")

    # This is funny, but the `async_save` below interacts with `distr` in some way such
    # that if we do the `gather_object` after it, it would deadlock. Unless we barrier,
    # which defeats the point of async. So, gather_object first.
    all_extras = u.gather_object_to(rank=0, obj={
        "step": step,
        "progress_before": progress.before,
        "progress_after": progress.after,
        **(extras or {}),
    })

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
            if not checkpoint_matches_schedule(prev_ckpt, keep_schedule):
                shutil.rmtree(prev_ckpt)


def checkpoint_matches_schedule(path, schedule):
    if not schedule:
        return False
    with open(pjoin(path, "extras.json"), "r") as f:
        extras = json.load(f)[0]
    return u.TrainingProgress.from_dicts(extras["progress_before"], extras["progress_after"]).crossed(schedule)


def load_ckpt(path, model, optim, weights_only=False):
    if not os.path.exists(path):
        raise ValueError(f"Checkpoint path was not found: {path}")

    prints0(f"Resuming from {path}")

    # Workaround for Python 3.13 + PyTorch DCP bug:
    # https://fb.workplace.com/groups/319878845696681/permalink/1657362868614932/
    _orig_wrap = dcp.utils._wrap_exception
    def _wrap_exception_fixed(exc):
        result = _orig_wrap(exc)
        for frame in result[1]:  # result is (exc, StackSummary)
            if hasattr(frame, '_code'):
                object.__setattr__(frame, '_code', None)
        return result
    dcp.utils._wrap_exception = _wrap_exception_fixed
    # Workaround end.

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

    c.maxtok = 8 * 4096

    c.data.name = "random_nouns"
    c.data.min_nouns = 128
    c.data.max_nouns = 256
    c.iter.maxtok = lambda: c.maxtok

    c.nsteps = 16
    c.warmup_nsteps = 3
    c.lr_adam = 1e-3
    c.lr_muon = 1e-3
    c.wd = lambda: c.lr_adam * 0.01

    c.muon.param_modes = {"muon_h": [r".*mlp.l[12].weight", r".*att.[qkvo].weight", r".*img_emb.proj.weight", r".*txt_unemb.head.weight"],
                          "embedding": [r".*txt_emb.emb.weight"],
                          "adam": [r".*"]}

    c.model.dim = 4096
    c.model.depth = 4
    c.model.txt_unemb.chunksz = 4096
    c.use_graph_trainer = False

    c.evals.pplx_val.type = "pplx"
    c.evals.pplx_val.at_steps = 10
    c.evals.pplx_val.data.name = lambda: c.data.name
    c.evals.pplx_val.data.seed = 31337  # "val split" content
    c.evals.pplx_val.data.n = 150  # "val split" size
    c.evals.pplx_val.data.tokenizer = lambda: getattr(c.data, "tokenizer", None)
    c.evals.pplx_val.iter.maxtok = lambda: c.maxtok

    return c


if __name__ == "__main__":
    if "RANK" in os.environ:  # Launched via bv2/tools/launch_local or torchrun
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

    # Get a stacktrace on crash (abort/segfault/...)
    import faulthandler
    faulthandler.enable()

    # Add rank to cache dir to avoid race-condition on the lock.
    # It means ranks don't share the compile cache, but it also
    # means we don't get the following startup crash randomly anymore:
    # torch._inductor.exc.InductorError: Timeout: The file lock [...] could not be acquired.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_{getuser()}_rank{local_rank}"

    # Don't squeeze tables! Can't protect this by isatty, because slurm-out is always a tty :(
    rich.reconfigure(width=500)

    u.filter_stderr("libpng warning: iCCP: ")

    # We need to "warmup" the einops backend cache; if we don't, then einops
    # has a multi-threading race-condition that makes it fail in our input pipeline.
    import einops
    einops._backends.get_backend(np.empty((1,1), np.uint8))
    einops.rearrange(np.empty((1,1), np.uint8), 'a b -> a b')

    prints = partial(print_stamped, rank=rank)
    print0 = print if rank == 0 else lambda *args, **kwargs: None
    prints0 = prints if rank == 0 else lambda *args, **kwargs: None

    prints0(f"{u.BLUE}Python{u.RESET} {sys.version.split()[0]}")
    prints0(f"{u.BLUE}PyTorch{u.RESET} {torch.__version__}")

    sws.run(partial(main, rank=rank, local_rank=local_rank, world_size=world_size))
