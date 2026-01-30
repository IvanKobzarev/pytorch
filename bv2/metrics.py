import functools
import gc
import io
import json
import os

import psutil


def only_on_rank0(func):
    def wrapper(self, *args, **kwargs):
        if self.rank == 0:
            return func(self, *args, **kwargs)

    return wrapper


class BytesWriter:
    """Writes ByteIOs on rank0 to files, pops them off the metrics so others never see them."""
    def __init__(self, rank, dir, first_step=0):
        self.step = first_step
        self.rank = rank
        self.dir = dir

    def log(self, data, flush=False):
        # NOTE: it's up to the metric creator to create only on rank0, or add rank to filename!
        for filename in [k for k, v in data.items() if isinstance(v, io.BytesIO)]:
            buf = data.pop(filename)  # Remove from data so downstream writers don't see it.
            filename = os.path.join(self.dir, filename)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            stepname = f"{filename}-{self.step:09d}"
            with open(stepname, "wb") as f:
                f.write(buf.getvalue())
            if os.path.islink(filename):
                os.unlink(filename)
            os.symlink(os.path.basename(stepname), filename)

    def end_step(self):
        self.step += 1

    def finish(self, training_done):
        pass

    def save_ckpt(self):
        return None


class WandbWriter:
    def __init__(self, config, rank, name, dir, first_step=0, entity="rigi", project="bv2", resume=None):
        self.step = first_step
        self.rank = rank
        if self.rank != 0:
            return

        # Ignore a warning-spam from pydantic via wandb
        import warnings  # noqa: E402
        warnings.filterwarnings("ignore", message=r".*The '(repr|frozen)'.*`Field\(\)`.*")
        import wandb  # noqa: E402

        wandb.login()

        config["env"] = {k: v for k, v in os.environ.items() if "key" not in k.lower()}
        config["PID"] = os.getpid()

        self.wandb_run = wandb.init(
            entity=entity,
            project=project,
            dir=dir,
            name=name,
            config={**config, "workdir": dir},
            settings=wandb.Settings(quiet=True),
            tags=[config.get("data_name", "N/A")],
            id=resume,
            resume="allow",
        )

        self._my_types = (wandb.sdk.data_types.utils.Media,)

    @only_on_rank0
    def log(self, data, flush=False):
        self.wandb_run.log(data, step=self.step, commit=False)
        # Remove wandb-specific types so downstream writers don't see them.
        for k in [k for k, v in data.items() if isinstance(v, self._my_types)]:
            data.pop(k)

    def end_step(self):
        if self.rank == 0:
            self.wandb_run.log({}, step=self.step, commit=True)
        self.step += 1

    @only_on_rank0
    def finish(self, training_done):
        self.wandb_run.finish()  # Wandb doesn't know "preempted state", so always finish.

    @only_on_rank0
    def save_ckpt(self):
        return self.wandb_run.id


class JsonlWriter:
    def __init__(self, rank, dir, first_step=0):
        self.step = first_step
        self.rank = rank
        self.fname = os.path.join(dir, "metrics.jsonl")
        self.step_metrics = {}

    @only_on_rank0
    def log(self, data, flush=False):
        self.step_metrics.update(data)

    def end_step(self):
        if self.rank == 0:
            self.step_metrics["step"] = self.step
            self._remove_invalid_json()
            self._round_floats()
            with open(self.fname, "a+") as f:
                f.write(json.dumps(self.step_metrics) + "\n")
            self.step_metrics = {}
        self.step += 1

    def finish(self, training_done):
        pass

    def save_ckpt(self):
        return None

    def _remove_invalid_json(self):
        for k, v in list(self.step_metrics.items()):
            if isinstance(v, (int, float, str, list, tuple, dict)) or v is None:
                continue
            else:
                raise ValueError(f"Not json'able and not ignored: {k} ({type(v)}): {v}")

    def _round_floats(self, sig_figs=7):
        self.step_metrics = self._round_floats_rec(self.step_metrics, sig_figs)

    def _round_floats_rec(self, obj, sig_figs):
        if isinstance(obj, float):
            return float(f"{obj:.{sig_figs}g}")
        elif isinstance(obj, dict):
            return {k: self._round_floats_rec(v, sig_figs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._round_floats_rec(item, sig_figs) for item in obj]
        return obj


class PlattliWriter:
    def __init__(self, rank, dir, first_step=0, hotsize=25):
        self.step = first_step
        self.rank = rank
        if self.rank != 0:
            return

        import plattli  # noqa: E402
        self.writer = plattli.CompactingWriter(dir, step=first_step, hotsize=hotsize)

    @only_on_rank0
    def log(self, data, flush=False):
        self.writer.write(flush=flush, **data)

    def end_step(self):
        if self.rank == 0:
            self.writer.end_step()
        self.step += 1

    @only_on_rank0
    def finish(self, training_done):
        # Only optimize plattli storage/zip when we're fully done, not when preempted.
        self.writer.finish(optimize=training_done, zip=training_done)

    def save_ckpt(self):
        return None


class MultiWriter:
    def __init__(self, **writers):
        self.writers = writers

    def log(self, data, flush=False):
        for w in self.writers.values():  # Writers may modify data in-place to remove handled entries.
            w.log(data, flush=flush)

    def end_step(self):
        for w in self.writers.values():
            w.end_step()

    def finish(self, training_done):
        for w in self.writers.values():
            w.finish(training_done)

    def save_ckpt(self):
        return {name: w.save_ckpt() for name, w in self.writers.items() if w is not None}


@functools.cache
def _get_gpu_handle(gpu_index=0):
    import pynvml
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(gpu_index), pynvml


def log_system_metrics(logger, gpu_index=0, prefix="sys"):
    """Log system metrics (CPU, RAM, GPU, disk, network) with given prefix."""
    # CPU
    load_1m, load_5m, load_15m = os.getloadavg()
    logger.log({
        f"{prefix}/cpu_percent": psutil.cpu_percent(),
        f"{prefix}/cpu_load_1m": load_1m,
        f"{prefix}/cpu_load_5m": load_5m,
        f"{prefix}/cpu_load_15m": load_15m,
    })

    # RAM
    mem = psutil.virtual_memory()
    logger.log({
        f"{prefix}/ram_used_gb": mem.used / 1e9,
        f"{prefix}/ram_percent": mem.percent,
    })
    logger.log({f"{prefix}/pygc_gen{i}": n for i, n in enumerate(gc.get_count())})

    # Disk I/O
    disk = psutil.disk_io_counters()
    logger.log({
        f"{prefix}/disk_read_gb": disk.read_bytes / 1e9,
        f"{prefix}/disk_write_gb": disk.write_bytes / 1e9,
    })

    # Network I/O
    net = psutil.net_io_counters()
    logger.log({
        f"{prefix}/net_sent_gb": net.bytes_sent / 1e9,
        f"{prefix}/net_recv_gb": net.bytes_recv / 1e9,
    })

    # GPU (skip if pynvml not available)
    try:
        handle, pynvml = _get_gpu_handle(gpu_index)
    except ImportError:
        return
    gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
    gpu_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    logger.log({
        f"{prefix}/gpu_power_w": gpu_power,
        f"{prefix}/gpu_power_percent": 100 * gpu_power / gpu_power_limit,
        f"{prefix}/gpu_temp_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        f"{prefix}/gpu_util_percent": gpu_util.gpu,
        f"{prefix}/gpu_mem_util_percent": gpu_util.memory,
        f"{prefix}/gpu_clock_sm_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        f"{prefix}/gpu_clock_mem_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
    })
