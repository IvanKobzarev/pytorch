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


def log_system_metrics(logger, gpu_index=0, prefix="sys/rank0", _base={}):
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
        f"{prefix}/proc_rss_gb": psutil.Process().memory_info().rss / 1e9,
    })
    logger.log({f"{prefix}/gc_count{i}": n for i, n in enumerate(gc.get_count())})

    # Disk I/O (cumulative since first call, not since boot)
    if disk := psutil.disk_io_counters():
        _base.setdefault("disk_r", disk.read_bytes)
        _base.setdefault("disk_w", disk.write_bytes)
        logger.log({
            f"{prefix}/disk_read_mb": (disk.read_bytes - _base["disk_r"]) / 1e6,
            f"{prefix}/disk_write_mb": (disk.write_bytes - _base["disk_w"]) / 1e6,
        })

    # VM pressure (cumulative since first call, not since boot)
    vmstat = {s[0]: int(s[1]) for line in open("/proc/vmstat") if len(s := line.split()) == 2}
    for k in ("allocstall_movable", "pgmajfault", "pgsteal_kswapd"):
        _base.setdefault(k, vmstat.get(k, 0))
        logger.log({f"{prefix}/{k}": vmstat.get(k, 0) - _base[k]})

    # Network I/O (cumulative since first call, not since boot)
    net = psutil.net_io_counters()
    _base.setdefault("net_s", net.bytes_sent)
    _base.setdefault("net_r", net.bytes_recv)
    logger.log({
        f"{prefix}/net_sent_mb": (net.bytes_sent - _base["net_s"]) / 1e6,
        f"{prefix}/net_recv_mb": (net.bytes_recv - _base["net_r"]) / 1e6,
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
