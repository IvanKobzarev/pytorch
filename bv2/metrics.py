import io
import json
import os

# fmt: off
# Ignore a warning-spam from pydantic via wandb
import warnings

warnings.filterwarnings("ignore", message=r".*The '(repr|frozen)'.*`Field\(\)`.*")
# fmt: on

import wandb  # noqa: E402


def only_on_rank0(func):
    def wrapper(self, *args, **kwargs):
        if self.rank == 0:
            return func(self, *args, **kwargs)

    return wrapper


class WandbLogger:
    def __init__(self, config, rank, name, dir, entity="rigi", project="bv2", first_step=0, resume=None):
        self.step = first_step
        self.rank = rank
        if self.rank != 0:
            return

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
            # I tried using both {resume,fork}_from arguments for resuming, but they
            # aren't supported in FAIR's current WandB instance, only in wandb.io ones.
            # However, this simple approach reusing ID and our manual steps, seems to work:
            id=resume,
            resume="allow",
        )

        self.step_metrics = {}
        self.dir = dir
        self.fname = os.path.join(dir, "metrics.jsonl")

    @only_on_rank0
    def log(self, data):
        # Dump BytesIO objects to a file inside workdir, do not attempt to log in W&B.
        for filename, buf in ((k, v) for k, v in data.items() if isinstance(v, io.BytesIO)):
            filename = os.path.join(self.dir, filename)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            # Save twice, with and without step name.
            for fname in [filename, f"{filename}-{self.step:09d}"]:
                with open(fname, "wb") as f:
                    f.write(buf.getvalue())

        # Filter out BytesIO fields and log whats left to W&B.
        data = {k: v for k, v in data.items() if not isinstance(v, io.BytesIO)}
        self.wandb_run.log(data, step=self.step, commit=False)
        self.step_metrics.update(data)

    def end_step(self):
        if self.rank == 0:
            self.wandb_run.log({}, step=self.step, commit=True)
            self._append_flush_jsonl()
        self.step += 1

    @only_on_rank0
    def log_file(self, f, policy="now"):
        self.wandb_run.save(f, policy=policy)

    @only_on_rank0
    def finish(self):
        self.wandb_run.finish()

    @only_on_rank0
    def _append_flush_jsonl(self):
        self.step_metrics["step"] = self.step
        remove_invalid_json_(self.step_metrics)
        self.step_metrics = round_floats_(self.step_metrics, sig_figs=7)
        js = json.dumps(self.step_metrics)
        with open(self.fname, "a+") as f:
            f.write(js + "\n")
        self.step_metrics = {}

    @only_on_rank0
    def save_ckpt(self):
        return self.wandb_run.id


def remove_invalid_json_(measurements):
    for k, v in list(measurements.items()):
        if isinstance(v, wandb.sdk.data_types.table.Table):
            del measurements[k]
        elif isinstance(v, (int, float, str, list, tuple, dict)) or v is None:
            # In our Python json dialect, None becomes "null", which works.
            # We sometimes get it for example for gradnorms of unused params.
            continue
        else:
            raise ValueError(f"Not json'able and not ignored: {k} ({type(v)}): {v}")
    return measurements


def round_floats_(obj, sig_figs=7):
    if isinstance(obj, float):
        return float(f"{obj:.{sig_figs}g}")
    elif isinstance(obj, dict):
        return {k: round_floats_(v, sig_figs) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats_(item, sig_figs) for item in obj]
    return obj
