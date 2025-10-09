import json
import os

# fmt: off
# Ignore a warning-spam from pydantic via wandb
import warnings
warnings.filterwarnings("ignore", message=r".*The '(repr|frozen)'.*`Field\(\)`.*")
# fmt: on

import wandb


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
            config=config,
            settings=wandb.Settings(quiet=True),
            tags=[config.get("data_name", "N/A")],
            # Interestingly, the two below ({resume,fork}_from) aren't supported yet, but
            # this simple approach reusing ID and our manual steps, seems to work already.
            id=resume,
            resume="allow",
            # resume_from=f"{resume}?_step={self.step}" if resume else None,
            # fork_from=f"{resume}?_step={self.step}" if resume else None,
        )

        self.step_metrics = {}
        self.fname = os.path.join(dir, "metrics.jsonl")

    @only_on_rank0
    def log(self, data):
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
        js = json.dumps(self.step_metrics)
        with open(self.fname, "a+") as f:
            f.write(js + "\n")
        self.step_metrics = {}

    @only_on_rank0
    def save_ckpt(self):
        return self.wandb_run.id


def remove_invalid_json_(measurements):
    def _is_jsonable(x):
        try:
            json.dumps(x, allow_nan=True)
            return True
        except TypeError:
            return False

    for k, v in list(measurements.items()):
        if not _is_jsonable(v):
            del measurements[k]
    return measurements
