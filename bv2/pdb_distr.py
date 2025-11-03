# dist_pdb.py
import functools
import io
import pdb
import sys

import torch.distributed as dist
from IPython.core.debugger import Pdb as iPdb


class _Sink(io.TextIOBase):
    def __init__(self):
        self._b = []

    def write(self, s):
        self._b.append(s)
        return len(s)

    def take(self):
        s = "".join(self._b)
        self._b.clear()
        return s


def _bcast(obj, group):
    x = [obj] if dist.get_rank(group=group) == 0 else [None]
    dist.broadcast_object_list(x, src=0, group=group)
    return x[0]


def _gather(obj, group):
    out = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(out, obj, group=group)
    return out


class DistributedPdb(iPdb):
    def __init__(self, group=None, prompt="(pdb) "):
        super().__init__(stdin=None, stdout=_Sink())
        self.group = group or dist.group.WORLD
        self.rank = dist.get_rank(group=self.group)
        self.world = dist.get_world_size(group=self.group)
        self._sink = self.stdout
        self.prompt = prompt

    def _drain_and_print(self):
        chunks = _gather(self._sink.take(), self.group)
        if self.rank == 0:
            for r, s in enumerate(chunks):
                if s:
                    # Print in rank order as a single block to avoid interleaving.
                    head = f"[rank {r}] " if self.world > 1 else ""
                    print(head + s, end="", flush=True)

    def _cmd_from_master(self):
        if self.rank == 0:
            try:
                s = input(self.prompt)  # real terminal input on rank 0
            except EOFError:
                s = "quit"
            s = s.rstrip("\r\n")
            if not s.strip():
                s = self.lastcmd or ""  # bare Enter reliably repeats the last command
        else:
            s = None
        return _bcast(s, self.group)

    def interaction(self, frame, tb):
        self.setup(frame, tb)
        try:
            self.print_stack_entry(self.stack[self.curindex])
            self._drain_and_print()
            while True:
                line = self._cmd_from_master()
                # Redirect *per command* so `!` statements and pp output are captured too.
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = self._sink
                try:
                    stop = self.onecmd(line)  # returns True on 'c', 'q', etc.
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
                self._drain_and_print()
                if self.world > 1:
                    dist.barrier(group=self.group)
                if stop:
                    break
        finally:
            self.forget()


def _caller_frame(skip=0):
    f = sys._getframe().f_back
    for _ in range(skip):
        f = f.f_back
    return f


def dbreakpoint(group=None, skip=0):
    """Call this at the place you want to stop. All ranks sync here and obey rank-0 PDB."""
    frame = _caller_frame(skip=skip)
    if not dist.is_initialized():
        return pdb.set_trace(frame)
    if group is None:
        group = dist.group.WORLD
    dist.barrier(group=group)  # make sure all ranks arrive together
    DistributedPdb(group=group).set_trace(frame)


def enable_as_default(group=None):
    # Note we sep skip=1 to go one frame up and skip the lambda wrapper.
    sys.breakpointhook = functools.partial(dbreakpoint, group=group, skip=1)
