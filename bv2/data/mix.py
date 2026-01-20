from copy import deepcopy

import bv2
import bv2.utils as u


class Dataset:
    def __init__(self, mix, *, seed=(), tokenizer=None, common={}, **datasets):
        # 1. Normalize the weights so we can simply sample from the mix.
        # 2. Sort the dataset names alphabetically such that the order of sampling each
        #    does not depend on the order in which they were defined, but only on their prob.
        self.mix = {name: mix[name] / sum(mix.values()) for name in sorted(mix)}
        self.mixseed = (*seed, "mixer")

        self.datasets = {name: bv2.simple_data.from_config({
            "seed": (*seed, name),
            **common,
            **datasets[name],
            "tokenizer": tokenizer,  # Force this to be the same, for now.
        }) for name in self.mix}

    def make_example(self, which, exid):
        ex = self.datasets[which].make_example(**exid)
        if "src" not in ex:  # TODO: maybe just overwrite altogether after moving finevision to this?
            ex["src"] = which
        return ex

    def make_exids(self, seed, start_offset=0, start_states={}, rank=0, **kw):
        # Basically, keep a generator for each dataset, and switch between them.
        # However, we also need to return the whole combined state_after for each
        # of them every single time, since we need to checkpoint them all!
        def make_generator(name):
            # Mix name into seed, so that using the same dataset twice (eg diff settings like qfmt)
            # doesn't result in walking the two in lock-step.
            return self.datasets[name].make_exids(
                seed=(seed, name), rank=rank, **kw, **start_states.get(name, {}))
        generators = {n: make_generator(n) for n in self.mix}
        states = deepcopy(start_states)

        for step in u.count(start_offset):
            which = u.rng(seed, rank, step).choice(list(self.mix), p=list(self.mix.values())).item()
            exid, states[which] = next(generators[which])
            # We need deepcopy of states to avoid them being overwritten by next example fetch
            # before the saving code was able to save them.
            yield {"which": which, "exid": exid}, {"start_offset": step + 1, "start_states": deepcopy(states)}

        # NOTE: We specifically don't shard the mixture components by rank, since that
        # would cause severe imbalances in terms of tokens if we use native resolution
        # and different components have very different resolutions. If each rank gets
        # samples from each component, then this is not an issue.

    @property
    def tt(self):
        # They all have forcibly the same tokenizer, so just return any:
        return next(iter(self.datasets.values())).tt

    def vocab_size(self):
        return self.tt.n_vocab

    # TODO: This is not generically possible here, because `data` is already packed and dpacked
    #       and it would be a lot of work to slice it into individual sub datasets etc.
    #       We should probably rather add tools to look at the raw batch we always dump already.
    # def vis_data_wandb(self, data):
