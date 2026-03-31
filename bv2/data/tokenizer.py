from functools import cache

import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Tiktoken:
    def __init__(self, first_N=None, path=None, regex="o200k"):
        path = path or "/checkpoint/rigi/bv2/l4_200k_base.model"

        # "pretokenization" step done via regexp
        pattern = {
            # borrowed from: https://www.internalfb.com/code/fbsource/[cd5f9614da86]/genai/xlformers/core/tokenizers/finetune.py?lines=281
            # bento notebook: https://fburl.com/anp/u3rlrljj.
            "o200k": (
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
                r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            ),
            # The GPT-4 regex, but split digits individually, and ignore english-specific 'nt etc.
            "gpt4-onedigit": r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        }[regex]

        # load actual tokens
        tokens = load_tiktoken_bpe(path)

        if first_N is not None:
            tokens = {k: val for k, val in tokens.items() if val < first_N}

        base_dict_size = len(tokens)
        self.bos = base_dict_size
        self.eos = base_dict_size + 1
        self.sep = base_dict_size + 2
        self.special_tokens = {
            "<|bos|>": self.bos,
            "<|eos|>": self.eos,
            "<|sep|>": self.sep,
        }

        self.tokenizer = tiktoken.Encoding(
            name="l4_200k_base",
            pat_str=pattern,
            mergeable_ranks=tokens,
            special_tokens=self.special_tokens,
        )

        self.n_vocab = self.tokenizer.n_vocab

    def encode(self, text):
        # This setting means all special tokens are encoded as plain text and not
        # treated specially. They also do not raise an exception - we don't want
        # to kill jobs far into training. Maybe we should print/log a warning though?
        # To use special tokens, just use tt.special_tokens["<|eos|>"] or tt.eos.
        return self.tokenizer.encode(text, disallowed_special=())

    def decode(self, text):
        return self.tokenizer.decode(text)


@cache
def get_tiktoken(first_N=None, path="/checkpoint/rigi/bv2/l4_200k_base.model", regex="o200k"):
    return Tiktoken(first_N=first_N, path=path, regex=regex)
