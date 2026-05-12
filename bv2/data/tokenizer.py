import os

os.environ.setdefault("TIKTOKEN_CACHE_DIR", "")  # Disable tiktoken fs-cache, it leads to mistakes!

from functools import cache

import regex
import tiktoken
from tiktoken.load import load_tiktoken_bpe

PATTERNS = {
    # borrowed from: https://www.internalfb.com/code/fbsource/[cd5f9614da86]/genai/xlformers/core/tokenizers/finetune.py?lines=281
    # bento notebook: https://fburl.com/anp/u3rlrljj.
    "o200k": (
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
        r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    # The GPT-4 regex, but split digits individually, and ignore english-specific 'nt etc.
    "gpt4-onedigit": r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    # Conservative code regex. Key choices are:
    # - identifiers keep '_' and trailing digits together
    # - common identifier prefixes like '.', '::', '->', '@', '#', '$' stay attached
    # - line-start indentation is isolated as its own chunk, even when it gets deep
    # - inline spaces attach to the following word/number/punctuation to avoid wrecking prose
    # - markup/css attribute names with '-', ':', '.' stay together when used like attrs
    # - PHP-specific forms keep namespace chains and property access intact
    # - numbers keep common code literal forms together, including separators and suffixes
    # - the fallback still catches leftover mid-line spaces/tabs such as double-spaces or trailing ws
    "code": (
        r"(?m)^[ \t]+|"
        r" ?[_\p{L}][_\p{L}\p{N}]*(?:[-:\.][_\p{L}\p{N}]+)+(?=\s*=)|"
        r" ?\$\{[$]?[_\p{L}][_\p{L}\p{N}]*\}|"
        r" ?\{\$[_\p{L}][_\p{L}\p{N}]*\}|"
        r" ?\\?[_\p{L}][_\p{L}\p{N}]*(?:\\[_\p{L}][_\p{L}\p{N}]*)+|"
        r" ?(?:\?->|->|::)\$[_\p{L}][_\p{L}\p{N}]*|"
        r" ?(?:[$@#]|\\|\.|::|->|\?->)?[_\p{L}][_\p{L}\p{N}]*|"
        r" ?0[xX][0-9A-Fa-f](?:[0-9A-Fa-f_']*[0-9A-Fa-f])?(?:\.[0-9A-Fa-f](?:[0-9A-Fa-f_']*[0-9A-Fa-f])?)?(?:[pP][+-]?[0-9](?:[0-9_']*[0-9])?)?[A-Za-z%]*|"
        r" ?0[bB][01](?:[01_']*[01])?[A-Za-z%]*|"
        r" ?0[oO][0-7](?:[0-7_']*[0-7])?[A-Za-z%]*|"
        r" ?[0-9](?:[0-9_']*[0-9])?(?:\.[0-9](?:[0-9_']*[0-9])?)?(?:[eE][+-]?[0-9](?:[0-9_']*[0-9])?)?[A-Za-z%]*|"
        r" ?\.[0-9](?:[0-9_']*[0-9])?(?:[eE][+-]?[0-9](?:[0-9_']*[0-9])?)?[A-Za-z%]*|"
        r"\r\n|[\r\n]|"
        r" ?(?:[^\s\p{L}\p{N}_$\\{]|[$](?![_\p{L}])|\\(?![_\p{L}])|\{(?!\$[_\p{L}]))+|"
        r"[ \t]+"
    ),
}


def get_pattern(regex_name="o200k"):
    return PATTERNS.get(regex_name, regex_name)


@cache
def get_pretok_regex(regex_name="o200k"):
    return regex.compile(get_pattern(regex_name))


def pretokenize(text, regex_name="o200k"):
    return get_pretok_regex(regex_name).findall(text)


class Tiktoken:
    def __init__(self, first_N=None, path=None, regex="o200k", extras=()):
        path = path or "/checkpoint/rigi/bv2/l4_200k_base.model"

        # "pretokenization" step done via regexp
        pattern = get_pattern(regex)
        self.pat_str = pattern

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
        for i, name in enumerate(extras):
            tok_id = base_dict_size + 3 + i
            self.special_tokens[f"<|{name}|>"] = tok_id
            setattr(self, name, tok_id)

        self.tokenizer = tiktoken.Encoding(
            name="l4_200k_base",
            pat_str=pattern,
            mergeable_ranks=tokens,
            special_tokens=self.special_tokens,
        )

        self.mergeable_ranks = tokens
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
def get_tiktoken(first_N=None, path="/checkpoint/rigi/bv2/l4_200k_base.model", regex="o200k", extras=()):
    return Tiktoken(first_N=first_N, path=path, regex=regex, extras=tuple(extras))
