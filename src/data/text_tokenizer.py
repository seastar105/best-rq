import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np


class MLSuperbTokenizer:
    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]],
        non_linguistic_symbols: Optional[Union[Path, str, Iterable[str]]] = None,
        remove_non_linguistic_symbols: bool = False,
        nonsplit_symbols: Optional[Iterable[str]] = None,
        unk_symbol: str = "<unk>",
        space_symbol: str = "<space>",
    ):
        self.tokenizer = EspnetCharTokenizer(
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
            nonsplit_symbols=nonsplit_symbols,
        )
        self.token_id_converter = TokenIDConverter(
            token_list=token_list,
            unk_symbol=unk_symbol,
        )

    def encode(self, text: str) -> Dict[str, List[int]]:
        tokens = self.tokenizer.text2tokens(text)
        token_ids = self.token_id_converter.tokens2ids(tokens)
        return {"input_ids": token_ids}

    def decode(self, token_ids: Union[np.ndarray, Iterable[int]]) -> str:
        tokens = self.token_id_converter.ids2tokens(token_ids)
        text = self.tokenizer.tokens2text(tokens)
        return text

    def __call__(self, text: str) -> Dict[str, List[int]]:
        return self.encode(text)

    def __len__(self) -> int:
        return self.token_id_converter.get_num_vocabulary_size()


class EspnetCharTokenizer:
    def __init__(
        self,
        non_linguistic_symbols: Optional[Union[Path, str, Iterable[str]]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
        nonsplit_symbols: Optional[Iterable[str]] = None,
    ):
        self.space_symbol = space_symbol
        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            try:
                with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                    self.non_linguistic_symbols = {line.rstrip() for line in f}
            except FileNotFoundError:
                warnings.warn(f"{non_linguistic_symbols} doesn't exist.")
                self.non_linguistic_symbols = set()
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols
        self.nonsplit_symbols = set() if nonsplit_symbols is None else {sym.split(":")[0] for sym in nonsplit_symbols}

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'space_symbol="{self.space_symbol}"'
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            f'nonsplit_symbols="{self.nonsplit_symbols}"'
            f")"
        )

    def text2tokens(self, line: str) -> List[str]:
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols.union(self.nonsplit_symbols):
                if line.startswith(w):
                    if w in self.nonsplit_symbols or not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(w)])
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                if t == " ":
                    t = self.space_symbol
                tokens.append(t)
                line = line[1:]
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        tokens = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(tokens)


class TokenIDConverter:
    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]],
        unk_symbol: str = "<unk>",
    ):
        if isinstance(token_list, (Path, str)):
            token_list = Path(token_list)
            self.token_list_repr = str(token_list)
            self.token_list: List[str] = []

            with token_list.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line[0] + line[1:].rstrip()
                    self.token_list.append(line)

        else:
            self.token_list: List[str] = list(token_list)
            self.token_list_repr = ""
            for i, t in enumerate(self.token_list):
                if i == 3:
                    break
                self.token_list_repr += f"{t}, "
            self.token_list_repr += f"... (NVocab={(len(self.token_list))})"

        self.token2id: Dict[str, int] = {}
        for i, t in enumerate(self.token_list):
            if t in self.token2id:
                raise RuntimeError(f'Symbol "{t}" is duplicated')
            self.token2id[t] = i

        self.unk_symbol = unk_symbol
        if self.unk_symbol not in self.token2id:
            raise RuntimeError(f"Unknown symbol '{unk_symbol}' doesn't exist in the token_list")
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2id.get(i, self.unk_id) for i in tokens]
