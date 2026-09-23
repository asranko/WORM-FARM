from __future__ import annotations
import json
from pathlib import Path

class CharTokenizer:
    def __init__(self, stoi: dict[str,int], itos: list[str]):
        self.stoi = stoi
        self.itos = itos
        self.unk = stoi.get('\ufffd', 0)

    @classmethod
    def build(cls, text: str, max_vocab: int = 1024):
        freq = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        chars = sorted(freq, key=lambda x: (-freq[x], x))[: max_vocab - 2]
        itos = ['\ufffd', '\n'] + [c for c in chars if c != '\n' and c != '\ufffd']
        itos = itos[:max_vocab]
        stoi = {c:i for i,c in enumerate(itos)}
        return cls(stoi, itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.unk) for c in text]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.itos[i] if 0 <= i < len(self.itos) else '\ufffd' for i in ids)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({'itos': self.itos}, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: str | Path):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        itos = d['itos']; return cls({c:i for i,c in enumerate(itos)}, itos)
