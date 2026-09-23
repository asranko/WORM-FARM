from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import torch
from .model import TinyLMConfig, WormLMMicro
from .tokenizer import CharTokenizer

@dataclass(frozen=True)
class SurprisalResult:
    text: str
    mean_nll: float
    perplexity: float
    byte_coverage: float
    novelty: float

class WormLMSurprisalSensor:
    """A real LM-backed anomaly/novelty sensor for the synthetic colony."""
    def __init__(self, checkpoint: str | Path, vocab_path: str | Path, baseline_path: str | Path | None = None):
        self.vocab = CharTokenizer.load(vocab_path)
        ck = torch.load(checkpoint, map_location='cpu')
        cfg = TinyLMConfig(vocab_size=len(self.vocab.itos), **{k:ck['config'][k] for k in ('d_model','n_layers','n_heads','d_ff','max_seq_len')})
        self.model = WormLMMicro(cfg)
        self.model.load_state_dict(ck['model'])
        self.model.eval()
        self.baseline_mean, self.baseline_std = 2.5, 0.5
        if baseline_path and Path(baseline_path).exists():
            d=json.loads(Path(baseline_path).read_text())
            self.baseline_mean=float(d['mean_nll']); self.baseline_std=max(1e-4,float(d['std_nll']))

    @torch.no_grad()
    def score(self, text: str) -> SurprisalResult:
        if not text.strip():
            raise ValueError('text must not be empty')
        ids=self.vocab.encode(text)
        if len(ids)<2:
            return SurprisalResult(text,0.0,1.0,1.0,0.0)
        x=torch.tensor([ids[:-1]],dtype=torch.long); y=torch.tensor([ids[1:]],dtype=torch.long)
        logits,_=self.model(x,y)
        logp=torch.log_softmax(logits,dim=-1)
        nll=-logp.gather(-1,y.unsqueeze(-1)).squeeze(-1).mean().item()
        ppl=float(torch.exp(torch.tensor(min(10.0,nll))).item())
        covered=sum(1 for c in text if c in self.vocab.stoi)/max(1,len(text))
        z=(nll-self.baseline_mean)/self.baseline_std
        novelty=max(0.0,min(1.0,0.5+0.125*z))
        return SurprisalResult(text,nll,ppl,covered,novelty)
