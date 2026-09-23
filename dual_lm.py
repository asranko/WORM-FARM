from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

@dataclass(frozen=True)
class DualConfig:
    vocab_size:int
    seq_len:int=64
    d_model:int=192
    n_heads:int=6
    n_layers:int=4
    d_ff:int=512
    mode_count:int=2

class DualBlock(nn.Module):
    def __init__(self,cfg: DualConfig):
        super().__init__(); self.cfg=cfg
        self.ln1=nn.LayerNorm(cfg.d_model); self.qkv=nn.Linear(cfg.d_model,3*cfg.d_model)
        self.proj=nn.Linear(cfg.d_model,cfg.d_model); self.ln2=nn.LayerNorm(cfg.d_model)
        self.fc1=nn.Linear(cfg.d_model,cfg.d_ff); self.fc2=nn.Linear(cfg.d_ff,cfg.d_model)
    def forward(self,x,causal:bool):
        h=self.ln1(x); q,k,v=self.qkv(h).chunk(3,-1)
        b,t,d=q.shape; hd=d//self.cfg.n_heads
        q=q.view(b,t,self.cfg.n_heads,hd).transpose(1,2)
        k=k.view(b,t,self.cfg.n_heads,hd).transpose(1,2)
        v=v.view(b,t,self.cfg.n_heads,hd).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=causal)
        y=y.transpose(1,2).contiguous().view(b,t,d); x=x+self.proj(y)
        h=self.ln2(x); return x+self.fc2(F.gelu(self.fc1(h)))

class WormDualLM(nn.Module):
    def __init__(self,cfg:DualConfig):
        super().__init__(); self.cfg=cfg
        self.emb=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.pos=nn.Embedding(cfg.seq_len,cfg.d_model)
        self.mode=nn.Embedding(cfg.mode_count,cfg.d_model)
        self.blocks=nn.ModuleList([DualBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln=nn.LayerNorm(cfg.d_model)
        self.head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.head.weight=self.emb.weight
    def forward(self,tokens,mode:int=0):
        b,t=tokens.shape
        if t>self.cfg.seq_len: raise ValueError('sequence too long')
        p=torch.arange(t,device=tokens.device)[None,:]
        x=self.emb(tokens)+self.pos(p)+self.mode.weight[mode][None,None,:]
        causal=(mode==0)
        for blk in self.blocks: x=blk(x,causal)
        return self.head(self.ln(x))

def build_from_checkpoint(path:str, device:str='cpu'):
    obj=torch.load(path,map_location=device,weights_only=False)
    cfg=DualConfig(**obj['config'])
    model=WormDualLM(cfg)
    model.load_state_dict(obj['state_dict'])
    model.to(device).eval()
    return model, obj

class SPTokenizer:
    def __init__(self, model_path:str):
        self.sp=spm.SentencePieceProcessor(model_file=model_path)
        self.pad_id=self.sp.pad_id(); self.bos_id=self.sp.bos_id(); self.eos_id=self.sp.eos_id(); self.mask_id=self.sp.piece_to_id('<mask>')
        if self.mask_id < 0: raise ValueError('tokenizer missing <mask>')
    def encode(self,text:str): return [self.bos_id]+self.sp.encode(text,out_type=int)+[self.eos_id]

@torch.no_grad()
def score_sequence(model:WormDualLM, ids:list[int], mask_id:int, mask_ratio:float, seed:int):
    ids=list(ids[:model.cfg.seq_len])
    if len(ids)<4: return {'ar_nll': math.log(model.cfg.vocab_size), 'diff_nll': math.log(model.cfg.vocab_size), 'ar_signal':0.0,'diff_signal':0.0}
    x=torch.tensor([ids],dtype=torch.long)
    ar_in=x[:,:-1]; ar_t=x[:,1:]
    ar_logits=model(ar_in,0)
    ar_nll=float(F.cross_entropy(ar_logits.reshape(-1,model.cfg.vocab_size), ar_t.reshape(-1)))
    g=torch.Generator(device='cpu'); g.manual_seed(int(seed))
    cand=torch.arange(1,len(ids)-1,dtype=torch.long)
    k=max(1,int(len(cand)*mask_ratio))
    perm=torch.randperm(len(cand),generator=g)
    pos=cand[perm[:k]]
    z=x.clone(); z[0,pos]=mask_id
    logits=model(z,1)
    target=x[0,pos]
    diff_nll=float(F.cross_entropy(logits[0,pos], target))
    base=math.log(model.cfg.vocab_size)
    ar_signal=max(0.0,min(1.0,(base-ar_nll)/base))
    diff_signal=max(0.0,min(1.0,(base-diff_nll)/base))
    return {'ar_nll':ar_nll,'diff_nll':diff_nll,'ar_signal':ar_signal,'diff_signal':diff_signal}
