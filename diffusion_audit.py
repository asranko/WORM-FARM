from __future__ import annotations
import json, math, re, hashlib
from pathlib import Path
from typing import Any, Dict
import torch

# The diffusion model code is deliberately embedded here to keep the Farm release self-contained.
from dataclasses import dataclass
from torch import nn
import torch.nn.functional as F

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)

@dataclass
class DiffusionConfig:
    vocab_size:int; mask_id:int; max_seq_len:int; d_model:int; n_heads:int; n_layers:int; d_ff:int; dropout:float=0.0

class _Block(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout):
        super().__init__(); assert d_model%n_heads==0
        self.ln1=nn.LayerNorm(d_model); self.qkv=nn.Linear(d_model,3*d_model,bias=False); self.proj=nn.Linear(d_model,d_model,bias=False)
        self.ln2=nn.LayerNorm(d_model); self.fc=nn.Linear(d_model,d_ff); self.fc2=nn.Linear(d_ff,d_model); self.drop=nn.Dropout(dropout)
        self.n_heads=n_heads; self.head_dim=d_model//n_heads
    def forward(self,x):
        B,T,C=x.shape; h=self.ln1(x); qkv=self.qkv(h).view(B,T,3,self.n_heads,self.head_dim).permute(2,0,3,1,4); q,k,v=qkv
        a=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim); a=F.softmax(a,dim=-1); y=a@v; y=y.transpose(1,2).contiguous().view(B,T,C)
        x=x+self.drop(self.proj(y)); h=self.ln2(x); return x+self.drop(self.fc2(F.gelu(self.fc(h))))

class _Model(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.emb=nn.Embedding(cfg.vocab_size,cfg.d_model); self.pos=nn.Embedding(cfg.max_seq_len,cfg.d_model)
        self.time_mlp=nn.Sequential(nn.Linear(64,cfg.d_model),nn.SiLU(),nn.Linear(cfg.d_model,cfg.d_model))
        self.blocks=nn.ModuleList([_Block(cfg.d_model,cfg.n_heads,cfg.d_ff,cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln=nn.LayerNorm(cfg.d_model); self.head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
    def tfeat(self,t):
        f=torch.exp(torch.linspace(math.log(1),math.log(1000),32,device=t.device)); a=t[:,None]*f[None,:]; return torch.cat([torch.sin(a),torch.cos(a)],-1)
    def forward(self,ids,t):
        B,T=ids.shape; pos=torch.arange(T,device=ids.device)[None,:]; x=self.emb(ids)+self.pos(pos)+self.time_mlp(self.tfeat(t))[:,None,:]
        for b in self.blocks:x=b(x)
        return self.head(self.ln(x))

class DiffusionAuditBrain:
    """Wrap an existing Brain and use a trained diffusion denoiser as a bidirectional coherence audit.

    It does not replace semantic truth checking. Its score is explicitly a model-domain coherence signal.
    """
    name="worm-diffusion-audit"
    def __init__(self,base,checkpoint,vocab_path,mask_rate=0.35,weight=0.15,seed=42):
        self.base=base; self.mask_rate=float(mask_rate); self.weight=float(weight); self.seed=int(seed); self.calls=0; self.last_scores={}
        c=torch.load(checkpoint,map_location='cpu',weights_only=False); self.cfg=DiffusionConfig(**c['config']); self.model=_Model(self.cfg); self.model.load_state_dict(c['state_dict']); self.model.eval()
        self.vocab=json.loads(Path(vocab_path).read_text()); self.stoi={t:i for i,t in enumerate(self.vocab)}
        self.model_digest=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    def _score(self,text,seed):
        toks=TOKEN_RE.findall(text.lower()); ids=[self.stoi.get('<bos>',1)]+[self.stoi.get(t,self.stoi.get('<unk>',3)) for t in toks]
        ids=ids[:self.cfg.max_seq_len]
        if len(ids)<4:return 0.0
        x=torch.tensor(ids,dtype=torch.long).unsqueeze(0); g=torch.Generator(); g.manual_seed(seed)
        mask=torch.rand_like(x,dtype=torch.float32,generator=g)<self.mask_rate; mask[:,:1]=False
        xt=x.clone(); xt[mask]=self.cfg.mask_id
        t=torch.full((1,),0.55)
        with torch.no_grad(): logits=self.model(xt,t); lp=torch.log_softmax(logits,-1)
        chosen=lp[0][mask[0],x[0][mask[0]]]
        if chosen.numel()==0:return 0.0
        nll=float(-chosen.mean())
        # Normalize relative to the random-token baseline so the score has usable dynamic range.
        random_nll=math.log(max(2,len(self.vocab)-1))
        z=(random_nll-nll)/1.25
        return float(1.0/(1.0+math.exp(-max(-12.0,min(12.0,z)))))
    def propose(self,*,claim,worm,parent,context):
        proposal=self.base.propose(claim=claim,worm=worm,parent=parent,context=context)
        text=' '.join([proposal.get('title',''),proposal.get('core_hypothesis',''),proposal.get('counterargument',''),proposal.get('falsifier','')])
        score=self._score(text,self.seed+self.calls*17); self.calls+=1; self.last_scores[worm.id]=score
        # Use the score only as a small evidence-discipline multiplier.
        proposal=dict(proposal)
        proposal['information_gain']=max(0.0,min(1.0,proposal.get('information_gain',0.5)*(0.85+0.15*score)))
        proposal['falsifiability']=max(0.0,min(1.0,proposal.get('falsifiability',0.5)*(0.90+0.10*score)))
        return proposal
    def learn(self,*,worm_id,reward,finding=None):
        score=self.last_scores.get(worm_id,0.5); scaled=float(reward)*(1.0-self.weight+self.weight*score)
        method=getattr(self.base,'learn',None)
        if callable(method): method(worm_id=worm_id,reward=scaled,finding=finding)
    def runtime_state(self):
        method=getattr(self.base,'runtime_state',None); return {'diffusion_calls':self.calls,'diffusion_scores':dict(self.last_scores),'model_digest':self.model_digest,'base':method() if callable(method) else {}}
    def load_runtime_state(self,state):
        self.calls=int(state.get('diffusion_calls',0)); self.last_scores=dict(state.get('diffusion_scores',{})); method=getattr(self.base,'load_runtime_state',None)
        if callable(method): method(state.get('base',{}))
    def fingerprint(self):
        method=getattr(self.base,'fingerprint',None); base=method() if callable(method) else ''
        return hashlib.sha256((base+self.model_digest).encode()).hexdigest()
