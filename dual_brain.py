from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
from .models import Claim, Finding, Worm
from .brain import BrainProtocolError, validate_proposal
from .dual_lm import build_from_checkpoint, SPTokenizer, score_sequence
from .mode_router import RoutingBandit, ModeDecision

class WormDualBrain:
    """Shared AR+masked-diffusion LM used as a per-worm mode-selection and uncertainty brain.

    The dual LM supplies cognitive signals; a delegated proposal brain renders the final structured artifact.
    This keeps the verified proposal/audit contract while making mode choice genuinely model-driven.
    """
    name = 'worm-dual-lm'

    def __init__(self, checkpoint_path:str, tokenizer_path:str, delegate:Any,
                 mask_ratio:float=0.35, threshold:float=0.08, device:str='cpu', seed:int=42, router:RoutingBandit|None=None):
        self.checkpoint_path=str(checkpoint_path); self.tokenizer_path=str(tokenizer_path)
        self.delegate=delegate; self.mask_ratio=float(mask_ratio); self.threshold=float(threshold)
        self.device=device; self.seed=int(seed)
        self.model,self.meta=build_from_checkpoint(self.checkpoint_path,self.device)
        self.tok=SPTokenizer(self.tokenizer_path)
        self.calls=0; self.ar_calls=0; self.diff_calls=0
        self.last_signal:dict[str,Any]={}
        self.last_replay_payload:dict[str,Any]={}
        self.router=router

    def _input_text(self, claim:Claim, worm:Worm, parent:Finding|None, context:Dict[str,Any]) -> str:
        parent_text='' if parent is None else f' {parent.title} {parent.detail[:700]}'
        ws=context.get('shared_workspace',[])
        # Canonical JSON makes the replay payload byte-stable across checkpoint
        # serialization, where dict insertion order and tuple/list conversion can differ.
        ws_text=' '.join(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))[:220] for x in ws[:4])
        current_generation = int(context.get('current_generation', worm.generation))
        return (f'SPECIALTY={worm.specialty} GENERATION={current_generation} DEPTH={worm.depth} '
                f'STRATEGY={worm.current_strategy} CLAIM={claim.text[:900]} '
                f'ASSUMPTIONS={json.dumps(claim.assumptions[:4], ensure_ascii=False, sort_keys=True)} '
                f'EVIDENCE={json.dumps(claim.evidence[:4], ensure_ascii=False, sort_keys=True)} '
                f'WORKSPACE={ws_text}{parent_text}')

    def propose(self, *, claim:Claim, worm:Worm, parent:Finding|None, context:Dict[str,Any]) -> Dict[str,Any]:
        text=self._input_text(claim,worm,parent,context)
        ids=self.tok.encode(text)
        seed=self.seed + 997*worm.generation + 13*worm.depth + sum(ord(c) for c in worm.specialty)
        signal=score_sequence(self.model,ids,self.tok.mask_id,self.mask_ratio,seed)
        traits=context.get('genome',{}) if isinstance(context.get('genome',{}),dict) else {}
        strategy=context.get('strategy_phenotype',{}) if isinstance(context.get('strategy_phenotype',{}),dict) else {}
        penetration=float(traits.get('penetration',0.5)); adversarial=float(traits.get('adversariality',0.5)); novelty=float(traits.get('novelty_bias',traits.get('novelty',0.5)))
        mode_score=(signal['diff_signal']-signal['ar_signal']) + 0.12*penetration + 0.10*adversarial + 0.05*novelty
        uncertainty=max(0.0,1.0-0.5*(signal['ar_signal']+signal['diff_signal']))
        if self.router is not None:
            decision = self.router.select(
                worm=worm, phenotype=traits, ar_signal=float(signal['ar_signal']), diff_signal=float(signal['diff_signal']),
                mode_score=float(mode_score), uncertainty=float(uncertainty), input_tokens=len(ids),
                strategy=strategy, workspace_state=context.get('shared_workspace', []),
                decision_generation=int(context.get('current_generation', worm.generation)),
            )
            mode = decision.chosen_mode
            routing = decision.as_dict()
        else:
            mode='DIFFUSION' if mode_score >= self.threshold else 'AR'
            routing = {}
        payload=dict(context)
        payload['dual_mode_signal']={**signal,'mode_score':mode_score,'selected_mode':mode,'uncertainty':uncertainty,'specialty':worm.specialty,'routing_decision':routing}
        self.calls+=1; self.ar_calls += int(mode=='AR'); self.diff_calls += int(mode=='DIFFUSION')
        self.last_signal=payload['dual_mode_signal']
        self.last_replay_payload = {
            'input_text': text, 'claim': {'id': claim.id, 'text': claim.text, 'assumptions': list(claim.assumptions), 'evidence': list(claim.evidence)},
            'worm': {'id': worm.id, 'generation': int(context.get('current_generation', worm.generation)), 'birth_generation': worm.generation, 'depth': worm.depth, 'specialty': worm.specialty, 'current_strategy': worm.current_strategy, 'routing_bias': float(getattr(worm, 'routing_bias', 0.0))},
            'parent': None if parent is None else {'id': parent.id, 'title': parent.title, 'detail': parent.detail[:900]},
            'context': dict(context), 'mask_ratio': self.mask_ratio,
        }
        # Delegate remains the contract owner; dual LM changes its context and therefore the proposal.
        proposal=self.delegate.propose(claim=claim,worm=worm,parent=parent,context=payload)
        out=validate_proposal(proposal).as_dict()
        # Small bounded modulation from the actual LM signal.
        out['novelty']=max(0.0,min(1.0,0.88*out['novelty']+0.12*signal['diff_signal']))
        out['information_gain']=max(0.0,min(1.0,0.88*out['information_gain']+0.12*uncertainty))
        out['falsifiability']=max(0.0,min(1.0,0.94*out['falsifiability']+0.06*signal['ar_signal']))
        return out

    def counterfactual_from_payload(self, replay_payload:dict[str,Any], forced_mode:str) -> dict[str,Any]:
        """Replay the exact saved input with the opposite mode, without mutating colony state."""
        if forced_mode not in {"AR", "DIFFUSION"}:
            raise ValueError("forced_mode must be AR or DIFFUSION")
        text = str(replay_payload["input_text"])
        ids = self.tok.encode(text)
        seed = self.seed + 997*int(replay_payload["worm"].get("generation",0)) + 13*int(replay_payload["worm"].get("depth",0)) + sum(ord(c) for c in str(replay_payload["worm"].get("specialty","")))
        signal = score_sequence(self.model, ids, self.tok.mask_id, self.mask_ratio, seed)
        traits = replay_payload.get("context", {}).get("genome", {}) if isinstance(replay_payload.get("context", {}).get("genome", {}), dict) else {}
        p = replay_payload.get("worm", {})
        penetration=float(traits.get("penetration",0.5)); adversarial=float(traits.get("adversariality",0.5)); novelty=float(traits.get("novelty",0.5))
        mode_score=(signal['diff_signal']-signal['ar_signal']) + 0.12*penetration + 0.10*adversarial + 0.05*novelty
        mode_signal=float(signal['diff_signal'] if forced_mode=="DIFFUSION" else signal['ar_signal'])
        uncertainty=max(0.0,1.0-0.5*(signal['ar_signal']+signal['diff_signal']))
        context=dict(replay_payload.get("context", {}))
        context['dual_mode_signal']={**signal,'mode_score':mode_score,'selected_mode':forced_mode,'uncertainty':uncertainty,'specialty':p.get('specialty','GENERALIST'),'counterfactual':True}
        try:
            proposal = self.delegate.propose(
                claim=type('ClaimProxy', (), replay_payload['claim'])(),
                worm=type('WormProxy', (), p)(), parent=None, context=context
            )
            validated = validate_proposal(proposal).as_dict()
        except Exception as exc:
            validated={'error': f'{type(exc).__name__}: {exc}'}
        return {'forced_mode':forced_mode,'signal':signal,'mode_score':mode_score,'uncertainty':uncertainty,'input_tokens':len(ids),'proposal':validated}

    def runtime_state(self)->dict[str,Any]:
        return {'calls':self.calls,'ar_calls':self.ar_calls,'diff_calls':self.diff_calls,'last_signal':self.last_signal,'router': self.router.runtime_state() if self.router else {}}

    def load_runtime_state(self,state:dict[str,Any])->None:
        if not state: return
        self.calls=int(state.get('calls',self.calls)); self.ar_calls=int(state.get('ar_calls',self.ar_calls)); self.diff_calls=int(state.get('diff_calls',self.diff_calls))
        self.last_signal=dict(state.get('last_signal',self.last_signal))
        if self.router is not None and state.get('router'): self.router.load_runtime_state(state['router'])

    def learn(self, *, worm_id:str, reward:float, finding:Finding|None=None):
        method=getattr(self.delegate,'learn',None)
        if callable(method): method(worm_id=worm_id,reward=reward,finding=finding)
