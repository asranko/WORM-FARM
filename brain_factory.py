from __future__ import annotations

import os

from .brain import CachedBrain, HTTPBrain, StaticBrain
from .models import FarmConfig
from .runtime import BrainAdapter, RuleBasedBrain


def make_brain(config: FarmConfig) -> BrainAdapter:
    provider = config.brain_provider.lower().strip()
    if provider in {"rule-based", "rule", "deterministic"}:
        return CachedBrain(RuleBasedBrain())
    if provider in {"static", "offline-test"}:
        return CachedBrain(StaticBrain())
    if provider in {"torch-neural", "neural", "pytorch"}:
        try:
            from .neural_brain import NeuralBrainConfig, TorchNeuralBrain
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError("torch-neural brain requires PyTorch. Install: pip install 'worm-farm[neural]'") from exc
            raise
        base = TorchNeuralBrain(NeuralBrainConfig(
            hidden_dim=config.brain_hidden_dim,
            learning_rate=config.brain_learning_rate,
            weight_decay=config.brain_weight_decay,
            entropy_bonus=config.brain_entropy_bonus,
            baseline_decay=config.brain_baseline_decay,
            temperature=config.brain_temperature_policy,
            seed=config.random_seed,
        ))
        return CachedBrain(base)
    if provider in {"worm-diffusion", "diffusion", "diffusion-audit"}:
        try:
            from .diffusion_audit import DiffusionAuditBrain
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError("worm-diffusion requires PyTorch. Install: pip install 'worm-farm[neural]'") from exc
            raise
        base_provider = config.diffusion_base_provider.lower().strip()
        if base_provider in {"torch-neural", "neural", "pytorch"}:
            from .neural_brain import NeuralBrainConfig, TorchNeuralBrain
            base = TorchNeuralBrain(NeuralBrainConfig(
                hidden_dim=config.brain_hidden_dim, learning_rate=config.brain_learning_rate,
                weight_decay=config.brain_weight_decay, entropy_bonus=config.brain_entropy_bonus,
                baseline_decay=config.brain_baseline_decay, temperature=config.brain_temperature_policy,
                seed=config.random_seed,
            ))
        elif base_provider in {"rule-based", "rule", "deterministic"}:
            base = RuleBasedBrain()
        else:
            raise ValueError(f"unsupported diffusion_base_provider: {config.diffusion_base_provider}")
        checkpoint = config.diffusion_checkpoint_path
        vocab = config.diffusion_vocab_path
        if not checkpoint or not vocab:
            raise ValueError("worm-diffusion requires diffusion_checkpoint_path and diffusion_vocab_path")
        return CachedBrain(DiffusionAuditBrain(base, checkpoint, vocab, mask_rate=config.diffusion_mask_rate, weight=config.diffusion_audit_weight, seed=config.diffusion_seed))
    if provider in {"worm-dual", "dual", "wormdual"}:
        try:
            from .dual_brain import WormDualBrain
        except ModuleNotFoundError as exc:
            if exc.name in {"torch", "sentencepiece"}:
                raise RuntimeError("worm-dual requires PyTorch and sentencepiece. Install: pip install 'worm-farm[dual]'") from exc
            raise
        if not config.dual_checkpoint_path or not config.dual_tokenizer_path:
            raise ValueError("worm-dual requires dual_checkpoint_path and dual_tokenizer_path")
        delegate_cfg = __import__("dataclasses").replace(config, brain_provider=config.dual_delegate_provider)
        delegate = make_brain(delegate_cfg)
        # Avoid recursively wrapping a CachedBrain because WormDualBrain is already cacheable at the farm boundary.
        if isinstance(delegate, CachedBrain):
            delegate = delegate.delegate
        return CachedBrain(WormDualBrain(
            checkpoint_path=config.dual_checkpoint_path,
            tokenizer_path=config.dual_tokenizer_path,
            delegate=delegate,
            mask_ratio=config.dual_mask_ratio,
            threshold=config.dual_mode_threshold,
            device=config.dual_device,
            seed=config.random_seed,
        ))
    if provider in {"ollama", "openai_compatible", "openai-compatible"}:
        base = HTTPBrain(
            base_url=config.brain_base_url,
            model=config.brain_model,
            api_key=os.getenv("WORM_BRAIN_API_KEY", ""),
            timeout=config.brain_timeout_seconds,
            temperature=config.brain_temperature,
            provider="ollama" if provider == "ollama" else "openai_compatible",
        )
        return CachedBrain(base)
    raise ValueError(f"unsupported brain_provider: {config.brain_provider}")
