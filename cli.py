from __future__ import annotations
import argparse
import json
from pathlib import Path

from .farm import WormFarm
from .brain_factory import make_brain
from .models import Claim, FarmConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WORM FARM v2.1 adversarial cognitive colony with neural brain support and named generation ledger")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a synthetic colony")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--generations", type=int, default=10)
    run.add_argument("--worms", type=int, default=72)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--brain-provider", default="rule-based", choices=["rule-based", "static", "ollama", "openai-compatible", "torch-neural"])
    run.add_argument("--brain-url", default="http://127.0.0.1:11434/v1")
    run.add_argument("--brain-model", default="qwen2.5:3b")
    resume = sub.add_parser("resume", help="resume from a checkpoint")
    resume.add_argument("checkpoint", type=Path)
    verify = sub.add_parser("verify", help="verify a checkpoint")
    verify.add_argument("checkpoint", type=Path)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "run":
        cfg = FarmConfig(random_seed=args.seed, max_generations=args.generations, max_worms=args.worms, brain_provider=args.brain_provider, brain_base_url=args.brain_url, brain_model=args.brain_model)
        farm = WormFarm(cfg, brain=make_brain(cfg))
        farm.seed(Claim(
            id="CLI",
            text="The current explanatory framing is complete.",
            evidence=["synthetic-A", "synthetic-B"],
            assumptions=["The search space is complete.", "No alternative explanation matters."],
        ))
        farm.run()
        if args.checkpoint:
            farm.save_checkpoint(args.checkpoint)
        print(json.dumps(farm.report(), ensure_ascii=False, indent=2))
    else:
        farm = WormFarm.from_checkpoint(args.checkpoint)
        farm.verify()
        if args.cmd == "resume":
            farm.run()
        print(json.dumps(farm.report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
