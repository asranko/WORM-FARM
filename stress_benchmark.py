from . import Claim, FarmConfig, WormFarm

CLAIM = Claim(
    id="BENCH",
    text="The current explanatory model fits the observation, therefore the solution space is complete.",
    evidence=["E1", "E2", "E3"],
    assumptions=["The search space is complete.", "No alternative explanation matters."],
)

for seed in range(10):
    cfg = FarmConfig(max_generations=12, max_worms=80, founder_pairs=4, random_seed=seed)
    farm = WormFarm(cfg)
    farm.seed(CLAIM)
    farm.run()
    r = farm.report()
    print(seed, r["worms"], r["findings"], r["genetic_births"], r["max_depth"], r["war_edges"], r["trust_edges"], r["omega_events"], r["extinction_events"], r["genome_diversity"])
