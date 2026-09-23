import json, csv, time
from pathlib import Path
from worm_farm import Claim, FarmConfig, WormFarm

ROOT=Path(__file__).resolve().parent/'worm_farm'/'assets'
OUT=Path(__file__).resolve().parent/'results'/'final_10x8_v350'; OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[42,77,101,113,211,313,401,509,607,709]

def make(seed):
    return FarmConfig(
        random_seed=seed,
        brain_provider='worm-dual',
        dual_checkpoint_path=str(ROOT/'wormdual_192x4.pt'),
        dual_tokenizer_path=str(ROOT/'wormdual_sp.model'),
        dual_delegate_provider='rule-based', dual_device='cpu',
        founder_pairs=10, max_generations=8, max_findings_per_generation=16,
        max_audits_per_generation=8, arena_matches_per_generation=2,
        routing_min_resolved_for_genetic_bias=10, routing_min_mode_observations=2,
        routing_replay_max_per_generation=12,
        routing_empirical_signal_enabled=True, routing_empirical_min_confidence=10,
        routing_signal_calibration_enabled=False,
        self_development_enabled=False,
        genetic_selection_mode='tournament', genetic_tournament_size=4, genetic_tournament_temperature=1.0,
        routing_fitness_lambda=0.0,
    )

def run(seed):
    farm=WormFarm(make(seed)); farm.seed(Claim(id='final-v350',text='A conventional framing may conceal a boundary condition.', assumptions=['the framing is complete'], evidence=['synthetic observation'])); t=time.time(); farm.run(); elapsed=time.time()-t
    resolved=[r for r in farm.state.routing_decisions if r.get('resolved')]
    outcomes=[int(bool(r.get('audit_pass')) and bool(r.get('downstream_survival'))) for r in resolved]
    return {
      'seed':seed,'elapsed_s':elapsed,'decisions':len(farm.state.routing_decisions),'resolved':len(resolved),
      'control':sum(bool(r.get('control_group')) for r in farm.state.routing_decisions),
      'ar':sum(r.get('chosen_mode')=='AR' for r in farm.state.routing_decisions),
      'diffusion':sum(r.get('chosen_mode')=='DIFFUSION' for r in farm.state.routing_decisions),
      'positive_outcomes':sum(outcomes),'control_fraction':sum(bool(r.get('control_group')) for r in farm.state.routing_decisions)/max(1,len(farm.state.routing_decisions)),
      'empirical_observation_pairs':len(farm.routing_book.bandit.empirical_signal.stats),
      'genome_diversity_g0': farm.state.generation_history[0].genome_diversity if farm.state.generation_history else None,
      'genome_diversity_g7': farm.state.generation_history[-1].genome_diversity if farm.state.generation_history else None,
    }
allr=[]
for seed in SEEDS:
    r=run(seed); allr.append(r); print(json.dumps(r), flush=True)
    json.dump({'seeds':SEEDS,'runs':allr},open(OUT/'FINAL_10X8_SUMMARY.json','w'),indent=2)
with open(OUT/'FINAL_10X8_SUMMARY.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=allr[0].keys()); w.writeheader(); w.writerows(allr)
print('DONE')
