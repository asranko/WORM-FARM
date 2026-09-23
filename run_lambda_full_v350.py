import json, csv, time
from pathlib import Path
from worm_farm import Claim, FarmConfig, WormFarm

ROOT=Path(__file__).resolve().parent/'worm_farm'/'assets'
OUT=Path(__file__).resolve().parent/'results'/'selector_lambda_full_v350'; OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[42,77,101,113,211,313,401,509,607,709]

CLAIM=Claim(id='lambda-v350',text='A conventional framing may conceal a boundary condition.', assumptions=['the framing is complete'], evidence=['synthetic observation'])

def make(seed, lam):
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
        routing_fitness_lambda=float(lam),
    )

def run(seed, lam):
    farm=WormFarm(make(seed,lam)); farm.seed(CLAIM); t=time.time(); farm.run(); elapsed=time.time()-t
    births=[r for r in farm.genetics.records if r.success]
    pairs_by_gen={}
    for r in births:
        pairs_by_gen.setdefault(int(r.generation),[]).append([r.parent_a,r.parent_b])
    for g in pairs_by_gen: pairs_by_gen[g]=sorted(tuple(sorted(p)) for p in pairs_by_gen[g])
    div=[float(x.genome_diversity) for x in farm.state.generation_history]
    return {
      'seed':seed,'lambda':lam,'elapsed_s':elapsed,
      'decisions':len(farm.state.routing_decisions),'resolved':sum(bool(r.get('resolved')) for r in farm.state.routing_decisions),
      'births':len(births),'parent_pairs_by_generation':pairs_by_gen,
      'final_population':len(farm.state.worms),'active_end':len(farm.state.active_worms()),
      'routing_fitness_mean':sum(float(w.routing_fitness) for w in farm.state.worms.values())/max(1,len(farm.state.worms)),
      'diversity_g0':div[0] if div else None,'diversity_g7':div[-1] if div else None,
    }

rows=[]
for lam in (0.0,5.0):
    for seed in SEEDS:
        r=run(seed,lam); rows.append(r); print(json.dumps({k:v for k,v in r.items() if k!='parent_pairs_by_generation'}), flush=True)
        json.dump(rows,open(OUT/'LAMBDA_FULL_RAW.json','w'),indent=2)
# paired compare
by={(r['seed'],r['lambda']):r for r in rows}
comp=[]
for seed in SEEDS:
    a=by[(seed,0.0)]; b=by[(seed,5.0)]
    comp.append({
      'seed':seed,
      'parent_pairs_identical': a['parent_pairs_by_generation']==b['parent_pairs_by_generation'],
      'births_lambda0':a['births'],'births_lambda5':b['births'],
      'routing_fitness_lambda0':a['routing_fitness_mean'],'routing_fitness_lambda5':b['routing_fitness_mean'],
      'diversity_g0_lambda0':a['diversity_g0'],'diversity_g7_lambda0':a['diversity_g7'],
      'diversity_g0_lambda5':b['diversity_g0'],'diversity_g7_lambda5':b['diversity_g7'],
      'final_population_lambda0':a['final_population'],'final_population_lambda5':b['final_population'],
    })
summary={
 'runs':rows,'comparison':comp,
 'identical_seed_count':sum(x['parent_pairs_identical'] for x in comp),
 'different_seed_count':sum(not x['parent_pairs_identical'] for x in comp),
 'acceptance_met':sum(not x['parent_pairs_identical'] for x in comp)>0,
}
json.dump(summary,open(OUT/'LAMBDA_FULL_SUMMARY.json','w'),indent=2)
with open(OUT/'LAMBDA_FULL_COMPARISON.csv','w',newline='') as f:
    fields=list(comp[0]); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(comp)
print('SUMMARY',json.dumps({k:summary[k] for k in ['identical_seed_count','different_seed_count','acceptance_met']}),flush=True)
