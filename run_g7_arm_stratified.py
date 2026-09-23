from __future__ import annotations
import csv, json, math, os, hashlib
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'worm_farm'/'assets'
OUT=ROOT/'results'/'claim_diverse_cost_parity_v2'/'g7_arm_stratified_v1'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS=[42,77,101,113,211,313,401,509,607,709,811,823,839,853,877]
TASKS=[
('T01_CAUSAL_MEDIATION','A factory introduced a new inspection step. Defect rates fell from 8% to 5%, while average operator training hours also increased. Determine which additional observation would most directly distinguish whether the inspection step caused the improvement, rather than the training increase.'),
('T02_COUNTERFACTUAL_CACHE','Two software services show the same latency today. Service A used a warmed cache for the last 20 minutes; Service B did not. State the counterfactual question that would isolate the cache effect, and identify the outcome that must remain comparable.'),
('T03_BOUNDARY_CONDITION','A rule says: every admitted case with at least three corroborating observations is accepted. Find a concrete boundary condition under which that rule can fail even though the three observations are individually accurate.'),
('T04_ASSUMPTION_HUNT','A team concludes that a model is robust because accuracy stays above 90% after five perturbation tests. Identify one hidden assumption required for that conclusion and explain what measurement would test it.'),
('T05_ANOMALY_DETECTION','A sensor normally reports values between 40 and 60 units. It suddenly reports 41, 42, 58, 59, 60, 41, 42, then 93 once. Decide whether the isolated 93 should be treated as an anomaly using only the stated evidence; distinguish detection from diagnosis.'),
('T06_EVIDENCE_SUFFICIENCY','A study reports that a treatment improved recovery from 50% to 54% in one uncontrolled cohort of 100 participants. What claim is justified by these data alone, and what stronger claim would require additional evidence?'),
('T07_MECHANISM_DISCRIMINATION','A population becomes more consistent after a policy change. Mechanism M1 predicts faster convergence with lower diversity; mechanism M2 predicts unchanged diversity with improved coordination. Which single measurement would most efficiently discriminate between the two mechanisms?'),
('T08_COMPETING_EXPLANATIONS','An automated reviewer rejects 18% of submissions this month instead of 11% last month. Give two non-equivalent explanations that fit the observation and one observation that would distinguish them.'),
('T09_UNCERTAINTY_CALIBRATION','A classifier assigns confidence 0.80 to 200 predictions, of which 150 are correct. Is the 0.80 score calibrated for this subgroup? State the empirical calibration test and distinguish calibration from accuracy.'),
('T10_ADVERSARIAL_CLAIM','Claim: Because every successful run used the same initialization protocol, the protocol is proven necessary. Identify the logical gap and construct the smallest counterexample that would defeat the claim.'),
('T11_NOVELTY_HYPOTHESIS','Propose a testable hypothesis about why two equally accurate reasoning strategies might differ in recovery after failure. Specify one measurable prediction that could falsify your hypothesis.'),
('T12_MULTI_STEP_INFERENCE','Three interventions are observed: A increases throughput by 10%, B increases throughput by 8%, and A+B increases throughput by 9%. Determine what can and cannot be inferred about interaction between A and B without assuming additivity. Give the minimal additional comparison needed.'),
]
EXCEPTIONS_A={'MEMORY_ANALYST','NOVELTY_FORGE'}
EXCEPTIONS_B={'ANOMALY_HUNTER','BLIND_SPOT_HUNTER','COUNTERFACTUAL','META_ARCHITECT','WORM_EATER'}
FOLDS={
 'fold_a': {'evaluation':set(x[0] for x in TASKS[6:]), 'exceptions':EXCEPTIONS_A},
 'fold_b': {'evaluation':set(x[0] for x in TASKS[:6]), 'exceptions':EXCEPTIONS_B},
}

def make_config(seed):
    from worm_farm import FarmConfig
    return FarmConfig(
        random_seed=seed, brain_provider='worm-dual',
        dual_checkpoint_path=str(ASSET/'wormdual_192x4.pt'), dual_tokenizer_path=str(ASSET/'wormdual_sp.model'),
        dual_delegate_provider='rule-based', dual_device='cpu', founder_pairs=10, max_generations=8, max_findings_per_generation=16,
        max_audits_per_generation=8, arena_matches_per_generation=2, routing_min_resolved_for_genetic_bias=10,
        routing_min_mode_observations=2, routing_replay_max_per_generation=12, routing_empirical_signal_enabled=True,
        routing_empirical_min_confidence=10, routing_signal_calibration_enabled=False, self_development_enabled=False,
        genetic_selection_mode='tournament', genetic_tournament_size=4, genetic_tournament_temperature=1.0,
        routing_fitness_lambda=0.0)

def apply_static_rule(farm, exceptions):
    original=farm.routing_book.bandit.select
    def select(**kwargs):
        d=original(**kwargs)
        if d.control_group:
            return d
        mode='AR' if kwargs['worm'].specialty in exceptions else 'DIFFUSION'
        return replace(d, chosen_mode=mode, other_mode=('DIFFUSION' if mode=='AR' else 'AR'))
    farm.routing_book.bandit.select=select

def run_one(task_id, claim_text, seed, policy, fold):
    os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
    from worm_farm import Claim, WormFarm
    farm=WormFarm(make_config(seed))
    if policy=='simple': apply_static_rule(farm, FOLDS[fold]['exceptions'])
    farm.seed(Claim(id=task_id,text=claim_text,assumptions=['synthetic task assumption'],evidence=['synthetic observation']))
    farm.run()
    g7=farm.state.generation_history[-1]
    run_g7=float(getattr(g7,'routing_mean_fitness',0.5))
    vals={'AR':[],'DIFFUSION':[]}
    for r in farm.state.routing_decisions:
        if int(r.get('generation',-1))!=7 or not r.get('resolved') or r.get('control_group'):
            continue
        w=farm.state.worms.get(r.get('worm_id'))
        if w is None: continue
        vals[r.get('chosen_mode','AR')].append(float(getattr(w,'routing_fitness',0.5)))
    row={'fold':fold,'task_id':task_id,'seed':seed,'policy':policy,'g7_routing_mean_fitness':run_g7}
    for mode,key in [('AR','ar'),('DIFFUSION','diff')]:
        xs=vals[mode]; n=len(xs); m=sum(xs)/n if n else None; sd=math.sqrt(sum((x-m)**2 for x in xs)/(n-1)) if n>1 else None
        row[f'g7_{key}_n']=n; row[f'g7_{key}_mean_worm_fitness']=m; row[f'g7_{key}_sd']=sd
    covered=sum(row[f'g7_{k}_n'] for k in ('ar','diff'))
    row['g7_stratified_coverage']=covered
    row['g7_reconciliation_weighted']=(sum((row[f'g7_{k}_n'] or 0)*(row[f'g7_{k}_mean_worm_fitness'] or 0) for k in ('ar','diff'))/covered) if covered else None
    row['reconciliation_error']=abs(run_g7-row['g7_reconciliation_weighted']) if row['g7_reconciliation_weighted'] is not None else None
    return row

def main():
    task_map=dict(TASKS)
    jobs=[]
    for fold in ('fold_a','fold_b'):
        for task_id in sorted(FOLDS[fold]['evaluation']):
            for seed in SEEDS:
                for policy in ('thompson','simple'):
                    key=f'{fold}|{task_id}|{seed}|{policy}'; digest=hashlib.sha256(key.encode()).hexdigest()[:16]
                    p=OUT/f'run_{digest}.json'
                    if not p.exists(): jobs.append((task_id,task_map[task_id],seed,policy,fold,digest))
    print('PENDING',len(jobs),flush=True)
    workers=min(6,len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(run_one,*j[:5]):j for j in jobs}
        for fut in as_completed(futs):
            j=futs[fut]
            try:
                row=fut.result(); (OUT/f'run_{j[5]}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
                print('OK',row['fold'],row['task_id'],row['seed'],row['policy'],flush=True)
            except Exception as e:
                print('FAIL',j[:5],type(e).__name__,str(e),flush=True)
    rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('run_*.json'))]
    if rows:
        with (OUT/'G7_ARM_STRATIFIED_RUNS.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('SAVED',len(rows),flush=True)
if __name__=='__main__': main()
