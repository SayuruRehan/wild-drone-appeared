"""Recompute numeric aggregation and chart from archived raw records."""
import csv
import json
from pathlib import Path

import numpy as np

out=Path(__file__).resolve().parent
root=out/'raw'
protocol=json.loads((root/'protocol.json').read_text())
rows=list(csv.DictReader((root/'validation_missions.csv').open()))
fits=json.loads((root/'fits.json').read_text())
seeds=protocol['fit_seeds'];budgets=protocol['budgets']
interval=protocol['coverage_config']['updates_per_round']
metrics=['return','weighted_freshness','total_losses','safe_returns','charge_completions',
         'wall_commands','collision_attempts']
per_fit=[]
for budget in budgets+['selected']:
    checkpoint='selected_coverage.pt' if budget=='selected' else f'round-{budget//interval}.pt'
    for seed in seeds:
        group=[r for r in rows if r['training_seed']==str(seed) and r['checkpoint']==checkpoint]
        assert len(group)==5
        opportunities=sum(int(r['critical_drone_steps']) for r in group)
        moves=sum(int(r['critical_return_moves']) for r in group)
        per_fit.append({'budget':budget,'training_seed':seed,'checkpoint':checkpoint,
                        **{m:float(np.mean([float(r[m]) for r in group])) for m in metrics},
                        'full_fleet_losses':sum(int(r['total_losses'])==3 for r in group),
                        'critical_drone_steps':opportunities,'critical_return_moves':moves,
                        'critical_return_move_fraction':moves/opportunities
                        if opportunities else None})
with (out/'per-fit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(per_fit[0]));w.writeheader();w.writerows(per_fit)

def stats(values):
    values=[v for v in values if v is not None]
    return {'mean':float(np.mean(values)) if values else None,
            'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'units':len(values)}

summary={}
for budget in budgets+['selected']:
    group=[r for r in per_fit if r['budget']==budget]
    summary[str(budget)]={m:stats([r[m] for r in group])
                         for m in metrics+['critical_return_move_fraction']}
    summary[str(budget)]['full_fleet_losses']=sum(r['full_fleet_losses'] for r in group)
baseline=list(csv.DictReader((root/'baseline-replay.csv').open()))
assert len(baseline)==5
summary['greedy']={m:stats([float(r[m]) for r in baseline]) for m in metrics}
summary['greedy']['sd_unit']='scenario'
fields=['joint_accuracy','live_command_accuracy','critical_command_accuracy',
        'critical_toward_base_command_fraction','charging_hold_fraction','cross_entropy']
label_rows=[]
for seed in seeds:
    records=list(csv.DictReader((root/f'seed-{seed}/label_metrics.csv').open()))
    label_rows.extend([r|{'training_seed':seed} for r in records])
summary['classification']={str(b):{split:{m:stats([float(r[m]) for r in label_rows if
                                               int(r['updates'])==b and r['split']==split])
                                        for m in fields} for split in ['train','validation']}
                           for b in budgets}
summary['paired_return_differences_vs_first_budget']={}
for b in budgets[1:]:
    differences=[]
    for seed in seeds:
        values={r['budget']:r['return'] for r in per_fit if r['training_seed']==seed}
        differences.append(values[b]-values[budgets[0]])
    summary['paired_return_differences_vs_first_budget'][str(b)]=stats(differences)
summary['selected_rounds']=[f['selected_round'] for f in fits]
summary['fit_seconds_total']=sum(f['fit_seconds'] for f in fits)
familiar=list(csv.DictReader((root/'posthoc-training-missions.csv').open()))
summary['posthoc_training_flights']={}
for policy in ['supervised-greedy-coverage-probe','greedy']:
    group=[r for r in familiar if r['policy']==policy]
    summary['posthoc_training_flights'][policy]={
        'missions':len(group),'full_fleet_losses':sum(int(r['total_losses'])==3 for r in group),
        **{m:float(np.mean([float(r[m]) for r in group])) for m in metrics}}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype']='none'
matplotlib.rcParams['svg.hashsalt']='fixed-corpus-fitting-budget-2026-10-04'
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for split,color in [('train','#2166ac'),('validation','#b35806')]:
    values=np.array([[float(r['joint_accuracy'])*100 for r in label_rows if
                      r['training_seed']==seed and r['split']==split] for seed in seeds])
    x=np.arange(protocol['coverage_config']['rounds']+1)*interval
    mean,sd=values.mean(0),values.std(0,ddof=1)
    axes[0].plot(x,mean,color=color,label=split)
    axes[0].fill_between(x,mean-sd,mean+sd,color=color,alpha=.15)
values=np.array([[r['return'] for b in budgets for r in per_fit if
                  r['budget']==b and r['training_seed']==seed] for seed in seeds])
mean,sd=values.mean(0),values.std(0,ddof=1)
axes[1].errorbar(budgets,mean,yerr=sd,color='#2166ac',marker='o',capsize=4,label='Supervised policy')
axes[1].axhline(summary['greedy']['return']['mean'],color='#555555',linestyle='--',label='Greedy mean')
axes[0].set_title('Fixed-corpus decision fitting');axes[0].set_ylabel('Joint-action agreement (%)')
axes[0].set_ylim(0,100)
axes[1].set_title('Autonomous validation flight');axes[1].set_ylabel('Mission return (higher is better)')
for ax in axes:
    ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Mean ± sample SD over three fits; five validation missions per fit')
fig.savefig(out/'budget-comparison.svg',metadata={'Date':None});plt.close(fig)
