"""Regenerate derived metrics and chart from this directory's archived raw records."""
import csv
import json
from pathlib import Path

import numpy as np

out = Path(__file__).resolve().parent
root = out/'raw'
rows = list(csv.DictReader((root/'validation_missions.csv').open()))
fits = json.loads((root/'fits.json').read_text())
settings = json.loads((root/'protocol.json').read_text())['coverage_config']
last_round = settings['rounds']
metrics = ['return', 'weighted_freshness', 'total_losses', 'safe_returns', 'charge_completions',
           'wall_commands', 'collision_attempts']
per_fit, per_round = [], []
for arm in ['fixed', 'teacher', 'dagger']:
    for seed in [0,1,2]:
        for checkpoint in [f'round-{last_round}.pt','selected_coverage.pt']:
            group = [r for r in rows if r['arm']==arm and r['training_seed']==str(seed) and
                     r['checkpoint']==checkpoint]
            opportunities = sum(int(r['critical_drone_steps']) for r in group)
            moves = sum(int(r['critical_return_moves']) for r in group)
            per_fit.append({'arm':arm,'training_seed':seed,'checkpoint':checkpoint,
                            **{m:float(np.mean([float(r[m]) for r in group])) for m in metrics},
                            'critical_drone_steps':opportunities,'critical_return_moves':moves,
                            'critical_return_move_fraction':moves/opportunities
                            if opportunities else None})
        for r in range(1,last_round+1):
            group=[row for row in rows if row['arm']==arm and row['training_seed']==str(seed)
                   and row['checkpoint']==f'round-{r}.pt']
            per_round.append({'arm':arm,'training_seed':seed,'round':r,
                              **{m:float(np.mean([float(row[m]) for row in group])) for m in metrics}})

def write_csv(name, data):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

write_csv('per-fit.csv',per_fit)
write_csv('per-round.csv',per_round)

def stats(values):
    values=[v for v in values if v is not None]
    return {'mean':float(np.mean(values)) if values else None,
            'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,
            'units':len(values)}

summary={}
classification=['joint_accuracy','live_command_accuracy','critical_joint_accuracy',
                'critical_command_accuracy','critical_toward_base_command_fraction',
                'charging_hold_fraction','cross_entropy']
for arm in ['fixed','teacher','dagger']:
    group=[f for f in fits if f['arm']==arm]
    summary[arm]={'final_training_labels':[f['final_training_states'] for f in group],
                  'selected_rounds':[f['selected_round'] for f in group],
                  'selected_validation_joint_accuracy':
                  [f['selected_validation_joint_accuracy'] for f in group],
                  'final_classification':{split:{m:stats([f[f'final_{split}_metrics'][m]
                                            for f in group]) for m in classification}
                                          for split in ['train','validation']}}
    for checkpoint in [f'round-{last_round}.pt','selected_coverage.pt']:
        a=[r for r in per_fit if r['arm']==arm and r['checkpoint']==checkpoint]
        summary[arm][checkpoint]={m:stats([r[m] for r in a])
                                 for m in metrics+['critical_return_move_fraction']}
    collections=[]
    for seed in [0,1,2]:
        path=root/arm/f'seed-{seed}'/'collections.json'
        records=json.loads(path.read_text()) if path.exists() else []
        live=sum(r['live_drone_states'] for r in records)
        unreachable=sum(r['unreachable_drone_states'] for r in records)
        collections.append({'seed':seed,'added_labels':sum(r['states'] for r in records),
                            'live_drone_states':live,'unreachable_drone_states':unreachable,
                            'unreachable_fraction':unreachable/live if live else None,
                            'critical_drone_states':sum(r['critical_drone_states'] for r in records),
                            'charging_drone_states':sum(r['charging_drone_states'] for r in records),
                            'all_stay_labels':sum(r['all_stay_labels'] for r in records)})
    summary[arm]['collection_per_fit']=collections
baseline=[r for r in rows if r['policy']=='greedy']
summary['greedy']={m:stats([float(r[m]) for r in baseline]) for m in metrics}
summary['greedy']['sd_unit']='scenario'
summary['paired_final_return_differences']={}
for arm in ['teacher','dagger']:
    differences=[]
    for seed in [0,1,2]:
        values={r['arm']:r['return'] for r in per_fit if r['training_seed']==seed and
                r['checkpoint']==f'round-{last_round}.pt'}
        differences.append(values[arm]-values['fixed'])
    summary['paired_final_return_differences'][f'{arm} minus fixed']=stats(differences)
summary['fit_and_collection_seconds_total']=sum(f['fit_and_collection_seconds'] for f in fits)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype']='none'
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for arm,color in [('fixed','#2166ac'),('teacher','#1b7837'),('dagger','#b35806')]:
    classification_values=[]
    for seed in [0,1,2]:
        records=list(csv.DictReader((root/arm/f'seed-{seed}'/'label_metrics.csv').open()))
        classification_values.append([100*float(r['joint_accuracy']) for r in records
                                      if r['split']=='validation'])
    values=np.array(classification_values)
    x=np.arange(last_round+1)
    mean,sd=values.mean(0),values.std(0,ddof=1)
    axes[0].plot(x,mean,color=color,label=arm)
    axes[0].fill_between(x,mean-sd,mean+sd,color=color,alpha=.12)
    values=np.array([[r['return'] for r in per_round if r['arm']==arm and
                     r['training_seed']==seed] for seed in [0,1,2]])
    mean,sd=values.mean(0),values.std(0,ddof=1)
    axes[1].plot(x[1:],mean,color=color,label=arm)
    axes[1].fill_between(x[1:],mean-sd,mean+sd,color=color,alpha=.12)
axes[1].axhline(summary['greedy']['return']['mean'],color='#555555',linestyle='--',
                label='Greedy teacher')
axes[0].set_title('Held-out teacher joint-action agreement')
axes[0].set_ylabel('Agreement (%)');axes[0].set_ylim(0,25)
axes[1].set_title('Autonomous validation return')
axes[1].set_ylabel('Mission return (higher is better)')
for ax in axes:
    ax.set_xlabel('Round (1,000 updates each)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Mean ± sample SD across three fits; five missions per fit')
fig.savefig(out/'comparison.svg');plt.close(fig)
