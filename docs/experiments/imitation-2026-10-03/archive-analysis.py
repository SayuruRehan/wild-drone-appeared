import csv
import hashlib
import json
from pathlib import Path

import numpy as np

out = Path(__file__).resolve().parent
root = out/'raw'
rows = list(csv.DictReader((root/'validation_missions.csv').open()))
metrics = ['return','weighted_freshness','total_losses','safe_returns','charge_completions',
           'wall_commands','collision_attempts']
per_fit = []
for checkpoint in ['initial_probe.pt', 'final_probe.pt', 'selected_probe.pt']:
    for seed in [0,1,2]:
        group = [r for r in rows if r['checkpoint']==checkpoint and r['training_seed']==str(seed)]
        count = sum(int(r['critical_drone_steps']) for r in group)
        per_fit.append({'checkpoint':checkpoint,'training_seed':seed,
                        **{m:float(np.mean([float(r[m]) for r in group])) for m in metrics},
                        'critical_drone_steps':count,
                        'critical_return_moves':sum(int(r['critical_return_moves']) for r in group),
                        'critical_return_move_fraction':
                        sum(int(r['critical_return_moves']) for r in group)/count if count else None})
with (out/'per-fit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(per_fit[0]));w.writeheader();w.writerows(per_fit)
summary = {}
for checkpoint in ['initial_probe.pt','final_probe.pt','selected_probe.pt']:
    group = [r for r in per_fit if r['checkpoint']==checkpoint]
    summary[checkpoint]={m:{'mean':float(np.mean([r[m] for r in group])),
                           'sample_sd':float(np.std([r[m] for r in group],ddof=1))}
                          for m in metrics+['critical_return_move_fraction']}
baseline=[r for r in rows if r['policy']=='greedy']
summary['greedy']={m:{'mean':float(np.mean([float(r[m]) for r in baseline])),
                      'scenario_sample_sd':float(np.std([float(r[m]) for r in baseline],ddof=1))}
                     for m in metrics}
summary['greedy']['critical_return_move_fraction']={
    'pooled':sum(int(r['critical_return_moves']) for r in baseline)/
    sum(int(r['critical_drone_steps']) for r in baseline)}
fits=json.loads((root/'fits.json').read_text())
summary['classification_final']={split:{m:{'mean':float(np.mean([f[f'final_{split}_metrics'][m]
                                          for f in fits])),
                              'sample_sd':float(np.std([f[f'final_{split}_metrics'][m]
                                          for f in fits],ddof=1))}
                                       for m in ['joint_accuracy','live_command_accuracy',
                                       'critical_joint_accuracy','critical_command_accuracy',
                                       'critical_toward_base_command_fraction',
                                       'charging_hold_fraction','cross_entropy']}
                                 for split in ['train','validation']}
summary['fit_seconds_total']=sum(f['fit_seconds'] for f in fits)
summary['selected_epochs']=[f['selected_epoch'] for f in fits]
summary['selected_validation_agreement']=[f['best_validation_joint_accuracy'] for f in fits]
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype']='none'
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
colors=['#2166ac','#b35806','#1b7837']
for seed,color in enumerate(colors):
    records=list(csv.DictReader((root/f'seed-{seed}'/'label_metrics.csv').open()))
    for split,style in [('train','-'),('validation','--')]:
        group=[r for r in records if r['split']==split]
        ax.plot([int(r['epoch']) for r in group],[100*float(r['joint_accuracy']) for r in group],
                style,color=color,label=f'Seed {seed}: {split}')
ax.set_xlabel('Supervised epochs'); ax.set_ylabel('Joint-action agreement (%)')
ax.set_title('Unchanged network: fit to greedy demonstrations')
ax.set_ylim(0,100); ax.grid(alpha=.2); ax.legend(ncol=2,fontsize=8)
fig.savefig(out/'label-agreement.svg'); plt.close(fig)
# Report prose is separately reviewed; this manifest binds raw and derived numeric artifacts.
records={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(out.rglob('*')) if p.is_file() and p.name not in
         {'README.md','archive-manifest.json'}}
(out/'archive-manifest.json').write_text(json.dumps({'sha256':records},indent=2)+'\n')
print(json.dumps(summary,indent=2))
