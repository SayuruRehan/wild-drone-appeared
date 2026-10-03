"""Plot one recorded mission's battery and necessary return-energy bound."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = list(csv.DictReader(args.trace.open()))
    config = json.loads(args.protocol.read_text())['environment']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.fonttype'] = 'none'
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained', sharey=True)
    for drone, ax in enumerate(axes):
        rows = [r for r in records if int(r['drone']) == drone and r['live_before'] == 'True']
        times = [int(r['step'])-1 for r in rows]
        ax.plot(times, [float(r['battery_before']) for r in rows], label='Battery', color='#2166ac')
        ax.plot(times, [int(r['distance_before'])*config['move_cost'] for r in rows],
                label='Minimum energy to base', color='#b35806')
        losses = [r for r in rows if r['new_loss'] == 'True']
        if losses:
            loss = losses[0]
            ax.scatter([int(loss['step'])], [float(loss['battery_after'])],
                       color='#b2182b', marker='x', label='Drone lost', zorder=3)
        ax.set_title(f'Drone {drone+1}')
        ax.set_xlabel('Mission step')
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Battery units')
    axes[0].legend(fontsize=8)
    fig.suptitle(f"One validation mission: scenario {records[0]['scenario_seed']}, "
                 f"training seed {records[0]['training_seed']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)


if __name__ == '__main__':
    main()
