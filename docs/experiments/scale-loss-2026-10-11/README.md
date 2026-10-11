# Frozen loss-only 40×40 comparison

Source: `b6825bbdaa176277f76efb3dc84ffe334cabe289`. Three new supervised fits, zero RL updates
and zero new worlds. All control fits/data are from the audited original pilot.
The original failed regression pilot and exact-reference diagnostic remain intact.

| Policy | Freshness ↑ | Overdue fraction ↓ | Return ↑ |
|---|---:|---:|---:|
| ranking | 0.135589 | 0.787513 | -21.511754 |
| centered | 0.192003 | 0.697632 | -17.844649 |
| greedy | 0.226995 | 0.643557 | -15.649919 |
| masked_random | 0.086493 | 0.885049 | -25.408448 |
| exact_inspection | 0.226350 | 0.644781 | -15.686822 |

Every-seed objective-effect decision: **False**.
Original every-seed ranking capability: **False**.
Original every-seed mission capability: **False**.
See `completion.json` for all separate checks; no best seed is selected.

The only fitting intervention is a tied-best-set log-probability loss, replacing
masked gain MSE. Architecture, initial tensors, exact sample-index sequence,
training packets, optimizer, clipping and 2,000-update budget match the control.
All 60 complete development missions are evaluated under the same feasible mask
and first-index argmax. Greedy's different native candidate/movement rule remains
contextual. Confirmation and final worlds remain unused. Logits are not calibrated
gains; their MSE is a diagnostic export, not comparable calibration evidence.
These descriptive seed/world means establish no significance or unique causal
mechanism. This is not trained DQN, REINFORCE or A2C or assignment completion.

Independent replay passed 5517 ranking rows, all 18000 new mission
steps and 240 paired comparisons. Every action matches exactly; checkpoint/data/
source bytes, initial tensors and sampling counts/digests were verified. Fixed
float32 tolerance: atol=1e-05, rtol=1e-06; gates/hashes/actions stay strict.
`comparison.csv` has every seed mean; `paired.csv` has every matched-world delta.
`summary.csv` reports means/sample SD across fitted seed means separately from
baseline world SD; repeated seed/world rows are not independent replicates.
`rankings/` and `missions/` preserve full compressed scores and action/reward/hash traces.

Full raw checkpoints are retained in Actions artifact
`frozen-40x40-loss-b6825bbdaa176277f76efb3dc84ffe334cabe289`; original training/control inputs
are in `frozen-40x40-pilot-7c9562aa6064f46d96cc9a163de3deb2ecaa3e19`
from run 38111928600. Artifact retention is 90 days. Download both artifacts and
extract their `runs/` folders into the repository, then replay to a fresh output:

```bash
python -m pip install -e '.[rl]'
python scripts/audit_scale_loss.py --run runs/scale-loss-2026-10-11 \
  --parent-root runs/scale-pilot-2026-10-11 --out runs/replayed-scale-loss
```

The prospective recipe is in `docs/scale-loss-protocol.md`. A negative result
stops this objective direction without extra fits. A positive result supports
designing actual RL comparison and separately reserved evaluation, not RL success.
