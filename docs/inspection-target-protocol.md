# Prospective all-action inspection capability diagnostic

Test whether the existing FleetNetwork can learn immediate inspection rankings
under direct supervision. This is separate from RL and from earlier imitation of
one Greedy command per state. Keep nine spatial channels, the 24-value vector,
125 joint outputs, action feasibility and full 20×20 development physics unchanged.
Use grid 20, horizon 160, battery 80, recharge 5, sensor radius 1, deadline 25,
maximum fuel 100, and all other environment defaults from the coordination pilot.

Publish this protocol and runner before collection/fitting. Collect natural
pre-action states on training worlds 0–39 and held-out development worlds
6000–6015, using two model-independent collectors: masked random seeded by world
and Greedy. Snapshot elapsed steps 16, 32, 64 and 96 (320 planned training and
128 planned held-out states). Log missing snapshots, including real early mission
termination. Retain all observed states; exclude zero feasible-gain-range states
only from fitting and informative ranking means, with denominators reported.
Split by world, not by snapshot. No state or target from held-out worlds enters
fitting. Worlds 5800–5819, prior diagnosis worlds 5900–5907 and final-test worlds
10000+ remain unopened in this experiment.

Targets are the exact original one-step freshness_gain component for every
feasible action, computed on isolated environment clones. Count overlapping
coverage once and include next inspection-age increment, motion, energy and
charging. Normalize feasible gains within each state to [0,1] using its minimum
and range, retaining tied best actions. Infeasible targets are excluded. This
normalization supplies known target scale during supervised training only; it
requires no target oracle when reading a trained model's scores.

Three fresh networks use seeds 0, 1 and 2. Supply the same fixed natural training
corpus upfront. Each fit receives exactly 2,000 Adam updates at rate .001,
batch size 64 sampled uniformly with replacement from informative training
states, gradient norm clip .5, CPU with one Torch thread. Minimize squared error
averaged over feasible actions within each state, then averaged over batch
states. No pairwise/teacher cross-entropy, new features, class weighting, reward
change, RL updates, initialization from prior weights or budget extension.
Record every batch loss. Export initial and final supervised checkpoints with
exact tensor roundtrip, format, budget, byte-size and SHA-256 checks. Export no
intermediate selectable policies; inspect held-out predictions only after all
three fixed fits finish. No checkpoint selection or retuning follows these data.

Compare initial and final scores against a state-blind training-action-mean
control: for each action, average normalized target over informative training
states where it is feasible; use zero for unseen actions. This control sees no
held-out targets. Also report analytical uniform-random expectations and the
exact one-step inspection reference. Read model scores and select first masked
argmax without adding target information at prediction time.

Primary measures are natural training and held-out normalized gain regret,
tied-best match rate and masked regression error. Exclude zero-range states from
ranking/error summaries, and show all counts. Average over observed informative
states within each fit, then report mean/sample SD across three
fit seeds; states are not independent training seeds. The descriptive capability
criterion requires, in every fit, training regret <= .05 and held-out regret
<= .20, improvement >= .15 over both that fit's initialization and the shared
state-blind control, and held-out tied-best match >= .25. These predeclared
thresholds are diagnostic criteria, not significance tests or mission gates.

Construct the previously defined balanced urgency and one-cell position probes
on a fixed subset of held-out natural states: worlds 6000–6007, both collectors,
steps 16 and 64 (32 planned cases). Use the existing model-blind pair-selection
rules from the ranking diagnostic, and construct all interventions before
fitting. Archive missing/ineligible reasons, input hashes, masks and full gains.
Compare within the common feasible set, separately from deployment-mask changes.
Record score/order/top changes, both sides' regret and tied-best match, plus
identical-state no-op controls. Artificial urgency distributions may be outside
natural training support. These secondary contrasts do not enter training,
checkpoint selection or the primary capability criterion.

No autonomous policy or mission-performance improvement is claimed. Good training
and poor held-out rankings suggest a generalization gap under this corpus/budget;
good held-out rankings demonstrate capacity for a narrow immediate target under
supervision, not that RL can learn long-term coordination. Failure to fit is not
proof of architectural impossibility. Review alongside prior imitation/budget
studies without pooling their models as extra replicates. No representation
change, warm start, larger fitting budget or 40×40 run follows automatically.

Archive source hashes, environment/protocol, exact data identities, observation
hashes, full action gains/masks and score vectors, every numeric row, per-fit and
aggregate results, checkpoint identities, completion/verification and checksums.
Compressed JSON preserves raw vectors without rounding. Binary observation
datasets and weights stay in ignored local runs; deterministic collector replay
and identities bind them. Check recomputation and export fidelity before merging
through the sole development branch `sr`.
