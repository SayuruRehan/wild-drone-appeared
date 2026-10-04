# Conditional action-learning diagnostic

Continue the approved correction sequence before scaling. The original network can
memorize a complete 20×20 teacher trajectory but fails across worlds. The matched
input and recovery-collection treatments did not establish useful autonomous behavior.
This study tests a conditional output-head recipe, using the existing supervised fitting
flow. It does not change RL defaults, observations, physics, rewards or deployment authority.

Before the full comparison, run a separate competence gate with training world 0 alone,
the same five validation worlds, three fit seeds per arm and 2,000 updates per fit. Use
only one trained round; final and agreement-selected exports then refer to that round.
The runner persists 65 validation missions including Greedy. Separately evaluate the six
final learners and Greedy on trained world 0: seven additional training-world flights.
Both arms must reach 100% training joint agreement and exactly match the non-timing
Greedy training-world mission metrics, including return/recharge, before the full study.
Keep this gate separate from validation and the full comparison; it establishes trajectory
memorization only. If it fails, investigate it rather than proceed to the full corpus.

Use the full fixed Greedy corpus from training worlds 0–99 (15,864 real states) and
teacher validation states on worlds 5000–5004 (800 states). Retain failures and all labels.
Both arms receive these same tensors, training sampler seeds and initial convolution/
encoder tensors, the original nine map channels and 24-value vector. Their output-head
initializations and parameter counts differ; report those differences. This is a comparison
of complete output-head recipes, not a pure equal-capacity factorization ablation.

`joint` retains the exact original FleetNetwork: 128 encoder units and one 125-way
categorical head. `conditional` keeps that encoder and replaces the output head with
p(a0|state), p(a1|state,a0), p(a2|state,a0,a1), in fixed drone order 0/1/2. The first head
is linear 128→5; the second and third concatenate preceding command one-hots and use
64-unit ReLU layers, then five logits. Initialize the common body by constructing the
original network under the same seed before replacing the conditional head. Do not
retain unused original-head parameters or claim equal full-model initialization.

Train the conditional arm with the sum of three teacher-context cross-entropies, exactly
complete joint negative log likelihood, without division by three. The teacher's preceding
commands supply contexts during fitting only. At inference score every one of the 125
combinations by the sum of conditional log probabilities and choose global joint argmax.
Do not greedily decode drones in sequence, filter actions, query Greedy, override return/
charging decisions or use privileged state at deployment. Normalize each conditional
categorical distribution; enumerate joint actions in a0*25+a1*5+a2 order.

Six CPU fits use seeds 0/1/2, Adam rate 0.001, batch64 uniformly sampled with replacement,
norm clip0.5, five continuous rounds of 4,000 updates (20,000 per fit), with unchanged
optimizer/sampler state. Greedy's action label remains one heuristic choice with ties;
agreement is not reward optimality. Do not add curriculum, recovery collection, class
weighting, extra inputs or broader training worlds in this comparison.

Save initialization, rounds1–5, final and agreement-selected exports with byte, model,
shared-encoder, dataset, metadata and source identities. Select maximum validation joint
agreement among trained rounds1–5 with earliest ties; initialization is ineligible. Never
select by autonomous return. Validation affects measurement/selection only, never training.
Use a fresh output folder with no resume interface; prevent accidental overwrite.

Evaluate all five trained rounds plus selected on the five reused validation worlds,
with one shared five-world Greedy reference:185 persisted missions. The example trace is
fixed as conditional fit0 final round5 world5000. Check paired initial encoder hashes and
persisted mission/baseline counts. Final worlds10000+ remain unused. Exact old trained
weights may differ across host runtimes; old results are historical context, not current
paired replicates. A small same-runtime test checks the joint arm matches the original
fixed fitter exactly.

Primary evidence is paired final conditional-minus-joint autonomous return, fleet losses,
inspection freshness, safe returns, charging completions and actual critical return moves.
Average worlds within each fit, then mean/sample SD over three fits; pair differences by
seed. Greedy SD is across five worlds. Nested checkpoints and reused validation worlds are
exploratory tuning evidence, not independent replicates, significance or final generalization.
Higher label accuracy or survival without inspection does not meet the improvement gate.

Archive all raw numerical/provenance records and original collection CSV bytes. Verify all
exports and source/dataset identities, independently replay missions and audit aggregates,
pass review and required CI, then merge sr→master and synchronize sr. Scale compute, data
or grid only after useful autonomous behavior improves; a failed head recipe remains a
published diagnostic and is not promoted into the RL defaults.
