# Prospective matched drone-centered inspection comparison

Compare `current`, `absolute` and `centered` input arms before changing RL.
All use exactly the existing FleetNetwork dimensions: nine spatial channels,
24 scalar inputs, 125 outputs, the same parameter count and matched initial
tensors per seed. Original RL/model defaults stay unchanged. This experiment
exports a separate supervised format with its arm/config embedded and applies
the same observation transform inside forward during fitting and prediction.

`current` is the unchanged five world maps, three drone one-hots and base one-hot.
The derived arms retain all five original world maps, base one-hot and 24-vector.
They replace the three drone one-hots with three copies of an observation-derived
cell inspection field:

`(priority/9)*(1+2*burning+frontier)*min((age+1)/deadline,1)`.

Derive frontier from four-neighbour burning cells, positive observed fuel and
nonburning status; derive priority/age from their observed normalized maps using
the fixed environment configuration. No hidden state, targets, action coverage,
joint-action reward, return/charging decision or feasible mask enters the transform.
This field deliberately makes current per-cell inspection urgency explicit, but
does not supply action-gain labels or solve joint footprint/energy/overlap effects.

`absolute` supplies the same field in world coordinates in each live drone's
auxiliary plane. `centered` translates one copy per live drone to place its
position at `(grid_size//2, grid_size//2)`. Crop beyond the 20×20 auxiliary plane
and zero-pad outside the original grid; never wrap. Lost-drone auxiliary planes
are zero in both derived arms. Keep north/south/west/east orientation fixed.
Original world maps still retain the entire scene and vector coordinates are
unchanged. Thus `centered` versus `absolute` isolates the translation/cropping
treatment; `absolute` versus `current` captures the cell-field/input-channel change.
Do not attribute a centered-versus-current improvement solely to centering.

Reuse the exact prior target corpus, verifying dataset SHA-256/bytes, launch
configuration and archived case/gain/mask identities before fitting: training
worlds 0–39, 319 observed states/303 informative; reused development worlds
6000–6015, 127 observed/124 informative; 160 existing counterfactual states and
their fixed 32 urgency/32 position/32 no-op pairs. Prior early-termination
exclusions remain unchanged. The previously examined 6000-world set is reused
development evidence, not a fresh test. No new state enters training.

For prospective primary evaluation, collect fresh development worlds 6200–6215
using the same world-seeded masked-random and Greedy collectors and elapsed
steps 16/32/64/96: 128 planned natural states. Generate and persist this corpus
before fitting. Log real early termination and zero-range states. All arms see
the same training and evaluation data. Confirmation worlds 5800–5819 and earlier
final-test worlds 10000+ remain unopened in this experiment.

For each training seed 0/1/2 and each arm, fit a fresh network for exactly 2,000
Adam updates at .001, batch size 64 sampled uniformly with replacement from
the same 303 informative training states, clip .5, CPU with one Torch thread.
Retain the prior all-feasible, per-state normalized squared-error objective.
Within seed, the three arms must have bit-identical initial tensors and identical
sample-index sequences; record a sequence digest. Verify the new current arm's
final tensors and every training loss against the preceding diagnostic. No extra
parameters, target/data weighting, budget extension, checkpoint choice or RL
updates. Export initial/final weights only, with exact tensor/metadata audits.
Read evaluation predictions only after all nine fixed fits complete.

Primary outcomes are fresh-world normalized gain regret and tied-best match
under each state's existing feasible mask. Use all tied best actions and exclude
zero-range states from informative means, reporting denominators. Average
informative observed states within each fit, then report mean/sample SD across
three seeds and paired centered-minus-control differences within seed. Also
report regression error, training/reused-development results, initial controls,
the shared training-action-mean control and analytic uniform-random expectations.
These states/worlds are not independent fitted models and paired differences
are descriptive, not significance tests.

The prospective capability criterion requires every centered fit to have:
training regret <= .05; fresh-world regret <= .20; fresh-world best match >= .25;
and fresh regret lower by >= .10 than both its matched current and absolute arms.
Keep each condition separately. This is a diagnostic criterion, not a long-term
mission gate. Secondary urgency/position results use the original common-feasible
pair rules and include deployment-mask changes, order flips and no-op controls.
These artificially constructed maps remain separate from natural evaluation.

Publish this protocol and source before generation/fitting. Archive exact source,
input/case/dataset/checkpoint identities, transforms and parameter/init/sampler
audits, all losses, numeric rows and complete prediction vectors. Use lossless
float32 NumPy archives (`allow_pickle=False`), without rounding; preserve original
gain/mask records in compressed JSON. Check numerical recomputation and transform
replay/export fidelity before merging through `sr` to `master`.

Good primary results would support considering an RL experiment using this input
hypothesis, not establish mission improvement or general success at 40×40. A
negative comparison does not prove drone-centered representations cannot help;
it tests this exact transform/corpus/budget. No default representation promotion,
RL warm start, extra fitting or 40×40 experiment follows automatically.
