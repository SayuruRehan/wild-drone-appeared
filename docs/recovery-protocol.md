# Teacher-assisted recovery collection diagnostic

The approved second correction improves learner-relevant experience before scaling. The
matched derived-input study did not produce useful autonomous behavior, so retain the
original FleetNetwork/24-vector transform and 125 joint outputs rather than promote that
input treatment. Fix this collection recipe before examining outcomes. It is supervised
imitation, not an RL advantage or an intervention supplied during evaluation.

Collect the full real Greedy corpus on worlds 0–99 upfront, plus teacher validation states
on 5000–5004. Use unchanged 20×20 development physics and 160-step horizon. All six fits
(three seeds 0/1/2, two arms) share this initial corpus, initial tensors, sampler seeds and
first 4,000 updates. Adam rate is 0.001, batch size 64 uniformly sampled with replacement,
norm clip 0.5. Continue optimizer/sampler state for five rounds of 4,000 updates: 20k per fit.

Before rounds 2–5, revisit training worlds 0–24, 25–49, 50–74 and 75–99. Greedy labels every
real pre-action state; the preceding-round learner supplies proposals. `learner` executes
only learner proposals. `assisted` executes the teacher's joint action whenever any alive
drone is away with battery minus necessary return energy at most six move costs, or any
alive drone is at base below full charge. Otherwise it executes the teacher with probability
0.5 and the learner with probability 0.5. Draw coins from an independent collection RNG at
every step, including gated steps; seed it as 100000+100*fit_seed+round. Never use live
environment RNG for that choice. Query Greedy once per state and record actual teacher/
learner proposals, execution source, gates, coins and return/charge/loss transition events.

This treatment changes a complete collection recipe (mixture plus energy/charging gates);
it does not isolate each component's effect. Assistance is joint and can affect all drones.
It is neither an optimal recovery oracle nor a guaranteed safety controller: Greedy can
fail, collisions can consume reserve, and previously unreachable states remain unreachable.
Retain all real labels, including failures. Do not teleport, restore battery, invent labels
after termination, filter failures or manufacture successful recovery. Revisited worlds may
contain duplicate teacher states; retain those too. Append all added states to the fixed
initial corpus and train uniformly. No weighting, new worlds, action factorization, reward
changes or RL updates belong here.

The two arms match update counts and queried world IDs, not oracle-query counts, labeled
states, successful charging or state coverage: early termination and assistance change
trajectory lengths/distributions. Record those differences. The initial teacher corpus is
shared; the revisited worlds are training worlds, never validation. A previous fixed-corpus
20k run is historical context, not an extra matched arm; compare its initial/4k tensor prefix
when available to check unchanged fitting. Prior pure-learner 5k results used different data
and update schedules, so are not pooled as current replicates.

Save initial, rounds 1–5, final and agreement-selected exports with exact tensor/metadata,
byte/model/data/source identities. Selected maximizes validation joint agreement among rounds
1–5, earliest ties, initialization ineligible; it is biased by repeated validation and not
chosen by autonomous return. Deploy both arms as unshielded argmax learners only: the teacher,
coins and gates are absent. Evaluate all trained rounds plus selected policies on the five
validation worlds and one shared five-world Greedy reference: 185 missions. Round checkpoints
are nested, not independent fitted models. Final worlds 10000+ remain unused.

Primary evidence is the paired final-arm difference in autonomous return, full-fleet losses,
freshness, safe returns, completed charging and actually executed critical return moves.
Better teacher-assisted collection or higher label accuracy alone is insufficient. Average
worlds within each fit, then mean/sample SD over the three seeds; pair arm differences by
seed. Baseline SD is across five scenarios. Preserve collection quality and actual execution
statistics separately from autonomous deployment outcomes. This is exploratory development
validation, not final generalization or significance evidence.

Fresh output folders only; no resume interface. Verify common initialization/first-round
tensors and persisted mission/baseline counts. Archive raw numeric and provenance records
and original collection CSV bytes, independently replay collection and missions, and pass
review/CI before merging sr→master. Scale the grid or compute only after useful autonomous
inspection/return/charging improves; survival without inspection is not success.
