# Drone RL comparison and assignment status

20 by 20 development results and 40 by 40 main study    11 October 2026

**The masked learners kept all three drones alive in the recorded evaluations, but they did not demonstrate strong learned surveillance. The frozen 40×40 experiment is complete. The consolidated technical notebook is complete; contributor attribution, independent assessed preparation, the recorded presentation and report remain. A 60×60 run is optional under the supplied brief.**

This is a private explanation and planning aid. The supplied assignment brief prohibits generative AI in submitted work; the group must independently prepare and understand its assessed submission. [1]

## What passing means

DQN, REINFORCE and A2C are three alternative controllers. Each controller commands the entire three-drone fleet. Drone survival, useful surveillance and completing the experiment are separate outcomes.

| Outcome | 20×20 historical masked study | 40×40 frozen masked study |
| --- | --- | --- |
| Drone survival | All three learners report zero mean losses | All 180 learner missions ended with all three drones alive |
| Learning performance | No convincing freshness advantage over masked random | 0 of 9 fitted policies passed either complete performance gate |
| Study completion | Historical test and tuning tables archived | All nine fits and 260 final missions completed and audited |

Zero losses does not establish that every drone finished at base. The recorded safe-return metric counts return events, which can happen repeatedly; it is not an all-drones-at-base success rate.

## Freshness comparison

Freshness measures how recent inspections remain, weighted by location priority and averaged over the complete mission. Higher is better. Percentages below express the 0–1 freshness score, not the percentage of land covered. [2, 3]

| Controller | 20×20 freshness | 40×40 freshness |
| --- | ---: | ---: |
| DQN | 14.59% | 8.44% |
| REINFORCE | 13.34% | 6.90% |
| A2C | 13.69% | 8.43% |
| Greedy | 32.20% | 22.08% |
| Sweep | 11.68% | 6.49% |
| Random | 9.45% | 5.96% |
| Masked random | 14.50% | 8.38% |

40×40 learner means average 20 missions within each fitted seed, then the three seed means. Baselines average 20 missions. These are two different studies; lower 40×40 scores do not isolate a grid-size effect or prove a decline caused by the corrected code.

## What improved and what remains weak

### Safety improved with feasibility rules

The historical 20×20 unmasked test reports mean losses of 1.70 drones for DQN, 2.00 for REINFORCE and 3.00 for A2C per mission. The masked versions report zero. In the 40×40 main study, each learner preserved the full fleet in 60 of 60 missions. [2, 3]

The feasibility mask uses battery and collision rules to restrict available commands. At 40×40, masked random also lost zero drones, while unmasked random lost all three in every one of its 20 missions. This control makes the contribution of those rules visible. Survival cannot be credited entirely to learned return or recharge coordination.

### Surveillance did not improve beyond the mask consistently

| 40×40 controller | Return ↑ | Freshness ↑ | Overdue ↓ | Losses ↓ |
| --- | ---: | ---: | ---: | ---: |
| DQN | -24.79 | 8.44% | 86.87% | 0 |
| REINFORCE | -25.81 | 6.90% | 89.40% | 0 |
| A2C | -24.83 | 8.43% | 86.95% | 0 |
| Greedy | -16.10 | 22.08% | 65.56% | 0 |
| Sweep | -26.07 | 6.49% | 90.12% | 0 |
| Random | -42.87 | 5.96% | 92.10% | 3 |
| Masked random | -25.46 | 8.38% | 88.60% | 0 |

DQN and A2C are close to masked random in freshness. Their differences are only +0.059 and +0.055 percentage points respectively. REINFORCE is 1.478 percentage points below masked random. Greedy has the best mean return and freshness in this comparison. These descriptive differences are not statistical significance claims.

The predeclared learning gate requires, for every fitted seed, freshness at least 0.01 above masked random, return at least 0.5 above it, and no worse losses or fleet failures. The separate Greedy gate includes at least 90% of Greedy freshness and comparable overdue and safety metrics. Every one of the nine fits failed both complete gates. [3]

### The comparison is more traceable

The 40×40 study froze corrected code, settings and a 300-episode budget before fitting. It used three independent training seeds per algorithm, final checkpoints only, and 20 fresh scenarios per fit. All nine fits completed 90,000 interactions each, for 810,000 training interactions in total. The audit replayed all 75,911 evaluation steps without neural-score drift. Earlier negative experiments were preserved. [3, 4]

### Why the grid comparison is historical

The development configuration uses horizon 160, battery 80, recharge 5, sensor radius 1 and inspection deadline 25. The main study uses horizon 300, battery 160, recharge 8, radius 2 and deadline 30. Test worlds and code revisions also differ. The October 10 correctness fixes did not rerun the historical 20×20 study. [4, 5]

A controlled claim about scaling would require both grids to be run under the same corrected implementation and a declared scaling protocol. That extra comparison is needed only if we want to make that particular claim; the present results support an honest historical comparison.

## Assignment completion check

The supplied COMP6008 brief requires three implemented RL methods, multiple runs and tuning, analysis, a single structured notebook, a recorded presentation and a report. It explicitly allows failed learners when the failure is scientifically explained. It does not prescribe 40×40 or 60×60, a minimum training budget, or beating Greedy. [1, pp. 2–5]

| Requirement | Current status |
| --- | --- |
| Explicit MDP and simulator | Implemented with documented state, actions, rewards and transitions |
| Three RL methods | DQN, REINFORCE and A2C implemented and run |
| Multiple runs and tuning evidence | Historical tuning preserved; nine frozen main fits completed |
| Unseen tests and baseline comparison | Completed at 40×40 with Greedy, sweep, random and masked random |
| Single consolidated technical notebook | Complete with saved outputs, the 40×40 evidence and verified review execution; contributor attribution and independent assessed preparation remain |
| Pseudocode linked to implementation | Source-resolved mappings and complete implementations included and verified in the main notebook |
| Recorded presentation | Not established as completed; maximum 20 minutes |
| IEEE report and contributions | Not established as completed; final writing and attribution remain |
| Coordinator topic approval | Retain or confirm the actual approval record |

### The main remaining work is the assessed deliverables

The consolidated [main notebook](../Wildfire_Drone_Fleet_clean.ipynb) now contains the frozen 40×40 results, historical evidence, learning curves, uncertainty units, failure interpretation and complete implementation listings. Its 42 code cells completed without errors in an independent clean-kernel GitHub Actions rerun. The separate Colab helper retains the older training workflow. The brief warns that results used in the presentation or report must also appear in the notebook. Check this before submission. [1, p. 5]

The technical notebook explains each learner and connects pseudocode to the actual source. It documents the MDP and feasibility mask, distinguishes safety from learned mission quality, and retains reproduction instructions and dependencies. The group still needs to independently understand and prepare its assessed submission. Describe convergence and sample efficiency within the fixed budget without claiming that 300 episodes prove convergence.

The report must contain a project description and critical review, proposals for improvements and extensions, and individual contributions. It is limited to eight A4 pages in IEEE two-column format; references, code listings and mathematical derivations go in separate appendices. [1, pp. 7–11]

Part 2 needs at least one proposed improvement and one distinct future-work direction per group member, supported by at least six academic sources. For a three-person group this means at least three improvements and three extensions. Hyperparameter changes do not count as improvements. The brief recommends at least 40% of the report for this part. These are justified proposals; executing all of them is not required. [1, p. 10]

The supplied brief lists 23 October 2026 for code and presentation, and 30 October 2026 for the report. Check Blackboard for later amendments because the brief says it may be superseded. The earlier technical planning draft proposed five seeds and 100 test missions; the executed scope is three seeds and 20 test missions, so disclose that reduction rather than implying that the draft protocol was fully met.

## Recommended next step and use of Colab

**Proceed to contributor attribution, independent assessed preparation, and planning the presentation and report. Preserve the completed main study as the fixed-budget result. Further loss tuning or a larger grid is not required to turn these negative findings into a scientifically useful assignment analysis.**

### When another experiment would answer a useful question

| Option | Purpose | Priority now |
| --- | --- | --- |
| Finalize current evidence | Explain observed safety, weak freshness and failure causes | Recommended |
| Corrected matched 20×20 run | Support a controlled 20×20 versus 40×40 scaling claim | Optional if making that claim |
| Longer 40×40 training | Test whether a larger fixed budget changes the outcome | Optional separate study |
| 60×60 training or evaluation | Measure additional scalability or transfer | Future extension |

A 60×60 run would not resolve why the learners currently remain close to masked random at 40×40. A useful optional GPU study would first ask a specific question on the existing task, such as whether a larger training budget improves freshness consistently across seeds. More compute is an experiment resource, not evidence of learning or an explanation for failure.

If the group independently chooses a new study, predeclare its code, mask, settings, training budget, seeds, evaluation mode and success criteria. Train all three methods with comparable budgets; include the same baselines. Keep the completed 300-episode study unchanged and use a new output folder. Choose settings using separate validation worlds and open a new final test set only after all fits finish. The 22000–22019 worlds are now used test evidence and should not guide new choices.

### GPU speed should be measured

The recorded 20×20 T4 benchmark found DQN faster on CUDA (106 versus 88 steps per second), while REINFORCE and A2C were faster on CPU in that short probe. This is a throughput measurement, not a learning comparison or a prediction for 40×40. The simulator still runs on CPU. Benchmark the actual workload before committing the GPU. [6]

Colab GPU availability and runtime limits vary, and selecting a GPU runtime does not automatically move computations to the GPU. Use explicit device settings and durable checkpoints if an optional run proceeds. No new fits were launched for this comparison. [7]

### Evidence used

[1] COMP6008 Project Guidelines(2).pdf, 29 July 2026. Supplied assignment brief; pages 1–5 and 7–11. Also checked the two supplied wildfire proposal documents.

[2] Historical 20×20 test table, [docs/experiments/final-2026-10-05/test_summary.csv](experiments/final-2026-10-05/test_summary.csv). Its aggregate rows are historical evidence; no equivalent complete replay audit is claimed.

[3] Frozen 40×40 archive, [docs/experiments/rl-main-2026-10-11](experiments/rl-main-2026-10-11). episodes.csv, summary.csv, completion.json, audit.json and README.md.

[4] [docs/scale-rl-protocol.md](scale-rl-protocol.md) and [configs/rl-main-frozen.json](../configs/rl-main-frozen.json). Frozen study code 855e5510ff46b16082ab7d551e7c228acab6f572.

[5] [docs/verification-2026-10-10.md](verification-2026-10-10.md) and [configs/development.json](../configs/development.json). Correctness fixes and development settings.

[6] [docs/colab1000-results.md](colab1000-results.md). Historical unmasked 1,000-episode T4 pilot and bounded CPU/CUDA benchmark.

[7] Google Colab FAQ, accessed 11 October 2026. [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

Repository evidence: github.com/SayuruRehan/wild-drone-appeared. Final study merged in PR 33; merge commit 139e74b5db88e48dbed873b2b62168a525ac1db8.
