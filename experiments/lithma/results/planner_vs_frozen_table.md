| Policy | Freshness ↑ | Overdue ↓ | Return ↑ | Energy | Losses ↓ | ms/decision |
|---|---:|---:|---:|---:|---:|---:|
| dqn | 0.0844 ± 0.0025 | 0.8687 ± 0.0063 | -24.79 ± 0.25 | 698 | 0.00 | 0.43 |
| reinforce | 0.0690 ± 0.0073 | 0.8940 ± 0.0157 | -25.81 ± 0.59 | 675 | 0.00 | 0.65 |
| a2c | 0.0843 ± 0.0116 | 0.8695 ± 0.0223 | -24.83 ± 0.92 | 723 | 0.00 | 0.69 |
| greedy | 0.2208 ± 0.0266 | 0.6556 ± 0.0377 | -16.10 ± 1.50 | 783 | 0.00 | 3.17 |
| planner | 0.2207 ± 0.0269 | 0.6668 ± 0.0370 | -16.55 ± 1.49 | 757 | 0.00 | 153.69 |
| sweep | 0.0649 ± 0.0154 | 0.9012 ± 0.0200 | -26.07 ± 0.81 | 465 | 0.00 | 0.21 |
| random | 0.0596 ± 0.0178 | 0.9210 ± 0.0236 | -42.87 ± 1.12 | 483 | 3.00 | 0.01 |
| masked_random | 0.0838 ± 0.0243 | 0.8860 ± 0.0331 | -25.46 ± 1.33 | 675 | 0.00 | 0.21 |

Mean ± sample SD. Learners: SD across 3 training-seed means; baselines and planner: SD across 20 worlds.
Frozen rows are copied from the archive; their ms/decision was measured on a GitHub runner, the planner on this machine with 5 parallel workers, so timing columns are indicative only.
