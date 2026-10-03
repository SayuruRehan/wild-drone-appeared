# Repository workflow

- Use `sr` as the only development branch. Do not create additional feature, fix,
  analysis, or worktree branches.
- Make changes and commits on `sr`; open pull requests from `sr` to `master`.
  Do not commit development changes directly to `master`.
- Before starting work, synchronize `sr` with the latest `master` and preserve any
  unmerged work. After a PR is merged, synchronize `sr` again so it has no leftover
  ahead/behind count before the next task.
- Prefer a merge commit for this continuing branch so its existing commits remain
  in `master`'s history and synchronization can fast-forward. If a PR is squash
  merged, verify that all intended changes are preserved before realigning `sr`.
- The repository owner's instruction to merge ready PRs remains in effect. Verify
  the reviewed revision and required checks before merging.
