# Mission Control — GitHub Pages deployment

This is a static deployment of the React/Vite frontend under
`https://sayururehan.github.io/wild-drone-appeared/`.

## One-time repository setup

1. Ensure the account's plan supports Pages on **private** repositories. Otherwise make a deliberate decision about visibility; do not change repository visibility simply to enable Pages.
2. Repository **Settings → Pages → Build and deployment → Source → GitHub Actions**.
3. After merging the deployment PR, open **Actions → Deploy mission control to GitHub Pages** and run the workflow if it has not triggered automatically.
4. Verify the page and mission replay data load on the deployed URL.

The workflow checks out `master`, runs `npm ci` and `npm test` in `web`, and builds with `VITE_BASE_PATH=/wild-drone-appeared/`. Assets and static replay/evidence data are served from the repository's Pages subpath.

## Runtime behavior

The overview, research, results, and mission explorer operate from committed static evidence, including gzipped replay bundles. No Python server is deployed. The Simulation Lab requires the separate local Python API at `/api` and remains unavailable on the Pages site; this is intentional, not a failed frontend deployment. The UI indicates that the local simulator is required.

For local development, the default base remains `/`, and the existing Vite proxy for `/api` is unchanged.

## Verification

- Pages entrypoint and JS/CSS assets load without 404s
- `/wild-drone-appeared/data/evidence.json` is reachable
- Select a Greedy/DQN/A2C mission and verify a gzipped replay loads
- Hash navigation works on reload
- Simulation Lab shows the local-simulator-required state on Pages
