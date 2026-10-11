# Stach Fantasy Football HQ — 2026

A responsive sports-style dashboard for ESPN fantasy league **2108436489**.

## What works

- Open `index.html` locally to see a **clearly labeled preview** with all 10 teams.
- Once connected, standings, roster summaries, average points, projected records, championship probabilities, playoff odds, byes, and projected playoff bracket come from a **Tuesday ESPN data snapshot**.
- The remaining fantasy schedule is **used in the simulation, but not shown** on the website.
- The simulation runs 12,000 Monte Carlo seasons each Tuesday and includes first-round byes and a three-round, six-team bracket.
- If the ESPN fetch fails, the GitHub Action fails rather than publishing invented results. The last successful snapshot remains on the website.

## Put it online (GitHub Pages)

1. Create a new **GitHub repository** (public is simplest for GitHub Pages).
2. Upload **the contents of this folder** to the repository root, including `.github/workflows/update.yml`.
3. Under **Settings → Pages**, select **Deploy from a branch**, choose `main` and `/ (root)`, and save.
4. Under **Actions → Update Stach Fantasy Dashboard**, choose **Run workflow** once to attempt the first ESPN sync.
5. After a successful run, open your Pages address: `https://YOUR-USERNAME.github.io/YOUR-REPO/`.
6. The workflow then runs each **Tuesday at 15:00 UTC** (10 AM EST / 11 AM EDT). GitHub scheduled workflows may run late and can be disabled after long inactivity.

### If the league is private

The ESPN unofficial API may return **401/403** for private leagues. This code supports the `ESPN_S2` and `ESPN_SWID` environment variables in **GitHub repository secrets**, but these are sensitive login session credentials. **Do not send them to ChatGPT, put them in files, or commit them to GitHub.** Ask a trusted adult or league commissioner to handle secure setup, or make the league publicly viewable if appropriate.

## Accuracy and caveats

- This is a **fan project**, not an official ESPN app. The unofficial ESPN API may change or stop working.
- The ESPN endpoint was **not reachable from the development environment**, so the first live sync is **not yet verified**.
- Team projections blend scoring history and available roster projection information. They **do not** guarantee a complete injury, bye, or waiver model.
- The site uses ESPN's actual schedule internally, not the illustrative schedule previously discussed. The dashboard hides the schedule as requested.
- Week 15/16/17 playoff round labels assume the league's six-team playoffs use those weeks; confirm in ESPN settings.
- A data refresh is a **snapshot**, not a live game tracker. Clicking Refresh reloads the latest snapshot but does not invoke ESPN or run the simulation.
- GitHub Actions and GitHub Pages may require account configuration. A file alone cannot activate hosting or scheduling.

## Files

- `index.html` — dashboard with clearly labeled preview fallback
- `update.py` — ESPN fetch, scoring model, Monte Carlo simulations
- `data/dashboard.json` — generated on first successful sync
- `.github/workflows/update.yml` — weekly automation
- `requirements.txt` — updater dependencies
