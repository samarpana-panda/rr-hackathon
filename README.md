# Who Delivers When It Matters: analysis files

Prepared by Samarpana Panda (4th-year student, Mukesh Patel School of Technology Management and Engineering) for the CRP SupeRR Selector Season 2 challenge.

These are the scripts and outputs listed in the report's appendix ("Appendix: reproducibility"). They run end to end on the supplied data pack and use fixed random seeds, so a re-run reproduces the final lists.

## Folders

- `scripts/`: the ten analysis steps, plus the two shell scripts that run them in order.
- `inputs/releases_2027.json`: the retention, release, trade, opt-out, retirement and ban lists used for the auction filter. The data pack doesn't contain these, so I compiled them from media reports.
- `outputs/`: the main results of the final run.

## Scripts, in run order

| Step | Script | What it does |
|---|---|---|
| 1 | `stakes_nrr.py` | Match stakes per team per match. Simulates the IPL league (4,000 draws, seed 7) with a net-run-rate tiebreak, enumerates bilateral series exactly and applies the knockout rules. |
| 2 | `pressure.py` | Cleaning, match state, targets, conditions factor, win-probability models and situational components. |
| 3 | `learn_pressure.py` | Consequence network: 4→12→12→1, written with autograd. |
| 4 | `difficulty.py` | Monotone difficulty models for batting and bowling, controlling for player level. |
| 5 | `blend.py` | Pressure = stakes^0.363 × (0.65 × consequence + 0.35 × difficulty). Marks the top 25% of balls per competition as high pressure. |
| 6 | `superover.py` | Adds the 38 super-over balls as extreme pressure. |
| 7 | `perf.py` | Expected values, value per ball (chase wickets valued by win probability), wicket-luck adjustment, ratings, score and eligibility. It runs once per competition list. |
| 8 | `reliability.py` | 300-draw match bootstrap (seed 11). |
| 9 | `avail.py` | Auction availability from squads, retentions, releases and trades. |
| 10 | `compare_sens.py` | Compares the six alternative settings with the final lists. |

`run_all.sh` runs steps 1 to 9 and the sensitivity variants. `run_boot.sh` runs the bootstraps.

## Running it

The scripts use the absolute folder paths from my working environment. Before running, replace them with your own, for example with find-and-replace:

- `/home/claude/data/pack`: the folder holding the supplied CSV files (`ball_by_ball.csv`, `matches.csv`, `competitions.csv`, `competition_team_player_mapping.csv`, `playing_eleven.csv`, `teams.csv`).
- `/home/claude/final_nrr`: the output folder for the run.
- `/home/claude/work`: the folder holding `avail.py` and `releases_2027.json`.
- `/home/claude/final_run`: the previous run's output folder, used as a reference in two places:
  - **`stakes_nrr.py`** reads innings totals from a cleaned ball file when it builds the net-run-rate margins. Those totals depend only on runs, balls and wickets, not on stakes. On a clean run, run `pressure.py` once first, or point this step at `balls_pressure.parquet` in the output folder.
  - **`compare_sens.py`** uses it as one of its comparison variants (the lists before the net-run-rate tiebreak).

Python 3 needs numpy, pandas, scikit-learn, autograd and pyarrow:

```bash
pip install numpy pandas scikit-learn autograd pyarrow
bash scripts/run_all.sh
python3 scripts/compare_sens.py
```

The full run takes about 30–60 minutes on a laptop. Most of that time goes on the bootstrap and the sensitivity variants.

## Outputs

| File | Contents |
|---|---|
| `submission_top10_ipl.csv` | The submitted top 10 batters and bowlers: IPL list, super overs included, available players only. |
| `final_available.csv` | Every ranked player in every list, with availability status, reason and availability rank. |
| `ipl/`, `international/`, `a_level/` | The final ratings (`rank_bat.csv`, `rank_bowl.csv`) and the bootstrap summaries (`boot_bat.csv`, `boot_bowl.csv`) for each competition group. |
| `stakes.csv` | Stakes multiplier per team per match. |
| `sensitivity_report.csv` | Overlap and rank correlation of each alternative setting with the final lists. |

The intermediate ball-level files (parquet, about 200,000 rows each) are left out because of their size. The scripts regenerate them.
