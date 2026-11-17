"""Sensitivity report: each variant vs the final (NRR) baseline, with super overs (v8).
For every list: overall top-10 overlap, Spearman rank correlation on players ranked in both, and the overlap of the
available-at-auction top 10 (same availability rules)."""
import subprocess, pandas as pd

BASE = '/home/claude/final_nrr'
VARIANTS = {'Before NRR (17 Nov final)': '/home/claude/final_run',
            'Consequence/difficulty 55/45': f'{BASE}/sens/a0.55', 'Consequence/difficulty 75/25': f'{BASE}/sens/a0.75',
            'HP cut-off top 20%': f'{BASE}/sens/q0.8', 'HP cut-off top 30%': f'{BASE}/sens/q0.7',
            'Stakes power 0.25': f'{BASE}/sens/g0.25', 'Stakes power 0.5': f'{BASE}/sens/g0.5'}
LISTS = [(c, s, i) for c in ('ipl', 'intl', 'alevel') for s, i in (('bat', 'striker_id'), ('bowl', 'bowler_id'))]

for root in VARIANTS.values():
    if root != '/home/claude/final_run':
        subprocess.run(['python3', '/home/claude/work/avail.py', root, f'{root}/final_available.csv'], capture_output=True)
av_base = pd.read_csv(f'{BASE}/final_available.csv')

rows = []
for name, root in VARIANTS.items():
    av = pd.read_csv(f'{root}/final_available.csv')
    for c, s, idc in LISTS:
        b = pd.read_csv(f'{BASE}/v8/{c}/rank_{s}.csv'); v = pd.read_csv(f'{root}/v8/{c}/rank_{s}.csv')
        m = b[[idc, 'rank']].merge(v[[idc, 'rank']], on=idc)
        top = lambda d: set(d.sort_values('rank')[idc][:10])
        f = lambda d: set(d[(d.version == 'v8') & (d.comp == c) & (d.side == s) & (d.avail_rank <= 10)].player_id)
        rows.append(dict(variant=name, list=f'{c} {s}', top10_overlap=len(top(b) & top(v)),
                         spearman=round(m['rank'].corr(m.rank_y if 'rank_y' in m else m['rank'], method='spearman'), 3) if False else
                         round(m.iloc[:, 1].corr(m.iloc[:, 2], method='spearman'), 3),
                         avail_top10_overlap=len(f(av_base) & f(av)), n_both=len(m), n_base=len(b), n_var=len(v)))
r = pd.DataFrame(rows)
r.to_csv(f'{BASE}/sensitivity_report.csv', index=False)
pd.set_option('display.width', 200)
for k in ('top10_overlap', 'spearman', 'avail_top10_overlap'):
    print('\n==', k); print(r.pivot(index='variant', columns='list', values=k).reindex(VARIANTS.keys()).to_string())
print('\n== eligible counts (base / variant)'); print(r.pivot(index='variant', columns='list', values='n_var').reindex(VARIANTS.keys()).to_string())
