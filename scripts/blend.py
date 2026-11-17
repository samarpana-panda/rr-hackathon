"""v4 pressure = stakes x (alpha * Consequence + (1 - alpha) * Difficulty)
Consequence C: learned net on situation inputs (share bowled, innings, own win prob, wickets down), mean 1.
Difficulty  D: data-driven shortfall, rescaled to the same spread as C (so alpha is the real relative weight), mean 1.
"""
import sys, json, numpy as np, pandas as pd
cfg = dict(alpha=0.65, out='/home/claude/final_nrr/v4'); cfg.update(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
b = pd.read_parquet('/home/claude/final_nrr/v4/consequence/balls_pressure.parquet')      # has g_bat, g_bowl, stakes
d = pd.read_parquet(cfg.get('diff', '/home/claude/final_nrr/v4/difficulty.parquet'))
assert (d.match_id.values == b.match_id.values).all() and (d.innings_ball_num.values == b.innings_ball_num.values).all()
for s in ('bat', 'bowl'):
    C = b[f'g_{s}'].values
    dd = d[f'd_{s}'].fillna(d[f'd_{s}'].mean()).values
    D = np.clip(1 + (dd - dd.mean()) * C.std() / dd.std(), 0.1, None)
    b[f'C_{s}'], b[f'D_{s}'] = C, D
    b[f'Q_{s}'] = cfg['alpha'] * C + (1 - cfg['alpha']) * D
if 'gamma' in cfg:
    gam = cfg['gamma']
elif cfg.get('rebalance', True):   # equal influence: stakes^gamma, gamma = sd(log in-match) / sd(log stakes), averaged over sides
    gam = np.mean([np.log(b[f'Q_{s}']).std() / np.log(b[f'stakes_{s}']).std() for s in ('bat', 'bowl')])
else:
    gam = 1.0
print('stakes gamma', round(gam, 3)); open(cfg['out'] + '_gamma.txt', 'w').write(str(gam))
for s in ('bat', 'bowl'):
    b[f'stakes_eff_{s}'] = b[f'stakes_{s}'] ** gam
    b[f'P_{s}'] = b[f'stakes_eff_{s}'] * b[f'Q_{s}']
    b[f'hp_{s}'] = b[f'P_{s}'] >= b[f'P_{s}'].quantile(0.75)
import os; os.makedirs(cfg['out'], exist_ok=True)
b.to_parquet(f"{cfg['out']}/balls_pressure.parquet")
if cfg.get('report', True):
    for s in ('bat', 'bowl'):
        print(s, 'corr(C,D)', round(np.corrcoef(b[f'C_{s}'], b[f'D_{s}'])[0, 1], 3),
              '| HP share from big games (stakes>=1.4):', round((b.loc[b[f'hp_{s}'], f'stakes_{s}'] >= 1.4).mean(), 2))
        print(b.groupby(['innings', 'over_idx'])[f'hp_{s}'].mean().unstack(0).mul(100).round(0).T.to_string())
        print(b.groupby(f'why_{s}')[f'hp_{s}'].mean().round(2).sort_values().to_dict())
