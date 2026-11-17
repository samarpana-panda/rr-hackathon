"""Add super-over balls to the v6 ball table (v8).
Each super-over ball is valued as a last-over state: over 20, (6 - legal balls bowled) left, 8 + wickets lost down
(two wickets end a super over), chase target = first super-over total + 1. Closeness = 1, no dead filter.
Stakes: max(match stakes, 1.8) before compression - a super over decides the match, so it is at least an eliminator.
Pressure: set to the maximum pressure score in its competition, so every super-over ball is high pressure.
"""
import numpy as np, pandas as pd, os
SRC = os.environ.get('SO_IN', '/home/claude/final_nrr/v6'); DST = os.environ.get('SO_OUT', '/home/claude/final_nrr/v8')
v6 = pd.read_parquet(f'{SRC}/balls_pressure.parquet')
raw = pd.read_csv('/home/claude/data/pack/ball_by_ball.csv', low_memory=False)
so = raw[raw.innings > 2].copy().sort_values(['match_id', 'innings', 'innings_ball_num'])
et = so.extras_type
so['wide'] = et.eq('wide'); so['noball'] = et.isin(['Overstep', 'other'])
so['legal'] = ~(so.wide | so.noball); so['faced'] = ~so.wide
ext = np.select([so.wide | so.noball, et.isin(['byes', 'legbyes']), et.eq('Penalty')], [1 + so.extras_2_runs, 1 + so.extras_2_runs, 5], 0)
so['total'] = so.runs + ext
so['bowler_runs'] = so.runs + np.where(so.wide | so.noball, 1 + so.extras_2_runs, 0)
so['wkt'] = so.is_wicket.astype(bool)
so['bowler_wkt'] = so.wkt & ~so.dismissal_type.isin(['RunOut', 'Retire Out', 'Obstracting the field', 'Obstructing the field'])
g = so.groupby(['match_id', 'innings'])
so['runs_before'] = g.total.cumsum() - so.total
w_before = g.wkt.cumsum() - so.wkt
lb = g.legal.cumsum() - so.legal
so['wkts_before'] = (8 + w_before).clip(upper=9)
so['legal_before'] = 114 + lb
so['inn_balls'] = 120; so['balls_left'] = (6 - lb).clip(lower=1); so['over_idx'] = 19
first_tot = so[so.innings == 3].groupby('match_id').total.sum()
so['target'] = np.where(so.innings == 4, so.match_id.map(first_tot) + 1, np.nan)
so['need'] = np.where(so.innings == 4, (so.target - so.runs_before).clip(lower=0), np.nan)
so['rrr'] = so.need / so.balls_left
so['innings'] = so.innings.map({3: 1, 4: 2})
mi = v6.groupby('match_id').first()
for c in ['grp', 'cf', 'match_date', 'venue_id']:
    so[c] = so.match_id.map(mi[c])
so['time'] = 1.0; so['close'] = 1.0; so['dead'] = 1.0; so['wp'] = 0.5
so['dom_bat'] = 0.0; so['dom_bowl'] = 0.0; so['dom_raw'] = 0.0
st = pd.read_csv('/home/claude/final_nrr/stakes.csv').set_index(['match_id', 'team_id']).stakes_mult
gam = float(open(SRC + '_gamma.txt').read())
for side, team in [('bat', 'batting_team_id'), ('bowl', 'bowling_team_id')]:
    base = np.array([st.get((m, t), 1.0) for m, t in zip(so.match_id, so[team])])
    so[f'stakes_{side}'] = np.maximum(base, 1.8)
    so[f'stakes_eff_{side}'] = so[f'stakes_{side}'] ** gam
    so[f'why_{side}'] = 'super_over'
for side in ('bat', 'bowl'):
    pmax = v6.groupby('grp')[f'P_{side}'].max()
    so[f'P_{side}'] = so.grp.map(pmax)
    so[f'hp_{side}'] = True
so['is_so'] = True
v6['is_so'] = False
cols = [c for c in v6.columns if c in so.columns]
out = pd.concat([v6, so[cols]], ignore_index=True)
os.makedirs(DST, exist_ok=True)
out.to_parquet(f'{DST}/balls_pressure.parquet')
print('added', len(so), 'super-over balls; by competition:', so.grp.value_counts().to_dict())
print(so[['match_id', 'innings', 'striker_name', 'bowler_name', 'runs', 'wkt', 'wkts_before', 'balls_left', 'need', 'stakes_bat']].to_string(index=False))
