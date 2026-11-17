"""Data-driven difficulty (v4).

Difficulty = how much a situation lowers what an average player achieves on that ball.
  1. Basic expectation E0 from state only (competition group, innings, over, wickets down, conditions).
  2. Residual value v (runs/wickets vs E0, wickets costed in runs) for the batter / bowler.
  3. Monotone GBM: v ~ difficulty factors + phase + player's usual level (leave-match-out, shrunk).
     Directions fixed by the user's definition (more squeeze / personal pressure can never make it easier);
     magnitudes learned from data.
  4. Difficulty d = -prediction with player level set to average (so it describes the situation, not the player).
Batting factors : squeeze (bowling side on top, last 2 overs), batter behind required/par rate, collapse arrival,
                  dot-ball streak, required run rate (chase).
Bowling factors : batting side on top (last 2 overs), expensive previous over, expensive current over, required rate.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as R, HistGradientBoostingClassifier as C

b = pd.read_parquet('/home/claude/final_nrr/balls_pressure.parquet')
b['grp_c'] = b.grp.map({'IPL': 0, 'T20I_full': 1, 'T20I_assoc': 2, 'A_teams': 3})
b['rrr_f'] = b.rrr.fillna(-1).clip(upper=4)
i1 = b[b.innings == 1].copy()
i1['rem'] = (i1.groupby('match_id').total.transform('sum') - i1.runs_before) / i1.cf
wc = R(max_iter=200, monotonic_cst=[1, -1], random_state=0).fit(i1[['balls_left', 'wkts_before']], i1.rem)
e0 = wc.predict(np.c_[b.balls_left, b.wkts_before]); e1 = wc.predict(np.c_[b.balls_left, np.minimum(b.wkts_before + 1, 10)])
b['wkt_cost'] = np.clip(np.where(b.wkts_before >= 9, e0, e0 - e1), 0.5, None) * b.cf
BASE = ['grp_c', 'innings', 'over_idx', 'wkts_before', 'cf']

def player_level(df, pid, k=300):
    """leave-match-out shrunk mean residual per ball for each player"""
    tot = df.groupby(pid).v.agg(['sum', 'count'])
    pm = df.groupby([pid, 'match_id']).v.agg(['sum', 'count'])
    s = tot['sum'].reindex(pm.index.get_level_values(0)).values - pm['sum'].values
    n = tot['count'].reindex(pm.index.get_level_values(0)).values - pm['count'].values
    lvl = pd.Series(s / (n + k), index=pm.index)
    return lvl.reindex(pd.MultiIndex.from_arrays([df[pid], df.match_id])).values

def fit_difficulty(df, feats, mono, pid):
    X = feats + ['over_idx', 'innings', 'lvl']
    m = R(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
          monotonic_cst=mono + [0, 0, 1], random_state=0).fit(df[X], df.v)
    neutral = df[X].copy(); neutral['lvl'] = 0.0
    return -m.predict(neutral), m

# ---------------- batting
fb = b[b.faced].copy()
fb['out'] = (fb.wkt & (fb.dismissed_player_id == fb.striker_id)).astype(int)
fb['e_runs'] = R(max_iter=300, random_state=0, categorical_features=[0]).fit(fb[BASE], fb.runs).predict(fb[BASE])
fb['e_out'] = C(max_iter=300, random_state=0, categorical_features=[0]).fit(fb[BASE], fb.out).predict_proba(fb[BASE])[:, 1]
fb['v'] = (fb.runs - fb.e_runs) - (fb.out - fb.e_out) * fb.wkt_cost
fb['squeeze'] = (fb.dom_raw / 15).clip(-2, 2)
fb['lvl'] = player_level(fb, 'striker_id')
FB = ['squeeze', 'p_sr', 'p_newbat', 'p_dots', 'rrr_f']
fb['d_bat'], mb = fit_difficulty(fb, FB, [-1, -1, -1, -1, 0], 'striker_id')

# ---------------- bowling
bw = b.copy(); bw['bwk'] = bw.bowler_wkt.astype(int)
bw['e_runs'] = R(max_iter=300, random_state=0, categorical_features=[0]).fit(bw[BASE], bw.bowler_runs).predict(bw[BASE])
bw['e_wk'] = C(max_iter=300, random_state=0, categorical_features=[0]).fit(bw[BASE], bw.bwk).predict_proba(bw[BASE])[:, 1]
bw['v'] = (bw.e_runs - bw.bowler_runs) + (bw.bwk - bw.e_wk) * bw.wkt_cost
bw['bat_on_top'] = (-bw.dom_raw / 15).clip(-2, 2)
bw['lvl'] = player_level(bw, 'bowler_id')
FW = ['bat_on_top', 'p_prev_over', 'p_cur_over', 'rrr_f']
bw['d_bowl'], mw = fit_difficulty(bw, FW, [-1, -1, -1, 0], 'bowler_id')

out = b[['match_id', 'innings', 'innings_ball_num']].copy()
out['d_bat'] = np.nan; out.loc[fb.index, 'd_bat'] = fb.d_bat
out['d_bowl'] = bw.d_bowl.values
out.to_parquet('/home/claude/final_nrr/v4/difficulty.parquet')

# ---------------- what the data says about each factor (partial effects, runs per ball of difficulty)
pd.set_option('display.width', 200)
def pd_curve(m, df, X, col, grid):
    base = df[X].sample(20000, random_state=0).copy(); base['lvl'] = 0.0; r = {}
    for gv in grid:
        t = base.copy(); t[col] = gv; r[gv] = -m.predict(t).mean()
    return pd.Series(r).round(3)
Xb = FB + ['over_idx', 'innings', 'lvl']; Xw = FW + ['over_idx', 'innings', 'lvl']
print('BATTING difficulty (runs/ball shortfall) by factor')
print('squeeze', pd_curve(mb, fb, Xb, 'squeeze', [-1, -0.5, 0, 0.5, 1, 1.5]).to_dict())
for c in ['p_sr', 'p_newbat', 'p_dots']: print(c, pd_curve(mb, fb, Xb, c, [0, .25, .5, .75, 1]).to_dict())
print('rrr (chase, runs/ball)', pd_curve(mb, fb[fb.innings == 2], Xb, 'rrr_f', [0.8, 1.2, 1.6, 2.0, 2.5, 3.0]).to_dict())
print('BOWLING difficulty by factor')
print('bat_on_top', pd_curve(mw, bw, Xw, 'bat_on_top', [-1, -0.5, 0, 0.5, 1, 1.5]).to_dict())
for c in ['p_prev_over', 'p_cur_over']: print(c, pd_curve(mw, bw, Xw, c, [0, .25, .5, .75, 1]).to_dict())
print('rrr (defending)', pd_curve(mw, bw[bw.innings == 2], Xw, 'rrr_f', [0.8, 1.2, 1.6, 2.0, 2.5, 3.0]).to_dict())
print('d summary'); print(out[['d_bat', 'd_bowl']].describe().round(3))
print('lvl effect check: corr(lvl, v)', round(np.corrcoef(fb.lvl, fb.v)[0, 1], 3), round(np.corrcoef(bw.lvl, bw.v)[0, 1], 3))
