"""Ball-level pressure index for batting side and bowling side.

P_side = Stakes(team) x InningsMult x DeadFilter x [0.5*Situation + 0.25*Dominance_side + 0.25*Personal_side]
  Situation  = Time x (0.4 + 0.6*Closeness)      Time: logistic in share of innings bowled, steep in last 4-5 overs
  Closeness  = 1 - |2*WP - 1|                    WP from out-of-fold GBM win-probability models (chase / 1st inns)
  DeadFilter = clip(min(WP,1-WP)/0.05, 0.1, 1)   lost causes / foregone conclusions fade out
  Dominance  = last 12 legal balls (2 overs): expected runs - actual runs + 7*(wickets - expected wickets), side-specific
  Personal   = batter: SR vs required rate (inns 2) or vs conditions-adjusted par (inns 1), new batter after
               a wicket, dot-ball streak; bowler: previous over or current over expensive vs expectation
  InningsMult = 0.85 first innings, 1.0 second innings
"""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

D = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/data/pack'
import os, json
W = json.loads(os.environ.get('PW', '{"sit": 0.5, "dom": 0.25, "pers": 0.25}'))
HPQ = float(os.environ.get('PQ', 0.75))
POUT = os.environ.get('POUT', '/home/claude/final_nrr/balls_pressure.parquet')
INN_MULT = {1: 0.85, 2: 1.0}
FULL = {'India', 'Australia', 'England', 'South Africa', 'New Zealand', 'Pakistan', 'Sri Lanka',
        'West Indies', 'Bangladesh', 'Afghanistan', 'Ireland', 'Zimbabwe'}

# ---------------------------------------------------------------- load
b = pd.read_csv(f'{D}/ball_by_ball.csv', low_memory=False)
m = pd.read_csv(f'{D}/matches.csv').merge(pd.read_csv(f'{D}/competitions.csv'), on='comp_id')
b = b[b.innings.isin([1, 2])].copy()  # super overs (6 matches, 38 balls) excluded

def group(r):
    if r.comp_name.startswith('IPL'): return 'IPL'
    if r.match_format == 'T20': return 'A_teams'
    return 'T20I_full' if (r.home_team in FULL and r.away_team in FULL) else 'T20I_assoc'
m['grp'] = m.apply(group, axis=1)
b = b.merge(m[['match_id', 'grp', 'venue_id', 'first_inn_revised_overs', 'second_inn_revised_overs',
               'revised_target', 'win_team_id', 'result', 'match_date']], on='match_id')

# ---------------------------------------------------------------- runs & legality
et = b.extras_type
b['wide'] = et.eq('wide'); b['noball'] = et.isin(['Overstep', 'other'])
b['legal'] = ~(b.wide | b.noball)
b['faced'] = ~b.wide
extra = np.select([b.wide | b.noball, et.isin(['byes', 'legbyes']), et.eq('Penalty')],
                  [1 + b.extras_2_runs, 1 + b.extras_2_runs, 5], 0)  # byes/legbyes amount not recorded: assume 1
b['total'] = b.runs + extra
b['bowler_runs'] = b.runs + np.where(b.wide | b.noball, 1 + b.extras_2_runs, 0)
b['wkt'] = b.is_wicket.astype(bool)
b['bowler_wkt'] = b.wkt & ~b.dismissal_type.isin(['RunOut', 'Retire Out', 'Obstracting the field', 'Obstructing the field'])
b = b.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)

g = b.groupby(['match_id', 'innings'])
b['runs_before'] = g.total.cumsum() - b.total
b['wkts_before'] = g.wkt.cumsum() - b.wkt
b['legal_before'] = g.legal.cumsum() - b.legal
rev = np.where(b.innings == 1, b.first_inn_revised_overs, b.second_inn_revised_overs)
b['inn_balls'] = np.where(pd.notna(rev), rev, 120)
b['balls_left'] = (b.inn_balls - b.legal_before).clip(lower=1)
b['over_idx'] = (b.legal_before // 6).clip(upper=19).astype(int)  # 0..19

# ---------------------------------------------------------------- targets
tot = b.groupby(['match_id', 'innings']).total.sum().unstack()
mm = m.set_index('match_id')
margin = mm.result.str.extract(r'[Ww]on by (\d+) [Rr]un', expand=False).astype(float)
target = tot[1] + 1
fix = margin.reindex(tot.index).notna() & mm.revised_target.reindex(tot.index).isna()
target[fix] = tot[2][fix] + margin.reindex(tot.index)[fix] + 1        # consistent with official margin
target = target.where(mm.revised_target.reindex(tot.index).isna(), mm.revised_target.reindex(tot.index))
b['target'] = b.match_id.map(target)
b['need'] = (b.target - b.runs_before).clip(lower=0)
b['rrr'] = np.where(b.innings == 2, b.need / b.balls_left, np.nan)       # runs per ball

# ---------------------------------------------------------------- expected rates & conditions
leg = b[b.legal | b.noball]
exp_tab = b.groupby(['grp', 'innings', 'over_idx']).total.sum() / b.groupby(['grp', 'innings', 'over_idx']).legal.sum()
bat_tab = b.groupby(['grp', 'innings', 'over_idx']).runs.sum() / b.groupby(['grp', 'innings', 'over_idx']).faced.sum()
b['exp_rpb_grp'] = exp_tab.reindex(pd.MultiIndex.from_frame(b[['grp', 'innings', 'over_idx']])).values
b['exp_bat_grp'] = bat_tab.reindex(pd.MultiIndex.from_frame(b[['grp', 'innings', 'over_idx']])).values
# match conditions factor: actual vs expected scoring across both innings, shrunk 50% toward 1
mf = b.groupby('match_id').total.sum() / b.groupby('match_id').apply(lambda d: (d.exp_rpb_grp * d.legal).sum())
# venue factor (leave-one-match-out, shrunk with k=5 matches)
ms = pd.DataFrame({'mf': mf}).join(m.set_index('match_id')[['venue_id']])
vs = ms.groupby('venue_id').mf.agg(['sum', 'count'])
ms['vf'] = ms.apply(lambda r: 1 + ((vs.loc[r.venue_id, 'sum'] - r.mf) - (vs.loc[r.venue_id, 'count'] - 1)) /
                    ((vs.loc[r.venue_id, 'count'] - 1) + 5), axis=1)
ms['cf'] = 0.5 * (1 + 0.5 * (ms.mf - 1)) + 0.5 * ms.vf
b['cf'] = b.match_id.map(ms.cf); b['vf'] = b.match_id.map(ms.vf)
b['exp_rpb'] = b.exp_rpb_grp * b.cf
wk_tab = b.groupby(['grp', 'innings', 'over_idx']).wkt.sum() / b.groupby(['grp', 'innings', 'over_idx']).legal.sum()
b['exp_wkt'] = wk_tab.reindex(pd.MultiIndex.from_frame(b[['grp', 'innings', 'over_idx']])).values
b['exp_bat'] = b.exp_bat_grp * b.cf

# ---------------------------------------------------------------- win probability (out-of-fold by match)
def fit_wp(df, feats, y, mono):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, monotonic_cst=mono,
                                          l2_regularization=1.0, random_state=0).fit(df[feats], y)

def oof_wp(df, feats, y, mono):
    p = np.full(len(df), np.nan)
    for tr, te in GroupKFold(5).split(df, y, df.match_id):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             monotonic_cst=mono, l2_regularization=1.0, random_state=0)
        clf.fit(df.iloc[tr][feats], y.iloc[tr]); p[te] = clf.predict_proba(df.iloc[te][feats])[:, 1]
    return p

b['bat_team_won'] = (b.win_team_id == b.batting_team_id).astype(float)
decided = b.win_team_id.notna() & (b.win_team_id != 0)
b['grp_ipl'] = (b.grp == 'IPL').astype(int)
b['wk_in_hand'] = 10 - b.wkts_before
c2 = b[(b.innings == 2) & decided].copy()
f2 = ['need', 'balls_left', 'wk_in_hand', 'rrr', 'target', 'grp_ipl']
b.loc[c2.index, 'wp'] = oof_wp(c2, f2, c2.bat_team_won, [-1, 1, 1, -1, 0, 0])
c1 = b[(b.innings == 1) & decided].copy()
c1['par'] = c1.vf * c1.groupby('grp').target.transform('mean')
c1['runs_vs_par_pace'] = c1.runs_before - c1.par * c1.legal_before / c1.inn_balls
f1 = ['runs_before', 'wk_in_hand', 'balls_left', 'runs_vs_par_pace', 'par', 'grp_ipl']
b.loc[c1.index, 'wp'] = oof_wp(c1, f1, c1.bat_team_won, [1, 1, 0, 1, -1, 0])
# no-result / abandoned matches: score with models fit on all decided matches
nr2 = b[(b.innings == 2) & ~decided]; nr1 = b[(b.innings == 1) & ~decided].copy()
if len(nr2): b.loc[nr2.index, 'wp'] = fit_wp(c2, f2, c2.bat_team_won, [-1, 1, 1, -1, 0, 0]).predict_proba(nr2[f2])[:, 1]
if len(nr1):
    nr1['par'] = nr1.vf * nr1.grp.map(c1.groupby('grp').target.mean())
    nr1['runs_vs_par_pace'] = nr1.runs_before - nr1.par * nr1.legal_before / nr1.inn_balls
    b.loc[nr1.index, 'wp'] = fit_wp(c1, f1, c1.bat_team_won, [1, 1, 0, 1, -1, 0]).predict_proba(nr1[f1])[:, 1]
b['wp'] = b.wp.fillna(0.5)

# ---------------------------------------------------------------- situation
x = b.legal_before / b.inn_balls
b['time'] = 1 / (1 + np.exp(-(x - 0.72) / 0.075))
b['close'] = 1 - (2 * b.wp - 1).abs()
b['dead'] = (np.minimum(b.wp, 1 - b.wp) / 0.05).clip(0.1, 1)
b['situation'] = b.time * (0.4 + 0.6 * b.close)

# ---------------------------------------------------------------- dominance (last 12 legal balls)
# rolling by legal-ball count: build per-innings arrays
dom = np.zeros(len(b))
for (mid, inn), idx in b.groupby(['match_id', 'innings']).indices.items():
    d = b.iloc[idx]
    lb = d.legal_before.values; tot_ = d.total.values; ex = (d.exp_rpb * d.legal).values
    wk = d.wkt.values.astype(float) - (d.exp_wkt * d.legal).values   # wickets above expectation
    ctot, cex, cwk = np.r_[0, np.cumsum(tot_)], np.r_[0, np.cumsum(ex)], np.r_[0, np.cumsum(wk)]
    # index of first delivery with legal_before >= lb-12
    start = np.searchsorted(lb, lb - 12, side='left')
    i = np.arange(len(d))
    deficit = (cex[i] - cex[start]) - (ctot[i] - ctot[start]) + 7 * (cwk[i] - cwk[start])
    dom[idx] = np.where(lb >= 12, deficit, 0)
b['dom_raw'] = dom
b['dom_bat'] = (b.dom_raw / 15).clip(0, 1)      # bowling side on top -> batters squeezed
b['dom_bowl'] = (-b.dom_raw / 15).clip(0, 1)    # batting side on top -> bowlers under fire

# ---------------------------------------------------------------- personal: batter
gb = b.groupby(['match_id', 'innings', 'striker_id'])
b['bf_before'] = gb.faced.cumsum() - b.faced
b['br_before'] = gb.runs.cumsum() - b.runs
b['_eb'] = b.exp_bat * b.faced
b['exp_bat_before'] = b.groupby(['match_id', 'innings', 'striker_id'])._eb.cumsum() - b._eb
# dot streak faced by this striker
dotball = (b.faced & (b.runs == 0) & ~b.wkt).astype(int)
streak = np.zeros(len(b), int)
for _, idx in gb.indices.items():
    s = 0
    for k in idx:
        streak[k] = s
        if b.faced.iat[k]: s = s + 1 if dotball.iat[k] else 0
b['dot_streak'] = streak
bench = np.where(b.innings == 2, np.minimum(b.rrr, 3) * 0.95,  # batter share of required (~5% comes from extras)
                 np.where(b.bf_before > 0, b.exp_bat_before / b.bf_before.clip(lower=1), b.exp_bat))
b['sr_bench'] = bench
srs = (b.br_before + 6 * bench) / (b.bf_before + 6)
b['p_sr'] = ((bench - srs) / np.maximum(bench, 0.3) / 0.5).clip(0, 1)
recent_wkt = np.zeros(len(b))
for (mid, inn), idx in b.groupby(['match_id', 'innings']).indices.items():
    d = b.iloc[idx]; lb = d.legal_before.values; wk = np.r_[0, np.cumsum(d.wkt.values)]
    st = np.searchsorted(lb, lb - 12, side='left'); recent_wkt[idx] = wk[np.arange(len(d))] - wk[st]
b['recent_wkts'] = recent_wkt
b['p_newbat'] = np.where((b.bf_before < 10) & (b.recent_wkts >= 2), 1 - b.bf_before / 10, 0)
b['p_dots'] = ((b.dot_streak - 1) / 3).clip(0, 1)
b['pers_bat'] = (np.maximum(b.p_sr, b.p_newbat) + 0.3 * b.p_dots).clip(0, 1)

# ---------------------------------------------------------------- personal: bowler
b['exp_bowl'] = b.exp_rpb * b.legal
ov = b.groupby(['match_id', 'innings', 'overid', 'bowler_id']).agg(r=('bowler_runs', 'sum'), e=('exp_bowl', 'sum'),
                                                                  n=('legal', 'sum')).reset_index()
ov['excess'] = ov.r - ov.e * 6 / ov.n.clip(lower=1)
ov = ov.sort_values(['match_id', 'innings', 'bowler_id', 'overid'])
ov['prev_excess'] = ov.groupby(['match_id', 'innings', 'bowler_id']).excess.shift(1)
b = b.merge(ov[['match_id', 'innings', 'overid', 'bowler_id', 'prev_excess']], on=['match_id', 'innings', 'overid', 'bowler_id'], how='left')
go = b.groupby(['match_id', 'innings', 'overid', 'bowler_id'])
cur_excess = (go.bowler_runs.cumsum() - b.bowler_runs) - (go.exp_bowl.cumsum() - b.exp_bowl)
b['p_prev_over'] = (b.prev_excess.fillna(0) / 8).clip(0, 1)
b['p_cur_over'] = (cur_excess / 8).clip(0, 1)
b['pers_bowl'] = np.maximum(b.p_prev_over, b.p_cur_over)

# ---------------------------------------------------------------- stakes & combine
st = pd.read_csv('/home/claude/final_nrr/stakes.csv')[['match_id', 'team_id', 'stakes_mult', 'stakes_reason']]
b = b.merge(st.rename(columns={'team_id': 'batting_team_id', 'stakes_mult': 'stakes_bat', 'stakes_reason': 'why_bat'}),
            on=['match_id', 'batting_team_id'], how='left')
b = b.merge(st.rename(columns={'team_id': 'bowling_team_id', 'stakes_mult': 'stakes_bowl', 'stakes_reason': 'why_bowl'}),
            on=['match_id', 'bowling_team_id'], how='left')
b[['stakes_bat', 'stakes_bowl']] = b[['stakes_bat', 'stakes_bowl']].fillna(1.0)
im = b.innings.map(INN_MULT)
b['raw_bat'] = W['sit'] * b.situation + W['dom'] * b.dom_bat + W['pers'] * b.pers_bat
b['raw_bowl'] = W['sit'] * b.situation + W['dom'] * b.dom_bowl + W['pers'] * b.pers_bowl
b['P_bat'] = b.raw_bat * im * b.dead * b.stakes_bat
b['P_bowl'] = b.raw_bowl * im * b.dead * b.stakes_bowl
for s in ('bat', 'bowl'):
    b[f'hp_{s}'] = b[f'P_{s}'] >= b[f'P_{s}'].quantile(HPQ)
b.drop(columns=['delivery_ist_time']).to_parquet(POUT)
print(b[['P_bat', 'P_bowl', 'situation', 'dom_bat', 'dom_bowl', 'pers_bat', 'pers_bowl', 'wp']].describe().round(3).T)
