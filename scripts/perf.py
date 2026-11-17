"""Player performance under pressure.

Per-ball value vs an average player in the same state:
  batter: (runs - E[runs]) - (out - E[out]) * wicket_cost
  bowler: (E[runs conceded] - runs conceded) + (bowler wicket - E[wkt]) * wicket_cost
  wicket_cost(over, wickets lost) = expected remaining runs lost by one more wicket (first-innings table, monotone GBM).
  E[.] from gradient-boosted models on match state only (no player identity):
    competition group, innings, over, balls bowled, wickets down, required rate, runs needed, conditions factor,
    time, closeness, dead filter, dominance, stakes.
Player ratings (weights = competition weight x recency x pressure intensity on high-pressure balls):
  HP rating   = shrunk weighted mean value per ball on high-pressure balls (x100 -> runs per 100 balls)
  Base rating = same on ordinary balls
  Delta       = HP rating - Base rating
  Consistency = share of matches (>=4 HP balls) with positive HP value, shrunk toward the population mean
  Composite   = 0.60 z(HP) + 0.25 z(Consistency) + 0.15 z(Delta)
"""
import sys, os, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

CFG = dict(comp_w={'IPL': 1.0, 'T20I_full': 1.0, 'A_teams': 0.7, 'T20I_assoc': 0.5},
           recency=0.85, min_hp_balls=120, min_hp_matches=10, comp_weights=(0.60, 0.25, 0.15), groups=None, hp_q=0.75, big_neff=50)
if len(sys.argv) > 1: CFG.update(json.loads(sys.argv[1]))
SRC = CFG.get('balls', '/home/claude/final_nrr/balls_pressure.parquet')
OUT = CFG.get('out', '/home/claude/final_nrr'); os.makedirs(OUT, exist_ok=True)

b = pd.read_parquet(SRC)
if 'is_so' not in b: b['is_so'] = False
b['is_so'] = b.is_so.fillna(False).astype(bool)
b['year'] = pd.to_datetime(b.match_date).dt.year
b['grp_c'] = b.grp.map({'IPL': 0, 'T20I_full': 1, 'T20I_assoc': 2, 'A_teams': 3})
b['rrr_f'] = b.rrr.fillna(-1).clip(upper=4); b['need_f'] = np.where(b.innings == 2, b.need, -1)

# ---------------------------------------------------------------- wicket cost
i1 = b[(b.innings == 1) & ~b.is_so].copy()   # super overs never train the wicket-cost curve
i1['final'] = i1.groupby('match_id').total.transform('sum')
i1['remaining'] = i1.final - i1.runs_before
i1['rem_scaled'] = i1.remaining / i1.cf
wc = HistGradientBoostingRegressor(max_iter=200, monotonic_cst=[1, -1], random_state=0)
wc.fit(i1[['balls_left', 'wkts_before']], i1.rem_scaled)
def wicket_cost(df):
    bl = df.balls_left.values
    e0 = wc.predict(np.c_[bl, df.wkts_before]); e1 = wc.predict(np.c_[bl, np.minimum(df.wkts_before + 1, 10)])
    cost = np.where(df.wkts_before >= 9, e0, e0 - e1)
    return np.clip(cost, 0.5, None) * df.cf.values
b['wkt_cost'] = wicket_cost(b)

# ---------------------------------------------------------------- chase wickets valued by the win-probability they cost (v11, 17 Nov)
# In a chase a wicket's cost is the drop in the batting side's chance of winning, turned into runs at the
# exchange rate of that moment (how much one run changes the chance of winning). First innings keep the run-based cost.
if CFG.get('chase_wp', True):
    i2 = b[(b.innings == 2) & ~b.is_so & b.bat_team_won.notna()]
    WPX = ['need', 'balls_left', 'wkts_before', 'cf', 'grp_c']
    wpm = HistGradientBoostingClassifier(max_iter=300, random_state=0, monotonic_cst=[-1, 1, -1, 0, 0], categorical_features=[4])
    wpm.fit(i2[WPX], i2.bat_team_won.astype(int))
    def wp_at(df, need_shift=0, wk_shift=0):
        X = df[WPX].copy(); X['need'] = (X.need + need_shift).clip(lower=0); X['wkts_before'] = (X.wkts_before + wk_shift).clip(upper=9)
        return wpm.predict_proba(X)[:, 1]
    c2 = b.innings == 2
    s = b[c2]
    wp0 = wp_at(s)
    wp_w = np.where(s.wkts_before >= 9, 0.0, wp_at(s, wk_shift=1))          # the tenth wicket ends the chase
    wp_cost = np.clip(wp0 - wp_w, 0, None)
    # runs-equivalent: how many extra runs added to the target would cut the chance of winning as much as the wicket does.
    # Found by search over 0..40 extra runs (a direct comparison, so flat steps in the model can't blow it up).
    MAXR = 40
    grid = np.column_stack([wp_at(s, need_shift=k) for k in range(MAXR + 1)])     # chance of winning with k more runs needed
    reached = grid <= wp_w[:, None] + 1e-9
    first = np.where(reached.any(axis=1), reached.argmax(axis=1), MAXR).astype(float)
    # where the chase is already decided (the wicket barely moves the chance), keep the run-based cost
    decided = wp_cost < 0.005
    b.loc[c2, 'wkt_cost'] = np.where(decided, b.loc[c2, 'wkt_cost'], np.clip(first, 0.5, MAXR))
    b.loc[c2, 'wkt_wp_cost'] = wp_cost

# ---------------------------------------------------------------- expected outcome models
S = ['grp_c', 'innings', 'over_idx', 'legal_before', 'wkts_before', 'rrr_f', 'need_f', 'cf', 'time', 'close', 'dead']
fb = b[b.faced].copy()
fb['out'] = (fb.wkt & (fb.dismissed_player_id == fb.striker_id)).astype(int)
Xb = S + ['dom_bat', 'stakes_bat']
tr = ~fb.is_so   # expected-outcome models train on normal innings only, then score super-over balls as last-over states
fb['e_runs'] = HistGradientBoostingRegressor(max_iter=300, random_state=0, categorical_features=[0]).fit(fb.loc[tr, Xb], fb.runs[tr]).predict(fb[Xb])
fb['e_out'] = HistGradientBoostingClassifier(max_iter=300, random_state=0, categorical_features=[0]).fit(fb.loc[tr, Xb], fb.out[tr]).predict_proba(fb[Xb])[:, 1]
fb['val'] = (fb.runs - fb.e_runs) - (fb.out - fb.e_out) * fb.wkt_cost

bw = b.copy()
Xw = S + ['dom_bowl', 'stakes_bowl']
bw['bwk'] = bw.bowler_wkt.astype(int)
tw = ~bw.is_so
bw['e_runs'] = HistGradientBoostingRegressor(max_iter=300, random_state=0, categorical_features=[0]).fit(bw.loc[tw, Xw], bw.bowler_runs[tw]).predict(bw[Xw])
bw['e_wkt'] = HistGradientBoostingClassifier(max_iter=300, random_state=0, categorical_features=[0]).fit(bw.loc[tw, Xw], bw.bwk[tw]).predict_proba(bw[Xw])[:, 1]
bw['val'] = (bw.e_runs - bw.bowler_runs) + (bw.bwk - bw.e_wkt) * bw.wkt_cost

# ---------------------------------------------------------------- optional competition split (expected models stay trained on all data)
if CFG['groups']:
    fb, bw = fb[fb.grp.isin(CFG['groups'])].copy(), bw[bw.grp.isin(CFG['groups'])].copy()
    fb['hp_bat'] = fb.P_bat >= fb.P_bat.quantile(CFG['hp_q'])           # high pressure judged within this competition
    bw['hp_bowl'] = bw.P_bowl >= bw.P_bowl[bw.legal].quantile(CFG['hp_q'])
# ---------------------------------------------------------------- ratings
def rate(df, pid, pname, hp, Pcol, ball_mask, kind):
    df = df.copy()
    df['w_base'] = df.grp.map(CFG['comp_w']) * CFG['recency'] ** (2026 - df.year)
    df['w'] = df.w_base * np.where(df[hp], df[Pcol], 1.0)
    df['cnt'] = ball_mask(df).astype(int)          # balls that count toward sample size
    # ---- wicket luck (v11b, 17 Nov): wickets/dismissals are rare, so each player's count is pulled toward the number an
    # average player would get on the same balls, by an amount learned from the data (Poisson-Gamma empirical Bayes).
    # theta = (W + k*mu) / (E + k); k = 1 / (spread of W/E beyond chance). No real spread -> k infinite -> everyone at mu.
    wk_obs, wk_exp = ('out', 'e_out') if kind == 'bat' else ('bwk', 'e_wkt')
    run_part = (df.runs - df.e_runs) if kind == 'bat' else (df.e_runs - df.bowler_runs)
    sign = -1 if kind == 'bat' else 1
    df['val_raw'] = df.val
    df['wk_theta'] = 1.0
    wk_info = {}
    if CFG.get('wicket_eb', True):
        for flag in (True, False):
            sub = df[df[hp] == flag]
            g = sub.groupby(pid).agg(W=(wk_obs, 'sum'), E=(wk_exp, 'sum'))
            g = g[g.E >= 1]
            mu = g.W.sum() / g.E.sum()
            ratio = g.W / g.E
            extra = np.average((ratio - mu) ** 2, weights=g.E) - np.average(mu / g.E, weights=g.E)
            k = 1 / extra if extra > 1e-4 else np.inf
            allg = sub.groupby(pid).agg(W=(wk_obs, 'sum'), E=(wk_exp, 'sum'))
            theta = (mu if np.isinf(k) else (allg.W + k * mu) / (allg.E + k))
            th = pd.Series(theta, index=allg.index) if np.isscalar(theta) else theta
            m = df[hp] == flag
            df.loc[m, 'wk_theta'] = df.loc[m, pid].map(th).values
            wk_info['hp' if flag else 'base'] = dict(k=float(k), mu=float(mu), extra_var=float(extra))
        df['val'] = run_part + sign * (df.wk_theta * df[wk_exp] - df[wk_exp]) * df.wkt_cost
    H, N = df[df[hp]], df[~df[hp]]

    def shrunk(sub):
        g = sub.groupby(pid).apply(lambda d: pd.Series({'sw': d.w.sum(), 'sw2': (d.w ** 2).sum(), 'swv': (d.w * d.val).sum(),
                                                         'swv2': (d.w * d.val ** 2).sum(), 'n': d.cnt.sum()}))
        mean = g.swv / g.sw
        neff = g.sw ** 2 / g.sw2
        within = np.average((g.swv2 / g.sw - mean ** 2).clip(lower=1e-6), weights=g.sw)   # pooled per-ball variance
        big = neff >= CFG['big_neff']
        mu = np.average(mean[big], weights=neff[big])
        tau2 = max(np.average((mean[big] - mu) ** 2, weights=neff[big]) - np.average(within / neff[big], weights=neff[big]), 1e-6)
        rel = tau2 / (tau2 + within / neff)                     # reliability of each player's mean
        return mu + rel * (mean - mu), g.n, within / tau2, mean, within / neff, neff, tau2 > 1e-6
    hp_r, hp_n, k_hp, hp_raw, hp_var, hp_neff, hp_sig = shrunk(H)
    base_r, base_n, k_b, base_raw, base_var, _, base_sig = shrunk(N)
    # pressure effect: shrink each player's own (pressure - normal) difference toward the population, by its noise
    both = hp_raw.index.intersection(base_raw.index)
    d_raw = hp_raw[both] - base_raw[both]; d_var = hp_var[both] + base_var[both]
    bigd = hp_neff[both] >= CFG['big_neff']
    mu_d = np.average(d_raw[bigd], weights=1 / d_var[bigd])
    tau2_d = max(np.average((d_raw[bigd] - mu_d) ** 2, weights=1 / d_var[bigd]) - np.average(d_var[bigd], weights=1 / d_var[bigd]), 1e-6)
    delta = mu_d + tau2_d / (tau2_d + d_var) * (d_raw - mu_d)
    # match-level consistency on HP balls
    pm = H.groupby([pid, 'match_id']).agg(v=('val', 'sum'), n=('cnt', 'sum'))
    pm = pm[pm.n >= 4]
    pos = (pm.v > 0).groupby(level=0).agg(['sum', 'count'])
    p0 = pos['sum'].sum() / pos['count'].sum()
    cons = (pos['sum'] + 5 * p0) / (pos['count'] + 5)
    names = df.groupby(pid)[pname].agg(lambda s: s.mode().iat[0])
    team_last = df.sort_values('match_date').groupby(pid).last()
    r = pd.DataFrame({'player': names, 'hp_balls': hp_n, 'hp_matches': pos['count'], 'base_balls': base_n,
                      'hp_rating': hp_r * 100, 'base_rating': base_r * 100, 'consistency': cons,
                      'last_seen': team_last.match_date, 'last_grp': team_last.grp})
    r['delta'] = delta * 100
    # wicket luck: counts and the pulled-in rate on HP balls
    hw = H.groupby(pid).agg(W=(wk_obs, 'sum'), E=(wk_exp, 'sum'), th=('wk_theta', 'first'))
    r['hp_wk_obs'] = hw.W; r['hp_wk_exp'] = hw.E; r['hp_wk_theta'] = hw.th
    # traditional HP stats
    if kind == 'bat':
        t = H.groupby(pid).agg(runs=('runs', 'sum'), balls=('cnt', 'sum'), outs=('out', 'sum'),
                               fours=('is_four', 'sum'), sixes=('is_six', 'sum'), exp_runs=('e_runs', 'sum'))
        r['hp_sr'] = 100 * t.runs / t.balls; r['hp_avg'] = t.runs / t.outs.replace(0, np.nan)
        r['hp_4s'] = t.fours; r['hp_6s'] = t.sixes; r['hp_bd%'] = 100 * (t.fours + t.sixes) / t.balls
        r['hp_balls_per_bd'] = t.balls / (t.fours + t.sixes).replace(0, np.nan)
        r['hp_exp_sr'] = 100 * t.exp_runs / t.balls
        tb = N.groupby(pid).agg(runs=('runs', 'sum'), balls=('cnt', 'sum'))
        r['base_sr'] = 100 * tb.runs / tb.balls
    else:
        t = H.groupby(pid).agg(runs=('bowler_runs', 'sum'), balls=('cnt', 'sum'), wk=('bwk', 'sum'), exp_runs=('e_runs', 'sum'),
                               dots=('dotb', 'sum'))
        r['hp_econ'] = 6 * t.runs / t.balls; r['hp_exp_econ'] = 6 * t.exp_runs / t.balls
        r['hp_wkts'] = t.wk; r['hp_bpw'] = t.balls / t.wk.replace(0, np.nan); r['hp_dot%'] = 100 * t.dots / t.balls
        tb = N.groupby(pid).agg(runs=('bowler_runs', 'sum'), balls=('cnt', 'sum'))
        r['base_econ'] = 6 * tb.runs / tb.balls
    el = r[(r.hp_balls >= CFG['min_hp_balls']) & (r.hp_matches >= CFG['min_hp_matches'])].copy()
    z = lambda s: (s - s.mean()) / s.std()
    a, c, d = CFG['comp_weights']
    zd = z(el.delta) if tau2_d > 1e-6 else 0.0      # no detectable player-specific pressure effect -> term carries no information
    zh = z(el.hp_rating) if hp_sig else 0.0
    el['score'] = a * zh + c * z(el.consistency) + d * zd
    el = el.sort_values('score', ascending=False); el.insert(0, 'rank', range(1, len(el) + 1))
    return el, r, df[['val', 'val_raw', 'wk_theta']], dict(wk=wk_info, k_hp=k_hp, k_base=k_b, p0=p0, hp_signal=bool(hp_sig), delta_tau=float(np.sqrt(tau2_d) * 100), delta_mu=float(mu_d * 100))

bw['dotb'] = (bw.legal & (bw.bowler_runs == 0)).astype(int)
bat, bat_all, adj_b, bat_meta = rate(fb, 'striker_id', 'striker_name', 'hp_bat', 'P_bat', lambda d: d.faced, 'bat')
bowl, bowl_all, adj_w, bowl_meta = rate(bw, 'bowler_id', 'bowler_name', 'hp_bowl', 'P_bowl', lambda d: d.legal, 'bowl')
fb[['val', 'val_raw', 'wk_theta']] = adj_b.loc[fb.index]; bw[['val', 'val_raw', 'wk_theta']] = adj_w.loc[bw.index]
bat.to_csv(f'{OUT}/rank_bat.csv'); bowl.to_csv(f'{OUT}/rank_bowl.csv')
fb[['match_id', 'striker_id', 'hp_bat', 'P_bat', 'val', 'runs', 'e_runs', 'out', 'e_out', 'wkt_cost', 'grp', 'year', 'faced', 'val_raw', 'wk_theta']].to_parquet(f'{OUT}/val_bat.parquet')
bw[['match_id', 'bowler_id', 'hp_bowl', 'P_bowl', 'val', 'bowler_runs', 'e_runs', 'bwk', 'e_wkt', 'wkt_cost', 'grp', 'year', 'legal', 'val_raw', 'wk_theta']].to_parquet(f'{OUT}/val_bowl.parquet')
print('meta', bat_meta, bowl_meta, 'eligible', len(bat), len(bowl))
pd.set_option('display.width', 250)
cols_b = ['rank', 'player', 'hp_balls', 'hp_matches', 'hp_rating', 'base_rating', 'delta', 'consistency', 'hp_sr', 'hp_exp_sr', 'hp_avg', 'last_seen']
cols_w = ['rank', 'player', 'hp_balls', 'hp_matches', 'hp_rating', 'base_rating', 'delta', 'consistency', 'hp_econ', 'hp_exp_econ', 'hp_wkts', 'hp_dot%', 'last_seen']
print(bat[cols_b].head(25).round(2).to_string()); print(bowl[cols_w].head(25).round(2).to_string())
