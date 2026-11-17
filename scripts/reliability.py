"""Bootstrap (Poisson, by match) of the composite ranking. Recomputes shrinkage, consistency, z-scores each draw."""
import sys, json, numpy as np, pandas as pd
CFG = dict(comp_w={'IPL': 1.0, 'T20I_full': 1.0, 'A_teams': 0.7, 'T20I_assoc': 0.5}, recency=0.85,
           min_hp_balls=120, min_hp_matches=10, comp_weights=(0.60, 0.25, 0.15), big_neff=50, B=300, dir='/home/claude/final_nrr')
if len(sys.argv) > 1: CFG.update(json.loads(sys.argv[1]))
rng = np.random.default_rng(11)

def prep(path, pid, hp, P, cnt):
    v = pd.read_parquet(path)
    v['w'] = v.grp.map(CFG['comp_w']) * CFG['recency'] ** (2026 - v.year) * np.where(v[hp], v[P], 1.0)
    v['cnt'] = v[cnt].astype(int)
    agg = v.groupby([pid, 'match_id', hp]).apply(lambda d: pd.Series({
        'sw': d.w.sum(), 'sw2': (d.w ** 2).sum(), 'swv': (d.w * d.val).sum(), 'swv2': (d.w * d.val ** 2).sum(),
        'n': d.cnt.sum(), 'v': d.val.sum()})).reset_index()
    return agg

def score(agg, pid, hp, mw):
    a = agg.assign(**{c: agg[c] * agg.match_id.map(mw) for c in ['sw', 'sw2', 'swv', 'swv2', 'n']})
    out = {}
    for flag in (True, False):
        g = a[a[hp] == flag].groupby(pid)[['sw', 'sw2', 'swv', 'swv2', 'n']].sum()
        g = g[g.sw > 0]
        mean = g.swv / g.sw; neff = g.sw ** 2 / g.sw2
        within = np.average((g.swv2 / g.sw - mean ** 2).clip(lower=1e-6), weights=g.sw)
        big = neff >= CFG['big_neff']; mu = np.average(mean[big], weights=neff[big])
        tau2 = max(np.average((mean[big] - mu) ** 2, weights=neff[big]) - np.average(within / neff[big], weights=neff[big]), 1e-6)
        out[flag] = (100 * (mu + tau2 / (tau2 + within / neff) * (mean - mu)), g.n, mean, within / neff, neff)
    both = out[True][2].index.intersection(out[False][2].index)
    d_raw = out[True][2][both] - out[False][2][both]; d_var = out[True][3][both] + out[False][3][both]
    bigd = out[True][4][both] >= CFG['big_neff']
    mu_d = np.average(d_raw[bigd], weights=1 / d_var[bigd])
    tau2_d = max(np.average((d_raw[bigd] - mu_d) ** 2, weights=1 / d_var[bigd]) - np.average(d_var[bigd], weights=1 / d_var[bigd]), 1e-6)
    delta = 100 * (mu_d + tau2_d / (tau2_d + d_var) * (d_raw - mu_d))
    pm = a[a[hp] & (agg.n >= 4)].copy(); pm['mw'] = pm.match_id.map(mw)
    pos = pm.assign(p=(pm.v > 0) * pm.mw).groupby(pid)[['p', 'mw']].sum()
    p0 = pos.p.sum() / pos.mw.sum(); cons = (pos.p + 5 * p0) / (pos.mw + 5)
    r = pd.DataFrame({'hp': out[True][0], 'n': out[True][1], 'base': out[False][0], 'cons': cons, 'nm': pos.mw, 'delta': delta})
    r.attrs['dsig'] = tau2_d > 1e-6
    return r

def boot(path, pid, hp, P, cnt, final):
    agg = prep(path, pid, hp, P, cnt)
    elig = final.index
    mids = agg.match_id.unique(); ranks = []
    for _ in range(CFG['B']):
        mw = pd.Series(rng.poisson(1.0, len(mids)), index=mids)
        rr = score(agg, pid, hp, mw); dsig = rr.attrs['dsig']; r = rr.reindex(elig)
        z = lambda s: (s - s.mean()) / s.std()
        a, c, d = CFG['comp_weights']
        s = a * z(r.hp) + c * z(r.cons) + (d * z(r.delta) if dsig else 0)
        ranks.append(s.rank(ascending=False))
    R = pd.concat(ranks, axis=1)
    res = final[['rank', 'player']].copy()
    res['rank_p10'] = R.quantile(0.1, axis=1); res['rank_median'] = R.median(axis=1); res['rank_p90'] = R.quantile(0.9, axis=1)
    res['p_top10'] = (R <= 10).mean(axis=1); res['p_top20'] = (R <= 20).mean(axis=1)
    return res

D = CFG['dir']
bat = pd.read_csv(f'{D}/rank_bat.csv', index_col=0); bowl = pd.read_csv(f'{D}/rank_bowl.csv', index_col=0)
rb = boot(f'{D}/val_bat.parquet', 'striker_id', 'hp_bat', 'P_bat', 'faced', bat)
rw = boot(f'{D}/val_bowl.parquet', 'bowler_id', 'hp_bowl', 'P_bowl', 'legal', bowl)
rb.to_csv(f'{D}/boot_bat.csv'); rw.to_csv(f'{D}/boot_bowl.csv')
pd.set_option('display.width', 200)
print(rb.head(20).round(2).to_string()); print(rw.head(20).round(2).to_string())
print('next most likely top-10 outside current top 10:')
print(rb[rb['rank'] > 10].sort_values('p_top10', ascending=False).head(5).round(2).to_string())
print(rw[rw['rank'] > 10].sort_values('p_top10', ascending=False).head(5).round(2).to_string())
