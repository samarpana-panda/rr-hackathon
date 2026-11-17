"""Learn the pressure function from match outcomes (v3).

Idea: pressure = how much performing well in a situation matters to winning.
  For each ball, v = batting team's value vs an average side in the same state
      v = (runs - E[runs]) - (wicket - E[wicket]) * wicket_cost       (state-only GBMs, no team/player identity)
  Each side sees the ball through its own state; one shared network g (softplus MLP) maps a side's view to a pressure weight:
      batting view : [share of innings bowled, 2nd-innings flag, side's win prob, wickets down, dominance (+ = squeezed), personal_bat]
      bowling view : [same,                    same,               1 - win prob,   same,         -dominance,                personal_bowl]
      ball weight w = (g(bat view) + g(bowl view)) / 2
  Match model:  logit P(team batting first wins) = a + sum_balls sign * w * v   (sign +1 1st innings, -1 2nd)
  g is learned end-to-end; it can only fit results by up-weighting the situations where performance decides matches.
Training: 5-fold CV by match vs two baselines (flat weights; hand-set v1 formula). Final g = mean of 5 seeds on all data.
Stakes multipliers stay rule-based: P_side = g(side view) * stakes(team).
"""
import json, os, numpy as onp, pandas as pd
import autograd.numpy as np
from autograd import grad
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score

SRC = '/home/claude/final_nrr/balls_pressure.parquet'
OUT = os.environ.get('LP_OUT', '/home/claude/final_nrr/v3')
SEL = json.loads(os.environ.get('LP_FEATS', '[0,1,2,3,4,5]'))
os.makedirs(OUT, exist_ok=True)
b = pd.read_parquet(SRC)
b['grp_c'] = b.grp.map({'IPL': 0, 'T20I_full': 1, 'T20I_assoc': 2, 'A_teams': 3})
b['rrr_f'] = b.rrr.fillna(-1).clip(upper=4); b['need_f'] = onp.where(b.innings == 2, b.need, -1)

# ---------------------------------------------------------------- per-ball team value
i1 = b[b.innings == 1].copy()
i1['rem'] = (i1.groupby('match_id').total.transform('sum') - i1.runs_before) / i1.cf
wc = HistGradientBoostingRegressor(max_iter=200, monotonic_cst=[1, -1], random_state=0).fit(i1[['balls_left', 'wkts_before']], i1.rem)
e0 = wc.predict(onp.c_[b.balls_left, b.wkts_before]); e1 = wc.predict(onp.c_[b.balls_left, onp.minimum(b.wkts_before + 1, 10)])
b['wkt_cost'] = onp.clip(onp.where(b.wkts_before >= 9, e0, e0 - e1), 0.5, None) * b.cf
S = ['grp_c', 'innings', 'over_idx', 'legal_before', 'wkts_before', 'rrr_f', 'need_f', 'cf']
b['e_tot'] = HistGradientBoostingRegressor(max_iter=300, random_state=0, categorical_features=[0]).fit(b[S], b.total).predict(b[S])
b['e_wk'] = HistGradientBoostingClassifier(max_iter=300, random_state=0, categorical_features=[0]).fit(b[S], b.wkt).predict_proba(b[S])[:, 1]
b['v'] = (b.total - b.e_tot) - (b.wkt.astype(float) - b.e_wk) * b.wkt_cost

# ---------------------------------------------------------------- side views
x = (b.legal_before / b.inn_balls).values
inn2 = (b.innings == 2).astype(float).values
wk = (b.wkts_before / 10).values
dom = onp.clip(b.dom_raw.values / 15, -2, 2)
Fbat = onp.c_[x, inn2, b.wp.values, wk, dom, b.pers_bat.values]
Fbowl = onp.c_[x, inn2, 1 - b.wp.values, wk, -dom, b.pers_bowl.values]
FEATS = ['share_bowled', 'second_innings', 'own_win_prob', 'wickets_down', 'dominance_against', 'personal']
Fbat, Fbowl, FEATS = Fbat[:, SEL], Fbowl[:, SEL], [FEATS[i] for i in SEL]
mu_f, sd_f = Fbat.mean(0), Fbat.std(0)

# ---------------------------------------------------------------- pad into match tensors
dec = b.win_team_id.notna() & (b.win_team_id != 0)
first_bat = b[b.innings == 1].groupby('match_id').batting_team_id.first()
mids = onp.array(sorted(set(b.loc[dec, 'match_id']) & set(first_bat.index) & set(b.loc[b.innings == 2, 'match_id'])))
mpos = {m: i for i, m in enumerate(mids)}
sub = b[b.match_id.isin(mpos)]
y = onp.array([float(b.loc[b.match_id == m, 'win_team_id'].iat[0] == first_bat[m]) for m in mids])
L = sub.groupby('match_id').size().max(); M = len(mids)
idx_m = sub.match_id.map(mpos).values; idx_l = sub.groupby('match_id').cumcount().values
def pad(a, fill=0.0):
    out = onp.full((M, L) + a.shape[1:], fill); out[idx_m, idx_l] = a; return out
r = sub.index.values
FB, FW = pad((Fbat[r] - mu_f) / sd_f).astype('float32'), pad((Fbowl[r] - mu_f) / sd_f).astype('float32')
SV = pad((onp.where(sub.innings == 1, 1, -1) * sub.v.values).astype(float))   # signed value, 0 on padding
MASK = pad(onp.ones(len(sub)))
HAND = pad((0.5 * (sub.raw_bat + sub.raw_bowl) * sub.innings.map({1: .85, 2: 1}) * sub.dead).values)
print('matches', M, 'max balls', L, 'batting-first win rate', y.mean().round(3))

# ---------------------------------------------------------------- model
H = 12
def init(seed):
    rs = onp.random.RandomState(seed); F = FB.shape[-1]
    return [rs.randn(F, H) * 0.3, onp.zeros(H), rs.randn(H, H) * 0.3, onp.zeros(H), onp.zeros((H, 1)), onp.zeros(1), onp.zeros(1),
            onp.array([C0])]   # last layer starts at zero -> g flat at start = flat baseline

def g(p, F):
    h = np.tanh(np.dot(F, p[0]) + p[1]); h = np.tanh(np.dot(h, p[2]) + p[3])
    z = np.dot(h, p[4])[..., 0] + p[5][0]
    return np.logaddexp(0, z) * 0.1 / np.log(2)      # softplus, ~0.1 at init (scale of ~1 run of value per ball)

def logits(p, ix):
    w = 0.5 * (g(p, FB[ix]) + g(p, FW[ix])) * MASK[ix]
    w = w / (np.sum(w) / np.sum(MASK[ix]))           # shape only: mean weight 1 over the batch
    return p[6][0] + np.exp(p[7][0]) * np.sum(w * SV[ix], axis=1)

def loss(p, ix, lam):
    z = logits(p, ix); yy = y[ix]
    nll = np.mean(np.logaddexp(0, z) - yy * z)
    reg = lam * sum(np.sum(q ** 2) for q in (p[0], p[2], p[4]))
    return nll + reg

dloss = grad(loss)
def adam(p, ix, iters, lam, lr=0.005, track=None, every=50):
    m = [onp.zeros_like(q) for q in p]; v = [onp.zeros_like(q) for q in p]; hist = []
    for t in range(1, iters + 1):
        gr = dloss(p, ix, lam)
        for i in range(len(p)):
            m[i] = 0.9 * m[i] + 0.1 * gr[i]; v[i] = 0.999 * v[i] + 0.001 * gr[i] ** 2
            p[i] = p[i] - lr * (m[i] / (1 - 0.9 ** t)) / (onp.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        if track is not None and t % every == 0:
            hist.append(track(p))
    return p, hist

def fit_scale(wpad, tr):  # baselines: logit = a + c * sum(w * SV)
    s = onp.sum(wpad * SV, axis=1)
    from sklearn.linear_model import LogisticRegression
    lr_ = LogisticRegression(C=1e6).fit(s[tr, None], y[tr]); return lr_, s

from sklearn.linear_model import LogisticRegression
C0 = float(onp.log(LogisticRegression(C=1e6).fit(onp.sum(SV, 1)[:, None], y).coef_[0, 0]))
# ---------------------------------------------------------------- cross-validation
folds = list(GroupKFold(5).split(mids, y, mids))
LAMS, ITERS, EVERY = [1e-3], 600, 25
cv = {'flat': onp.zeros(M), 'hand': onp.zeros(M)}
for name, wpad in (('flat', MASK), ('hand', HAND)):
    for tr, te in folds:
        lr_, s = fit_scale(wpad, tr); cv[name][te] = lr_.predict_proba(s[te, None])[:, 1]
curves, BESTP = {}, {}
for lam in LAMS:
    P = onp.zeros((ITERS // EVERY, M))
    for k, (tr, te) in enumerate(folds):
        p = init(k)
        _, hist = adam(p, tr, ITERS, lam, track=lambda q: 1 / (1 + onp.exp(-logits(q, te))), every=EVERY)
        for j, h in enumerate(hist): P[j, te] = h
    BESTP[lam] = P
    curves[lam] = [log_loss(y, onp.clip(P[j], 1e-6, 1 - 1e-6)) for j in range(len(P))]
    print('lambda', lam, 'cv logloss by checkpoint', onp.round(curves[lam], 4).tolist())
best_lam = min(LAMS, key=lambda l: min(curves[l])); best_j = int(onp.argmin(curves[best_lam])); best_it = (best_j + 1) * EVERY
cv['learned'] = BESTP[best_lam][best_j]
res = {n: dict(logloss=round(log_loss(y, onp.clip(q, 1e-6, 1 - 1e-6)), 4), auc=round(roc_auc_score(y, q), 4),
               acc=round(((q > .5) == y).mean(), 4)) for n, q in cv.items()}
res['best'] = dict(lam=best_lam, iters=best_it)
print(json.dumps(res, indent=1))

# ---------------------------------------------------------------- final ensemble on all data, apply to every ball
allix = onp.arange(M); gb, gw = onp.zeros(len(b)), onp.zeros(len(b)); params = []
Fb_all, Fw_all = (Fbat - mu_f) / sd_f, (Fbowl - mu_f) / sd_f
for seed in range(3):
    p, _ = adam(init(100 + seed), allix, best_it, best_lam); params.append(p)
    gb += g(p, Fb_all) / 3; gw += g(p, Fw_all) / 3
norm = onp.mean(0.5 * (gb + gw)); gb, gw = gb / norm, gw / norm
b['g_bat'], b['g_bowl'] = gb, gw
b['P_bat'] = b.g_bat * b.stakes_bat; b['P_bowl'] = b.g_bowl * b.stakes_bowl
for s in ('bat', 'bowl'):
    b[f'hp_{s}'] = b[f'P_{s}'] >= b[f'P_{s}'].quantile(0.75)
b.drop(columns=['e_tot', 'e_wk', 'v']).to_parquet(f'{OUT}/balls_pressure.parquet')
b[['match_id', 'innings', 'over_idx', 'legal_before', 'inn_balls', 'wp', 'dom_raw', 'pers_bat', 'pers_bowl', 'wkts_before', 'g_bat', 'g_bowl', 'v']].to_parquet(f'{OUT}/g_ball.parquet')
onp.save(f'{OUT}/params.npy', onp.array([onp.concatenate([q.ravel() for q in p]) for p in params]))
json.dump(dict(cv=res, feats=FEATS, mu=mu_f.tolist(), sd=sd_f.tolist(), hidden=H), open(f'{OUT}/model_meta.json', 'w'), indent=1)
