"""Match-stakes multiplier per (match, team).  [NRR version, 18 Nov 2026]

IPL ties on points are broken by net run rate: real NRR to date, plus simulated margins for future games
(each simulated result reuses the runs/overs line of a real decided IPL league match). Everything else as stakes.py.


IPL league: Monte Carlo qualification maths over the actual remaining fixtures
(coin-flip results, random tiebreak since NRR is unknown).
  stake = dP(top4) + 0.5*dP(top2), where dP = P(q | win this game) - P(q | lose).
  Dead (eliminated even with a win, or nothing changes) -> 0.1 at any time.
  Must-win (a loss eliminates) -> 1.8 at any time.
  Window: team's final third of league games, or earlier if losing this and the
  next game would drop P(top4) below 10%. In window: stake<0.05 -> 0.1..1.0,
  else 1.0..1.8 scaled by stake. Outside window: 1.0.
Knockouts: IPL Q1 1.5, Eliminator/Q2/Semi 1.8, Final 2.0 (T20I finals/semis same).
Bilateral series & single-table tournaments (tri-series, qualifiers):
  qualify = win series / reach final. Dead (result can't change qualification) -> 0.1,
  must-win (a loss eliminates) -> 1.4, else 1.0.
Group-stage multi-group events (Asia Cups etc.): 1.0 (group structure not in data).
"""
import numpy as np, pandas as pd

RNG = np.random.default_rng(7)
NSIM = 4000


def _points_table(teams, played):
    pts = {t: 0.0 for t in teams}
    for h, a, w in played:
        if pd.isna(w) or w == 0:
            pts[h] += 1; pts[a] += 1
        else:
            pts[int(w)] += 2
    return pts


def _sim_qualify(teams, pts, remaining, forced, ks, nsim=NSIM):
    """forced: list of (home, away, winner) applied first. Returns {k: {team: P(top k)}}."""
    idx = {t: i for i, t in enumerate(teams)}
    base = np.array([pts[t] for t in teams], float)
    P = np.tile(base, (nsim, 1))
    for h, a, w in forced:
        P[:, idx[w]] += 2
    for h, a in remaining:
        hw = RNG.random(nsim) < 0.5
        P[hw, idx[h]] += 2
        P[~hw, idx[a]] += 2
    P = P + RNG.random(P.shape) * 0.01  # random tiebreak (NRR unknown)
    rank = (-P).argsort(1).argsort(1)
    return {k: {t: (rank[:, idx[t]] < k).mean() for t in teams} for k in ks}


LINES = {}      # match_id -> {team: (runs scored, balls faced, runs conceded, balls bowled)}; NRR rules: all out = full quota
POOL = None     # array of real decided IPL league lines: (winner runs, winner balls, loser runs, loser balls)


def _sim_qualify_nrr(teams, pts, agg, remaining, forced, ks, nsim=NSIM):
    """Like _sim_qualify, but ties on points are broken by simulated end-of-league NRR."""
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    P = np.tile(np.array([pts[t] for t in teams], float), (nsim, 1))
    RS = np.tile(np.array([agg[t][0] for t in teams], float), (nsim, 1)); BF = np.tile(np.array([agg[t][1] for t in teams], float), (nsim, 1))
    RC = np.tile(np.array([agg[t][2] for t in teams], float), (nsim, 1)); BB = np.tile(np.array([agg[t][3] for t in teams], float), (nsim, 1))
    games = [(h, a, w) for h, a, w in forced] + [(h, a, None) for h, a in remaining]
    rows = np.arange(nsim)
    for h, a, w in games:
        hw = np.full(nsim, w == h) if w is not None else RNG.random(nsim) < 0.5
        win = np.where(hw, idx[h], idx[a]); lose = np.where(hw, idx[a], idx[h])
        P[rows, win] += 2
        L = POOL[RNG.integers(0, len(POOL), nsim)]
        RS[rows, win] += L[:, 0]; BF[rows, win] += L[:, 1]; RC[rows, win] += L[:, 2]; BB[rows, win] += L[:, 3]
        RS[rows, lose] += L[:, 2]; BF[rows, lose] += L[:, 3]; RC[rows, lose] += L[:, 0]; BB[rows, lose] += L[:, 1]
    nrr = 6 * RS / np.maximum(BF, 1) - 6 * RC / np.maximum(BB, 1)
    key = P + 0.001 * np.tanh(nrr / 5)          # points first; NRR only separates equal points
    rank = (-key).argsort(1).argsort(1)
    return {k: {t: (rank[:, idx[t]] < k).mean() for t in teams} for k in ks}


def ipl_stakes(lg):
    """lg: league matches of one IPL season sorted chronologically."""
    teams = sorted(set(lg.home_team_id) | set(lg.away_team_id))
    rows, games_played = [], {t: 0 for t in teams}
    recs = list(lg[['match_id', 'home_team_id', 'away_team_id', 'win_team_id']].itertuples(index=False))
    n_games = {t: ((lg.home_team_id == t) | (lg.away_team_id == t)).sum() for t in teams}
    for i, (mid, h, a, w) in enumerate(recs):
        pts = _points_table(teams, [(r[1], r[2], r[3]) for r in recs[:i]])
        agg = {t: [0.0, 0.0, 0.0, 0.0] for t in teams}
        for r in recs[:i]:
            for t, line in LINES.get(r[0], {}).items():
                if t in agg:
                    agg[t] = [x + y for x, y in zip(agg[t], line)]
        rem = [(r[1], r[2]) for r in recs[i + 1:]]
        res_h = _sim_qualify_nrr(teams, pts, agg, rem, [(h, a, h)], [2, 4])
        res_a = _sim_qualify_nrr(teams, pts, agg, rem, [(h, a, a)], [2, 4])
        for team, opp in [(h, a), (a, h)]:
            games_played[team] += 1
            win = res_h if team == h else res_a
            lose = res_a if team == h else res_h
            d4 = win[4][team] - lose[4][team]
            d2 = win[2][team] - lose[2][team]
            stake = max(d4, 0) + 0.5 * max(d2, 0)
            # danger: lose this and the next game
            nxt = [r for r in recs[i + 1:] if team in (r[1], r[2])][:1]
            forced = [(h, a, opp)] + [(r[1], r[2], r[2] if r[1] == team else r[1]) for r in nxt]
            rem2 = [(r[1], r[2]) for r in recs[i + 1:] if not (nxt and r[0] == nxt[0][0])]
            p_danger = _sim_qualify_nrr(teams, pts, agg, rem2, forced, [4], nsim=1500)[4][team]
            final_third = games_played[team] > n_games[team] * 2 / 3
            in_window = final_third or p_danger < 0.10
            p4w, p4l = win[4][team], lose[4][team]
            if p4w < 0.002:
                mult, why = 0.1, 'ipl_dead'                     # eliminated even with a win
            elif p4l < 0.02:
                mult, why = 1.8, 'ipl_must_win'                 # a loss eliminates: a de facto eliminator
            elif stake < 0.005:
                mult, why = 0.1, 'ipl_dead'                     # result changes nothing (already through, top-2 settled)
            elif not in_window:
                mult, why = 1.0, 'ipl_normal'
            elif stake < 0.05:
                mult, why = 0.1 + 0.9 * stake / 0.05, 'ipl_low_stakes'
            else:
                mult, why = 1.0 + 0.8 * min((stake - 0.05) / 0.45, 1), 'ipl_race'
            rows.append(dict(match_id=mid, team_id=team, stake=stake, p_top4_if_win=win[4][team],
                             p_top4_if_lose=lose[4][team], p_top4_lose2=p_danger,
                             in_window=in_window, stakes_mult=mult, stakes_reason=why))
    return rows


def table_stakes(lg, k, bilateral):
    teams = sorted(set(lg.home_team_id) | set(lg.away_team_id))
    recs = list(lg[['match_id', 'home_team_id', 'away_team_id', 'win_team_id']].itertuples(index=False))
    rows = []
    for i, (mid, h, a, w) in enumerate(recs):
        pts = _points_table(teams, [(r[1], r[2], r[3]) for r in recs[:i]])
        rem = [(r[1], r[2]) for r in recs[i + 1:]]
        if bilateral:  # series win = strictly more points; exact enumeration is cheap
            res = {}
            for winner in (h, a):
                p = dict(pts); p[winner] += 2
                n = len(rem); sc = {t: 0.0 for t in teams}
                for mask in range(2 ** n):
                    q = dict(p)
                    for j, (x, y) in enumerate(rem):
                        q[x if (mask >> j) & 1 else y] += 2
                    t1, t2 = teams
                    if q[t1] > q[t2]: sc[t1] += 1
                    elif q[t2] > q[t1]: sc[t2] += 1
                    else: sc[t1] += .5; sc[t2] += .5  # drawn series: shared
                res[winner] = {t: v / 2 ** n for t, v in sc.items()}
            win_p = {h: res[h][h], a: res[a][a]}; lose_p = {h: res[a][h], a: res[h][a]}
        else:
            rh = _sim_qualify(teams, pts, rem, [(h, a, h)], [k])[k]
            ra = _sim_qualify(teams, pts, rem, [(h, a, a)], [k])[k]
            win_p = {h: rh[h], a: ra[a]}; lose_p = {h: ra[h], a: rh[a]}
        for team in (h, a):
            wp, lp = win_p[team], lose_p[team]
            if abs(wp - lp) < 1e-9:
                mult, why = 0.1, 'dead'
            elif lp < 1e-9:
                mult, why = 1.4, 'must_win'
            else:
                mult, why = 1.0, 'normal'
            rows.append(dict(match_id=mid, team_id=team, stake=wp - lp, p_top4_if_win=wp,
                             p_top4_if_lose=lp, stakes_mult=mult, stakes_reason=why))
    return rows


KO = {'Final': 2.0, 'Eliminator': 1.8, 'Semi Final': 1.8}


def load_lines(balls_path, m, c):
    global POOL
    b = pd.read_parquet(balls_path, columns=['match_id', 'innings', 'batting_team_id', 'bowling_team_id', 'total', 'legal', 'wkt', 'inn_balls', 'is_so'])
    b = b[~b.is_so.fillna(False).astype(bool)]
    x = b.groupby(['match_id', 'innings', 'batting_team_id', 'bowling_team_id']).agg(
        runs=('total', 'sum'), balls=('legal', 'sum'), w=('wkt', 'sum'), quota=('inn_balls', 'first')).reset_index()
    x['balls_nrr'] = np.where(x.w >= 10, x.quota, x.balls)       # all out counts as the full quota of overs
    mm = m.set_index('match_id')
    decided = set(mm.index[mm.win_team_id.notna() & (mm.win_team_id != 0)])
    pool = []
    for mid, g in x.groupby('match_id'):
        if len(g) != 2 or mid not in decided:
            continue                                            # no-results and one-innings games don't count in NRR
        d = {}
        for r in g.itertuples():
            d.setdefault(r.batting_team_id, [0, 0, 0, 0]); d.setdefault(r.bowling_team_id, [0, 0, 0, 0])
            d[r.batting_team_id][0] += r.runs; d[r.batting_team_id][1] += r.balls_nrr
            d[r.bowling_team_id][2] += r.runs; d[r.bowling_team_id][3] += r.balls_nrr
        LINES[mid] = d
        w = mm.at[mid, 'win_team_id']
        comp = c.set_index('comp_id').comp_name.get(mm.at[mid, 'comp_id'], '')
        if str(comp).startswith('IPL') and mm.at[mid, 'stage'] == 'League' and int(w) in d:
            lo = [t for t in d if t != int(w)][0]
            pool.append(d[int(w)][:2] + d[lo][:2])
    POOL = np.array(pool, float)
    print('NRR lines', len(LINES), 'IPL margin pool', len(POOL))


def build(m, c):
    m = m.merge(c[['comp_id', 'comp_name', 'match_format']], on='comp_id')
    m['match_date'] = pd.to_datetime(m.match_date)
    m = m.sort_values(['comp_id', 'match_date', 'match_number'])
    rows = []
    for cid, g in m.groupby('comp_id'):
        name = g.comp_name.iloc[0]
        teams = set(g.home_team_id) | set(g.away_team_id)
        ko = g[g.stage.isin(['Final', 'Eliminator', 'Semi Final', 'Qualifier'])]
        q = ko[ko.stage == 'Qualifier'].sort_values('match_number')
        for j, r in enumerate(ko.itertuples()):
            if r.stage == 'Qualifier':
                mult = 1.5 if r.match_id == q.match_id.iloc[0] else 1.8
            else:
                mult = KO[r.stage]
            for t in (r.home_team_id, r.away_team_id):
                rows.append(dict(match_id=r.match_id, team_id=t, stakes_mult=mult, stakes_reason='knockout_' + r.stage))
        lg = g[~g.match_id.isin(ko.match_id)]
        if name.startswith('IPL'):
            rows += ipl_stakes(lg)
        elif len(teams) == 2:
            rows += table_stakes(lg, 1, bilateral=True)
        elif lg.stage.eq('League').all() and len(teams) <= 4 and len(lg) == len(teams) * (len(teams) - 1):
            # single-table double round robin (tri-series / quad qualifiers)
            k = 2 if len(ko) else 1
            rows += table_stakes(lg, k, bilateral=False)
        else:
            for r in lg.itertuples():
                for t in (r.home_team_id, r.away_team_id):
                    rows.append(dict(match_id=r.match_id, team_id=t, stakes_mult=1.0, stakes_reason='group_stage'))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import sys
    d = sys.argv[1]
    load_lines(sys.argv[2], pd.read_csv(f'{d}/matches.csv'), pd.read_csv(f'{d}/competitions.csv'))
    s = build(pd.read_csv(f'{d}/matches.csv'), pd.read_csv(f'{d}/competitions.csv'))
    s.to_csv(sys.argv[3] if len(sys.argv) > 3 else '/home/claude/final_nrr/stakes.csv', index=False)
    print(s.stakes_reason.value_counts())
    print(s.groupby('stakes_reason').stakes_mult.describe())
