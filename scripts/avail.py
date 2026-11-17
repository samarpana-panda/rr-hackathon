"""Auction availability filter for the final lists.
Rule (user, 14 Nov): the top 10 is drawn from players NOT part of an IPL team for 2027,
whether or not they have registered for the auction yet.
contracted_2027 = IPL 2026 squads - released on 17 Nov 2026 (trades keep players contracted).
Usage: python3 avail.py <root dir holding v6/ v8/> <out csv>"""
import json, re, sys
import pandas as pd

D = '/home/claude/data/pack'
ROOT = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/work'
OUT = sys.argv[2] if len(sys.argv) > 2 else f'{ROOT}/final_available.csv'
R = json.load(open('/home/claude/work/releases_2027.json'))
norm = lambda x: re.sub('[^a-z]', '', str(x).lower())

mp = pd.read_csv(f'{D}/competition_team_player_mapping.csv')
comps = pd.read_csv(f'{D}/competitions.csv')
sq = mp[mp.comp_name == 'IPL 2026'].copy()
sq['n'] = sq.player_name.map(norm)

# released ids, matched within the releasing team's 2026 squad
rel_ids, missing = {}, []
for team, names in R['released'].items():
    for nm in names:
        h = sq[(sq.team_name == team) & (sq.n == norm(nm))]
        if len(h) == 1:
            rel_ids[int(h.player_id.iloc[0])] = team
        else:
            missing.append((team, nm))
contracted = {int(p): t for p, t in zip(sq.player_id, sq.team_name) if int(p) not in rel_ids}

# nationality: most frequent senior national side in T20I competitions
t20i = set(comps[comps.match_format == 'T20I'].comp_id)
nat_rows = mp[mp.comp_id.isin(t20i) & ~mp.team_name.str.endswith(' A')]
nat = nat_rows.groupby('player_id').team_name.agg(lambda s: s.value_counts().index[0])
NAT_FIX = {'Faf du Plessis': 'South Africa'}  # IPL-only rows in the pack; T20I caps predate the data
a_rows = mp[mp.team_name.str.endswith(' A')]
nat_a = a_rows.groupby('player_id').team_name.agg(lambda s: s.value_counts().index[0].replace(' A', '').strip())

sets = {k: {norm(x) for x in R[k]} for k in ('opted_out_2027', 'retired_ipl', 'banned', 'skipped_2026_flag')}

def country_of(pid, name):
    return NAT_FIX.get(name, nat.get(pid, nat_a.get(pid, 'India')))

def status(pid, name, last_seen, comp):
    n = norm(name)
    if pid in contracted:
        return 'excluded', f'contracted: {contracted[pid]}'
    country = country_of(pid, name)
    if country == 'Pakistan':
        return 'excluded', 'Pakistani (not eligible)'
    if n in sets['retired_ipl']:
        return 'excluded', 'retired from IPL'
    if n in sets['banned']:
        return 'excluded', 'banned'
    if n in sets['opted_out_2027']:
        return 'excluded', 'opted out of IPL 2027'
    if comp != 'alevel' and str(last_seen) < '2025-01-01':
        return 'excluded', 'inactive since before 2025'
    tag = f'released 17 Nov by {rel_ids[pid]}' if pid in rel_ids else ''
    if country == 'Bangladesh':
        return 'flag', ('Bangladesh status; ' + tag).strip('; ')
    if n in sets['skipped_2026_flag']:
        return 'flag', ('skipped IPL 2026; ' + tag).strip('; ')
    return 'available', tag

rows = []
for v in ('v6', 'v8'):
    for comp in ('ipl', 'intl', 'alevel'):
        for side, idc in (('bat', 'striker_id'), ('bowl', 'bowler_id')):
            try:
                r = pd.read_csv(f'{ROOT}/{v}/{comp}/rank_{side}.csv')
            except FileNotFoundError:
                continue
            try:
                bt = pd.read_csv(f'{ROOT}/{v}/{comp}/boot_{side}.csv')
                r = r.merge(bt[[idc, 'p_top10']], on=idc, how='left')
            except Exception:
                r['p_top10'] = float('nan')
            for _, x in r.iterrows():
                s, why = status(int(x[idc]), x.player, x.last_seen, comp)
                rows.append(dict(version=v, comp=comp, side=side, player_id=int(x[idc]), player=x.player,
                                 overall_rank=int(x['rank']), score=round(x.score, 3), hp_balls=x.hp_balls,
                                 hp_matches=x.hp_matches, p_top10=x.p_top10,
                                 nat=country_of(int(x[idc]), x.player),
                                 status=s, note=why))
out = pd.DataFrame(rows)
out['avail_rank'] = out[out.status != 'excluded'].groupby(['version', 'comp', 'side']).overall_rank.rank(method='first')
out.to_csv(OUT, index=False)
print('released matched:', len(rel_ids), '| unmatched:', missing)
print('contracted 2027 (known so far):', len(contracted))
for (v, c, s), g in out[out.avail_rank <= 10].groupby(['version', 'comp', 'side']):
    print(f'\n== {v} {c} {s}')
    print(g.sort_values('avail_rank')[['avail_rank', 'overall_rank', 'player', 'nat', 'score', 'hp_balls', 'p_top10', 'status', 'note']].to_string(index=False))
