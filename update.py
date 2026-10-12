"""Stach Fantasy Football HQ: schedule-aware, injury-aware Monte Carlo model.

This is an independent estimate, not ESPN's official playoff projection.
"""
import datetime as dt
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

LEAGUE = 2108436489
YEAR = 2026
SIMS = 20000
OUT = Path(__file__).resolve().parent / 'data' / 'dashboard.json'
URL = f'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{YEAR}/segments/0/leagues/{LEAGUE}'
STARTER_SLOTS = {0, 2, 4, 6, 23, 16, 17}
POSITION_PRIORS = {1: 17.0, 2: 11.0, 3: 11.0, 4: 8.5, 5: 8.0, 16: 7.0}
FLEX_POSITIONS = (2, 3, 4)


def fetch():
    cookies = {}
    if os.getenv('ESPN_S2') and os.getenv('ESPN_SWID'):
        cookies = {'espn_s2': os.environ['ESPN_S2'], 'SWID': os.environ['ESPN_SWID']}
    views = ['mTeam', 'mRoster', 'mMatchupScore', 'mSettings', 'mStatus', 'mStandings']
    response = requests.get(URL, params=[('view', v) for v in views], cookies=cookies,
                            headers={'User-Agent': 'StachFantasyDashboard/3.0'}, timeout=40)
    response.raise_for_status()
    data = response.json()
    if len(data.get('teams', [])) != 10:
        raise ValueError('ESPN did not provide all ten teams')
    return data


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def name(team):
    return team.get('name') or (str(team.get('location', '')) + ' ' + str(team.get('nickname', ''))).strip() or f"Team {team['id']}"


def parse_players(team, current):
    result = []
    for entry in (team.get('roster') or {}).get('entries', []):
        player = ((entry.get('playerPoolEntry') or {}).get('player') or {})
        if not player:
            continue
        pos = int(player.get('defaultPositionId') or -1)
        if pos not in POSITION_PRIORS:
            continue
        status = str(player.get('injuryStatus') or 'ACTIVE').upper()
        projections = []
        actuals = []
        for stat in player.get('stats') or []:
            period = int(stat.get('scoringPeriodId') or 0)
            source = int(stat.get('statSourceId', -1) if stat.get('statSourceId') is not None else -1)
            value = stat.get('appliedTotal')
            if value is None:
                continue
            if source == 1 and period == current:
                projections.append(number(value))
            if source == 0 and 1 <= period < current:
                actuals.append(number(value))
        projection = projections[0] if projections else None
        prior = POSITION_PRIORS[pos]
        if actuals:
            history = float(np.mean(actuals[-5:]))
            historical = (len(actuals) * history + 3 * prior) / (len(actuals) + 3)
        else:
            historical = prior
        # A zero current-week projection is NOT evidence of zero rest-of-season value.
        # It can mean bye, injury, or missing ESPN projection.
        if projection is not None and projection > 0.5:
            expected = 0.60 * projection + 0.40 * historical
        else:
            expected = historical
        expected = max(1.0, min(35.0, expected))
        result.append({
            'name': player.get('fullName') or 'Unknown',
            'position': pos,
            'injury': status,
            'slot': entry.get('lineupSlotId'),
            'projected': round(projection, 2) if projection is not None else None,
            '_mean': expected,
        })
    return result


def lineup_value(players, available):
    """Optimize QB, 2 RB, 2 WR, TE, FLEX, DST, K without reusing players."""
    pool = defaultdict(list)
    for player, active in zip(players, available):
        if active:
            pool[player['position']].append(player['_mean'])
    for values in pool.values():
        values.sort(reverse=True)
    chosen = []
    for pos, count in ((1, 1), (2, 2), (3, 2), (4, 1), (16, 1), (5, 1)):
        values = pool[pos]
        chosen.extend(values[:count])
        pool[pos] = values[count:]
        # Replacement-level placeholders avoid treating an incomplete roster as 0 points.
        if len(values) < count:
            chosen.extend([POSITION_PRIORS[pos] * 0.65] * (count - len(values)))
    flex = max((pool[pos][0] for pos in FLEX_POSITIONS if pool[pos]), default=7.0)
    return float(sum(chosen) + flex)


def return_week(status, rng):
    """Draw an uncertain return delay. No claim of a confirmed medical timetable."""
    if status in ('INJURY_RESERVE', 'IR'):
        return int(rng.choice([3, 5, 7, 11, 20], p=[.12, .23, .25, .20, .20]))
    if status in ('OUT', 'DOUBTFUL'):
        return int(rng.choice([1, 2, 3, 5, 9], p=[.20, .35, .25, .15, .05]))
    if status in ('SUSPENSION',):
        return int(rng.choice([2, 4, 7, 12], p=[.25, .35, .25, .15]))
    if status in ('QUESTIONABLE', 'PROBABLE'):
        return int(rng.choice([0, 1, 2], p=[.65, .25, .10]))
    return 0


def compute(data):
    teams = data['teams']
    ids = [int(t['id']) for t in teams]
    if len(set(ids)) != 10:
        raise ValueError('Duplicate ESPN team IDs')
    lookup = {team_id: i for i, team_id in enumerate(ids)}
    settings = (data.get('settings') or {}).get('scheduleSettings') or {}
    regular = int(settings.get('matchupPeriodCount') or 14)
    playoff_slots = int(settings.get('playoffTeamCount') or 6)
    if regular != 14 or playoff_slots != 6:
        raise ValueError('League playoff settings have changed; review model')
    status = data.get('status') or {}
    current = max(1, min(regular + 1, int(status.get('currentMatchupPeriod') or 1)))

    wins = np.zeros(10, dtype=int)
    losses = np.zeros(10, dtype=int)
    ties = np.zeros(10, dtype=int)
    points_for = np.zeros(10)
    points_against = np.zeros(10)
    histories = [[] for _ in teams]
    future = defaultdict(list)
    completed = []
    for match in data.get('schedule') or []:
        home = match.get('home') or {}
        away = match.get('away') or {}
        hi, ai = home.get('teamId'), away.get('teamId')
        if hi not in lookup or ai not in lookup:
            continue
        week = int(match.get('matchupPeriodId') or 0)
        if not 1 <= week <= regular:
            continue
        h, a = lookup[hi], lookup[ai]
        winner = str(match.get('winner') or 'UNDECIDED')
        if week < current and winner in ('HOME', 'AWAY', 'TIE'):
            hs = number(home.get('totalPoints'))
            ass = number(away.get('totalPoints'))
            if winner == 'HOME':
                wins[h] += 1; losses[a] += 1
            elif winner == 'AWAY':
                wins[a] += 1; losses[h] += 1
            else:
                ties[h] += 1; ties[a] += 1
            points_for[h] += hs; points_for[a] += ass
            points_against[h] += ass; points_against[a] += hs
            histories[h].append((week, hs)); histories[a].append((week, ass))
            completed.append({'week': week, 'home': hi, 'away': ai,
                              'homeScore': hs, 'awayScore': ass})
        elif week >= current:
            future[week].append((h, a))

    # Do not silently simulate an incomplete or duplicated schedule.
    for week in range(current, regular + 1):
        games = future[week]
        participants = sorted(j for pair in games for j in pair)
        if len(games) != 5 or participants != list(range(10)):
            raise ValueError(f'ESPN schedule incomplete for week {week}')

    for i, team in enumerate(teams):
        overall = ((team.get('record') or {}).get('overall') or {})
        reported = int(overall.get('gamesPlayed') or 0)
        observed = int(wins[i] + losses[i] + ties[i])
        if reported > observed:
            raise ValueError('ESPN records and completed matchups disagree; refusing to publish misleading odds')

    players = [parse_players(t, current) for t in teams]
    games_played = wins + losses + ties
    ppg = np.array([points_for[i] / games_played[i] if games_played[i] else 120.0
                    for i in range(10)])
    league_mean = float(np.mean(ppg))
    roster_healthy = np.array([lineup_value(p, [True] * len(p)) for p in players])
    # Avoid overreacting to just four results; normalize uncertain roster estimates.
    roster_center = float(np.mean(roster_healthy))
    roster_adjusted = league_mean + (roster_healthy - roster_center) * 0.75
    baseline = np.zeros(10)
    for i in range(10):
        games = int(games_played[i])
        history_est = (games * ppg[i] + 5 * league_mean) / (games + 5)
        baseline[i] = 0.55 * history_est + 0.45 * roster_adjusted[i]

    # Reproducible simulations for identical inputs; no daily random fluctuations.
    seed = 20261012 + current + int(round(float(points_for.sum()) * 100))
    rng = np.random.default_rng(seed)
    playoff_counts = np.zeros(10, dtype=int)
    bye_counts = np.zeros(10, dtype=int)
    champ_counts = np.zeros(10, dtype=int)
    win_totals = np.zeros(10)
    rank_totals = np.zeros(10)
    seed_brackets = Counter()

    for _ in range(SIMS):
        sw = wins.copy()
        st = ties.copy()
        sp = points_for.copy()
        # Season-long team uncertainty, rather than pretending our means are exact.
        season_bias = rng.normal(0, 5, 10)
        returns = [[return_week(p['injury'], rng) for p in roster] for roster in players]
        weekly_means = {}
        for week in range(current, regular + 4):
            delta = week - current
            values = []
            for i, roster in enumerate(players):
                active = [delta >= delay for delay in returns[i]]
                lineup = lineup_value(roster, active)
                roster_delta = (lineup - roster_healthy[i]) * 0.70
                values.append(max(65.0, baseline[i] + roster_delta + season_bias[i]))
            weekly_means[week] = values

        for week in range(current, regular + 1):
            means = weekly_means[week]
            for h, a in future[week]:
                hs, ass = np.maximum(0, rng.normal([means[h], means[a]], [23, 23]))
                hs = round(float(hs), 2)
                ass = round(float(ass), 2)
                if hs > ass:
                    sw[h] += 1
                elif ass > hs:
                    sw[a] += 1
                else:
                    st[h] += 1; st[a] += 1
                sp[h] += hs; sp[a] += ass

        order = sorted(range(10), key=lambda j: (sw[j] + .5 * st[j], sp[j]), reverse=True)
        win_totals += sw + .5 * st
        for rank, j in enumerate(order):
            rank_totals[j] += rank + 1
        seeds = tuple(order[:6])
        seed_brackets[seeds] += 1
        for j in seeds:
            playoff_counts[j] += 1
        for j in seeds[:2]:
            bye_counts[j] += 1

        def game(x, y, week):
            mx, my = weekly_means[week][x], weekly_means[week][y]
            sx, sy = rng.normal([mx, my], [23, 23])
            return x if sx > sy else y if sy > sx else (x if rng.random() < .5 else y)

        w36 = game(seeds[2], seeds[5], regular + 1)
        w45 = game(seeds[3], seeds[4], regular + 1)
        f1 = game(seeds[0], w45, regular + 2)
        f2 = game(seeds[1], w36, regular + 2)
        champ_counts[game(f1, f2, regular + 3)] += 1

    if sum(playoff_counts) != 6 * SIMS or sum(bye_counts) != 2 * SIMS or sum(champ_counts) != SIMS:
        raise AssertionError('Probability accounting failed')
    projected = seed_brackets.most_common(1)[0][0]
    result_teams = []
    for i, team in enumerate(teams):
        public_roster = [{k: p[k] for k in ('name', 'injury', 'slot', 'projected')}
                         for p in players[i]]
        result_teams.append({
            'id': ids[i], 'name': name(team), 'wins': int(wins[i]),
            'losses': int(losses[i]), 'ties': int(ties[i]),
            'pointsFor': round(float(points_for[i]), 2),
            'pointsAgainst': round(float(points_against[i]), 2),
            'ppg': round(float(ppg[i]), 1),
            'strength': round(float(baseline[i]), 1),
            'projectedWins': round(float(win_totals[i] / SIMS), 1),
            'averageRank': round(float(rank_totals[i] / SIMS), 2),
            'playoffPct': round(float(playoff_counts[i] * 100 / SIMS), 1),
            'byePct': round(float(bye_counts[i] * 100 / SIMS), 1),
            'champPct': round(float(champ_counts[i] * 100 / SIMS), 1),
            'roster': public_roster,
        })
    result_teams.sort(key=lambda t: (t['wins'] + t['ties'] / 2, t['pointsFor']), reverse=True)
    return {
        'mode': 'live', 'leagueId': LEAGUE, 'season': YEAR,
        'leagueName': 'Stach Fantasy Football',
        'updatedAt': dt.datetime.now(dt.timezone.utc).isoformat(),
        'throughWeek': current - 1, 'currentWeek': current,
        'regularSeasonWeeks': regular, 'simulations': SIMS,
        'teams': result_teams, 'projectedSeeds': [ids[j] for j in projected],
        'completedMatchups': completed,
        'method': ('Independent 20,000-season Monte Carlo estimate using ESPN records, '
                   'remaining schedule, recent scoring, player projections, all-roster lineup '
                   'optimization, uncertain injury returns, and team-strength uncertainty. '
                   'Return dates are hypothetical, not confirmed. This is not ESPN\'s official projection.'),
    }


if __name__ == '__main__':
    result = compute(fetch())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(OUT)
    print(f"Updated {len(result['teams'])} teams, {result['simulations']:,} simulations")
