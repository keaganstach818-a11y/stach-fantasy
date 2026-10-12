"""Stach Fantasy Football HQ — ESPN season simulator.

Future injury returns are unknown: blend current healthy-lineup information
with historical scoring instead of projecting today's lineup unchanged.
"""
import datetime
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import requests

LEAGUE = 2108436489
YEAR = 2026
SIMS = 20000
SEED = 20261013
# Return probabilities are assumptions, not medical predictions.
RETURN_HAZARD = {
    "QUESTIONABLE": 0.75, "DOUBTFUL": 0.35, "OUT": 0.30,
    "INJURY_RESERVE": 0.12, "SUSPENSION": 0.12,
}
POSITION = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
OUT = Path(__file__).parent / 'data' / 'dashboard.json'
URL = (f'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/'
       f'seasons/{YEAR}/segments/0/leagues/{LEAGUE}')
STARTER_SLOTS = {0, 2, 4, 6, 23, 16, 17}
INACTIVE = {'OUT', 'INJURY_RESERVE', 'SUSPENSION'}


def load():
    views = ['mTeam', 'mRoster', 'mMatchupScore', 'mSettings',
             'mStatus', 'mStandings']
    cookies = {}
    if os.getenv('ESPN_S2') and os.getenv('ESPN_SWID'):
        cookies = {'espn_s2': os.environ['ESPN_S2'],
                   'SWID': os.environ['ESPN_SWID']}
    response = requests.get(URL, params=[('view', v) for v in views],
                            cookies=cookies,
                            headers={'User-Agent': 'StachFantasyDashboard/3.0'},
                            timeout=35)
    response.raise_for_status()
    data = response.json()
    if len(data.get('teams', [])) != 10:
        raise RuntimeError('ESPN did not return all 10 teams.')
    return data


def team_name(team):
    return (team.get('name') or
            (team.get('location', '') + ' ' +
             team.get('nickname', '')).strip() or
            f"Team {team['id']}")


def player_baseline(player, current_week):
    """Use weekly historical points and current projection when available."""
    stats = player.get('stats') or []
    projections = [float(s['appliedTotal']) for s in stats
                   if s.get('statSourceId') == 1
                   and s.get('scoringPeriodId') == current_week
                   and s.get('appliedTotal') is not None]
    actual = [float(s['appliedTotal']) for s in stats
              if s.get('statSourceId') == 0
              and isinstance(s.get('scoringPeriodId'), int)
              and 1 <= s['scoringPeriodId'] < current_week
              and s.get('appliedTotal') is not None]
    recent = actual[-4:]
    historical = float(np.mean(recent)) if recent else None
    projected = projections[0] if projections else None
    if projected is not None and historical is not None:
        result = 0.55 * projected + 0.45 * historical
    elif projected is not None:
        result = projected
    elif historical is not None:
        result = historical
    else:
        result = None
    return round(max(0.0, min(50.0, result)), 2) if result is not None else None


def lineup_total(players, week_offset, return_weeks):
    """Draw possible player availability and choose a legal fantasy lineup."""
    eligible = {position: [] for position in ('QB', 'RB', 'WR', 'TE', 'DST', 'K')}
    for player_index, player in enumerate(players):
        position = player.get('position')
        value = player.get('baseline')
        if position not in eligible or value is None:
            continue
        status = (player.get('injury') or 'ACTIVE').upper()
        if status in RETURN_HAZARD and week_offset < return_weeks.get(player_index, 999):
            continue
        eligible[position].append(float(value))
    for values in eligible.values():
        values.sort(reverse=True)
    required = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'DST': 1, 'K': 1}
    score = 0.0
    used = {}
    for position, count in required.items():
        selected = eligible[position][:count]
        used[position] = len(selected)
        score += sum(selected)
    flex = [v for position in ('RB', 'WR', 'TE')
            for v in eligible[position][used[position]:]]
    if flex:
        score += max(flex)
    # Missing position projections make a lineup estimate unreliable.
    filled = sum(used.values()) + bool(flex)
    return score if filled == 9 and 60 <= score <= 240 else None


def roster(team, current_week):
    players = []
    for entry in (team.get('roster') or {}).get('entries', []):
        player = ((entry.get('playerPoolEntry') or {}).get('player') or {})
        if not player:
            continue
        projections = [s for s in player.get('stats', [])
                       if s.get('statSourceId') == 1
                       and s.get('scoringPeriodId') == current_week
                       and s.get('appliedTotal') is not None]
        projection = (float(projections[0]['appliedTotal'])
                      if projections else None)
        players.append({
            'name': player.get('fullName', 'Unknown'),
            'injury': player.get('injuryStatus', 'ACTIVE'),
            'slot': entry.get('lineupSlotId'),
            'position': POSITION.get(player.get('defaultPositionId'), ''),
            'baseline': player_baseline(player, current_week),
            'projected': round(projection, 2)
                         if projection is not None else None,
        })
    return players


def estimate_strength(ppg, recent, league_mean, players):
    """Past scoring anchors the model; future rosters vary by week."""
    return float(0.60 * ppg + 0.25 * recent + 0.15 * league_mean)


def run(data):
    teams = data['teams']
    ids = [int(t['id']) for t in teams]
    index = {team_id: i for i, team_id in enumerate(ids)}
    n = len(ids)
    schedule_settings = ((data.get('settings') or {})
                         .get('scheduleSettings') or {})
    regular_weeks = int(schedule_settings.get('matchupPeriodCount') or 14)
    playoff_teams = int(schedule_settings.get('playoffTeamCount') or 6)
    if n != 10 or playoff_teams != 6 or regular_weeks != 14:
        raise RuntimeError('League format has changed.')

    status = data.get('status') or {}
    current = int(status.get('currentMatchupPeriod') or 1)
    current = max(1, min(current, regular_weeks + 1))
    wins = np.zeros(n, dtype=int)
    losses = np.zeros(n, dtype=int)
    ties = np.zeros(n, dtype=int)
    points_for = np.zeros(n, dtype=float)
    points_against = np.zeros(n, dtype=float)
    history = {team_id: [] for team_id in ids}
    future = []
    completed = []

    for matchup in data.get('schedule') or []:
        home = matchup.get('home') or {}
        away = matchup.get('away') or {}
        home_id, away_id = home.get('teamId'), away.get('teamId')
        if home_id not in index or away_id not in index:
            continue
        week = int(matchup.get('matchupPeriodId') or 0)
        if not 1 <= week <= regular_weeks:
            continue
        h, a = index[home_id], index[away_id]
        winner = matchup.get('winner', 'UNDECIDED')
        hs = float(home.get('totalPoints') or 0)
        as_ = float(away.get('totalPoints') or 0)
        if week < current and winner in ('HOME', 'AWAY', 'TIE'):
            if winner == 'HOME':
                wins[h] += 1
                losses[a] += 1
            elif winner == 'AWAY':
                wins[a] += 1
                losses[h] += 1
            else:
                ties[h] += 1
                ties[a] += 1
            points_for[h] += hs
            points_for[a] += as_
            points_against[h] += as_
            points_against[a] += hs
            history[home_id].append((week, hs))
            history[away_id].append((week, as_))
            completed.append({'week': week, 'home': home_id,
                              'away': away_id, 'homeScore': hs,
                              'awayScore': as_})
        elif week >= current:
            future.append((week, h, a))

    for team in teams:
        i = index[int(team['id'])]
        record = ((team.get('record') or {}).get('overall') or {})
        played = int(record.get('gamesPlayed') or 0)
        observed = int(wins[i] + losses[i] + ties[i])
        if observed == 0 and played > 0:
            wins[i] = int(record.get('wins') or 0)
            losses[i] = int(record.get('losses') or 0)
            ties[i] = int(record.get('ties') or 0)
            points_for[i] = float(record.get('pointsFor') or 0)
            points_against[i] = float(record.get('pointsAgainst') or 0)
        elif played > observed:
            raise RuntimeError('ESPN schedule is missing completed results.')

    for week in range(current, regular_weeks + 1):
        games = [(h, a) for w, h, a in future if w == week]
        participants = sorted(j for pair in games for j in pair)
        if len(games) != n // 2 or participants != list(range(n)):
            raise RuntimeError(f'Incomplete ESPN schedule for week {week}.')

    roster_data = [roster(t, current) for t in teams]
    played = wins + losses + ties
    ppg = np.array([points_for[i] / played[i]
                    if played[i] else 120.0 for i in range(n)])
    league_mean = float(np.mean(ppg))
    strength = np.zeros(n, dtype=float)
    for i in range(n):
        recent_games = sorted(history[ids[i]],
                              key=lambda x: x[0], reverse=True)[:3]
        if recent_games:
            weights = [0.55, 0.30, 0.15][:len(recent_games)]
            recent = float(np.average([s for _, s in recent_games],
                                      weights=weights))
        else:
            recent = float(ppg[i])
        strength[i] = estimate_strength(float(ppg[i]), recent,
                                        league_mean, roster_data[i])

    rng = np.random.default_rng(SEED + current + int(np.sum(points_for)))
    playoff_counts = np.zeros(n)
    bye_counts = np.zeros(n)
    championship_counts = np.zeros(n)
    win_totals = np.zeros(n)
    rank_totals = np.zeros(n)
    seed_counts = [Counter() for _ in range(6)]
    bracket_counts = Counter()
    # Compute injury-aware team strength for each future week once per
    # simulation. Return dates are sampled rather than assumed certain.
    remaining_weeks = sorted({week for week, _, _ in future})

    def playoff_game(x, y):
        sx, sy = rng.normal([strength[x], strength[y]], [23, 23])
        return x if sx > sy else y if sy > sx else (x if rng.random() < 0.5 else y)

    for _ in range(SIMS):
        sim_wins = wins.copy()
        sim_ties = ties.copy()
        sim_points = points_for.copy()
        # Each injured player gets one possible return week per season,
        # so a return remains in effect in later simulated weeks.
        return_scenarios = []
        for players in roster_data:
            returns = {}
            for k, player in enumerate(players):
                status = (player.get('injury') or 'ACTIVE').upper()
                if status in RETURN_HAZARD:
                    delay = 3 if status in ('INJURY_RESERVE', 'SUSPENSION') else 0
                    returns[k] = delay + int(rng.geometric(RETURN_HAZARD[status])) - 1
            return_scenarios.append(returns)
        weekly_strength = {}
        for week in remaining_weeks:
            offset = week - current
            estimates = []
            for j in range(n):
                lineup = lineup_total(roster_data[j], offset, return_scenarios[j])
                if lineup is None:
                    estimates.append(strength[j])
                else:
                    # The historical anchor prevents an optimistic future
                    # best-lineup estimate from overwhelming real results.
                    estimates.append(0.70 * strength[j] + 0.30 * lineup)
            weekly_strength[week] = estimates
        for week, home, away in future:
            hs, as_ = np.maximum(0, rng.normal(
                [weekly_strength[week][home],
                 weekly_strength[week][away]], [23, 23]))
            hs, as_ = round(float(hs), 2), round(float(as_), 2)
            if hs > as_:
                sim_wins[home] += 1
            elif as_ > hs:
                sim_wins[away] += 1
            else:
                sim_ties[home] += 1
                sim_ties[away] += 1
            sim_points[home] += hs
            sim_points[away] += as_

        order = sorted(range(n), key=lambda j: (
            sim_wins[j] + 0.5 * sim_ties[j], sim_points[j]), reverse=True)
        win_totals += sim_wins + 0.5 * sim_ties
        for rank, j in enumerate(order):
            rank_totals[j] += rank + 1
        seeds = order[:6]
        bracket_counts[tuple(seeds)] += 1
        for position, j in enumerate(seeds):
            playoff_counts[j] += 1
            seed_counts[position][j] += 1
        bye_counts[seeds[0]] += 1
        bye_counts[seeds[1]] += 1
        wc36 = playoff_game(seeds[2], seeds[5])
        wc45 = playoff_game(seeds[3], seeds[4])
        finalist1 = playoff_game(seeds[0], wc45)
        finalist2 = playoff_game(seeds[1], wc36)
        championship_counts[playoff_game(finalist1, finalist2)] += 1

    output_teams = []
    for i in range(n):
        output_teams.append({
            'id': ids[i], 'name': team_name(teams[i]),
            'wins': int(wins[i]), 'losses': int(losses[i]),
            'ties': int(ties[i]),
            'pointsFor': round(float(points_for[i]), 2),
            'pointsAgainst': round(float(points_against[i]), 2),
            'ppg': round(float(ppg[i]), 1),
            'strength': round(float(strength[i]), 1),
            'projectedWins': round(float(win_totals[i] / SIMS), 1),
            'averageRank': round(float(rank_totals[i] / SIMS), 2),
            'playoffPct': round(float(playoff_counts[i] / SIMS * 100), 1),
            'byePct': round(float(bye_counts[i] / SIMS * 100), 1),
            'champPct': round(float(championship_counts[i] / SIMS * 100), 1),
            'roster': roster_data[i],
        })
    output_teams.sort(key=lambda t: (
        t['wins'] + 0.5 * t['ties'], t['pointsFor']), reverse=True)

    # A real simulated seed combination, not six independent picks.
    projected_seeds = [ids[j] for j in bracket_counts.most_common(1)[0][0]]
    assert int(np.sum(championship_counts)) == SIMS
    assert int(np.sum(playoff_counts)) == 6 * SIMS
    assert int(np.sum(bye_counts)) == 2 * SIMS

    return {
        'mode': 'live', 'leagueId': LEAGUE, 'season': YEAR,
        'leagueName': 'Stach Fantasy Football',
        'updatedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'throughWeek': max(0, current - 1), 'currentWeek': current,
        'regularSeasonWeeks': regular_weeks, 'simulations': SIMS,
        'teams': output_teams, 'projectedSeeds': projected_seeds,
        'completedMatchups': completed,
        'method': (
            '20,000 Monte Carlo seasons using ESPN matchups, past scoring, '
            'recent performance, and available healthy starter projections. '
            'Future weekly lineups are sampled for all teams using player '
            'history, projections, and uncertain injury-return scenarios. '
            'Return chances are assumptions, not verified return dates. '
            'Six playoff spots, two byes, points-for standings tiebreaker, '
            'and a fixed playoff bracket without reseeding. '
            'These are estimates, not guarantees.'
        ),
    }


if __name__ == '__main__':
    result = run(load())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(OUT)
    print(f"Updated {len(result['teams'])} teams "
          f"through week {result['throughWeek']}; "
          f"{result['simulations']:,} simulations")
