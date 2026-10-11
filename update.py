
"""Stach Fantasy Football HQ — ESPN prediction updater."""
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
SEED = 20261011
OUT = Path(__file__).parent / "data" / "dashboard.json"
URL = (
    f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
    f"seasons/{YEAR}/segments/0/leagues/{LEAGUE}"
)
STARTER_SLOTS = {0, 2, 4, 6, 23, 16, 17}
INACTIVE = {"OUT", "INJURY_RESERVE", "SUSPENSION"}


def load():
    views = [
        "mTeam", "mRoster", "mMatchupScore",
        "mSettings", "mStatus", "mStandings"
    ]
    cookies = {}
    if os.getenv("ESPN_S2") and os.getenv("ESPN_SWID"):
        cookies = {
            "espn_s2": os.environ["ESPN_S2"],
            "SWID": os.environ["ESPN_SWID"]
        }

    response = requests.get(
        URL,
        params=[("view", v) for v in views],
        cookies=cookies,
        headers={"User-Agent": "StachFantasyDashboard/2.0"},
        timeout=35
    )
    response.raise_for_status()
    data = response.json()

    if len(data.get("teams", [])) != 10:
        raise RuntimeError("ESPN did not return all 10 teams.")

    return data


def team_name(team):
    return (
        team.get("name")
        or (
            team.get("location", "")
            + " "
            + team.get("nickname", "")
        ).strip()
        or f"Team {team['id']}"
    )


def roster(team, current_week):
    players = []

    for entry in (team.get("roster") or {}).get("entries", []):
        player = (
            (entry.get("playerPoolEntry") or {}).get("player")
            or {}
        )
        if not player:
            continue

        projections = [
            s for s in player.get("stats", [])
            if s.get("statSourceId") == 1
            and s.get("scoringPeriodId") == current_week
            and s.get("appliedTotal") is not None
        ]

        projected = (
            float(projections[0]["appliedTotal"])
            if projections else None
        )

        players.append({
            "name": player.get("fullName", "Unknown"),
            "injury": player.get("injuryStatus", "ACTIVE"),
            "slot": entry.get("lineupSlotId"),
            "projected": (
                round(projected, 2)
                if projected is not None else None
            )
        })

    return players


def run(data):
    teams = data["teams"]
    ids = [int(t["id"]) for t in teams]
    index = {team_id: i for i, team_id in enumerate(ids)}
    n = len(ids)

    settings = data.get("settings") or {}
    schedule_settings = settings.get("scheduleSettings") or {}

    regular_weeks = int(
        schedule_settings.get("matchupPeriodCount") or 14
    )
    playoff_teams = int(
        schedule_settings.get("playoffTeamCount") or 6
    )

    if n != 10 or playoff_teams != 6 or regular_weeks != 14:
        raise RuntimeError("League format has changed.")

    status = data.get("status") or {}
    current = int(status.get("currentMatchupPeriod") or 1)
    current = max(1, min(current, regular_weeks + 1))

    wins = np.zeros(n, dtype=int)
    losses = np.zeros(n, dtype=int)
    ties = np.zeros(n, dtype=int)
    points_for = np.zeros(n, dtype=float)
    points_against = np.zeros(n, dtype=float)

    history = {team_id: [] for team_id in ids}
    future = []
    completed = []

    for matchup in data.get("schedule") or []:
        home = matchup.get("home") or {}
        away = matchup.get("away") or {}

        home_id = home.get("teamId")
        away_id = away.get("teamId")

        if home_id not in index or away_id not in index:
            continue

        week = int(matchup.get("matchupPeriodId") or 0)

        if not 1 <= week <= regular_weeks:
            continue

        h = index[home_id]
        a = index[away_id]
        winner = matchup.get("winner", "UNDECIDED")

        home_score = float(home.get("totalPoints") or 0)
        away_score = float(away.get("totalPoints") or 0)

        if week < current and winner in (
            "HOME", "AWAY", "TIE"
        ):
            if winner == "HOME":
                wins[h] += 1
                losses[a] += 1
            elif winner == "AWAY":
                wins[a] += 1
                losses[h] += 1
            else:
                ties[h] += 1
                ties[a] += 1

            points_for[h] += home_score
            points_for[a] += away_score
            points_against[h] += away_score
            points_against[a] += home_score

            history[home_id].append((week, home_score))
            history[away_id].append((week, away_score))

            completed.append({
                "week": week,
                "home": home_id,
                "away": away_id,
                "homeScore": home_score,
                "awayScore": away_score
            })

        elif week >= current:
            future.append((week, h, a))

    for team in teams:
        i = index[int(team["id"])]
        record = (
            (team.get("record") or {}).get("overall") or {}
        )

        played = int(record.get("gamesPlayed") or 0)
        observed = int(wins[i] + losses[i] + ties[i])

        if observed == 0 and played > 0:
            wins[i] = int(record.get("wins") or 0)
            losses[i] = int(record.get("losses") or 0)
            ties[i] = int(record.get("ties") or 0)
            points_for[i] = float(
                record.get("pointsFor") or 0
            )
            points_against[i] = float(
                record.get("pointsAgainst") or 0
            )

        elif played > observed:
            raise RuntimeError(
                "ESPN schedule is missing completed results."
            )

    # Validate the complete remaining regular-season schedule.
    for week in range(current, regular_weeks + 1):
        games = [
            (h, a) for w, h, a in future
            if w == week
        ]

        participants = sorted(
            j for pair in games for j in pair
        )

        if (
            len(games) != n // 2
            or participants != list(range(n))
        ):
            raise RuntimeError(
                f"Incomplete ESPN schedule for week {week}."
            )

    roster_data = [
        roster(team, current) for team in teams
    ]

    played = wins + losses + ties

    ppg = np.array([
        points_for[i] / played[i]
        if played[i] else 120.0
        for i in range(n)
    ])

    league_mean = float(np.mean(ppg))
    strength = np.zeros(n, dtype=float)

    # Estimate team strength using scoring history
    # and current-week starters when projections exist.
    for i in range(n):
        recent_games = sorted(
            history[ids[i]],
            key=lambda x: x[0],
            reverse=True
        )[:3]

        if recent_games:
            weights = [0.55, 0.30, 0.15][
                :len(recent_games)
            ]
            recent = float(np.average(
                [score for _, score in recent_games],
                weights=weights
            ))
        else:
            recent = float(ppg[i])

        starters = [
            p for p in roster_data[i]
            if p["slot"] in STARTER_SLOTS
        ]

        valid = [
            p["projected"]
            for p in starters
            if p["projected"] is not None
            and p["injury"] not in INACTIVE
        ]

        roster_estimate = (
            sum(valid)
            if len(starters) == 9 and len(valid) == 9
            else None
        )

        base = (
            0.50 * ppg[i]
            + 0.25 * recent
            + 0.25 * league_mean
        )

        strength[i] = (
            0.72 * base + 0.28 * roster_estimate
            if (
                roster_estimate is not None
                and 70 < roster_estimate < 220
            )
            else base
        )

    rng = np.random.default_rng(
        SEED + current + int(np.sum(points_for))
    )

    playoff_counts = np.zeros(n)
    bye_counts = np.zeros(n)
    championship_counts = np.zeros(n)
    win_totals = np.zeros(n)
    rank_totals = np.zeros(n)

    seed_counts = [Counter() for _ in range(6)]

    def playoff_game(x, y):
        score_x, score_y = rng.normal(
            [strength[x], strength[y]],
            [23, 23]
        )

        if score_x == score_y:
            return x if rng.random() < 0.5 else y

        return x if score_x > score_y else y

    # Run 20,000 possible seasons.
    for _ in range(SIMS):
        sim_wins = wins.copy()
        sim_ties = ties.copy()
        sim_points = points_for.copy()

        for week, home, away in future:
            hs, as_ = np.maximum(
                0,
                rng.normal(
                    [strength[home], strength[away]],
                    [23, 23]
                )
            )

            hs = round(float(hs), 2)
            as_ = round(float(as_), 2)

            if hs > as_:
                sim_wins[home] += 1
            elif as_ > hs:
                sim_wins[away] += 1
            else:
                sim_ties[home] += 1
                sim_ties[away] += 1

            sim_points[home] += hs
            sim_points[away] += as_

        # Ties count as half a win.
        # Total points scored breaks standings ties.
        order = sorted(
            range(n),
            key=lambda j: (
                sim_wins[j] + 0.5 * sim_ties[j],
                sim_points[j]
            ),
            reverse=True
        )

        win_totals += sim_wins + 0.5 * sim_ties

        for rank, team_index in enumerate(order):
            rank_totals[team_index] += rank + 1

        seeds = order[:6]

        for seed_position, team_index in enumerate(seeds):
            playoff_counts[team_index] += 1
            seed_counts[seed_position][team_index] += 1

        bye_counts[seeds[0]] += 1
        bye_counts[seeds[1]] += 1

        # Six-team playoff bracket with no reseeding.
        wild_36 = playoff_game(seeds[2], seeds[5])
        wild_45 = playoff_game(seeds[3], seeds[4])

        finalist_1 = playoff_game(seeds[0], wild_45)
        finalist_2 = playoff_game(seeds[1], wild_36)

        champion = playoff_game(
            finalist_1, finalist_2
        )
        championship_counts[champion] += 1

    output_teams = []

    for i in range(n):
        output_teams.append({
            "id": ids[i],
            "name": team_name(teams[i]),
            "wins": int(wins[i]),
            "losses": int(losses[i]),
            "ties": int(ties[i]),
            "pointsFor": round(
                float(points_for[i]), 2
            ),
            "pointsAgainst": round(
                float(points_against[i]), 2
            ),
            "ppg": round(float(ppg[i]), 1),
            "strength": round(
                float(strength[i]), 1
            ),
            "projectedWins": round(
                float(win_totals[i] / SIMS), 1
            ),
            "averageRank": round(
                float(rank_totals[i] / SIMS), 2
            ),
            "playoffPct": round(
                float(playoff_counts[i] / SIMS * 100), 1
            ),
            "byePct": round(
                float(bye_counts[i] / SIMS * 100), 1
            ),
            "champPct": round(
                float(championship_counts[i] / SIMS * 100), 1
            ),
            "roster": roster_data[i]
        })

    output_teams.sort(
        key=lambda t: (
            t["wins"] + t["ties"] / 2,
            t["pointsFor"]
        ),
        reverse=True
    )

    # Build a projected bracket from simulated seed frequencies.
    chosen = set()
    projected_seeds = []

    for counts in seed_counts:
        choices = sorted(
            range(n),
            key=lambda j: (
                counts[j],
                -rank_totals[j]
            ),
            reverse=True
        )

        pick = next(
            j for j in choices if j not in chosen
        )

        chosen.add(pick)
        projected_seeds.append(ids[pick])

    # Check that all simulation outcomes are accounted for.
    assert int(np.sum(championship_counts)) == SIMS
    assert int(np.sum(playoff_counts)) == 6 * SIMS
    assert int(np.sum(bye_counts)) == 2 * SIMS

    return {
        "mode": "live",
        "leagueId": LEAGUE,
        "season": YEAR,
        "leagueName": "Stach Fantasy Football",
        "updatedAt": (
            datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        ),
        "throughWeek": max(0, current - 1),
        "currentWeek": current,
        "regularSeasonWeeks": regular_weeks,
        "simulations": SIMS,
        "teams": output_teams,
        "projectedSeeds": projected_seeds,
        "completedMatchups": completed,
        "method": (
            "Monte Carlo simulations using ESPN matchups, "
            "scoring history, current-week starting-lineup "
            "projections when available, win-percentage "
            "seeding and points-for tiebreakers. "
            "Injuries and weekly scoring variability "
            "are simplified."
        )
    }


if __name__ == "__main__":
    result = run(load())

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Write safely so failed updates cannot overwrite good data.
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )
    temporary.replace(OUT)

    print(
        f"Updated {len(result['teams'])} teams "
        f"through week {result['throughWeek']}; "
        f"{result['simulations']:,} simulations"
    )
