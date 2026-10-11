"""ESPN 2026 Stach league updater. Unofficial ESPN endpoint; no credentials required for public leagues.
Private leagues: set ESPN_S2 and ESPN_SWID as GitHub Actions repository secrets.
"""
import os, json, math, datetime
from pathlib import Path
import requests
import numpy as np

LEAGUE=2108436489
YEAR=2026
URL=f'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{YEAR}/segments/0/leagues/{LEAGUE}'
OUT=Path(__file__).parent/'data/dashboard.json'
SEED=20261011
SIMS=12000

def load():
    params=[('view',v) for v in ('mTeam','mRoster','mMatchupScore','mSettings','mStatus','mStandings')]
    cookies={}
    if os.getenv('ESPN_S2') and os.getenv('ESPN_SWID'):
        cookies={'espn_s2':os.environ['ESPN_S2'],'SWID':os.environ['ESPN_SWID']}
    r=requests.get(URL,params=params,cookies=cookies,headers={'User-Agent':'StachFantasyDashboard/1.0'},timeout=35)
    r.raise_for_status()
    d=r.json()
    if len(d.get('teams',[]))<2: raise RuntimeError('ESPN returned no teams; check league visibility and credentials.')
    return d

def team_name(t):
    return t.get('name') or (t.get('location','')+' '+t.get('nickname','')).strip() or f"Team {t['id']}"

def roster(t):
    arr=[]
    for ent in (t.get('roster') or {}).get('entries',[]):
        pe=ent.get('playerPoolEntry') or {}
        p=pe.get('player') or {}
        if not p: continue
        stats=p.get('stats') or []
        proj=[]
        for s in stats:
            if s.get('statSourceId')==1 and s.get('appliedTotal') is not None:
                proj.append((s.get('scoringPeriodId',0),float(s['appliedTotal'])))
        arr.append({'name':p.get('fullName','Unknown'),'injury':p.get('injuryStatus','ACTIVE'),
                    'slot':ent.get('lineupSlotId'), 'projected':round(max(proj,key=lambda x:x[0])[1],1) if proj else None})
    return arr

def run(d):
    teams=d['teams']; ids=[int(t['id']) for t in teams]; n=len(ids); idx={k:i for i,k in enumerate(ids)}
    status=d.get('status') or {}; current=int(status.get('currentMatchupPeriod') or 5)
    settings=d.get('settings') or {}; schedule=d.get('schedule') or []
    reg=int((settings.get('scheduleSettings') or {}).get('matchupPeriodCount') or 14)
    playoff_n=int((settings.get('scheduleSettings') or {}).get('playoffTeamCount') or 6)
    if n!=10 or playoff_n!=6: raise RuntimeError(f'Expected 10 teams and 6 playoff spots; got {n} and {playoff_n}')
    # ESPN record is authoritative; if current week has not finished, it is typically not included yet.
    wins=np.zeros(n,dtype=int); losses=np.zeros(n,dtype=int); pf=np.zeros(n); pa=np.zeros(n)
    history={i:[] for i in ids}; future=[]; completed=[]
    for m in schedule:
        a=m.get('away') or {}; h=m.get('home') or {}; aid=a.get('teamId'); hid=h.get('teamId')
        if aid not in idx or hid not in idx: continue
        week=int(m.get('matchupPeriodId') or 0)
        if week>reg or week<1: continue
        winner=m.get('winner','UNDECIDED')
        av=float(a.get('totalPoints') or 0); hv=float(h.get('totalPoints') or 0)
        if winner in ('HOME','AWAY') and week<current:
            wins[idx[hid if winner=='HOME' else aid]]+=1
            losses[idx[aid if winner=='HOME' else hid]]+=1
            pf[idx[aid]]+=av;pf[idx[hid]]+=hv;pa[idx[aid]]+=hv;pa[idx[hid]]+=av
            history[aid].append((week,av));history[hid].append((week,hv))
            completed.append({'week':week,'home':hid,'away':aid,'homeScore':hv,'awayScore':av})
        elif week>=current:
            future.append((week,idx[hid],idx[aid]))
    # Fallback if ESPN doesn't provide complete historical schedule: use records, without fabricating scores.
    for t in teams:
        i=idx[int(t['id'])]; rec=(t.get('record') or {}).get('overall') or {}
        if wins[i]+losses[i]==0 and rec.get('gamesPlayed',0)>0:
            wins[i]=int(rec.get('wins',0));losses[i]=int(rec.get('losses',0));pf[i]=float(rec.get('pointsFor',0));pa[i]=float(rec.get('pointsAgainst',0))
    names=[team_name(t) for t in teams]
    roster_data=[roster(t) for t in teams]
    # Blend observed scoring with league mean, and active-player projections when available.
    games=np.maximum(1,wins+losses)
    avg=np.where((wins+losses)>0,pf/games,120)
    league_mean=float(np.mean(avg))
    strength=np.zeros(n)
    for i in range(n):
        past=sorted(history[ids[i]],reverse=True)
        recent=np.average([v for _,v in past[:3]],weights=[0.55,0.30,0.15][:len(past)]) if past else avg[i]
        active_proj=sorted([p['projected'] for p in roster_data[i] if p['projected'] is not None and p['injury'] not in ('OUT','INJURY_RESERVE') and p['slot'] not in (20,21)],reverse=True)
        roster_est=sum(active_proj[:9]) if len(active_proj)>=7 else None
        base=0.50*avg[i]+0.25*recent+0.25*league_mean
        strength[i]=0.72*base+0.28*roster_est if roster_est and 70<roster_est<220 else base
    # Missing future matchups? Keep the site operational, but don't pretend simulations use the full schedule.
    future_weeks={w for w,_,_ in future}; complete_schedule=all(sum(1 for x in future if x[0]==w)==5 for w in range(current,reg+1))
    if not complete_schedule:
        # Cannot accurately simulate actual schedule; fail rather than silently invent it.
        raise RuntimeError('ESPN did not return the complete remaining schedule. Refusing to publish misleading projections.')
    rng=np.random.default_rng(SEED+current+int(sum(pf)))
    playoffs=np.zeros(n); byes=np.zeros(n); champs=np.zeros(n); sumwins=np.zeros(n); ranks=np.zeros(n)
    future=sorted(future)
    for _ in range(SIMS):
        w=wins.copy(); p=pf.copy()
        for week,home,away in future:
            hs=max(0,rng.normal(strength[home],23));as_=max(0,rng.normal(strength[away],23))
            w[home if hs>as_ else away]+=1;p[home]+=hs;p[away]+=as_
        order=sorted(range(n),key=lambda j:(w[j],p[j]),reverse=True)
        sumwins+=w
        for rank,j in enumerate(order): ranks[j]+=rank+1
        seeds=order[:6]
        for j in seeds:playoffs[j]+=1
        byes[seeds[0]]+=1;byes[seeds[1]]+=1
        def game(x,y):
            # 22 point single-week volatility; stronger roster wins more often, not guaranteed.
            return x if rng.normal(strength[x],23)>rng.normal(strength[y],23) else y
        q1=game(seeds[2],seeds[5]);q2=game(seeds[3],seeds[4])
        # Standard 6-team bracket: 1 vs winner of 4/5, 2 vs winner of 3/6.
        s1=game(seeds[0],q2);s2=game(seeds[1],q1)
        champs[game(s1,s2)]+=1
    out=[]
    for i,t in enumerate(teams):
        out.append({'id':ids[i],'name':names[i],'wins':int(wins[i]),'losses':int(losses[i]),
                    'pointsFor':round(float(pf[i]),2),'pointsAgainst':round(float(pa[i]),2),
                    'ppg':round(float(avg[i]),1),'strength':round(float(strength[i]),1),
                    'projectedWins':round(float(sumwins[i]/SIMS),1),
                    'averageRank':round(float(ranks[i]/SIMS),2),
                    'playoffPct':round(float(playoffs[i]/SIMS*100),1),
                    'byePct':round(float(byes[i]/SIMS*100),1),
                    'champPct':round(float(champs[i]/SIMS*100),1),
                    'roster':roster_data[i]})
    out.sort(key=lambda t:(t['wins'],t['pointsFor']),reverse=True)
    prediction=sorted(out,key=lambda t:(t['projectedWins'],-t['averageRank']),reverse=True)
    result={'mode':'live','leagueId':LEAGUE,'season':YEAR,'leagueName':'Stach Fantasy Football',
            'updatedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'throughWeek':max(0,current-1),'currentWeek':current,'regularSeasonWeeks':reg,
            'simulations':SIMS,'teams':out,'projectedSeeds':[t['id'] for t in prediction],
            'completedMatchups':completed,'method':'Monte Carlo using ESPN remaining matchups, team scoring history, and available roster projections; simplified injury/bye modeling.'}
    return result

if __name__=='__main__':
    result=run(load())
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2))
    print('Updated',len(result['teams']),'teams through week',result['throughWeek'])
