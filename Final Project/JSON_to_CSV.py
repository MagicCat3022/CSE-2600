import json, math, datetime, pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

def extract_target_rows(obj: Dict[str, Any], fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Given the full JSON object (as in single_line.json), return rows for CSV for the target player.
    Each 'target_role_matches' entry is a match; we find the participant with puuid == obj['puuid'].
    """
    target_puuid = obj.get("puuid")
    riot_id = obj.get("riotId")
    matches = obj.get("target_role_matches", [])
    target_role_found = obj.get("target_role_found", -1)
    matches_examined = obj.get("matches_examined", -1)
    role_density = obj.get("role_density", -1.0)
    ranked_snapshot = obj.get("ranked_snapshot", [{}])
    if len(ranked_snapshot) > 0:
        ranked_snapshot = ranked_snapshot[0]
    leagueId = ranked_snapshot.get("leagueId", -1)
    tier = ranked_snapshot.get("tier", "-1")
    rank = ranked_snapshot.get("rank", -1)
    leaguePoints = ranked_snapshot.get("leaguePoints", -1)
    wins = ranked_snapshot.get("wins", -1)
    losses = ranked_snapshot.get("losses", -1)
    
    rows: List[Dict[str, Any]] = []
    targets: Dict[str, Any] = {}
    
    with open('output_targets.json', 'r', encoding='utf-8') as f:
        targets = json.load(f)
    
    for m in matches:
        md = m.get("metadata", {})
        info = m.get("info", {})
        match_id = md.get("matchId") or info.get("gameId")
        game_creation = info.get("gameCreation")
        game_duration = info.get("gameDuration")
        
        # Find the target participant
        p: Optional[Dict[str, Any]] = None
        for part in info.get("participants", []):
            if part.get("puuid") == target_puuid:
                p = part
                break
        if p is None:
            # If not found, skip
            continue
        
        out_row = {
            "puuid": target_puuid,
            "riotId": riot_id,
            "matchId": match_id,
            "gameCreation_ms": game_creation,
            "gameDuration_s": game_duration,
            "target_role_found": target_role_found,
            "matches_examined": matches_examined,
            "role_density": role_density,
            "leagueId": leagueId,
            "tier": tier,
            "rank": rank,
            "tier_int": None,
            "rank_int": None,
            "leaguePoints": leaguePoints,
            "wins": wins,
            "losses": losses
            }
        
        for key, value in targets.items():
            if key == 'challenges':
                cast(Dict[str, Any], value)
                for challenge_key in value.keys():
                    out_row[f"challenge_{challenge_key}"] = p.get('challenges', {}).get(challenge_key)
            else:
                out_row[key] = p.get(key)

        
        # If specific fields requested, filter; otherwise keep base_row
        if fields:
            row = {col: out_row.get(col) for col in fields}
        else:
            row = out_row
        
        rows.append(row)
    return rows

def matches_to_csv(obj: Dict[str, Any], out_path: str, fields: Optional[List[str]] = None) -> Path:
    rows = extract_target_rows(obj, fields=fields)
    df = pd.DataFrame(rows)
    # Sort by gameCreation if available (newest first)
    if "gameCreation_ms" in df.columns:
        df = df.sort_values(["uuid", "gameCreation_ms"], ascending=False)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def get_all_matches(obj: Dict[str, Any]):
    target_puuid = obj.get("puuid")
    matches = obj.get("target_role_matches", [])
    p: Optional[Dict[str, Any]] = None
    
    for m in matches:
        info = m.get("info", {})
        # Find the target participant
        for part in info.get("participants", []):
            if part.get("puuid") == target_puuid:
                p = part
                cast(Dict[str, Any], p)
                break
            
    with open('output.txt', 'w', encoding='utf-8') as wf:
        for key, value in p.items():
            if key == 'challenges':
                wf.write(f"{key}:\n")
                for challenge_key, challenge_value in value.items():
                    wf.write(f"  {challenge_key}: {challenge_value}\n")
            else:
                wf.write(f"{key}: {value}\n")
                
                
def ndjson_to_csv(ndjson_path: Path, out_path: Path, fields: Optional[List[str]] = None) -> Path:
    rows: List[Dict[str, Any]] = []
    with ndjson_path.open('r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            match_rows = extract_target_rows(obj, fields=fields)
            rows.extend(match_rows)
    df = pd.DataFrame(rows)
    
    # Correct variable names for clarity
    tier_mapping = {"IRON": 1, "BRONZE": 2, "SILVER": 3, "GOLD": 4, "PLATINUM": 5, "DIAMOND": 6, "MASTER": 7, "GRANDMASTER": 8, "CHALLENGER": 9}
    rank_mapping = {"IV": 1, "III": 2, "II": 3, "I": 4}
    
    # Convert to string first to handle mixed types
    df['tier_str'] = df['tier'].astype(str).str.upper()
    df['rank_str'] = df['rank'].astype(str).str.upper()
    
    # Apply correct mappings to correct columns
    df['tier_int'] = df['tier_str'].map(tier_mapping).fillna(-1).astype(int)
    df['rank_int'] = df['rank_str'].map(rank_mapping).fillna(-1).astype(int)
    
    # Drop temporary columns
    df = df.drop(columns=['tier_str', 'rank_str'])
    
    df = df.sort_values(["tier_int", "rank_int", "leaguePoints", "puuid"], ascending=[False, False, False, False])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out

if __name__ == "__main__":
    csv_path = ndjson_to_csv(Path("Data/JSON/oct_22_home_crawl.ndjson"), Path("Data/CSV/oct_22_home_crawl.csv"))
    print(f"CSV saved to: {csv_path}")
