import os
import json
from typing import List, cast, Optional
import time
from collections import deque
from typing import Dict, Set, Tuple

# Optional: load from a local .env file for development
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; if not installed the environment variables must be set by the user
    pass

import riotwatcher
from riotwatcher import LolWatcher, RiotWatcher


JSON_folder: Optional[str] = os.getenv('JSON_Output_Path')
if not JSON_folder:
    raise RuntimeError(
        "JSON_Output_Path is not set. Set it in your environment or create a .env file with JSON_Output_Path=your_path"
    )

CSV_Folder: Optional[str] = os.getenv('CSV_Output_Path')
if not CSV_Folder:
    raise RuntimeError(
        "CSV_Output_Path is not set. Set it in your environment or create a .env file with CSV_Output_Path=your_path"
    )

os.makedirs(JSON_folder, exist_ok=True)
os.makedirs(CSV_Folder, exist_ok=True)


def _get_env_api_key() -> str:
    key = os.getenv('RIOT_API_KEY')
    if not key:
        raise RuntimeError(
            "RIOT_API_KEY is not set. Set it in your environment or create a .env file with RIOT_API_KEY=your_key"
        )
    return key


def get_clients():
    api_key = _get_env_api_key()
    Riot_region = os.getenv('RIOT_REGION', 'americas')
    Lol_region = os.getenv('LOL_REGION', 'na1')

    Lol = LolWatcher(api_key)
    Riot = RiotWatcher(api_key)
    return Lol, Riot, Lol_region, Riot_region

def get_summoner_puuid(summoner_name: str, tagline: str) -> str:
    """Fetch the puuid for a given summoner (by Riot ID).

    Args:
        summoner_name: the summoner name portion of the Riot ID
        tagline: the tagline portion of the Riot ID

    Returns:
        The puuid of the summoner.
    """
    Lol, Riot, Lol_region, Riot_region = get_clients()

    response = Riot.account.by_riot_id(Riot_region, summoner_name, tagline)
    account = cast(dict, response)

    puuid = account.get('puuid')
    if not puuid:
        raise RuntimeError('Failed to obtain puuid for summoner')

    return puuid

# -------- Constants & helpers --------

ROLE_CANON = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}

def _sleep_backoff(attempt: int) -> None:
    # naive exponential backoff with cap ~ 20s
    time.sleep(min(0.5 * (2 ** attempt), 20.0))

def _with_retries(fn, *args, max_attempts: int = 5, **kwargs):
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            # Riot's 429/5xx handling: riotwatcher retries some, but be safe
            if any(code in msg for code in ["429", "502", "503", "504"]) and attempt < max_attempts - 1:
                attempt += 1
                _sleep_backoff(attempt)
                continue
            raise

def _riot_id_from_puuid(puuid: str) -> Tuple[str, str]:
    """Best-effort reverse lookup of Riot ID -> (gameName, tagLine)."""
    _, Riot, _, Riot_region = get_clients()
    try:
        acct = _with_retries(Riot.account.by_puuid, Riot_region, puuid) or {}
        acct = cast(dict, acct)
        return str(acct.get("gameName") or ""), str(acct.get("tagLine") or "")
    except Exception:
        return "", ""

def _participant_for_puuid(match: dict, puuid: str) -> dict:
    '''Return the participant dict for the given puuid in the match.'''
    info = match.get("info", {})
    for p in info.get("participants", []):
        if p.get("puuid") == puuid:
            return p
    return {}

def _is_ranked_solo(match: dict) -> bool:
    return (match.get("info", {}) or {}).get("queueId") == 420

def _has_target_role(match: dict, puuid: str, target_role: str) -> bool:
    p = _participant_for_puuid(match, puuid)
    # API uses 'teamPosition' for role; normalize to canonical uppercase
    return (p.get("teamPosition") or "").upper() == target_role

def _iter_match_ids_for_puuid(puuid: str, max_to_scan: int, page_size: int = 100):
    """Generator yielding match IDs up to max_to_scan, paginated."""
    page_size = min(page_size, 100)  # Riot max page size is 100
    Lol, _, Lol_region, _ = get_clients()
    fetched = 0
    start = 0
    while fetched < max_to_scan:
        batch = _with_retries(
            Lol.match.matchlist_by_puuid,
            Lol_region, puuid, start=start, count=min(page_size, max_to_scan - fetched), queue=420) or []
        if not batch:
            break
        for mid in batch:
            yield str(mid)
        got = len(batch)
        fetched += got
        start += got
        if got == 0:
            break

def _fetch_match(match_id: str) -> dict:
    Lol, _, Lol_region, _ = get_clients()
    return cast(dict, _with_retries(Lol.match.by_id, Lol_region, match_id))

def _collect_lobby_puuids(match: dict) -> Set[str]:
    info = match.get("info", {}) or {}
    return {p.get("puuid") for p in info.get("participants", []) if p.get("puuid")}

def _fetch_ranked_snapshot(puuid: str) -> dict:
    Lol, _, Lol_region, _ = get_clients()
    try:
        return cast(dict, _with_retries(Lol.league.by_puuid, Lol_region, puuid) or {})
    except Exception:
        return {}

# -------- Per-player role-qualified match collection --------

def collect_player_role_sample(
    puuid: str,
    target_role: str,
    target_role_matches_needed: int = 10,
    max_history_to_scan: int = 20,
    collect_full_matches: bool = True,
) -> Tuple[Dict, Set[str]]:
    """
    Returns (player_record, puuids_from_their_ranked_solo_matches).
    player_record schema:
      {
        "puuid": str,
        "riotId": "gameName#tagLine" | "",
        "target_role": str,
        "target_role_found": int,
        "matches_examined": int,
        "history_cap_reached": bool,
        "role_density": float,  # found/examined (0 if examined=0)
        "target_role_matches": [match or id, ...],
        "ranked_snapshot": dict
      }
    """
    target_role = target_role.upper()
    if target_role not in ROLE_CANON:
        raise ValueError(f"target_role must be one of {sorted(ROLE_CANON)}")

    found = 0
    examined = 0
    role_matches = []
    lobby_accumulator: Set[str] = set()

    for mid in _iter_match_ids_for_puuid(puuid, max_to_scan=max_history_to_scan):
        match = _fetch_match(mid)
        # We only care about ranked solo/duo (queue 420)
        if not _is_ranked_solo(match):
            continue

        examined += 1
        # Harvest lobby for propagation (unique players)
        lobby_accumulator |= _collect_lobby_puuids(match)

        # Count only if this player played the target role in this match
        if _has_target_role(match, puuid, target_role):
            role_matches.append(match if collect_full_matches else mid)
            found += 1
            if found >= target_role_matches_needed:
                break

    gameName, tagLine = _riot_id_from_puuid(puuid)
    ranked_snapshot = _fetch_ranked_snapshot(puuid)

    record = {
        "puuid": puuid,
        "riotId": f"{gameName}#{tagLine}" if (gameName and tagLine) else "",
        "target_role": target_role,
        "target_role_found": found,
        "matches_examined": examined,
        "history_cap_reached": examined >= max_history_to_scan and found < target_role_matches_needed,
        "role_density": (found / examined) if examined else 0.0,
        "target_role_matches": role_matches,
        "ranked_snapshot": ranked_snapshot,
    }
    return record, lobby_accumulator

# -------- BFS propagation crawl --------

def crawl_ranked_role_graph(
    seed_summoner_name: str,
    seed_tagline: str,
    target_role: str,
    target_role_matches_needed: int = 10,
    max_history_to_scan: int = 20,
    max_players_to_process: int = 300,
    out_jsonl_path: Optional[str] = None,
    collect_full_matches: bool = True,
) -> None:
    """
    BFS crawl starting from a single Riot ID, expanding across lobbies.

    Writes NDJSON records as it goes (append mode) to avoid losing work
    if interrupted.
    """
    if out_jsonl_path is None:
        out_jsonl_path = os.path.join(cast(str, JSON_folder), "ranked_role_crawl.ndjson")

    seed_puuid = get_summoner_puuid(seed_summoner_name, seed_tagline)

    visited: Set[str] = set()
    q: deque[str] = deque([seed_puuid])

    processed_count = 0

    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)

    # Open in append mode so resuming doesn't overwrite prior data
    with open(out_jsonl_path, "a", encoding="utf-8") as out_f:
        while q and processed_count < max_players_to_process:
            puuid = q.popleft()
            if puuid in visited:
                continue
            visited.add(puuid)

            try:
                rec, neighbors = collect_player_role_sample(
                    puuid=puuid,
                    target_role=target_role,
                    target_role_matches_needed=target_role_matches_needed,
                    max_history_to_scan=max_history_to_scan,
                    collect_full_matches=collect_full_matches,
                )
            except Exception as e:
                # Record the error but keep crawling
                rec = {
                    "puuid": puuid,
                    "error": str(e),
                    "target_role": target_role.upper(),
                }
                neighbors = set()

            # Append result immediately (NDJSON)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            processed_count += 1

            # Add neighbors (propagation). Maintain a modest cap so we don't explode too quickly.
            for n in neighbors:
                if n and (n not in visited):
                    q.append(n)

    print(f"[crawl] Processed {processed_count} player(s). Output -> {out_jsonl_path}")

# -------- Example: run a small crawl --------

if __name__ == "__main__":
    # Example seed (override via envs TEST_SUMMONER_NAME / TEST_TAGLINE)
    seed_name = os.getenv('TEST_SUMMONER_NAME', 'MagicCat3022')
    seed_tag = os.getenv('TEST_TAGLINE', '3022')

    # Tunable knobs (safe defaults shown)
    # crawl_ranked_role_graph(
    #     seed_summoner_name=seed_name,
    #     seed_tagline=seed_tag,
    #     target_role=os.getenv("TARGET_ROLE", "MIDDLE"),
    #     target_role_matches_needed=int(os.getenv("ROLE_MATCHES_NEEDED", "10")),
    #     max_history_to_scan=int(os.getenv("MAX_HISTORY_SCAN", "20")),
    #     max_players_to_process=int(os.getenv("MAX_PLAYERS", "400")),
    #     out_jsonl_path=os.path.join(JSON_folder, "oct_21_night_crawl.ndjson"),
    #     collect_full_matches=bool(int(os.getenv("COLLECT_FULL_MATCHES", "1"))),
    # )
