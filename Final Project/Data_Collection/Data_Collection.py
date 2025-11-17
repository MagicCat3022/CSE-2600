import os
import json
from typing import List, cast, Optional
import time
from collections import deque
from typing import Dict, Set, Tuple
import errno
import socket

import requests

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


def _get_env_api_key(thread: int) -> str:
    key = os.getenv(f'RIOT_API_KEY{thread}')
    if not key:
        raise RuntimeError(
            "RIOT_API_KEY is not set. Set it in your environment or create a .env file with RIOT_API_KEY=your_key"
        )
    return key


def get_clients(thread: int):
    api_key = _get_env_api_key(thread)
    Riot_region = os.getenv('RIOT_REGION', 'americas')
    Lol_region = os.getenv('LOL_REGION', 'na1')

    Lol = LolWatcher(api_key)
    Riot = RiotWatcher(api_key)
    return Lol, Riot, Lol_region, Riot_region

def get_summoner_puuid(summoner_name: str, tagline: str, thread: int) -> str:
    """Fetch the puuid for a given summoner (by Riot ID).

    Args:
        summoner_name: the summoner name portion of the Riot ID
        tagline: the tagline portion of the Riot ID

    Returns:
        The puuid of the summoner.
    """
    Lol, Riot, Lol_region, Riot_region = get_clients(thread)

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

_TRANSIENT_ERRNOS = {
    code
    for code in (
        getattr(errno, "ECONNRESET", None),
        getattr(errno, "ECONNREFUSED", None),
        getattr(errno, "ECONNABORTED", None),
        getattr(errno, "EPIPE", None),
        getattr(errno, "ETIMEDOUT", None),
        getattr(errno, "EHOSTUNREACH", None),
        getattr(errno, "ENETUNREACH", None),
        getattr(errno, "EAI_AGAIN", None),
        getattr(errno, "EAI_FAIL", None),
        getattr(errno, "EAI_NONAME", None),
        getattr(errno, "WSAECONNRESET", None),
        getattr(errno, "WSAETIMEDOUT", None),
        getattr(errno, "WSAECONNREFUSED", None),
        getattr(errno, "WSAEHOSTUNREACH", None),
        getattr(errno, "WSATRY_AGAIN", None),
    )
    if isinstance(code, int)
}
_TRANSIENT_ERRNOS.add(11001)


def _is_transient_network_error(exc: Exception) -> bool:
    stack = [exc]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        if isinstance(cur, requests.exceptions.RequestException):
            return True
        if isinstance(cur, socket.gaierror) and (cur.errno in _TRANSIENT_ERRNOS or cur.errno is None):
            return True
        if isinstance(cur, OSError) and cur.errno in _TRANSIENT_ERRNOS:
            return True
        stack.append(getattr(cur, "__cause__", None))
        stack.append(getattr(cur, "__context__", None))
    return False


def _should_retry(exc: Exception, attempt: int, max_attempts: int) -> bool:
    if attempt >= max_attempts - 1:
        return False
    msg = str(exc)
    if any(code in msg for code in ["429", "502", "503", "504"]):
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) in {408, 429, 500, 502, 503, 504}:
        return True
    return _is_transient_network_error(exc)

def _with_retries(fn, *args, max_attempts: int = 5, **kwargs):
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _should_retry(e, attempt, max_attempts):
                raise
            attempt += 1
            _sleep_backoff(attempt)

def _riot_id_from_puuid(puuid: str, thread: int) -> Tuple[str, str]:
    """Best-effort reverse lookup of Riot ID -> (gameName, tagLine)."""
    _, Riot, _, Riot_region = get_clients(thread)
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

def _iter_match_ids_for_puuid(puuid: str, max_to_scan: int, page_size: int, thread: int):
    """Generator yielding match IDs up to max_to_scan, paginated."""
    page_size = min(page_size, 100)  # Riot max page size is 100
    Lol, _, Lol_region, _ = get_clients(thread)
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

def _fetch_match(match_id: str, thread: int) -> dict:
    Lol, _, Lol_region, _ = get_clients(thread)
    return cast(dict, _with_retries(Lol.match.by_id, Lol_region, match_id))

def _collect_lobby_puuids(match: dict) -> Set[str]:
    info = match.get("info", {}) or {}
    return {p.get("puuid") for p in info.get("participants", []) if p.get("puuid")}

def _fetch_ranked_snapshot(puuid: str, thread: int) -> dict:
    Lol, _, Lol_region, _ = get_clients(thread)
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
    thread: int = -1,
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

    for mid in _iter_match_ids_for_puuid(puuid, max_history_to_scan, 100, thread):
        match = _fetch_match(mid, thread)
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

    gameName, tagLine = _riot_id_from_puuid(puuid, thread)
    ranked_snapshot = _fetch_ranked_snapshot(puuid, thread)

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
    target_written_amount: int = 100,
    out_jsonl_path: Optional[str] = None,
    collect_full_matches: bool = True,
    thread: int = -1,
) -> None:
    """
    BFS crawl starting from a single Riot ID, expanding across lobbies.

    Writes NDJSON records as it goes (append mode) to avoid losing work
    if interrupted. Only writes records for players who have at least
    target_role_matches_needed games in the target role.
    
    Parameters:
        seed_summoner_name: Starting player's summoner name
        seed_tagline: Starting player's tagline
        target_role: The role to filter for (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)
        target_role_matches_needed: Min number of matches in target role needed to be written
        max_history_to_scan: Max number of matches to scan per player
        max_players_to_process: Hard limit on total players to process
        target_written_amount: Target number of valid players to write (stops after reaching this)
        out_jsonl_path: Path to write the NDJSON output
        collect_full_matches: Whether to include full match data or just IDs
    """
    if out_jsonl_path is None:
        out_jsonl_path = os.path.join(cast(str, JSON_folder), "ranked_role_crawl.ndjson")

    seed_puuid = get_summoner_puuid(seed_summoner_name, seed_tagline, thread)

    visited: Set[str] = set()
    q: deque[str] = deque([seed_puuid])

    processed_count = 0
    written_count = 0

    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)

    # Open in append mode so resuming doesn't overwrite prior data
    with open(out_jsonl_path, "a", encoding="utf-8") as out_f, open(error_jsonl_path, "a", encoding="utf-8") as err_f:
        while q and processed_count < max_players_to_process and written_count < target_written_amount:
            puuid = q.popleft()
            if puuid in visited:
                continue

            try:
                rec, neighbors = collect_player_role_sample(
                    puuid=puuid,
                    target_role=target_role,
                    target_role_matches_needed=target_role_matches_needed,
                    max_history_to_scan=max_history_to_scan,
                    collect_full_matches=collect_full_matches,
                    thread=thread,
                )
                retry_counts.pop(puuid, None)
            except Exception as e:
                if _is_transient_network_error(e) and retry_counts.get(puuid, 0) < max_transient_retries:
                    retry_counts[puuid] = retry_counts.get(puuid, 0) + 1
                    _sleep_backoff(retry_counts[puuid])
                    q.append(puuid)
                    continue
                attempts = retry_counts.get(puuid, 0) + 1
                rec = {
                    "puuid": puuid,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "attempts": attempts,
                    "target_role": target_role.upper(),
                }
                retry_counts.pop(puuid, None)
                neighbors = set()

            visited.add(puuid)

            target_role_found = int(rec.get("target_role_found", 0))
            if "error" in rec:
                err_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                err_f.flush()
            elif target_role_found >= target_role_matches_needed:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                written_count += 1

            processed_count += 1

            for n in neighbors:
                if n and (n not in visited):
                    q.append(n)

    print(
        f"[crawl{thread}] Processed {processed_count} player(s), wrote {written_count} to output. "
        f"Target was {target_written_amount} written. Errors logged -> {error_jsonl_path}"
    )
# -------- Example: run a small crawl --------
from concurrent.futures import ThreadPoolExecutor
if __name__ == "__main__":
    # Example seed (override via envs TEST_SUMMONER_NAME / TEST_TAGLINE)
    seed_name1 = "Kitsune"
    seed_tag1 = "Yippe"
    
    seed_name2 = "raccoonlover"
    seed_tag2 = "balls"

    seed_name3 = "Isukiri"
    seed_tag3 = "9513"
    
    list_of_seeds = [(seed_name1, seed_tag1), (seed_name2, seed_tag2), (seed_name3, seed_tag3)]
    i=4
    crawl_ranked_role_graph(
            seed_summoner_name=seed_name2,
            seed_tagline=seed_tag2,
            target_role="MIDDLE",
            target_role_matches_needed=10,
            max_history_to_scan=20,
            max_players_to_process=2000,
            target_written_amount=200,
            out_jsonl_path=os.path.join(JSON_folder, f"oct_23_home_night_crawl_{i}.ndjson"),
            collect_full_matches=True,
            thread=i,
        )
    with open('crawl_complete_DEV.txt', 'w') as f:
        f.write('Crawls complete.\n')
