import os
import json
import logging
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from typing import Dict, Set, Tuple, Optional, cast
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

def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("data_collection")
    if logger.handlers:
        return logger

    desired_path = os.getenv("LOG_Output_Path")
    if desired_path:
        if desired_path.lower().endswith(".log"):
            log_path = desired_path
        else:
            os.makedirs(desired_path, exist_ok=True)
            log_path = os.path.join(desired_path, "data_collection.log")
    else:
        log_path = os.path.join(cast(str, JSON_folder), "data_collection.log")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s [%(threadName)s] %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    if os.getenv("LOG_TO_STDOUT", "0").lower() in {"1", "true", "yes"}:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.debug("Logger configured. Output -> %s", log_path)
    return logger

LOGGER = _configure_logger()

def _get_env_api_key(thread: int) -> str:
    LOGGER.debug("Fetching RIOT_API_KEY for thread index %s", thread)
    key = os.getenv(f'RIOT_API_KEY{thread}')
    if not key:
        LOGGER.error("RIOT_API_KEY%s missing from environment.", thread)
        raise RuntimeError(
            "RIOT_API_KEY is not set. Set it in your environment or create a .env file with RIOT_API_KEY=your_key"
        )
    LOGGER.debug("RIOT_API_KEY%s retrieved (length=%s).", thread, len(key))
    return key

def get_clients(thread: int):
    LOGGER.debug("Initializing Riot clients for thread %s", thread)
    api_key = _get_env_api_key(thread)
    Riot_region = os.getenv('RIOT_REGION', 'americas')
    Lol_region = os.getenv('LOL_REGION', 'na1')
    LOGGER.debug(
        "Using regions Lol=%s Riot=%s for thread %s", Lol_region, Riot_region, thread
    )
    Lol = LolWatcher(api_key)
    Riot = RiotWatcher(api_key)
    LOGGER.debug("Riot clients ready for thread %s", thread)
    return Lol, Riot, Lol_region, Riot_region

def get_summoner_puuid(summoner_name: str, tagline: str, thread: int) -> str:
    LOGGER.info(
        "[thread %s] Resolving PUUID for Riot ID %s#%s",
        thread,
        summoner_name,
        tagline,
    )
    Lol, Riot, Lol_region, Riot_region = get_clients(thread)
    response = Riot.account.by_riot_id(Riot_region, summoner_name, tagline)
    account = cast(dict, response)
    puuid = account.get('puuid')
    if not puuid:
        LOGGER.error(
            "[thread %s] Failed to obtain PUUID for Riot ID %s#%s",
            thread,
            summoner_name,
            tagline,
        )
        raise RuntimeError('Failed to obtain puuid for summoner')
    LOGGER.info(
        "[thread %s] Retrieved PUUID %s for Riot ID %s#%s",
        thread,
        puuid,
        summoner_name,
        tagline,
    )
    return puuid

ROLE_CANON = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}

def _sleep_backoff(attempt: int) -> None:
    duration = min(0.5 * (2 ** attempt), 20.0)
    LOGGER.debug("Sleeping for backoff: attempt=%s sleep=%.2fs", attempt, duration)
    time.sleep(duration)

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
    callable_name = getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
    while True:
        try:
            LOGGER.debug(
                "Invoking %s attempt=%s args=%s kwargs=%s",
                callable_name,
                attempt + 1,
                args,
                kwargs,
            )
            result = fn(*args, **kwargs)
            LOGGER.debug("Call to %s succeeded on attempt %s", callable_name, attempt + 1)
            return result
        except Exception as e:
            LOGGER.warning(
                "Call to %s failed on attempt %s with %s",
                callable_name,
                attempt + 1,
                e,
            )
            if _should_retry(e, attempt, max_attempts):
                attempt += 1
                _sleep_backoff(attempt)
                continue
            LOGGER.error(
                "Call to %s exceeded retry limit (%s). Raising.",
                callable_name,
                max_attempts,
                exc_info=True,
            )
            raise

def _riot_id_from_puuid(puuid: str, thread: int) -> Tuple[str, str]:
    LOGGER.debug("[thread %s] Reverse lookup Riot ID for PUUID %s", thread, puuid)
    _, Riot, _, Riot_region = get_clients(thread)
    try:
        acct = _with_retries(Riot.account.by_puuid, Riot_region, puuid) or {}
        acct = cast(dict, acct)
        game = str(acct.get("gameName") or "")
        tag = str(acct.get("tagLine") or "")
        LOGGER.debug(
            "[thread %s] Reverse lookup success for PUUID %s -> %s#%s",
            thread,
            puuid,
            game,
            tag,
        )
        return game, tag
    except Exception:
        LOGGER.exception(
            "[thread %s] Reverse lookup failed for PUUID %s", thread, puuid
        )
        return "", ""

def _participant_for_puuid(match: dict, puuid: str) -> dict:
    info = match.get("info", {})
    for p in info.get("participants", []):
        if p.get("puuid") == puuid:
            return p
    LOGGER.debug("Participant with PUUID %s not found in match %s", puuid, match.get("metadata", {}).get("matchId"))
    return {}

def _is_ranked_solo(match: dict) -> bool:
    queue_id = (match.get("info", {}) or {}).get("queueId")
    return queue_id == 420

def _has_target_role(match: dict, puuid: str, target_role: str) -> bool:
    participant = _participant_for_puuid(match, puuid)
    role = (participant.get("teamPosition") or "").upper()
    match_id = match.get("metadata", {}).get("matchId")
    LOGGER.debug(
        "Match %s participant %s played role %s (target=%s)",
        match_id,
        puuid,
        role,
        target_role,
    )
    return role == target_role

def _iter_match_ids_for_puuid(puuid: str, max_to_scan: int, page_size: int, thread: int):
    LOGGER.debug(
        "[thread %s] Iterating matches for PUUID %s (max=%s, page_size=%s)",
        thread,
        puuid,
        max_to_scan,
        page_size,
    )
    page_size = min(page_size, 100)
    Lol, _, Lol_region, _ = get_clients(thread)
    fetched = 0
    start = 0
    while fetched < max_to_scan:
        count = min(page_size, max_to_scan - fetched)
        batch = _with_retries(
            Lol.match.matchlist_by_puuid,
            Lol_region,
            puuid,
            start=start,
            count=count,
            queue=420,
        ) or []
        LOGGER.debug(
            "[thread %s] Retrieved %s match IDs starting at index %s for PUUID %s",
            thread,
            len(batch),
            start,
            puuid,
        )
        if not batch:
            break
        for mid in batch:
            yield str(mid)
        fetched += len(batch)
        start += len(batch)
        if len(batch) < count:
            LOGGER.debug("[thread %s] Batch smaller than requested; ending iteration.", thread)
            break

def _fetch_match(match_id: str, thread: int) -> dict:
    LOGGER.debug("[thread %s] Fetching match %s", thread, match_id)
    Lol, _, Lol_region, _ = get_clients(thread)
    match = cast(dict, _with_retries(Lol.match.by_id, Lol_region, match_id))
    LOGGER.debug(
        "[thread %s] Retrieved match %s (gameVersion=%s)",
        thread,
        match_id,
        match.get("info", {}).get("gameVersion"),
    )
    return match

def _collect_lobby_puuids(match: dict) -> Set[str]:
    match_id = match.get("metadata", {}).get("matchId")
    participants = match.get("info", {}).get("participants", []) or []
    puuids = {p.get("puuid") for p in participants if p.get("puuid")}
    LOGGER.debug(
        "Match %s lobby PUUID count=%s", match_id, len(puuids)
    )
    return puuids

def _fetch_ranked_snapshot(puuid: str, thread: int) -> dict:
    LOGGER.debug("[thread %s] Fetching ranked snapshot for %s", thread, puuid)
    Lol, _, Lol_region, _ = get_clients(thread)
    try:
        snapshot = cast(dict, _with_retries(Lol.league.by_puuid, Lol_region, puuid) or {})
        LOGGER.debug(
            "[thread %s] Ranked snapshot retrieved for %s with %s entries",
            thread,
            puuid,
            len(snapshot),
        )
        return snapshot
    except Exception:
        LOGGER.exception("[thread %s] Failed to fetch ranked snapshot for %s", thread, puuid)
        return {}

def collect_player_role_sample(
    puuid: str,
    target_role: str,
    target_role_matches_needed: int = 10,
    max_history_to_scan: int = 20,
    collect_full_matches: bool = True,
    thread: int = -1,
) -> Tuple[Dict, Set[str]]:
    LOGGER.info(
        "[thread %s] Collecting role sample for PUUID %s target_role=%s",
        thread,
        puuid,
        target_role,
    )
    target_role = target_role.upper()
    if target_role not in ROLE_CANON:
        LOGGER.error("Invalid target role %s requested.", target_role)
        raise ValueError(f"target_role must be one of {sorted(ROLE_CANON)}")

    found = 0
    examined = 0
    role_matches = []
    lobby_accumulator: Set[str] = set()

    for mid in _iter_match_ids_for_puuid(puuid, max_history_to_scan, 100, thread):
        LOGGER.debug(
            "[thread %s] Examining match %s for PUUID %s", thread, mid, puuid
        )
        match = _fetch_match(mid, thread)
        if not _is_ranked_solo(match):
            LOGGER.debug("[thread %s] Match %s skipped (non-ranked-solo).", thread, mid)
            continue

        examined += 1
        lobby_members = _collect_lobby_puuids(match)
        lobby_accumulator |= lobby_members
        LOGGER.debug(
            "[thread %s] Match %s lobby size=%s aggregate=%s",
            thread,
            mid,
            len(lobby_members),
            len(lobby_accumulator),
        )

        if _has_target_role(match, puuid, target_role):
            role_matches.append(match if collect_full_matches else mid)
            found += 1
            LOGGER.info(
                "[thread %s] Match %s counts towards target role %s (found=%s/%s)",
                thread,
                mid,
                target_role,
                found,
                target_role_matches_needed,
            )
            if found >= target_role_matches_needed:
                LOGGER.debug("[thread %s] Target role quota met for %s", thread, puuid)
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
    LOGGER.info(
        "[thread %s] Role sample complete for %s: examined=%s found=%s density=%.3f",
        thread,
        puuid,
        examined,
        found,
        record["role_density"],
    )
    return record, lobby_accumulator

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
    LOGGER.info(
        "[thread %s] Starting crawl: seed=%s#%s target_role=%s",
        thread,
        seed_summoner_name,
        seed_tagline,
        target_role,
    )
    if out_jsonl_path is None:
        out_jsonl_path = os.path.join(cast(str, JSON_folder), "ranked_role_crawl.ndjson")

    seed_puuid = get_summoner_puuid(seed_summoner_name, seed_tagline, thread)
    LOGGER.debug("[thread %s] Seed PUUID=%s", thread, seed_puuid)

    visited: Set[str] = set()
    q: deque[str] = deque([seed_puuid])

    processed_count = 0
    written_count = 0
    retry_counts: Dict[str, int] = {}
    max_transient_retries = 3
    error_jsonl_path = f"{out_jsonl_path}.errors.ndjson"

    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    LOGGER.debug(
        "[thread %s] Output path ready at %s", thread, out_jsonl_path
    )

    with open(out_jsonl_path, "a", encoding="utf-8") as out_f, open(error_jsonl_path, "a", encoding="utf-8") as err_f:
        LOGGER.debug("[thread %s] Output and error files opened in append mode.", thread)
        while q and processed_count < max_players_to_process and written_count < target_written_amount:
            puuid = q.popleft()
            LOGGER.debug(
                "[thread %s] Dequeued PUUID %s (queue_size=%s)",
                thread,
                puuid,
                len(q),
            )
            if puuid in visited:
                LOGGER.debug("[thread %s] PUUID %s already visited; skipping.", thread, puuid)
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
                    LOGGER.warning(
                        "[thread %s] Transient error for %s (attempt %s/%s); requeueing.",
                        thread,
                        puuid,
                        retry_counts[puuid],
                        max_transient_retries,
                    )
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
                LOGGER.exception(
                    "[thread %s] Final failure collecting sample for %s after %s attempt(s).",
                    thread,
                    puuid,
                    attempts,
                )
                retry_counts.pop(puuid, None)
                neighbors = set()

            visited.add(puuid)

            if "error" in rec:
                err_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                err_f.flush()
                LOGGER.info(
                    "[thread %s] Error record logged for %s (processed=%s).",
                    thread,
                    puuid,
                    processed_count + 1,
                )
            elif int(rec.get("target_role_found", 0)) >= target_role_matches_needed:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                written_count += 1
                LOGGER.info(
                    "[thread %s] Record written for %s (written=%s/%s).",
                    thread,
                    puuid,
                    written_count,
                    target_written_amount,
                )

            processed_count += 1
            LOGGER.debug(
                "[thread %s] Processed count=%s, visited=%s",
                thread,
                processed_count,
                len(visited),
            )

            for n in neighbors:
                if n and (n not in visited):
                    q.append(n)
            LOGGER.debug(
                "[thread %s] Added %s neighbors; queue_size=%s",
                thread,
                len(neighbors),
                len(q),
            )

    LOGGER.info(
        "[thread %s] Crawl complete. processed=%s written=%s target_written=%s output=%s errors=%s",
        thread,
        processed_count,
        written_count,
        target_written_amount,
        out_jsonl_path,
        error_jsonl_path,
    )

from concurrent.futures import ThreadPoolExecutor

if __name__ == "__main__":
    LOGGER.info("=== Crawl execution started ===")
    seed_name1 = "sorrow"
    seed_tag1 = "uma"
    seed_name2 = "stillborn"
    seed_tag2 = "S925"
    seed_name3 = "prodion"
    seed_tag3 = "NA1"
    seed_name4 = "prodion"
    seed_tag4 = "NA1"

    list_of_seeds = [(seed_name1, seed_tag1), (seed_name2, seed_tag2), (seed_name3, seed_tag3), (seed_name4, seed_tag4)]

    with ThreadPoolExecutor(max_workers=5) as executor:
        LOGGER.debug("ThreadPoolExecutor started with max_workers=5")
        for i, (seed_name, seed_tag) in enumerate(list_of_seeds):
            LOGGER.debug(
                "Submitting crawl task index=%s seed=%s#%s",
                i,
                seed_name,
                seed_tag,
            )
            executor.submit(
                crawl_ranked_role_graph,
                seed_summoner_name=seed_name,
                seed_tagline=seed_tag,
                target_role="MIDDLE",
                target_role_matches_needed=11,
                max_history_to_scan=20,
                max_players_to_process=2500,
                target_written_amount=300,
                out_jsonl_path=os.path.join(JSON_folder, f"oct_25_home_day_crawl_{i}.ndjson"),
                collect_full_matches=True,
                thread=i,
            )

        executor.shutdown(wait=True)
        LOGGER.debug("ThreadPoolExecutor shutdown complete.")

    with open('crawl_complete.txt', 'w') as f:
        f.write('Crawls complete.\n')
    LOGGER.info("crawl_complete.txt written.")
    LOGGER.info("=== Crawl execution finished ===")
