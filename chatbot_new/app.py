
from __future__ import annotations

import os
import json
import random
import time
import re
from datetime import datetime
from collections import deque
import concurrent.futures
from typing import List, Dict, Optional, Tuple

import queue
import threading
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from dotenv import load_dotenv
load_dotenv()

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH)
app = Flask(__name__)

# ---------------------------
# Config
# ---------------------------

DEFAULT_WORKERS = 2
GROQ_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
MODEL = (os.getenv("GENERATION_MODEL") or "gemini-2.0-flash").strip()

GROQ_MAX_COMPLETION_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS") or "600")
GROQ_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE") or "0.4")
GROQ_TOP_P = float(os.getenv("GROQ_TOP_P") or "1")
GROQ_REASONING_EFFORT = (os.getenv("GROQ_REASONING_EFFORT") or "").strip()

TURN_SLEEP_SECONDS = float(os.getenv("TURN_SLEEP_SECONDS") or "0")
CONVERSATION_SLEEP_SECONDS = float(os.getenv("CONVERSATION_SLEEP_SECONDS") or "0")

TARGET_CONVERSATIONS_DEFAULT = int(os.getenv("TARGET_CONVERSATIONS") or "10000")
PARALLEL_WORKERS = max(1, min(8, int(os.getenv("PARALLEL_WORKERS") or "8")))
GEN_MODE = (os.getenv("GEN_MODE") or "MULTITURN_SINGLE_CALL").strip().upper()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "runs_manifest.json")

MERGED_RAW_PATH = os.path.join(BASE_DIR, "dataset_raw_merged.json")
MERGED_RL_PATH = os.path.join(BASE_DIR, "dataset_rl_training_merged.jsonl")

GROQ_CLIENT = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else Groq()

RATE_LIMITS = {
    "rpm": 1000,
    "rpd": 500000,
    "tpm": 250000,
}

# ---------------------------
# State
# ---------------------------
generation_queue = queue.Queue()
is_generating = False
dataset: List[Dict] = []
TARGET_CONVERSATIONS: Optional[int] = TARGET_CONVERSATIONS_DEFAULT
CURRENT_RUN_ID: Optional[str] = None
generation_thread: Optional[threading.Thread] = None
RUN_STARTED_AT: Optional[float] = None

REQUEST_EVENTS: deque[Tuple[float, int]] = deque()
REQUESTS_TOTAL = 0
TOKENS_PROMPT_TOTAL = 0
TOKENS_COMPLETION_TOTAL = 0
TOKENS_TOTAL = 0
LAST_REQUEST_USAGE: Dict[str, int] = {}
USAGE_LOCK = threading.Lock()



MOODS = ["anxious", "frustrated", "scared", "angry", "tired", "sad", "confused", "grateful", "desperate", "stressed", "exhausted", "shy", "hesitant", "determined"]

BACKGROUNDS = ["I recently lost my job", "I just got out of the hospital", "I experienced domestic violence", "I lost contact with my family", "I left my housing due to debt", "I've been living on the street for weeks", "I have chronic health issues", "I suffer from severe anxiety", "I'm facing legal problems", "I'm new to the city and don't know the area", "I was recently released from detention", "I'm recovering from addiction"]

NEEDS = ["a safe place to sleep tonight", "a warm meal", "medical care", "mental health support", "financial assistance", "legal advice", "addiction treatment", "help with official documents", "warm clothing", "access to shower and hygiene", "a phone to make calls", "internet access", "storage for belongings", "help registering for services"]

QUESTION_TYPES = ["request for help", "request for information", "explaining my situation", "asking about services", "urgent question", "follow-up question"]

TIME_OF_DAY = ["early morning", "afternoon", "evening", "midnight"]
WEATHER = ["cold", "rainy", "hot", "windy", "mild"]
SEASONS = ["winter", "spring", "summer", "autumn"]


CITIES = [
    "Rotterdam Centraal", "Zuidplein", "Delfshaven", "Kralingen",
    "Lombardijen", "Feijenoord", "Charlois", "Schiedam"
]



# ---------------------------
# Helpers
# ---------------------------
def normalize_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def sanitize_for_training(text: str) -> str:
    t = normalize_text(text)
    t = t.replace("…", ".").replace("...", ".")
    t = re.sub(r"\s+\.", ".", t)
    return t.strip()


def strip_sensitive_numbers(text: str) -> str:
    t = (text or "")
    t = re.sub(r"\bhttps?://\S+\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\S+@\S+\.\S+\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d{3,}([-\s]?\d+)*\b", "", t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t


def normalize_bullets(text: str) -> str:
    t = text or ""
    t = re.sub(r"(الخطوات التالية:)\s*-", r"\1\n-", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+-\s+", "\n- ", t)
    return t.strip()


def new_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(4))
    return f"{ts}_{rnd}"


def run_paths(run_id: str) -> Tuple[str, str]:
    raw_path = os.path.join(BASE_DIR, f"dataset_raw_{run_id}.json")
    rl_path = os.path.join(BASE_DIR, f"dataset_rl_training_{run_id}.jsonl")
    return raw_path, rl_path


def load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {"runs": []}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"runs": []}


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def add_run_to_manifest(run_id: str, raw_path: str, rl_path: str, target: Optional[int]) -> None:
    manifest = load_manifest()
    manifest["runs"].append({
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "model": MODEL,
        "max_completion_tokens": GROQ_MAX_COMPLETION_TOKENS,
        "temperature": GROQ_TEMPERATURE,
        "target": target,
        "raw_path": raw_path,
        "rl_path": rl_path,
        "status": "running",
    })
    save_manifest(manifest)


def mark_run_complete(run_id: str, total_conversations: int, total_training_samples: int) -> None:
    manifest = load_manifest()
    for r in manifest.get("runs", []):
        if r.get("run_id") == run_id:
            r["status"] = "complete"
            r["completed_at"] = datetime.now().isoformat()
            r["total_conversations"] = total_conversations
            r["total_training_samples"] = total_training_samples
            break
    save_manifest(manifest)


def mark_run_stopped(run_id: str, total_conversations: int, total_training_samples: int) -> None:
    manifest = load_manifest()
    for r in manifest.get("runs", []):
        if r.get("run_id") == run_id:
            r["status"] = "stopped"
            r["stopped_at"] = datetime.now().isoformat()
            r["total_conversations"] = total_conversations
            r["total_training_samples"] = total_training_samples
            break
    save_manifest(manifest)


def generate_scenario() -> dict:
    primary = random.choice(NEEDS)
    secondary = random.choice([n for n in NEEDS if n != primary])
    return {
        "mood": random.choice(MOODS),
        "background": random.choice(BACKGROUNDS),
        "primary_need": primary,
        "secondary_need": secondary,
        "question_type": random.choice(QUESTION_TYPES),
        "time_of_day": random.choice(TIME_OF_DAY),
        "weather": random.choice(WEATHER),
        "season": random.choice(SEASONS),
        "location": random.choice(CITIES),
    }

def parse_target(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        target_int = int(value)
    except Exception:
        return None
    if target_int <= 0:
        return None
    return max(1, min(200000, target_int))

def progress_payload() -> dict:
    target = TARGET_CONVERSATIONS
    completed = len(dataset)
    if target is None:
        return {
            "completed": completed,
            "target": 0,
            "percent": 0.0,
            "label": f"{completed}/0",
        }
    percent = round(100.0 * completed / target, 2) if target else 0.0
    return {
        "completed": completed,
        "target": target,
        "percent": percent,
        "label": f"{completed}/{target}",
    }

def should_auto_start() -> bool:
    value = (os.getenv("AUTO_START_GENERATION") or "1").strip().lower()
    return value in ("1", "true", "yes", "on")

def _usage_value(usage: object, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(name) or 0)
    return int(getattr(usage, name, 0) or 0)

def record_usage(usage: object) -> None:
    global REQUESTS_TOTAL, TOKENS_PROMPT_TOTAL, TOKENS_COMPLETION_TOTAL, TOKENS_TOTAL, LAST_REQUEST_USAGE

    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens") or (prompt_tokens + completion_tokens)

    with USAGE_LOCK:
        REQUESTS_TOTAL += 1
        TOKENS_PROMPT_TOTAL += prompt_tokens
        TOKENS_COMPLETION_TOTAL += completion_tokens
        TOKENS_TOTAL += total_tokens
        LAST_REQUEST_USAGE = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        REQUEST_EVENTS.append((time.time(), total_tokens))

def estimate_tokens(messages: list[dict]) -> int:
    total_chars = 0
    for msg in messages:
        total_chars += len(str(msg.get("content", "")))
    estimated_prompt_tokens = max(1, total_chars // 4)
    return estimated_prompt_tokens + GROQ_MAX_COMPLETION_TOKENS

def wait_for_rate_limits(estimated_tokens: int) -> None:
    rpm_limit = RATE_LIMITS.get("rpm") or 0
    rpd_limit = RATE_LIMITS.get("rpd") or 0
    tpm_limit = RATE_LIMITS.get("tpm") or 0

    while True:
        with USAGE_LOCK:
            now = time.time()
            cutoff_day = now - 86400
            cutoff_minute = now - 60

            while REQUEST_EVENTS and REQUEST_EVENTS[0][0] < cutoff_day:
                REQUEST_EVENTS.popleft()

            events_last_minute = []
            tokens_last_minute = 0
            for ts, tokens in REQUEST_EVENTS:
                if ts >= cutoff_minute:
                    events_last_minute.append((ts, tokens))
                    tokens_last_minute += tokens

            if rpd_limit and len(REQUEST_EVENTS) >= rpd_limit:
                raise RuntimeError("RPD limit reached. Try again later.")

            wait_seconds = 0.0
            if rpm_limit and len(events_last_minute) >= rpm_limit:
                oldest_ts = events_last_minute[0][0]
                wait_seconds = max(wait_seconds, (oldest_ts + 60) - now)

            if tpm_limit and tokens_last_minute + estimated_tokens > tpm_limit:
                running_tokens = tokens_last_minute
                for ts, tokens in events_last_minute:
                    running_tokens -= tokens
                    if running_tokens + estimated_tokens <= tpm_limit:
                        wait_seconds = max(wait_seconds, (ts + 60) - now)
                        break
                if events_last_minute and running_tokens + estimated_tokens > tpm_limit:
                    wait_seconds = max(wait_seconds, (events_last_minute[-1][0] + 60) - now)

            if wait_seconds <= 0:
                return
        time.sleep(min(wait_seconds, 2.0))

def stats_payload() -> dict:
    with USAGE_LOCK:
        now = time.time()
        cutoff_day = now - 86400
        cutoff_minute = now - 60

        while REQUEST_EVENTS and REQUEST_EVENTS[0][0] < cutoff_day:
            REQUEST_EVENTS.popleft()

        rpm = 0
        tpm = 0
        for ts, tokens in REQUEST_EVENTS:
            if ts >= cutoff_minute:
                rpm += 1
                tpm += tokens

        return {
            "requests_total": REQUESTS_TOTAL,
            "tokens_prompt_total": TOKENS_PROMPT_TOTAL,
            "tokens_completion_total": TOKENS_COMPLETION_TOTAL,
            "tokens_total": TOKENS_TOTAL,
            "rpm": rpm,
            "rpd": len(REQUEST_EVENTS),
            "tpm": tpm,
            "last_request": LAST_REQUEST_USAGE,
        }

def eta_seconds() -> Optional[int]:
    if TARGET_CONVERSATIONS is None:
        return None
    completed = len(dataset)
    remaining = TARGET_CONVERSATIONS - completed
    if remaining <= 0:
        return 0
    if not RUN_STARTED_AT:
        return None
    elapsed = time.time() - RUN_STARTED_AT
    if elapsed <= 0 or completed <= 0:
        return None
    rate = completed / elapsed
    if rate <= 0:
        return None
    return int(remaining / rate)


def call_groq(messages: list[dict], max_retries: int = 3) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Set it in your .env.")

    last_err = None
    for attempt in range(max_retries):
        try:
            wait_for_rate_limits(estimate_tokens(messages))
            request_args = {
                "model": MODEL,
                "messages": messages,
                "temperature": GROQ_TEMPERATURE,
                "max_completion_tokens": GROQ_MAX_COMPLETION_TOKENS,
                "top_p": GROQ_TOP_P,
            }
            if GROQ_REASONING_EFFORT:
                request_args["reasoning_effort"] = GROQ_REASONING_EFFORT
            response = GROQ_CLIENT.chat.completions.create(**request_args)
            record_usage(getattr(response, "usage", None))
            return sanitize_for_training(response.choices[0].message.content)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Groq call failed: {last_err}")


def build_pair_prompt(scenario: dict) -> list[dict]:
    system = (
    "Homeless support chatbot training data. JSON only: {\"turns\":[{\"role\":\"homeless\"|\"helper\",\"content\":\"...\"}]}.\n"
    "English. No addresses/phones/URLs. Emergency: 'Call emergency services immediately.'\n"
    "4-6 exchanges: 1) Person situation+question, 2) Helper empathy+triage, 3) Person reply, 4) Helper plan+follow-up.\n"
    "Services: shelter, municipality, community center, doctor, social services."
)

    scenario = {
    "location": random.choice(CITIES),
    "time_of_day": random.choice(TIME_OF_DAY),
    "season": random.choice(SEASONS),
    "weather": random.choice(WEATHER),
    "mood": random.choice(MOODS),
    "background": random.choice(BACKGROUNDS),
    "primary_need": random.choice(NEEDS),
    "secondary_need": random.choice(NEEDS),
    "question_type": random.choice(QUESTION_TYPES)
}

    user = (
    f"City:{scenario['location']}|Time:{scenario['time_of_day']} ({scenario['season']}, {scenario['weather']})|"
    f"Mood:{scenario['mood']}|Background:{scenario['background']}|"
    f"Primary:{scenario['primary_need']}|Secondary:{scenario['secondary_need']}|"
    f"Type:{scenario['question_type']}\n"
    "Return JSON only."
)

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_pair(text: str, scenario: dict) -> List[Dict]:
    t = sanitize_for_training(text)
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()

    start = t.find("{")
    t2 = t[start:] if start != -1 else t

    try:
        obj, _ = json.JSONDecoder().raw_decode(t2)
        turns = obj.get("turns", [])
        cleaned: List[Dict] = []
        for turn in turns:
            role = (turn.get("role") or "").strip().lower()
            content = sanitize_for_training(str(turn.get("content", "")).strip())
            if not role or not content:
                continue

            if role in ("user", "person", "homeless"):
                role = "homeless"
            elif role in ("assistant", "helper"):
                role = "helper"
            else:
                continue

            if role == "helper":
                content = strip_sensitive_numbers(content)
                content = normalize_bullets(content)

            cleaned.append({"role": role, "content": content})

        if any(x["role"] == "homeless" for x in cleaned) and any(x["role"] == "helper" for x in cleaned):
            return cleaned
    except Exception:
        pass

    person_1 = f"I'm in {scenario['location']} and I need help with {scenario['primary_need']}. I don't know where to start."
    helper_1 = "I understand you're going through a tough time. Are you in a safe place right now? Do you need a place to sleep tonight? Are you alone or with children?"
    
    person_2 = "I'm alone. I need a place to stay tonight and I don't have many belongings."
    helper_2 = (
    "Thank you for explaining. Here are possible next steps:\n"
    "- Look for a shelter or community center for initial intake.\n"
    "- Request urgent assistance for tonight's stay, then discuss options for the coming days.\n"
    "- If you have issues with documents, ask social services for guidance.\n"
    "- If there is immediate danger, call emergency services right away.\n"
    "What area are you closest to right now?"
)

    return [
        {"role": "homeless", "content": sanitize_for_training(person_1)},
        {"role": "helper", "content": sanitize_for_training(helper_1)},
        {"role": "homeless", "content": sanitize_for_training(person_2)},
        {"role": "helper", "content": sanitize_for_training(helper_2)},
    ]


def make_training_records(conversations: list[dict]) -> list[dict]:
    out: List[Dict] = []

    base_system = (
    "You are a respectful and cooperative social support assistant.\n"
    "Speak in clear, practical, and balanced English.\n"
    "Do not invent addresses, phone numbers, or links.\n"
    "If you are unsure, say so and ask a specific clarifying question.\n"
    "In case of immediate danger or medical emergency: say 'Call emergency services immediately.'\n"
)

    for conv in conversations:
        turns = conv.get("turns") or []
        if len(turns) < 2:
            continue

        history_lines: List[str] = []
        for idx, turn in enumerate(turns):
            role = turn.get("role")
            content = turn.get("content", "")

            if role == "homeless":
                history_lines.append(f"Person: {content}")
                continue

            if role == "helper":
                instruction = base_system + "\ncontext :\n" + "\n".join(history_lines) + "\n\n repsonse:"

                last_user = ""
                for j in range(len(history_lines) - 1, -1, -1):
                    if history_lines[j].startswith("Person: "):
                        last_user = history_lines[j].replace("Person: ", "", 1)
                        break

                out.append({
                    "instruction": instruction,
                    "input": last_user,
                    "output": content,
                    "scenario": conv.get("scenario") or {},
                    "messages": (
                        [{"role": "system", "content": base_system}] +
                        [
                            {"role": ("user" if t.get("role") == "homeless" else "assistant"), "content": t.get("content", "")}
                            for t in turns[:idx + 1]
                            if t.get("content")
                        ]
                    ),
                })

                history_lines.append(f"Assistant: {content}")

    return out


def init_run_files(run_id: str) -> Tuple[str, str]:
    raw_path, rl_path = run_paths(run_id)
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write("[\n")
    with open(rl_path, "w", encoding="utf-8"):
        pass
    return raw_path, rl_path


def append_conversation(raw_path: str, rl_path: str, conv: dict, raw_written: int) -> int:
    with open(raw_path, "a", encoding="utf-8") as f:
        if raw_written > 0:
            f.write(",\n")
        f.write(json.dumps(conv, ensure_ascii=False))
        f.flush()

    training = make_training_records([conv])
    with open(rl_path, "a", encoding="utf-8") as f:
        for row in training:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()

    return len(training)


def finalize_raw_file(raw_path: str) -> None:
    with open(raw_path, "a", encoding="utf-8") as f:
        f.write("\n]\n")


def generate_one(conv_id: int) -> dict:
    scenario = generate_scenario()
    messages = build_pair_prompt(scenario)
    text = call_groq(messages)
    turns = parse_pair(text, scenario)

    return {
        "id": conv_id,
        "scenario": scenario,
        "turns": turns,
        "timestamp": datetime.now().isoformat(),
        "mode": GEN_MODE,
        "model": MODEL,
    }

def start_generation_run(target_value: Optional[str]) -> dict:
    global is_generating, TARGET_CONVERSATIONS, generation_thread, RUN_STARTED_AT

    if is_generating:
        raise RuntimeError("Generation already running")

    parsed_target = parse_target(target_value) if target_value is not None else None
    TARGET_CONVERSATIONS = parsed_target

    run_id = new_run_id()
    raw_path, rl_path = run_paths(run_id)
    add_run_to_manifest(run_id, raw_path, rl_path, TARGET_CONVERSATIONS)

    is_generating = True
    RUN_STARTED_AT = time.time()

    while not generation_queue.empty():
        try:
            generation_queue.get_nowait()
        except queue.Empty:
            break

    generation_thread = threading.Thread(target=generation_worker, args=(run_id,), daemon=True)
    generation_thread.start()

    return {
        "status": "started",
        "run_id": run_id,
        "target": TARGET_CONVERSATIONS if TARGET_CONVERSATIONS is not None else 0,
        "model": MODEL,
        "manifest": MANIFEST_PATH,
    }

# ---------------------------
# Worker
# ---------------------------
def generation_worker(run_id: str):
    global is_generating, dataset, TARGET_CONVERSATIONS, CURRENT_RUN_ID, generation_thread, RUN_STARTED_AT

    CURRENT_RUN_ID = run_id
    dataset = []
    conv_id = 0
    raw_written = 0
    training_written = 0

    raw_path, rl_path = init_run_files(run_id)

    generation_queue.put({
        "type": "status",
        "message": (
            f"Started run: {run_id} (model={MODEL}, target={TARGET_CONVERSATIONS}, "
            f"workers={PARALLEL_WORKERS})"
        )
    })

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, PARALLEL_WORKERS))
    pending: Dict[concurrent.futures.Future, int] = {}
    try:
        while is_generating and (TARGET_CONVERSATIONS is None or len(dataset) < TARGET_CONVERSATIONS):
            while (
                is_generating
                and len(pending) < max(1, PARALLEL_WORKERS)
                and (
                    TARGET_CONVERSATIONS is None
                    or (len(dataset) + len(pending) < TARGET_CONVERSATIONS)
                )
            ):
                conv_id += 1
                future = executor.submit(generate_one, conv_id)
                pending[future] = conv_id

            if not pending:
                break

            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED, timeout=1
            )
            if not done:
                continue

            for future in done:
                pending.pop(future, None)
                try:
                    conv = future.result()
                except Exception as e:
                    generation_queue.put({"type": "error", "conv_id": len(dataset) + 1, "error": str(e)[:180]})
                    time.sleep(0.1)
                    continue

                if TURN_SLEEP_SECONDS > 0:
                    time.sleep(TURN_SLEEP_SECONDS)

                dataset.append(conv)
                raw_written += 1
                training_written += append_conversation(raw_path, rl_path, conv, raw_written - 1)

                generation_queue.put({
                    "type": "new_conversation",
                    "conv_id": len(dataset),
                    "total": TARGET_CONVERSATIONS if TARGET_CONVERSATIONS is not None else -1,
                    "scenario": conv.get("scenario", {})
                })

                for t in conv.get("turns", []):
                    generation_queue.put({
                        "type": "turn",
                        "conv_id": len(dataset),
                        "role": t.get("role"),
                        "content": t.get("content")
                    })

                generation_queue.put({
                    "type": "conversation_complete",
                    "conv_id": len(dataset),
                    "total_complete": len(dataset),
                    "progress": progress_payload(),
                    "usage": stats_payload(),
                    "eta_seconds": eta_seconds(),
                })

                if CONVERSATION_SLEEP_SECONDS > 0:
                    time.sleep(CONVERSATION_SLEEP_SECONDS)
    finally:
        executor.shutdown(wait=is_generating, cancel_futures=not is_generating)

    finalize_raw_file(raw_path)
    c = len(dataset)
    s = training_written

    if is_generating:
        mark_run_complete(run_id, c, s)
    else:
        mark_run_stopped(run_id, c, s)

    generation_queue.put({
        "type": "complete",
        "total_conversations": c,
        "total_training_samples": s,
        "raw_path": raw_path,
        "rl_path": rl_path
    })

    CURRENT_RUN_ID = None
    is_generating = False
    generation_thread = None
    RUN_STARTED_AT = None

# ---------------------------
# Merge
# ---------------------------
def merge_all_runs() -> dict:
    manifest = load_manifest()
    runs = manifest.get("runs", [])

    rl_files = [r["rl_path"] for r in runs if r.get("rl_path") and os.path.exists(r["rl_path"])]
    raw_files = [r["raw_path"] for r in runs if r.get("raw_path") and os.path.exists(r["raw_path"])]

    total_lines = 0
    with open(MERGED_RL_PATH, "w", encoding="utf-8") as out:
        for fp in rl_files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.write(line + "\n")
                    total_lines += 1

    merged_raw: List[dict] = []
    for fp in raw_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                arr = json.load(f)
                if isinstance(arr, list):
                    merged_raw.extend(arr)
        except Exception:
            pass

    with open(MERGED_RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_raw, f, indent=2, ensure_ascii=False)

    return {
        "runs_count": len(runs),
        "merged_rl_path": MERGED_RL_PATH,
        "merged_raw_path": MERGED_RAW_PATH,
        "merged_training_lines": total_lines,
        "merged_raw_records": len(merged_raw),
    }

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dataset Generator</title>
  <style>
    :root { color-scheme: light; }
    body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
    .card { max-width: 720px; padding: 16px; border: 1px solid #ddd; border-radius: 8px; }
    .bar { height: 16px; background: #eee; border-radius: 8px; overflow: hidden; }
    .bar > div { height: 100%; width: 0%; background: #1a73e8; transition: width 200ms linear; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 8px; }
    .muted { color: #666; font-size: 0.9em; }
    .status { margin-top: 12px; }
    .log { margin-top: 12px; font-family: Consolas, monospace; font-size: 0.9em; white-space: pre-wrap; }
    .stats { margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; }
    .stats div { font-size: 0.9em; }
  </style>
</head>
<body>
  <div class="card">
    <h2>dataset generator</h2>
    <div class="bar"><div id="bar"></div></div>
    <div class="row">
      <div id="label" class="muted">0/0</div>
      <div id="percent" class="muted">0%</div>
    </div>
    <div id="status" class="status">Waiting for data...</div>
    <div class="stats">
      <div>Requests: <span id="reqTotal">0</span></div>
      <div>RPM: <span id="rpm">0</span> / <span id="rpmLimit">0</span></div>
      <div>RPD: <span id="rpd">0</span> / <span id="rpdLimit">0</span></div>
      <div>TPM: <span id="tpm">0</span> / <span id="tpmLimit">0</span></div>
      <div>Total tokens: <span id="tokensTotal">0</span></div>
      <div>Last request tokens: <span id="tokensLast">0</span></div>
      <div>ETA: <span id="eta">--</span></div>
    </div>
    <div id="log" class="log"></div>
  </div>
  <script>
    const rateLimits = """ + json.dumps(RATE_LIMITS) + """;
    const bar = document.getElementById("bar");
    const label = document.getElementById("label");
    const percent = document.getElementById("percent");
    const statusEl = document.getElementById("status");
    const logEl = document.getElementById("log");
    const reqTotal = document.getElementById("reqTotal");
    const rpm = document.getElementById("rpm");
    const rpd = document.getElementById("rpd");
    const tpm = document.getElementById("tpm");
    const tokensTotal = document.getElementById("tokensTotal");
    const tokensLast = document.getElementById("tokensLast");
    const eta = document.getElementById("eta");
    const rpmLimit = document.getElementById("rpmLimit");
    const rpdLimit = document.getElementById("rpdLimit");
    const tpmLimit = document.getElementById("tpmLimit");

    if (rateLimits) {
      rpmLimit.textContent = rateLimits.rpm ?? 0;
      rpdLimit.textContent = rateLimits.rpd ?? 0;
      tpmLimit.textContent = rateLimits.tpm ?? 0;
    }

    function setProgress(progress) {
      const completed = Number(progress.completed || 0);
      const target = Number(progress.target || 0);
      const pct = Number(progress.percent || 0);
      if (target > 0) {
        bar.style.width = Math.max(0, Math.min(100, pct)) + "%";
        label.textContent = completed + "/" + target;
        percent.textContent = pct + "%";
      } else {
        bar.style.width = "0%";
        label.textContent = completed + "/unlimited";
        percent.textContent = "0%";
      }
    }

    function addLog(text) {
      if (!text) return;
      const next = (logEl.textContent + "\\n" + text).trim();
      logEl.textContent = next.slice(-2000);
    }

    function setUsage(usage) {
      if (!usage) return;
      reqTotal.textContent = usage.requests_total ?? 0;
      rpm.textContent = usage.rpm ?? 0;
      rpd.textContent = usage.rpd ?? 0;
      tpm.textContent = usage.tpm ?? 0;
      tokensTotal.textContent = usage.tokens_total ?? 0;
      tokensLast.textContent = (usage.last_request && usage.last_request.total_tokens) ? usage.last_request.total_tokens : 0;
    }

    function formatEta(seconds) {
      if (seconds === null || seconds === undefined) return "--";
      if (seconds <= 0) return "0s";
      const hrs = Math.floor(seconds / 3600);
      const mins = Math.floor((seconds % 3600) / 60);
      const secs = Math.floor(seconds % 60);
      if (hrs > 0) return hrs + "h " + mins + "m";
      if (mins > 0) return mins + "m " + secs + "s";
      return secs + "s";
    }

    fetch("/status")
      .then(r => r.json())
      .then(data => {
        if (data && data.is_generating !== undefined) {
          statusEl.textContent = data.is_generating ? "Generating..." : "Idle";
        }
        if (data) {
          setProgress(data);
          setUsage(data.usage);
          eta.textContent = formatEta(data.eta_seconds);
          if (data.rate_limits) {
            rpmLimit.textContent = data.rate_limits.rpm ?? 0;
            rpdLimit.textContent = data.rate_limits.rpd ?? 0;
            tpmLimit.textContent = data.rate_limits.tpm ?? 0;
          }
        }
      })
      .catch(() => {});

    const es = new EventSource("/stream");
    es.onmessage = (event) => {
      let data = null;
      try { data = JSON.parse(event.data); } catch (_) {}
      if (!data) return;
      if (data.type === "status") {
        statusEl.textContent = data.message || "Generating...";
        addLog(data.message);
      }
      if (data.type === "conversation_complete" && data.progress) {
        setProgress(data.progress);
        setUsage(data.usage);
        eta.textContent = formatEta(data.eta_seconds);
        statusEl.textContent = "Generating...";
      }
      if (data.type === "complete") {
        statusEl.textContent = "Complete";
        addLog("Complete. Conversations: " + data.total_conversations);
      }
      if (data.type === "error") {
        statusEl.textContent = "Error";
        addLog("Error: " + data.error);
      }
    };
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


@app.route("/start", methods=["POST", "GET"])
def start_generation():
    global is_generating
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = {}

    target = (
        request.args.get("target")
        or payload.get("target")
        or str(TARGET_CONVERSATIONS_DEFAULT)
    )
    try:
        return jsonify(start_generation_run(target))
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/stop", methods=["POST"])
def stop_generation():
    global is_generating
    is_generating = False
    return jsonify({"status": "stopping"})


@app.route("/status")
def status():
    payload = progress_payload()
    payload.update({
        "is_generating": is_generating,
        "model": MODEL,
        "current_run_id": CURRENT_RUN_ID,
        "manifest": MANIFEST_PATH,
        "usage": stats_payload(),
        "rate_limits": RATE_LIMITS,
        "eta_seconds": eta_seconds(),
    })
    return jsonify(payload)


@app.route("/progress")
def progress():
    return jsonify(progress_payload())


@app.route("/runs")
def runs():
    return jsonify(load_manifest())


@app.route("/merge", methods=["POST"])
def merge():
    if is_generating:
        return jsonify({"error": "Stop generation before merge."}), 400
    result = merge_all_runs()
    return jsonify({"status": "merged", **result})


@app.route("/stream")
def stream():
    def event_stream():
        while True:
            try:
                data = generation_queue.get(timeout=30)
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                if data.get("type") == "complete":
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'}, ensure_ascii=False)}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    port = int(os.getenv("PORT") or "5000")
    if should_auto_start() and not is_generating:
        try:
            start_generation_run(str(TARGET_CONVERSATIONS_DEFAULT))
        except Exception as e:
            print(f"Auto-start failed: {e}")
    try:
        app.run(debug=False, threaded=True, port=port)
    except KeyboardInterrupt:
        is_generating = False
        if generation_thread and generation_thread.is_alive():
            generation_thread.join(timeout=5)
