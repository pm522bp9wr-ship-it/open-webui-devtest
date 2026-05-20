"""
title: Production Prompt Enhancer
author: bezz
version: 5.0.0
description: Intent-aware LLM-based filter that enhances user prompts for better responses. Features adaptive style picking, optional self-critique pass, dynamic intent exemplars, multimodal/tool/output-format awareness, per-user overrides, TTL-aware caching with full-config keys, request coalescing, config-driven intents, and smart skip logic.
required_open_webui_version: 0.9.1
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, Field
from fastapi import Request
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.misc import get_last_user_message
from open_webui.models.users import Users
from open_webui.constants import TASKS

logger = logging.getLogger("prompt_enhancer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# LRU + TTL cache for prompt enhancements
# ---------------------------------------------------------------------------


class _PromptCache:
    """LRU cache keyed by (full config signature + prompt) with optional TTL."""

    def __init__(self, maxsize: int = 128, ttl_seconds: float = 0.0):
        self._cache: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds

    def configure(self, maxsize: Optional[int] = None, ttl_seconds: Optional[float] = None) -> None:
        if maxsize is not None and maxsize > 0:
            self._maxsize = maxsize
        if ttl_seconds is not None and ttl_seconds >= 0:
            self._ttl = ttl_seconds

    def _key(self, signature: str, prompt: str) -> str:
        raw = f"{signature}\x00{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _expired(self, ts: float) -> bool:
        return self._ttl > 0 and (time.time() - ts) > self._ttl

    def get(self, signature: str, prompt: str) -> Optional[str]:
        key = self._key(signature, prompt)
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, ts = entry
        if self._expired(ts):
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return value

    def put(self, signature: str, prompt: str, enhanced: str) -> None:
        key = self._key(signature, prompt)
        self._cache[key] = (enhanced, time.time())
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


_prompt_cache = _PromptCache(maxsize=128)

# In-flight enhancement futures, keyed by the same key the cache uses.
# Concurrent identical requests share a single LLM call instead of N.
_inflight: "dict[str, asyncio.Future]" = {}


async def _coalesce(key: str, factory: "Callable[[], Awaitable[Optional[str]]]") -> Optional[str]:
    existing = _inflight.get(key)
    if existing is not None:
        return await existing

    loop = asyncio.get_event_loop()
    fut: "asyncio.Future" = loop.create_future()
    _inflight[key] = fut
    try:
        result = await factory()
        if not fut.done():
            fut.set_result(result)
        return result
    except BaseException as exc:
        if not fut.done():
            fut.set_exception(exc)
        raise
    finally:
        _inflight.pop(key, None)
        # If no other waiter consumed a stored exception, mark it retrieved
        # so asyncio doesn't log a spurious "exception was never retrieved".
        if fut.done() and not fut.cancelled():
            fut.exception()


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_THINKING_RE = re.compile(
    r"<(think|thinking|reason|reasoning|thought)>.*?</\1>"
    r"|"
    r"\|begin_of_thought\|.*?\|end_of_thought\|",
    re.DOTALL | re.IGNORECASE,
)

# A lone opening reasoning tag with no matching close (truncated reasoning):
# strip from the tag to end-of-string so raw chain-of-thought never leaks.
_DANGLING_THINK_RE = re.compile(
    r"<(?:think|thinking|reason|reasoning|thought)>.*\Z"
    r"|"
    r"\|begin_of_thought\|.*\Z",
    re.DOTALL | re.IGNORECASE,
)

_ARTIFACT_PATTERNS = [
    re.compile(
        r"^\s*(?:enhanced\s+prompt|here(?:'s| is) (?:the |your )?enhanced (?:prompt|version))[\s:]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:sure|certainly|of course|absolutely)[!,.]?\s*(?:here(?:'s| is))?\s*(?:the enhanced (?:prompt|version))?[\s:]*",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*\*\*enhanced prompt:?\*\*\s*", re.IGNORECASE),
]


def _clean_llm_output(text: str) -> str:
    cleaned = _THINKING_RE.sub("", text)
    cleaned = _DANGLING_THINK_RE.sub("", cleaned).strip()
    for pat in _ARTIFACT_PATTERNS:
        cleaned = pat.sub("", cleaned).strip()
    if len(cleaned) > 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        inner = cleaned[1:-1].strip()
        if len(inner) > 10:
            cleaned = inner
    return cleaned


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------

_FOLLOWUP_RE = re.compile(
    r"^(?:now |also |instead |change |modify |update |add |remove |make it |"
    r"try |use |switch |but |and |then |what about |how about |can you also )",
    re.IGNORECASE,
)

_DEICTIC_RE = re.compile(
    r"\b(it|its|that|this|these|those|them|they|the same|instead|again|"
    r"the above|previous|former|latter)\b",
    re.IGNORECASE,
)


def _has_prior_assistant(messages: list[dict]) -> bool:
    return any(m.get("role") == "assistant" for m in messages[:-1])


def _is_followup(messages: list[dict], user_message: str) -> bool:
    if len(messages) < 3 or not _has_prior_assistant(messages):
        return False
    word_count = len(user_message.split())
    if word_count > 40:
        return False
    if _FOLLOWUP_RE.match(user_message):
        return True
    # Short message that leans on prior context (deictic reference) — not just
    # any short message, to avoid treating new short questions as follow-ups.
    if word_count <= 8 and _DEICTIC_RE.search(user_message):
        return True
    return False


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------


def _is_well_structured(text: str) -> bool:
    indicators = 0
    if re.search(r"^\s*#{1,3}\s", text, re.MULTILINE):
        indicators += 1
    if len(re.findall(r"^\s*[-*]\s", text, re.MULTILINE)) >= 3:
        indicators += 1
    if len(re.findall(r"^\s*\d+\.\s", text, re.MULTILINE)) >= 3:
        indicators += 1
    if "```" in text:
        indicators += 1
    if len(text.split()) > 150:
        indicators += 1
    if re.search(r"\b(you are|act as|role:)\b", text, re.IGNORECASE):
        indicators += 1
    return indicators >= 3


# ---------------------------------------------------------------------------
# Code-only detection — skip prompts that are mostly code
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)


def _is_code_only(text: str) -> bool:
    code_blocks = _CODE_BLOCK_RE.findall(text)
    if not code_blocks:
        return False
    code_len = sum(len(b) for b in code_blocks)
    total_len = len(text.strip())
    if total_len == 0:
        return False
    non_code = total_len - code_len
    return non_code < 40 and (code_len / total_len) > 0.85


# ---------------------------------------------------------------------------
# Adaptive style scoring
# ---------------------------------------------------------------------------

_VAGUE_WORDS_RE = re.compile(
    r"\b(something|anything|stuff|things?|maybe|kind of|sort of|whatever)\b",
    re.IGNORECASE,
)
_CONSTRAINT_WORDS_RE = re.compile(
    r"\b(format|length|words?|chars?|return|must|should|do not|don'?t|"
    r"only|exactly|at most|at least|within|under|over|step[- ]by[- ]step)\b",
    re.IGNORECASE,
)
_STRUCTURE_MARKER_RE = re.compile(
    r"(^\s*#{1,3}\s|^\s*[-*]\s|^\s*\d+\.\s|```)",
    re.MULTILINE,
)


def _score_prompt(text: str) -> dict[str, Any]:
    words = text.split()
    word_count = len(words)
    return {
        "word_count": word_count,
        "vague_hits": len(_VAGUE_WORDS_RE.findall(text)),
        "constraint_hits": len(_CONSTRAINT_WORDS_RE.findall(text)),
        "has_structure": bool(_STRUCTURE_MARKER_RE.search(text)),
        "question_marks": text.count("?"),
    }


def _pick_style(score: dict[str, Any]) -> str:
    word_count = score["word_count"]
    constraints = score["constraint_hits"]
    vague = score["vague_hits"]
    structured = score["has_structure"]

    # Already-clear prompts: short, low-vagueness, has explicit constraints
    # and/or structural markers — keep changes minimal.
    if structured and constraints >= 1 and vague == 0:
        return "concise"
    if word_count <= 25 and constraints >= 2 and vague == 0:
        return "concise"

    # Very short and/or vague prompts benefit from a thorough expansion.
    if word_count <= 10 or vague >= 2:
        return "detailed"
    if word_count <= 20 and constraints == 0 and vague >= 1:
        return "detailed"

    return "standard"


# ---------------------------------------------------------------------------
# Output format detection
# ---------------------------------------------------------------------------

# Ordered: more specific patterns first.
_FORMAT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "markdown-table",
        re.compile(r"\b(markdown\s+table|as\s+(?:a\s+)?(?:md|markdown)\s+table)\b", re.IGNORECASE),
    ),
    (
        "json",
        re.compile(r"\b(?:as|in|return(?:ed)?(?:\s+as)?|output(?:\s+as)?)\s+json\b", re.IGNORECASE),
    ),
    (
        "yaml",
        re.compile(r"\b(?:as|in|return(?:ed)?(?:\s+as)?|output(?:\s+as)?)\s+yaml\b", re.IGNORECASE),
    ),
    (
        "csv",
        re.compile(r"\b(?:as|in|return(?:ed)?(?:\s+as)?|output(?:\s+as)?)\s+csv\b", re.IGNORECASE),
    ),
    (
        "code-only",
        re.compile(r"\b(code\s+only|just\s+(?:the\s+)?code|only\s+(?:the\s+)?code)\b", re.IGNORECASE),
    ),
    (
        "bullets",
        re.compile(r"\b(bullet\s+points?|as\s+bullets|in\s+bullets)\b", re.IGNORECASE),
    ),
]


def _detect_output_format(text: str) -> Optional[str]:
    for name, pat in _FORMAT_PATTERNS:
        if pat.search(text):
            return name
    return None


# ---------------------------------------------------------------------------
# Multimodal hint — count image parts in the last user message
# ---------------------------------------------------------------------------


def _count_image_parts(messages: list[dict]) -> int:
    if not messages:
        return 0
    last = messages[-1]
    if last.get("role") != "user":
        return 0
    content = last.get("content")
    if not isinstance(content, list):
        return 0
    count = 0
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if isinstance(ptype, str) and ptype.startswith("image"):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

INTENT_DEFS: dict[str, dict[str, Any]] = {
    "debugging": {
        "priority": 95,
        "patterns": [
            r"\b(error|exception|traceback|stack ?trace|bug|crash|panic|segfault)\b",
            r"\b(not working|doesn'?t work|fails?|failing|broken)\b",
            r"\b(debug|debugging|troubleshoot)\b",
        ],
        "hint": (
            "This is a debugging request. The enhanced prompt should ask the AI to:\n"
            "  - Identify the root cause from the error/symptoms described\n"
            "  - Provide the corrected code or exact fix steps\n"
            "  - Explain why the fix works\n"
            "  - Suggest how to prevent similar issues"
        ),
    },
    "coding": {
        "priority": 90,
        "patterns": [
            r"\b(code|program|function|class|method|script)\b",
            r"\b(python|javascript|typescript|java|rust|go(?:lang)?|ruby|php|sql)\b",
            r"\bc\+\+",
            r"\b(refactor|optimize|compile|unit test|pytest|jest)\b",
            r"\b(api|endpoint|rest|graphql)\b",
            r"```[\s\S]*?```",
        ],
        "hint": (
            "This is a coding request. The enhanced prompt should specify:\n"
            "  - Target language/framework if not already stated\n"
            "  - Expected input/output behavior\n"
            "  - Error handling and edge case requirements\n"
            "  - Whether tests, type hints, or documentation are expected\n"
            "  - Code should be in fenced blocks with language tags"
        ),
    },
    "creative": {
        "priority": 75,
        "patterns": [
            r"\b(write|compose|craft|draft) (a |an |me a |some )?(story|poem|song|novel|chapter|scene|dialogue|script|essay|blog post)\b",
            r"\b(fiction|fantasy|sci[- ]?fi|short story|narrative)\b",
            r"\b(character|plot|setting|prose|verse)\b",
        ],
        "hint": (
            "This is a creative writing request. The enhanced prompt should specify:\n"
            "  - Tone and mood (dark, whimsical, serious, etc.)\n"
            "  - Approximate length or structure\n"
            "  - Point of view and tense preferences\n"
            "  - Any thematic elements to emphasize\n"
            "  - Do NOT over-constrain — leave room for creative expression"
        ),
    },
    "analysis": {
        "priority": 70,
        "patterns": [
            r"\b(analy[sz]e|analysis|evaluate|assess|examine|investigate|review)\b",
            r"\b(root cause|implications?|interpret|significance)\b",
        ],
        "hint": (
            "This is an analysis request. The enhanced prompt should ask for:\n"
            "  - Structured breakdown with clear sections\n"
            "  - Evidence-based reasoning with cited sources where possible\n"
            "  - Consideration of counterarguments or alternative interpretations\n"
            "  - A clear, confidence-rated conclusion"
        ),
    },
    "explanation": {
        "priority": 65,
        "patterns": [
            r"\b(explain|describe|clarify|elaborate)\b",
            r"\b(what (is|are|does)|how (does|do|is|are))\b",
            r"\b(teach me|help me understand|walk me through|eli5)\b",
        ],
        "hint": (
            "This is an explanation request. The enhanced prompt should ask for:\n"
            "  - An intuitive overview before diving into details\n"
            "  - Concrete examples and analogies\n"
            "  - Definitions of jargon on first use\n"
            "  - Common misconceptions addressed"
        ),
    },
    "comparison": {
        "priority": 78,
        "patterns": [
            r"\b(compare|comparison|contrast|versus)\b",
            r"\bvs\b\.?",
            r"\b(difference|differences|similarities?) (between|in)\b",
            r"\b(pros? and cons?|trade[- ]offs?)\b",
        ],
        "hint": (
            "This is a comparison request. The enhanced prompt should ask for:\n"
            "  - A structured comparison table on key dimensions\n"
            "  - Specific use-case scenarios for each option\n"
            "  - A clear recommendation with justification"
        ),
    },
    "planning": {
        "priority": 72,
        "patterns": [
            r"\b(plan|roadmap|schedule|timeline|milestones?)\b",
            r"\b(break ?down|step[- ]by[- ]step|steps to)\b",
            r"\b(project|strategy|outline|organi[sz]e)\b",
        ],
        "hint": (
            "This is a planning request. The enhanced prompt should ask for:\n"
            "  - Numbered, sequenced action items\n"
            "  - Effort estimates and dependencies\n"
            "  - Risks, blockers, and milestones\n"
            "  - Clear scope boundaries"
        ),
    },
    "summarization": {
        "priority": 85,
        "patterns": [
            r"\b(summari[sz]e|summary|tl;?dr|brief overview|executive summary)\b",
            r"\b(key points?|main points?|takeaways?|recap)\b",
        ],
        "hint": (
            "This is a summarization request. The enhanced prompt should ask for:\n"
            "  - A one-sentence TL;DR followed by key points\n"
            "  - Faithfulness to the original — no added opinions\n"
            "  - Proportionate length — shorter is usually better"
        ),
    },
    "research": {
        "priority": 68,
        "patterns": [
            r"\b(research|find (info|information|sources?))\b",
            r"\b(sources?|citations?|references?|studies|papers?)\b",
            r"\b(latest|recent|current|state[- ]of[- ]the[- ]art)\b",
        ],
        "hint": (
            "This is a research request. The enhanced prompt should ask for:\n"
            "  - Specific, citable sources where possible\n"
            "  - Clear distinction between established facts and emerging findings\n"
            "  - Explicit flagging of uncertain or unverifiable claims\n"
            "  - Recency-aware information"
        ),
    },
    "brainstorming": {
        "priority": 60,
        "patterns": [
            r"\b(brainstorm|ideate|ideas? for|suggestions? for|come up with)\b",
            r"\b(list (of )?ideas?|options?|alternatives?)\b",
        ],
        "hint": (
            "This is a brainstorming request. The enhanced prompt should ask for:\n"
            "  - A diverse range of ideas (safe to bold)\n"
            "  - Brief rationale for each idea\n"
            "  - A ranked shortlist of the strongest options"
        ),
    },
    "translation": {
        "priority": 88,
        "patterns": [
            r"\btranslate\b",
            r"\b(from|to|into|in) (english|spanish|french|german|italian|portuguese|russian|chinese|japanese|korean|arabic|hindi)\b",
        ],
        "hint": (
            "This is a translation request. The enhanced prompt should specify:\n"
            "  - Natural, fluent phrasing over literal word-for-word\n"
            "  - Appropriate register (formal/informal)\n"
            "  - Translator notes for idioms or culturally-bound terms"
        ),
    },
    "problem_solving": {
        "priority": 80,
        "patterns": [
            r"\b(solve|solution|fix|resolve|figure out)\b",
            r"\b(problem|issue|challenge|obstacle)\b",
            r"\b(stuck|can'?t|cannot|unable to)\b",
        ],
        "hint": (
            "This is a problem-solving request. The enhanced prompt should ask for:\n"
            "  - Restated problem and constraints\n"
            "  - Multiple solution options with trade-offs\n"
            "  - A recommended approach with implementation steps\n"
            "  - Verification and rollback strategies"
        ),
    },
    "data_science": {
        "priority": 82,
        "patterns": [
            r"\b(data ?set|dataframe|pandas|numpy|scipy|sklearn|scikit[- ]learn)\b",
            r"\b(machine learning|deep learning|neural net(work)?|model training)\b",
            r"\b(regression|classification|clustering|feature engineering)\b",
            r"\b(visualization|matplotlib|seaborn|plotly)\b",
            r"\b(csv|parquet|json[l]?)\b.*\b(load|read|parse|import)\b",
        ],
        "hint": (
            "This is a data science request. The enhanced prompt should specify:\n"
            "  - Data shape, types, and size if relevant\n"
            "  - Target metric or success criterion\n"
            "  - Whether exploratory analysis or production-ready code is expected\n"
            "  - Visualization or reporting requirements\n"
            "  - Library preferences (pandas, polars, etc.)"
        ),
    },
    "math": {
        "priority": 76,
        "patterns": [
            r"\b(calculate|compute|derive|prove|equation|formula)\b",
            r"\b(integral|derivative|matrix|vector|probability|statistics)\b",
            r"\b(algebra|calculus|geometry|trigonometry|linear algebra)\b",
            r"\b(theorem|proof|conjecture|lemma)\b",
        ],
        "hint": (
            "This is a math request. The enhanced prompt should ask for:\n"
            "  - Step-by-step working shown clearly\n"
            "  - LaTeX formatting for equations where appropriate\n"
            "  - Verification of the answer with a sanity check\n"
            "  - Intuitive explanation alongside formal derivation"
        ),
    },
    "devops": {
        "priority": 74,
        "patterns": [
            r"\b(docker|kubernetes|k8s|helm|terraform|ansible|ci/?cd)\b",
            r"\b(deploy|deployment|infrastructure|pipeline|container)\b",
            r"\b(aws|azure|gcp|cloud|serverless|lambda)\b",
            r"\b(nginx|apache|load ?balancer|reverse ?proxy)\b",
            r"\b(monitoring|observability|prometheus|grafana|logs?)\b",
        ],
        "hint": (
            "This is a DevOps/infrastructure request. The enhanced prompt should specify:\n"
            "  - Target platform and environment constraints\n"
            "  - Security and access control requirements\n"
            "  - Scalability and high-availability needs\n"
            "  - Rollback and disaster recovery considerations\n"
            "  - Configuration as code where applicable"
        ),
    },
    "security": {
        "priority": 86,
        "patterns": [
            r"\b(security|vulnerabilit(y|ies)|exploit|attack|threat)\b",
            r"\b(authentication|authorization|oauth|jwt|rbac)\b",
            r"\b(encrypt(ion)?|hash(ing)?|ssl|tls|certificate)\b",
            r"\b(xss|csrf|injection|owasp|penetration|pentest)\b",
        ],
        "hint": (
            "This is a security request. The enhanced prompt should ask for:\n"
            "  - Threat model and attack vectors considered\n"
            "  - Defense-in-depth approach\n"
            "  - Compliance standards if applicable (SOC2, GDPR, etc.)\n"
            "  - Concrete remediation steps, not just theory\n"
            "  - Code examples with secure defaults"
        ),
    },
    "design": {
        "priority": 66,
        "patterns": [
            r"\b(ui|ux|design|wireframe|mockup|prototype|figma)\b",
            r"\b(layout|color ?scheme|typography|responsive|mobile[- ]first)\b",
            r"\b(accessibility|a11y|wcag|user experience|usability)\b",
        ],
        "hint": (
            "This is a design request. The enhanced prompt should specify:\n"
            "  - Target audience and platform (web, mobile, desktop)\n"
            "  - Accessibility requirements\n"
            "  - Brand guidelines or visual constraints\n"
            "  - Interaction patterns and user flow considerations"
        ),
    },
}


def _compile_intent_defs(defs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    compiled: dict[str, dict[str, Any]] = {}
    for name, cfg in defs.items():
        try:
            patterns = [re.compile(p, re.IGNORECASE) for p in cfg.get("patterns", [])]
        except re.error:
            continue
        if not patterns:
            continue
        compiled[name] = {
            "priority": cfg.get("priority", 50),
            "patterns": patterns,
            "hint": cfg.get("hint", ""),
        }
    return compiled


COMPILED_INTENTS: dict[str, dict[str, Any]] = _compile_intent_defs(INTENT_DEFS)


# Curated (original -> enhanced) exemplars by intent. Used when
# `dynamic_examples` is enabled to inject a single small examples block into
# the system prompt for the detected intent. Keep these short — they live in
# every enhancement call's system prompt when triggered.
INTENT_EXEMPLARS: dict[str, list[tuple[str, str]]] = {
    "debugging": [
        (
            "fix this TypeError when I run my script",
            "Diagnose the TypeError. Quote the exact failing line and traceback "
            "frame, identify the root cause (e.g. wrong type passed, None where "
            "an object was expected), propose the corrected code in a fenced "
            "block with the language tag, explain why the fix works, and list "
            "two ways to prevent the same class of bug.",
        ),
    ],
    "coding": [
        (
            "write a script to dedupe a list",
            "Write a Python 3 function `dedupe(items: Iterable[T]) -> list[T]` "
            "that preserves first-seen order. Handle non-hashable elements by "
            "falling back to an O(n^2) equality scan with a clear comment. "
            "Include type hints, a docstring, and three pytest cases covering "
            "ints, strings, and dicts.",
        ),
    ],
    "summarization": [
        (
            "summarize this article",
            "Summarize the article in two layers: (1) a single-sentence TL;DR; "
            "(2) 3–5 bulleted key points in the article's own order. Stay "
            "strictly faithful to the source — no added opinions or context. "
            "Keep the total summary under 25% of the original length.",
        ),
    ],
    "comparison": [
        (
            "compare postgres and mysql",
            "Compare PostgreSQL and MySQL for a mid-sized web application. "
            "Provide a markdown table on: SQL feature coverage, JSON/document "
            "support, replication, extensibility, and ecosystem maturity. "
            "Follow the table with two short scenario-based recommendations "
            "(read-heavy analytics vs. high-write OLTP).",
        ),
    ],
    "translation": [
        (
            "translate this to french",
            "Translate the following text to French. Prefer natural, fluent "
            "phrasing over literal word-for-word. Match the original register "
            "(formal vs. informal). For any idioms or culturally-bound terms, "
            "add a brief translator's note in square brackets.",
        ),
    ],
    "analysis": [
        (
            "analyze this code",
            "Analyze the code along four dimensions: correctness (any bugs or "
            "edge cases missed), readability (naming, structure, comments), "
            "performance (complexity hotspots), and maintainability. For each "
            "dimension, cite specific line ranges and rate severity (low/med/"
            "high). End with the top 3 actionable improvements.",
        ),
    ],
    "explanation": [
        (
            "explain how DNS works",
            "Explain how DNS works for a developer comfortable with HTTP but "
            "new to networking internals. Start with an intuitive overview of "
            "the resolution flow when a user opens a URL, then cover recursive "
            "resolvers, root/TLD/authoritative servers, and TTL-based caching. "
            "Use a concrete example resolving `www.example.com` end-to-end and "
            "define jargon on first use.",
        ),
    ],
    "planning": [
        (
            "plan a project to migrate to k8s",
            "Produce a Kubernetes migration plan as numbered phases, each with "
            "concrete deliverables, effort estimate (S/M/L), prerequisites, "
            "and rollback strategy. Cover: containerization, observability, "
            "ingress/networking, secrets management, and CI/CD. Highlight "
            "risks and milestones, and explicitly mark scope boundaries.",
        ),
    ],
}


# Memoized compilation of admin-supplied custom intents (keyed by raw string).
_custom_intent_cache: "dict[str, dict[str, dict[str, Any]]]" = {}


def _parse_extra_intents(raw: str) -> dict[str, dict[str, Any]]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    if raw in _custom_intent_cache:
        return _custom_intent_cache[raw]
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("extra_intent_patterns must be a JSON object")
        compiled = _compile_intent_defs(
            {
                str(name): {
                    "priority": cfg.get("priority", 50),
                    "patterns": cfg.get("patterns", []),
                    "hint": cfg.get("hint", ""),
                }
                for name, cfg in data.items()
                if isinstance(cfg, dict)
            }
        )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Invalid extra_intent_patterns, ignoring: %s", exc)
        compiled = {}
    _custom_intent_cache[raw] = compiled
    return compiled


TRIVIAL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*(hi|hello|hey|yo|sup|howdy|greetings|hiya)[\s!.?]*$",
        r"^\s*(thanks?|thank you|thx|ty|cheers)[\s!.?]*$",
        r"^\s*(ok|okay|k|cool|nice|great|awesome|got it|alright)[\s!.?]*$",
        r"^\s*(bye|goodbye|cya|see ya|later)[\s!.?]*$",
        r"^\s*(yes|no|yep|nope|yeah|nah|sure|maybe)[\s!.?]*$",
        r"^\s*(test|ping|hello world)[\s!.?]*$",
    ]
]


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_STYLE_INSTRUCTIONS = {
    "concise": (
        "\nStyle: CONCISE — enhance the prompt minimally. Add only the most "
        "critical missing details. Keep the result close to the original length. "
        "Prefer tightening over expanding."
    ),
    "standard": "",
    "detailed": (
        "\nStyle: DETAILED — produce a thorough, comprehensive enhanced prompt. "
        "Add context, constraints, format requirements, edge cases, and quality "
        "criteria. The result can be significantly longer than the original, up to "
        "3-4x, but every addition must add value."
    ),
}

BASE_SYSTEM_PROMPT = """\
You are an expert prompt engineer. Your task is to enhance the given prompt \
by making it more detailed, specific, and effective while preserving the \
user's original intent and voice.

Guidelines:
- Return ONLY the enhanced prompt. No headers, no "Enhanced Prompt:", no \
  wrapper text, no introductory phrases.
- Preserve the original language (English, Spanish, etc.).
- Make the prompt more specific and actionable without changing what the user \
  is asking for.
- Add details about expected format, depth, constraints, and quality where \
  the original is vague.
- Keep the enhanced prompt concise — improve clarity, don't add bloat. The \
  result should be at most 2-3x the original length.
- Preserve code blocks, URLs, and technical terms exactly as written.
- Do not add requirements the user didn't imply.

Examples:

Original: "Explain how DNS works"
Enhanced: "Explain how DNS works, starting with a plain-language overview of what happens when a user types a URL into their browser. Cover the role of recursive resolvers, root servers, TLD servers, and authoritative nameservers. Include a concrete example tracing the resolution of a real domain name. Define any technical terms on first use."

Original: "Write a Python script to rename files"
Enhanced: "Write a Python script that batch-renames files in a given directory. Accept a source directory path and a naming pattern (e.g., prefix + sequential number) as command-line arguments. Handle edge cases: empty directories, permission errors, and filename collisions. Use pathlib and argparse. Include type hints and a brief usage example in a docstring."

Original: "Compare React and Vue"
Enhanced: "Compare React and Vue.js for building a mid-sized single-page application. Cover: learning curve, ecosystem maturity, performance characteristics, state management approaches, TypeScript support, and community/job market. Use a comparison table for key dimensions, then provide a scenario-based recommendation for different team profiles."

IMPORTANT: Return ONLY the enhanced prompt text. Nothing else.\
"""

FOLLOWUP_SYSTEM_PROMPT = """\
You are an expert prompt engineer. The user is sending a follow-up message in \
an ongoing conversation. Your task is to enhance this follow-up while keeping \
it contextual — do NOT try to make it a standalone prompt.

Guidelines:
- Return ONLY the enhanced follow-up. No headers, no wrapper text.
- Keep the conversational tone — this is a continuation, not a new request.
- Add specificity about what "it", "that", "this" refer to when clear from context.
- If the user is asking for a modification, clarify what aspects to change and \
  what to preserve.
- Keep it brief — follow-ups should stay concise.
- Do not repeat information from earlier in the conversation.

Example:
Context: User asked for a Python CSV parser, assistant provided one.
Original follow-up: "now add error handling"
Enhanced follow-up: "Add robust error handling to the CSV parser: handle missing files (FileNotFoundError), malformed rows (skip and log them), and encoding issues (try UTF-8, fall back to latin-1). Add type hints to any new functions."

IMPORTANT: Return ONLY the enhanced follow-up text. Nothing else.\
"""

WELL_STRUCTURED_SYSTEM_PROMPT = """\
You are an expert prompt engineer. The user has written a detailed, \
well-structured prompt. It needs only light refinement — do NOT restructure \
or significantly expand it.

Guidelines:
- Return ONLY the refined prompt. No headers, no wrapper text.
- Make minimal, high-impact improvements only: fill obvious gaps, sharpen \
  vague requirements, fix ambiguities.
- Preserve the user's structure, formatting, and voice exactly.
- Do NOT add sections, change the organization, or significantly increase \
  the length.
- If the prompt is already excellent, return it nearly unchanged.

IMPORTANT: Return ONLY the refined prompt text. Nothing else.\
"""


# ---------------------------------------------------------------------------
# Enhancement context
# ---------------------------------------------------------------------------


@dataclass
class EnhancementContext:
    """Single source of truth for everything that affects the enhanced output.

    Also used to derive the cache signature, so the cache can never serve a
    result produced under a different configuration.
    """

    style: str = "standard"
    intents: list[str] = field(default_factory=list)
    is_followup: bool = False
    is_well_structured: bool = False
    custom_system_prompt: str = ""
    additional_instructions: str = ""
    model: str = ""
    temperature: float = 0.7
    user_id: str = ""
    self_critique: bool = False
    dynamic_examples: bool = False
    context_facts: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        payload = {
            "v": 5,
            "style": self.style,
            "intents": sorted(self.intents),
            "followup": self.is_followup,
            "structured": self.is_well_structured,
            "custom": self.custom_system_prompt.strip(),
            "extra": self.additional_instructions.strip(),
            "model": self.model,
            "temp": round(float(self.temperature), 3),
            "uid": self.user_id,
            "critique": self.self_critique,
            "examples": self.dynamic_examples,
            "facts": json.loads(
                json.dumps(self.context_facts, sort_keys=True, default=str)
            ),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _detect_intents(
    text: str,
    threshold: float = 0.55,
    intents: Optional[dict[str, dict[str, Any]]] = None,
) -> list[str]:
    catalog = intents if intents is not None else COMPILED_INTENTS
    results: list[tuple[str, float]] = []
    total_words = max(1, len(text.split()))
    length_factor = min(1.0, max(0.3, len(text) / 200.0)) if len(text) < 30 else 1.0

    for intent, cfg in catalog.items():
        total_hits = 0
        unique_hits = 0
        for pattern in cfg["patterns"]:
            found = pattern.findall(text)
            if found:
                unique_hits += 1
                total_hits += len(found)
        if total_hits == 0:
            continue

        sqrt_norm = min(1.0, (total_hits * 5) / (total_words**0.5))
        base = min(1.0, 0.35 + 0.15 * unique_hits + 0.05 * min(total_hits, 5))
        base = (base + sqrt_norm) / 2.0
        priority = cfg["priority"] / 100.0
        confidence = min(1.0, base * 0.7 + priority * 0.3) * length_factor
        results.append((intent, round(confidence, 3)))

    results.sort(key=lambda x: x[1], reverse=True)
    return [name for name, conf in results if conf >= threshold][:3]


def _is_trivial(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return any(p.match(stripped) for p in TRIVIAL_PATTERNS)


def _build_system_prompt(ctx: EnhancementContext, intents_catalog: dict[str, dict[str, Any]]) -> str:
    has_custom = bool(ctx.custom_system_prompt.strip())
    if has_custom:
        base = ctx.custom_system_prompt.strip()
    elif ctx.is_followup:
        base = FOLLOWUP_SYSTEM_PROMPT
    elif ctx.is_well_structured:
        base = WELL_STRUCTURED_SYSTEM_PROMPT
    else:
        base = BASE_SYSTEM_PROMPT

    parts = [base]

    if not has_custom:
        style_instr = _STYLE_INSTRUCTIONS.get(ctx.style, "")
        if style_instr:
            parts.append(style_instr)

        if ctx.intents:
            hints = []
            for intent in ctx.intents:
                cfg = intents_catalog.get(intent)
                if cfg and cfg.get("hint"):
                    hints.append(cfg["hint"])
            if hints:
                parts.append("Intent-specific guidance:\n" + "\n\n".join(hints))

        if ctx.dynamic_examples and ctx.intents:
            exemplars: list[tuple[str, str]] = []
            for intent in ctx.intents:
                for pair in INTENT_EXEMPLARS.get(intent, []):
                    exemplars.append(pair)
                    if len(exemplars) >= 2:
                        break
                if len(exemplars) >= 2:
                    break
            if exemplars:
                ex_lines = ["Examples for the detected intent(s):"]
                for original, enhanced in exemplars:
                    ex_lines.append(f'Original: "{original}"')
                    ex_lines.append(f'Enhanced: "{enhanced}"')
                    ex_lines.append("")
                parts.append("\n".join(ex_lines).rstrip())

        images = ctx.context_facts.get("images")
        if isinstance(images, int) and images > 0:
            noun = "image" if images == 1 else "images"
            parts.append(
                f"The user has attached {images} {noun}. The enhanced prompt "
                "must explicitly leverage vision (e.g., 'examine the "
                f"{noun}, then ...') and reference relevant image content "
                "where it informs the answer."
            )

        tools = ctx.context_facts.get("tools")
        if isinstance(tools, list) and tools:
            tool_list = ", ".join(str(t) for t in tools)
            parts.append(
                f"Available tools: {tool_list}. The enhanced prompt must "
                "direct the AI to call the appropriate tool(s) in the right "
                "order (e.g., 'first call X to fetch Y, then ...'). Do not "
                "assume tools that aren't listed."
            )

        fmt = ctx.context_facts.get("format")
        if isinstance(fmt, str) and fmt:
            parts.append(
                f"Requested output format: {fmt}. The enhanced prompt must "
                "require exactly this format, with a concrete schema, columns, "
                "or keys where applicable."
            )

    if ctx.additional_instructions.strip():
        parts.append(
            "Additional instructions from the user:\n"
            + ctx.additional_instructions.strip()
        )

    return "\n\n".join(parts)


def _build_user_prompt(
    user_message: str,
    messages: list[dict],
    tool_ids: Optional[list] = None,
    include_datetime: bool = True,
) -> str:
    parts: list[str] = []

    prior = [
        msg for msg in messages[:-1][-6:] if msg.get("role") in ("user", "assistant")
    ]
    if prior:
        context_lines = []
        for msg in prior:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if isinstance(content, str) and content.strip():
                snippet = content.strip()
                if len(snippet) > 300:
                    snippet = snippet[:297] + "..."
                context_lines.append(f"{role}: {snippet}")
        if context_lines:
            parts.append(
                'Conversation context:\n"""\n' + "\n".join(context_lines) + '\n"""'
            )

    meta_bits: list[str] = []
    if include_datetime:
        meta_bits.append(f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if tool_ids:
        meta_bits.append(f"Available tools: {', '.join(str(t) for t in tool_ids)}")
    if meta_bits:
        parts.append("\n".join(meta_bits))

    parts.append(f'Prompt to enhance:\n"""{user_message}"""')
    return "\n\n".join(parts)


def _resolve_model(
    valves_model_id: Optional[str], __model__: Optional[dict], body: dict
) -> str:
    if valves_model_id:
        return valves_model_id
    if __model__:
        base = __model__.get("base_model_id", "")
        if base:
            return base
        info = __model__.get("info")
        if isinstance(info, dict) and info.get("id"):
            return info["id"]
        if __model__.get("id"):
            return __model__["id"]
    return body.get("model", "") or ""


def _set_last_user_message_text(messages: list[dict], new_text: str) -> None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "user":
            continue
        content = messages[i].get("content", "")
        if isinstance(content, str):
            messages[i]["content"] = new_text
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = new_text
                    return
            # No text part — prepend one so image/other parts are preserved.
            content.insert(0, {"type": "text", "text": new_text})
        else:
            messages[i]["content"] = new_text
        return


def _matches_custom_skip(text: str, patterns_str: str) -> bool:
    if not patterns_str.strip():
        return False
    for line in patterns_str.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if re.search(line, text, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _extract_content(response: Any) -> Optional[str]:
    """Defensively pull assistant content from a chat-completion response.

    Returns None for streaming responses, error payloads, or any unexpected
    shape so callers fall back to the original prompt instead of raising.
    """
    if not isinstance(response, dict):
        return None
    try:
        choices = response.get("choices")
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return None


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for filter ordering in Open WebUI.",
        )
        enabled: bool = Field(
            default=True,
            description="Master on/off switch.",
        )
        model_id: Optional[str] = Field(
            default=None,
            description="Model for enhancement. Leave empty to use the chat model.",
        )
        show_status: bool = Field(
            default=False,
            description="Show status indicators during enhancement.",
        )
        show_enhanced_prompt: bool = Field(
            default=False,
            description=(
                "Display a comparison of the original and enhanced prompt in chat. "
                "Do not use with custom pipes."
            ),
        )

        # --- Skip controls ---
        min_prompt_length: int = Field(
            default=12,
            description="Skip prompts shorter than this (characters).",
        )
        max_prompt_length: int = Field(
            default=8000,
            description="Skip prompts longer than this (characters).",
        )
        skip_followups: bool = Field(
            default=False,
            description="Skip enhancement entirely for follow-up messages in a conversation.",
        )
        skip_code_only: bool = Field(
            default=True,
            description="Skip enhancement for prompts that are predominantly code blocks.",
        )
        custom_skip_patterns: str = Field(
            default="",
            description=(
                "Additional regex patterns (one per line) to skip enhancement. "
                "Lines starting with # are ignored."
            ),
        )

        # --- Enhancement tuning ---
        enhancement_style: str = Field(
            default="standard",
            description=(
                "Enhancement depth: 'concise' (minimal changes), "
                "'standard' (balanced), 'detailed' (thorough expansion)."
            ),
        )
        temperature: float = Field(
            default=0.7,
            ge=0.0,
            le=2.0,
            description="LLM temperature for the enhancement call. Lower = more consistent, higher = more creative.",
        )
        intent_threshold: float = Field(
            default=0.55,
            ge=0.0,
            le=1.0,
            description="Minimum confidence to apply intent-specific hints.",
        )
        enable_intent_detection: bool = Field(
            default=True,
            description="Use regex intent detection for domain-specific hints.",
        )
        extra_intent_patterns: str = Field(
            default="",
            description=(
                "JSON object of custom intents merged over the built-ins. "
                'Format: {"name": {"priority": 70, "patterns": ["regex", ...], '
                '"hint": "guidance text"}}.'
            ),
        )
        include_tool_context: bool = Field(
            default=True,
            description="Pass available tool IDs to the enhancer for context.",
        )
        include_datetime: bool = Field(
            default=True,
            description="Include current date/time in enhancer context.",
        )
        max_enhanced_length: int = Field(
            default=4000,
            description=(
                "Maximum character length for the enhanced prompt. "
                "If exceeded, the original prompt is used instead."
            ),
        )
        enable_cache: bool = Field(
            default=True,
            description="Cache enhanced prompts to avoid duplicate LLM calls for identical inputs.",
        )
        cache_ttl_seconds: int = Field(
            default=3600,
            ge=0,
            description="Seconds before a cached enhancement expires. 0 = never expire.",
        )
        cache_maxsize: int = Field(
            default=128,
            ge=1,
            description="Maximum number of cached enhancements (LRU eviction).",
        )
        retry_on_failure: bool = Field(
            default=True,
            description="Retry once on transient LLM failure before falling back to original.",
        )

        # --- Smarter quality ---
        adaptive_style: bool = Field(
            default=False,
            description=(
                "Auto-pick enhancement style (concise/standard/detailed) from "
                "the prompt's vagueness, structure, and explicit constraints. "
                "User overrides always win."
            ),
        )
        self_critique: bool = Field(
            default=False,
            description=(
                "After enhancing, run a second LLM pass to critique and "
                "revise the result. Doubles enhancement latency; off by default."
            ),
        )
        self_critique_temperature: float = Field(
            default=0.3,
            ge=0.0,
            le=2.0,
            description="Temperature for the self-critique/revise pass.",
        )
        dynamic_examples: bool = Field(
            default=True,
            description=(
                "Inject 1–2 curated (original→enhanced) exemplars matching the "
                "detected intent into the system prompt."
            ),
        )

        # --- Context awareness ---
        multimodal_aware: bool = Field(
            default=True,
            description=(
                "When the user attaches images, instruct the enhancer to make "
                "the enhanced prompt explicitly leverage vision."
            ),
        )
        tool_directive: bool = Field(
            default=True,
            description=(
                "When tools are available, instruct the enhancer to make the "
                "enhanced prompt directively call them in the right order."
            ),
        )
        auto_format: bool = Field(
            default=True,
            description=(
                "Detect output-format cues in the original prompt (JSON, YAML, "
                "CSV, markdown table, bullets, code-only) and require them in "
                "the enhanced prompt."
            ),
        )

        # --- Prompt customization ---
        custom_system_prompt: str = Field(
            default="",
            description="Fully replace the default enhancement system prompt.",
        )
        additional_instructions: str = Field(
            default="",
            description=(
                "Extra instructions appended to the system prompt. "
                "Use this to steer enhancement style without replacing the whole prompt. "
                "Example: 'Always ask the AI to show its reasoning step by step.'"
            ),
        )

        debug: bool = Field(
            default=False,
            description="Verbose debug logging.",
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Enable prompt enhancement for your messages.",
        )
        enhancement_style: Optional[str] = Field(
            default=None,
            description=(
                "Override the enhancement style for your messages: "
                "'concise', 'standard', or 'detailed'. Leave empty to use admin default."
            ),
        )
        show_enhanced_prompt: Optional[bool] = Field(
            default=None,
            description="Override whether to show the enhanced prompt comparison. Leave empty for admin default.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def _call_llm(
        self,
        request: Optional[Request],
        payload: dict,
        user,
    ) -> Optional[str]:
        try:
            response = await generate_chat_completion(
                request, payload, user=user, bypass_filter=True
            )
        except Exception:
            return None
        return _extract_content(response)

    async def _call_llm_with_retry(
        self,
        request: Optional[Request],
        payload: dict,
        user,
    ) -> Optional[str]:
        result = await self._call_llm(request, payload, user)
        if result is not None:
            return result
        if not self.valves.retry_on_failure:
            return None
        await asyncio.sleep(1.0)
        return await self._call_llm(request, payload, user)

    async def _critique_and_revise(
        self,
        original: str,
        candidate: str,
        request: Optional[Request],
        user,
        model: str,
    ) -> str:
        """One-shot critique+revise pass. Returns the candidate on any failure."""
        system = (
            "You are revising an enhanced prompt. Compare the candidate "
            "against the user's original. If the candidate is faithful, "
            "specific, and high-quality, return it unchanged. Otherwise "
            "return a single revised prompt that fixes the issues — keeping "
            "the user's intent, voice, and language.\n\n"
            "Return ONLY the final prompt text. No headers, no commentary."
        )
        user_msg = (
            f'Original prompt:\n"""{original}"""\n\n'
            f'Candidate enhancement:\n"""{candidate}"""'
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "temperature": float(self.valves.self_critique_temperature),
        }
        try:
            raw = await self._call_llm(request, payload, user)
        except Exception:
            return candidate
        if not raw:
            return candidate
        cleaned = _clean_llm_output(raw).strip()
        if not cleaned:
            return candidate
        return cleaned

    def _resolve_user_overrides(self, user_valves) -> tuple[str, bool, bool]:
        """Returns (style, show_embed, user_overrode_style)."""
        style = self.valves.enhancement_style
        show_embed = self.valves.show_enhanced_prompt
        user_overrode_style = False
        if isinstance(user_valves, self.UserValves):
            if user_valves.enhancement_style in ("concise", "standard", "detailed"):
                style = user_valves.enhancement_style
                user_overrode_style = True
            if user_valves.show_enhanced_prompt is not None:
                show_embed = user_valves.show_enhanced_prompt
        if style not in ("concise", "standard", "detailed"):
            style = "standard"
        return style, show_embed, user_overrode_style

    def _should_skip(
        self, user_message: str, messages: list[dict], followup: bool
    ) -> Optional[str]:
        if _is_trivial(user_message):
            return "trivial"
        if len(user_message) < self.valves.min_prompt_length:
            return "too short"
        if len(user_message) > self.valves.max_prompt_length:
            return "too long"
        if self.valves.skip_code_only and _is_code_only(user_message):
            return "code-only"
        if _matches_custom_skip(user_message, self.valves.custom_skip_patterns):
            return "custom skip pattern"
        if followup and self.valves.skip_followups:
            return "follow-up"
        return None

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __task__=None,
        __request__: Optional[Request] = None,
    ) -> dict:
        if not self.valves.enabled:
            return body

        user_valves = __user__.get("valves", None) if __user__ else None
        if isinstance(user_valves, self.UserValves) and not user_valves.enabled:
            return body

        if __task__ and __task__ != TASKS.DEFAULT:
            return body

        messages = body.get("messages", [])
        if not messages or messages[-1].get("role") != "user":
            return body

        user_message = get_last_user_message(messages)
        if not user_message:
            return body

        followup = _is_followup(messages, user_message)

        skip_reason = self._should_skip(user_message, messages, followup)
        if skip_reason:
            if self.valves.debug:
                logger.info("Skipped: %s", skip_reason)
            return body

        well_structured = _is_well_structured(user_message)
        style, show_embed, user_overrode_style = self._resolve_user_overrides(
            user_valves
        )

        # Adaptive style: only when the valve is on AND the user did not pick
        # their own style. Skipped for follow-ups and well-structured prompts
        # since those already use specialized system prompts.
        if (
            self.valves.adaptive_style
            and not user_overrode_style
            and not followup
            and not well_structured
        ):
            style = _pick_style(_score_prompt(user_message))

        # --- Intent detection (built-ins + admin custom intents) ---
        intents_catalog = dict(COMPILED_INTENTS)
        intents_catalog.update(_parse_extra_intents(self.valves.extra_intent_patterns))
        active_intents: list[str] = []
        if self.valves.enable_intent_detection:
            active_intents = _detect_intents(
                user_message, self.valves.intent_threshold, intents_catalog
            )

        model_to_use = _resolve_model(self.valves.model_id, __model__, body)
        if not model_to_use:
            if self.valves.debug:
                logger.info("Skipped: no model could be resolved")
            return body

        # --- Context awareness facts ---
        context_facts: dict[str, Any] = {}
        if self.valves.multimodal_aware:
            image_count = _count_image_parts(messages)
            if image_count > 0:
                context_facts["images"] = image_count
        if self.valves.tool_directive:
            tool_ids = body.get("tool_ids")
            if isinstance(tool_ids, list) and tool_ids:
                context_facts["tools"] = sorted(str(t) for t in tool_ids)
        if self.valves.auto_format:
            fmt = _detect_output_format(user_message)
            if fmt:
                context_facts["format"] = fmt

        ctx = EnhancementContext(
            style=style,
            intents=active_intents,
            is_followup=followup,
            is_well_structured=well_structured,
            custom_system_prompt=self.valves.custom_system_prompt,
            additional_instructions=self.valves.additional_instructions,
            model=model_to_use,
            temperature=self.valves.temperature,
            user_id=str(__user__.get("id", "")) if __user__ else "",
            self_critique=self.valves.self_critique,
            dynamic_examples=self.valves.dynamic_examples,
            context_facts=context_facts,
        )
        signature = ctx.signature()

        # Keep the shared cache configured to the current admin valves.
        _prompt_cache.configure(
            maxsize=self.valves.cache_maxsize,
            ttl_seconds=float(self.valves.cache_ttl_seconds),
        )

        # --- Cache check ---
        if self.valves.enable_cache:
            cached = _prompt_cache.get(signature, user_message)
            if cached is not None:
                if self.valves.debug:
                    logger.info("Cache hit for prompt (len=%d)", len(user_message))
                _set_last_user_message_text(messages, cached)
                body["messages"] = messages
                if self.valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Prompt enhanced (cached).",
                                "done": True,
                            },
                        }
                    )
                return body

        system_prompt = _build_system_prompt(ctx, intents_catalog)

        if self.valves.debug:
            logger.info(
                "Enhancing | intents=%s | followup=%s | structured=%s | style=%s | len=%d",
                active_intents,
                followup,
                well_structured,
                style,
                len(user_message),
            )

        if self.valves.show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Enhancing prompt...", "done": False},
                }
            )

        user = await Users.get_user_by_id(__user__["id"]) if __user__ else None

        tool_ids = body.get("tool_ids") if self.valves.include_tool_context else None
        user_prompt = _build_user_prompt(
            user_message=user_message,
            messages=messages,
            tool_ids=tool_ids,
            include_datetime=self.valves.include_datetime,
        )

        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "temperature": self.valves.temperature,
        }

        t_start = time.perf_counter()

        async def _produce() -> Optional[str]:
            raw = await self._call_llm_with_retry(__request__, payload, user)
            if raw is None:
                return None
            cleaned = _clean_llm_output(raw)
            if not cleaned.strip():
                return None
            if self.valves.self_critique:
                cleaned = await self._critique_and_revise(
                    original=user_message,
                    candidate=cleaned,
                    request=__request__,
                    user=user,
                    model=model_to_use,
                )
                if not cleaned.strip():
                    return None
            if len(cleaned) > self.valves.max_enhanced_length:
                if self.valves.debug:
                    logger.warning(
                        "Enhanced prompt too long (%d > %d) — keeping original",
                        len(cleaned),
                        self.valves.max_enhanced_length,
                    )
                return None
            if self.valves.enable_cache:
                _prompt_cache.put(signature, user_message, cleaned)
            return cleaned

        coalesce_key = hashlib.sha256(
            f"{signature}\x00{user_message}".encode("utf-8")
        ).hexdigest()

        try:
            if self.valves.enable_cache:
                enhanced = await _coalesce(coalesce_key, _produce)
            else:
                enhanced = await _produce()

            if not enhanced:
                if self.valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Enhancement skipped — using original prompt.",
                                "done": True,
                            },
                        }
                    )
                return body

            _set_last_user_message_text(messages, enhanced)
            body["messages"] = messages

            elapsed_ms = int((time.perf_counter() - t_start) * 1000)

            if self.valves.debug:
                logger.info(
                    "Enhanced (%d chars, %dms): %s",
                    len(enhanced),
                    elapsed_ms,
                    enhanced[:200],
                )

            if self.valves.show_status:
                intent_tag = f" [{', '.join(active_intents)}]" if active_intents else ""
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Prompt enhanced ({elapsed_ms}ms){intent_tag}.",
                            "done": True,
                        },
                    }
                )

            if show_embed:
                original_escaped = _escape_html(user_message)
                enhanced_escaped = _escape_html(enhanced)
                intent_label = (
                    f' — {", ".join(active_intents)}' if active_intents else ""
                )
                mode_label = (
                    " (follow-up)"
                    if followup
                    else " (light refinement)" if well_structured else ""
                )
                style_label = f" [{style}]" if style != "standard" else ""

                embed_html = (
                    '<div style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;'
                    "border:1px solid rgba(0,0,0,0.1);border-radius:12px;overflow:hidden;"
                    'margin:8px 0;font-size:13px;line-height:1.5;color:#333;">'
                    '<div style="background:rgba(0,0,0,0.03);padding:8px 16px;'
                    "border-bottom:1px solid rgba(0,0,0,0.08);font-size:11px;"
                    f'font-weight:600;color:#666;text-transform:uppercase;letter-spacing:0.5px;">'
                    f"Prompt Enhanced{intent_label}{mode_label}{style_label}</div>"
                    '<div style="padding:10px 16px;border-bottom:1px solid rgba(0,0,0,0.06);">'
                    '<div style="font-size:10px;font-weight:600;color:#999;'
                    'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Original</div>'
                    f'<div style="color:#888;font-style:italic;">{original_escaped}</div></div>'
                    '<div style="padding:10px 16px;">'
                    '<div style="font-size:10px;font-weight:600;color:#999;'
                    'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Enhanced</div>'
                    f'<div style="color:#333;">{enhanced_escaped}</div></div>'
                    "</div>"
                )

                await __event_emitter__(
                    {"type": "embeds", "data": {"embeds": [embed_html]}}
                )

        except Exception as e:
            logger.exception("Enhancement failed: %s", e)
            if self.valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Enhancement error — using original prompt.",
                            "done": True,
                        },
                    }
                )

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
        __request__: Optional[Request] = None,
    ) -> dict:
        return body
