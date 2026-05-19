import asyncio


# ---------------------------------------------------------------------------
# _clean_llm_output
# ---------------------------------------------------------------------------


def test_clean_strips_balanced_thinking(pe):
    out = pe._clean_llm_output("<think>secret reasoning</think>Real prompt here")
    assert out == "Real prompt here"


def test_clean_strips_case_insensitive_and_pipe_form(pe):
    out = pe._clean_llm_output("<Thinking>x</Thinking>Hello")
    assert out == "Hello"
    out2 = pe._clean_llm_output("|begin_of_thought|deep|end_of_thought| Final")
    assert out2 == "Final"


def test_clean_strips_dangling_unclosed_thinking(pe):
    # No closing tag — must not leak the chain-of-thought.
    out = pe._clean_llm_output("Good prompt.\n<think>now I will keep rambling")
    assert out == "Good prompt."
    assert "rambling" not in out


def test_clean_strips_artifact_prefixes(pe):
    assert pe._clean_llm_output("Enhanced Prompt: Do the thing") == "Do the thing"
    assert pe._clean_llm_output("Sure! Here is the enhanced prompt: Go") == "Go"


def test_clean_unwraps_surrounding_quotes(pe):
    assert pe._clean_llm_output('"A sufficiently long quoted prompt"') == (
        "A sufficiently long quoted prompt"
    )


def test_clean_keeps_short_quoted_text(pe):
    # Inner too short to confidently unwrap.
    assert pe._clean_llm_output('"hi"') == '"hi"'


# ---------------------------------------------------------------------------
# _detect_intents
# ---------------------------------------------------------------------------


def test_detect_intents_basic(pe):
    intents = pe._detect_intents("I have a Python TypeError, please debug this error")
    assert "debugging" in intents


def test_detect_intents_caps_at_three(pe):
    text = (
        "debug this error, write a python function, analyze the data, "
        "compare options, summarize the result, translate to french"
    )
    assert len(pe._detect_intents(text)) <= 3


def test_detect_intents_threshold_filters(pe):
    # Very high threshold => nothing qualifies.
    assert pe._detect_intents("debug this error", threshold=0.999) == []


def test_detect_intents_with_custom_catalog(pe):
    custom = pe._parse_extra_intents(
        '{"legal": {"priority": 99, "patterns": ["\\\\bcontract\\\\b"], '
        '"hint": "legal hint"}}'
    )
    catalog = dict(pe.COMPILED_INTENTS)
    catalog.update(custom)
    intents = pe._detect_intents(
        "please review this lengthy contract clause carefully", intents=catalog
    )
    assert "legal" in intents


# ---------------------------------------------------------------------------
# _parse_extra_intents
# ---------------------------------------------------------------------------


def test_parse_extra_intents_invalid_json_is_ignored(pe):
    assert pe._parse_extra_intents("{not valid json") == {}


def test_parse_extra_intents_empty(pe):
    assert pe._parse_extra_intents("   ") == {}


def test_parse_extra_intents_memoized(pe):
    raw = '{"x": {"priority": 50, "patterns": ["foo"], "hint": "h"}}'
    first = pe._parse_extra_intents(raw)
    second = pe._parse_extra_intents(raw)
    assert first is second  # cached object identity


# ---------------------------------------------------------------------------
# _is_followup
# ---------------------------------------------------------------------------


def _conv(*texts):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": t} for i, t in enumerate(texts)
    ]


def test_followup_needs_prior_assistant(pe):
    msgs = [{"role": "user", "content": "hi"}, {"role": "user", "content": "now do x"}]
    assert pe._is_followup(msgs, "now do x") is False


def test_followup_regex_match(pe):
    msgs = _conv("write a parser", "here you go", "now add error handling")
    assert pe._is_followup(msgs, "now add error handling") is True


def test_followup_short_with_deictic(pe):
    msgs = _conv("explain X", "explanation", "make it shorter")
    assert pe._is_followup(msgs, "make it shorter") is True


def test_short_new_question_is_not_followup(pe):
    msgs = _conv("explain X", "explanation", "what is quantum entanglement")
    assert pe._is_followup(msgs, "what is quantum entanglement") is False


def test_long_message_is_not_followup(pe):
    long_msg = "word " * 50
    msgs = _conv("a", "b", long_msg)
    assert pe._is_followup(msgs, long_msg) is False


# ---------------------------------------------------------------------------
# _is_well_structured / _is_code_only / _is_trivial
# ---------------------------------------------------------------------------


def test_well_structured_detects_rich_prompt(pe):
    text = (
        "# Role\nYou are an expert.\n\n"
        "- point one\n- point two\n- point three\n\n"
        "```python\nprint('x')\n```"
    )
    assert pe._is_well_structured(text) is True


def test_plain_prompt_not_well_structured(pe):
    assert pe._is_well_structured("just explain dns to me") is False


def test_code_only_detection(pe):
    assert pe._is_code_only("```python\n" + "x = 1\n" * 20 + "```") is True
    assert pe._is_code_only("Explain this:\n```\nx=1\n```\nin detail please ok") is False
    assert pe._is_code_only("no code here") is False


def test_trivial_detection(pe):
    for t in ["hi", "thanks!", "ok", "bye", "yes", "  "]:
        assert pe._is_trivial(t) is True
    assert pe._is_trivial("explain recursion") is False


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------


def test_resolve_model_prefers_valve(pe):
    assert pe._resolve_model("valve-model", {"id": "m"}, {}) == "valve-model"


def test_resolve_model_from_model_dict(pe):
    assert pe._resolve_model(None, {"base_model_id": "b"}, {}) == "b"
    assert pe._resolve_model(None, {"info": {"id": "i"}}, {}) == "i"
    assert pe._resolve_model(None, {"id": "x"}, {}) == "x"


def test_resolve_model_falls_back_to_body_and_empty(pe):
    assert pe._resolve_model(None, None, {"model": "body-m"}) == "body-m"
    assert pe._resolve_model(None, None, {}) == ""


# ---------------------------------------------------------------------------
# _set_last_user_message_text
# ---------------------------------------------------------------------------


def test_set_text_string_content(pe):
    msgs = [{"role": "user", "content": "old"}]
    pe._set_last_user_message_text(msgs, "new")
    assert msgs[0]["content"] == "new"


def test_set_text_list_with_text_part(pe):
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "old"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        }
    ]
    pe._set_last_user_message_text(msgs, "new")
    assert msgs[0]["content"][0]["text"] == "new"
    assert msgs[0]["content"][1]["type"] == "image_url"


def test_set_text_list_without_text_preserves_images(pe):
    msgs = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
    ]
    pe._set_last_user_message_text(msgs, "new")
    assert msgs[0]["content"][0] == {"type": "text", "text": "new"}
    assert any(p.get("type") == "image_url" for p in msgs[0]["content"])


# ---------------------------------------------------------------------------
# _matches_custom_skip / _escape_html / _extract_content
# ---------------------------------------------------------------------------


def test_matches_custom_skip(pe):
    patterns = "# comment line\n^/debug\nfoo.*bar"
    assert pe._matches_custom_skip("/debug now", patterns) is True
    assert pe._matches_custom_skip("foo zzz bar", patterns) is True
    assert pe._matches_custom_skip("nothing", patterns) is False


def test_matches_custom_skip_bad_regex_ignored(pe):
    assert pe._matches_custom_skip("anything", "([unclosed") is False


def test_escape_html(pe):
    assert pe._escape_html('<a href="x">&') == "&lt;a href=&quot;x&quot;&gt;&amp;"


def test_extract_content_variants(pe):
    assert pe._extract_content({"choices": [{"message": {"content": "hi"}}]}) == "hi"
    assert (
        pe._extract_content(
            {"choices": [{"message": {"content": [{"type": "text", "text": "a"}]}}]}
        )
        == "a"
    )
    assert pe._extract_content("not a dict") is None
    assert pe._extract_content({}) is None
    assert pe._extract_content({"choices": []}) is None


# ---------------------------------------------------------------------------
# _PromptCache
# ---------------------------------------------------------------------------


def test_cache_put_get_and_signature_isolation(pe):
    c = pe._PromptCache(maxsize=10)
    c.put("sigA", "prompt", "enhancedA")
    assert c.get("sigA", "prompt") == "enhancedA"
    # Different signature must not collide.
    assert c.get("sigB", "prompt") is None


def test_cache_lru_eviction(pe):
    c = pe._PromptCache(maxsize=2)
    c.put("s", "p1", "e1")
    c.put("s", "p2", "e2")
    c.get("s", "p1")  # p1 now most-recent
    c.put("s", "p3", "e3")  # evicts least-recent (p2)
    assert c.get("s", "p1") == "e1"
    assert c.get("s", "p2") is None
    assert c.get("s", "p3") == "e3"


def test_cache_ttl_expiry(pe, monkeypatch):
    c = pe._PromptCache(maxsize=10, ttl_seconds=100)
    now = [1000.0]
    monkeypatch.setattr(pe.time, "time", lambda: now[0])
    c.put("s", "p", "e")
    assert c.get("s", "p") == "e"
    now[0] += 101
    assert c.get("s", "p") is None


def test_context_signature_changes_with_config(pe):
    a = pe.EnhancementContext(style="standard", model="m")
    b = pe.EnhancementContext(style="detailed", model="m")
    c = pe.EnhancementContext(style="standard", model="m", custom_system_prompt="X")
    assert a.signature() != b.signature()
    assert a.signature() != c.signature()
    assert a.signature() == pe.EnhancementContext(style="standard", model="m").signature()


# ---------------------------------------------------------------------------
# Filter.inlet (async)
# ---------------------------------------------------------------------------


def _body(text):
    return {"messages": [{"role": "user", "content": text}], "model": "test-model"}


async def test_inlet_skips_trivial(pe, emitter):
    f = pe.Filter()
    body = _body("hi")
    out = await f.inlet(body, emitter)
    assert out["messages"][0]["content"] == "hi"


async def test_inlet_skips_short(pe, emitter):
    f = pe.Filter()
    out = await f.inlet(_body("short"), emitter)
    assert out["messages"][0]["content"] == "short"


async def test_inlet_skips_when_no_model(pe, emitter):
    f = pe.Filter()
    body = {"messages": [{"role": "user", "content": "explain recursion in depth"}]}
    out = await f.inlet(body, emitter)
    assert out["messages"][0]["content"] == "explain recursion in depth"


async def test_inlet_successful_enhancement(pe, emitter, monkeypatch):
    f = pe.Filter()
    f.valves.show_status = True

    async def fake_call(self, request, payload, user):
        return "a much more detailed version of the request"

    monkeypatch.setattr(pe.Filter, "_call_llm", fake_call)
    out = await f.inlet(_body("explain how dns resolution works"), emitter)
    assert out["messages"][0]["content"] == (
        "a much more detailed version of the request"
    )
    assert any("enhanced" in (d or "").lower() for d in emitter.descriptions())


async def test_inlet_cache_hit_second_call(pe, emitter, monkeypatch):
    f = pe.Filter()
    calls = {"n": 0}

    async def fake_call(self, request, payload, user):
        calls["n"] += 1
        return "the enhanced detailed prompt text"

    monkeypatch.setattr(pe.Filter, "_call_llm", fake_call)
    await f.inlet(_body("explain how dns resolution works"), emitter)
    await f.inlet(_body("explain how dns resolution works"), emitter)
    assert calls["n"] == 1  # second served from cache


async def test_inlet_retry_then_success(pe, emitter, monkeypatch):
    f = pe.Filter()
    seq = iter([None, "the recovered enhanced prompt"])

    async def fake_low(self, request, payload, user):
        return next(seq)

    async def _noop_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(pe.Filter, "_call_llm", fake_low)
    monkeypatch.setattr(pe.asyncio, "sleep", _noop_sleep)
    out = await f.inlet(_body("explain how dns resolution works"), emitter)
    assert out["messages"][0]["content"] == "the recovered enhanced prompt"


async def test_inlet_oversized_output_falls_back(pe, emitter, monkeypatch):
    f = pe.Filter()
    f.valves.max_enhanced_length = 20

    async def fake_call(self, request, payload, user):
        return "x" * 500

    monkeypatch.setattr(pe.Filter, "_call_llm", fake_call)
    original = "explain how dns resolution works in detail"
    out = await f.inlet(_body(original), emitter)
    assert out["messages"][0]["content"] == original


async def test_inlet_llm_exception_falls_back(pe, emitter, monkeypatch):
    f = pe.Filter()

    async def boom(self, request, payload, user):
        raise RuntimeError("upstream down")

    # _call_llm swallows; here patch the retry layer to ensure graceful path.
    monkeypatch.setattr(pe.Filter, "_call_llm_with_retry", boom)
    original = "explain how dns resolution works in detail"
    out = await f.inlet(_body(original), emitter)
    assert out["messages"][0]["content"] == original


async def test_inlet_request_coalescing(pe, emitter, monkeypatch):
    f = pe.Filter()
    calls = {"n": 0}
    gate = asyncio.Event()

    async def slow_call(self, request, payload, user):
        calls["n"] += 1
        await gate.wait()
        return "the single shared enhanced prompt result"

    monkeypatch.setattr(pe.Filter, "_call_llm", slow_call)

    t1 = asyncio.create_task(f.inlet(_body("explain dns resolution thoroughly"), emitter))
    t2 = asyncio.create_task(
        f.inlet(_body("explain dns resolution thoroughly"), emitter)
    )
    await asyncio.sleep(0.05)
    gate.set()
    r1, r2 = await asyncio.gather(t1, t2)

    assert calls["n"] == 1  # coalesced into a single LLM call
    assert r1["messages"][0]["content"] == "the single shared enhanced prompt result"
    assert r2["messages"][0]["content"] == "the single shared enhanced prompt result"
