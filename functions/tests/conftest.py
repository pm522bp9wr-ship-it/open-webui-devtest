"""Test bootstrap for the standalone prompt_enhancer filter.

The filter lives outside the backend package (Open WebUI loads functions as
single uploaded modules), so it imports `open_webui.*`. We stub those modules
with lightweight fakes so the pure helpers and the Filter can be tested without
the full backend.
"""

import sys
import types
from pathlib import Path

import pytest

FUNCTIONS_DIR = Path(__file__).resolve().parents[1]
if str(FUNCTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(FUNCTIONS_DIR))


def _install_stub_modules() -> None:
    # open_webui package
    pkg = types.ModuleType("open_webui")
    pkg.__path__ = []  # mark as package
    sys.modules["open_webui"] = pkg

    utils_pkg = types.ModuleType("open_webui.utils")
    utils_pkg.__path__ = []
    sys.modules["open_webui.utils"] = utils_pkg

    models_pkg = types.ModuleType("open_webui.models")
    models_pkg.__path__ = []
    sys.modules["open_webui.models"] = models_pkg

    # open_webui.utils.chat.generate_chat_completion
    chat_mod = types.ModuleType("open_webui.utils.chat")

    async def generate_chat_completion(request, form_data, user=None, **kwargs):
        # Default: behaves as if the LLM echoed nothing. Tests monkeypatch
        # prompt_enhancer.generate_chat_completion or Filter._call_llm.
        return {"choices": [{"message": {"content": ""}}]}

    chat_mod.generate_chat_completion = generate_chat_completion
    sys.modules["open_webui.utils.chat"] = chat_mod

    # open_webui.utils.misc.get_last_user_message
    misc_mod = types.ModuleType("open_webui.utils.misc")

    def get_last_user_message(messages):
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    return " ".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                return content if isinstance(content, str) else None
        return None

    misc_mod.get_last_user_message = get_last_user_message
    sys.modules["open_webui.utils.misc"] = misc_mod

    # open_webui.models.users.Users
    users_mod = types.ModuleType("open_webui.models.users")

    class _Users:
        @staticmethod
        async def get_user_by_id(user_id, db=None):
            return {"id": user_id}

    users_mod.Users = _Users
    sys.modules["open_webui.models.users"] = users_mod

    # open_webui.constants.TASKS
    constants_mod = types.ModuleType("open_webui.constants")

    class TASKS:
        DEFAULT = "default"
        TITLE_GENERATION = "title_generation"

    constants_mod.TASKS = TASKS
    sys.modules["open_webui.constants"] = constants_mod

    # fastapi.Request (only used as a type hint / passthrough)
    if "fastapi" not in sys.modules:
        fastapi_mod = types.ModuleType("fastapi")

        class Request:  # noqa: D401 - minimal stub
            pass

        fastapi_mod.Request = Request
        sys.modules["fastapi"] = fastapi_mod


_install_stub_modules()


@pytest.fixture()
def pe():
    """The imported prompt_enhancer module with a clean cache per test."""
    import prompt_enhancer as module

    module._prompt_cache.clear()
    module._prompt_cache.configure(maxsize=128, ttl_seconds=0.0)
    module._inflight.clear()
    module._custom_intent_cache.clear()
    return module


class EventCollector:
    """Fake __event_emitter__ that records emitted events."""

    def __init__(self):
        self.events = []

    async def __call__(self, event):
        self.events.append(event)

    def descriptions(self):
        return [e.get("data", {}).get("description") for e in self.events]


@pytest.fixture()
def emitter():
    return EventCollector()
