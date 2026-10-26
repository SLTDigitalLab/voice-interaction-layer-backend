"""The three places a reply can come from, behind one small interface.

Each tier wraps an existing agent class (GeminiAgent, LiveKitInferenceAgent,
LlamaOllamaAgent) instead of copying it, and imports it lazily: agent.py will
import this package, and the cloud SDKs are heavy and optional.

A tier's stream() either yields text or raises TierUnavailable (the provider
could not do it) or SafetyBlocked (the provider declined). Whatever the
wrapped agent raises is translated here, so the chain only ever sees those two.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import config

from .errors import MISSING_KEY, SafetyBlocked, TierUnavailable, unavailable_from_exception

GEMINI = "gemini"
LIVEKIT = "livekit"
GPT = "gpt"
LOCAL = "local"

LOCAL_MODE = "local single agent (Llama via Ollama)"

_PROBE_PROMPT = "Reply with one word."


@dataclass
class Turn:
    """Everything one reply needs, independent of which tier answers."""

    text: str
    system_prompt: Optional[str] = None
    user_name: Optional[str] = None
    language: Optional[str] = None
    tool_context: Optional[str] = None
    # True when the text carries a customer's account data: only tiers that run on
    # this machine may see it (see config.CUSTOMER_DATA_TO_CLOUD_LLM).
    private: bool = False


def _cloud_prompts(turn: Turn) -> tuple[str, str]:
    """(system prompt, user message) for a cloud tier. A caller that built its
    own system prompt wins; otherwise the same composer prompt the Orchestrator
    uses, so a reply does not change character depending on the tier."""
    import agent
    from knowledge.prompts import rules_for

    system_prompt = turn.system_prompt
    if system_prompt is None:
        system_prompt = (
            f"{agent._personalize(agent._COMPOSER_ROLE_PROMPT, turn.user_name, turn.language)}"
            f"\n\n{rules_for(turn.tool_context)}"
        )
    message = f"User said: {turn.text}"
    if turn.tool_context:
        message += f"\n\nKnowledge base excerpts (from the tool/retrieval stage):\n{turn.tool_context}"
    return system_prompt, message


def _guarded(pieces: Iterator[str]) -> Iterator[str]:
    """Passes text through and turns any failure into the two outcomes the
    chain understands. GeneratorExit is a BaseException and is not caught, so
    closing the stream early still works."""
    try:
        yield from pieces
    except (SafetyBlocked, TierUnavailable):
        raise
    except Exception as exc:  # noqa: BLE001
        raise unavailable_from_exception(exc) from exc


class Tier(ABC):
    name: str = ""
    # False only for a model that runs on this machine.
    cloud: bool = True

    @abstractmethod
    def label(self) -> str:
        """The mode string shown in /health and the log."""

    @abstractmethod
    def configured(self) -> bool:
        """True when credentials exist. No network call."""

    @abstractmethod
    def _open_stream(self, turn: Turn) -> Iterator[str]:
        ...

    def stream(self, turn: Turn) -> Iterator[str]:
        if not self.configured():
            raise TierUnavailable(MISSING_KEY)
        return _guarded(self._open_stream(turn))

    def probe(self) -> None:
        """One tiny real call; raises TierUnavailable on failure."""
        for _ in self.stream(Turn(text="Hi", system_prompt=_PROBE_PROMPT)):
            break


AgentFactory = Callable[[str], object]


class GeminiTier(Tier):
    name = GEMINI

    def __init__(self, agent_factory: Optional[AgentFactory] = None):
        self._agent_factory = agent_factory

    def label(self) -> str:
        return "cloud (gemini)"

    def configured(self) -> bool:
        return bool(config.GEMINI_API_KEY)

    def _open_stream(self, turn: Turn) -> Iterator[str]:
        system_prompt, message = _cloud_prompts(turn)
        if self._agent_factory is not None:
            agent_obj = self._agent_factory(system_prompt)
        else:
            from agent import GeminiAgent

            agent_obj = GeminiAgent(system_prompt=system_prompt)
        return agent_obj.respond_stream(message)


class LiveKitTier(Tier):
    name = LIVEKIT

    def __init__(self, agent_factory: Optional[AgentFactory] = None):
        self._agent_factory = agent_factory

    def label(self) -> str:
        return f"livekit inference ({config.LIVEKIT_LLM_MODEL})"

    def configured(self) -> bool:
        return config.livekit_configured()

    def _open_stream(self, turn: Turn) -> Iterator[str]:
        system_prompt, message = _cloud_prompts(turn)
        if self._agent_factory is not None:
            agent_obj = self._agent_factory(system_prompt)
        else:
            from livekit_inference import LiveKitInferenceAgent

            agent_obj = LiveKitInferenceAgent(system_prompt=system_prompt)
        return agent_obj.respond_stream(message)


class LocalTier(Tier):
    """Ollama on this machine. The last resort: it needs no key, and its agent
    reports its own problems as text instead of raising, so a failure here is
    spoken to the user rather than falling further."""

    name = LOCAL
    cloud = False

    def __init__(self, agent_factory: Optional[Callable[[Turn], object]] = None):
        self._agent_factory = agent_factory

    def label(self) -> str:
        return LOCAL_MODE

    def configured(self) -> bool:
        return True

    def _open_stream(self, turn: Turn) -> Iterator[str]:
        if self._agent_factory is not None:
            agent_obj = self._agent_factory(turn)
        else:
            from agent import LlamaOllamaAgent

            # The chain already holds this turn's retrieved context, so the
            # agent's own retrieval hook just hands it back.
            agent_obj = LlamaOllamaAgent(tool_callback=lambda _text: turn.tool_context)
        return agent_obj.respond_stream(turn.text, turn.user_name, turn.language)

    def probe(self) -> None:
        start = time.perf_counter()
        try:
            import agent

            response = agent.requests.get(f"{config.LOCAL_LLM_HOST.rstrip('/')}/api/tags", timeout=3)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise unavailable_from_exception(exc) from exc
        finally:
            print(f"LLM chain probe (local): {time.perf_counter() - start:.2f}s")


class GptTier(Tier):
    name = GPT

    def __init__(self, agent_factory: Optional[AgentFactory] = None):
        self._agent_factory = agent_factory

    def label(self) -> str:
        return "cloud (gpt)"

    def configured(self) -> bool:
        return bool(config.OPENAI_API_KEY)

    def _open_stream(self, turn: Turn) -> Iterator[str]:
        system_prompt, message = _cloud_prompts(turn)
        if self._agent_factory is not None:
            agent_obj = self._agent_factory(system_prompt)
        else:
            from agent import GPTAgent

            agent_obj = GPTAgent(system_prompt=system_prompt)
        return agent_obj.respond_stream(message)


def default_tiers() -> dict[str, Tier]:
    tiers: dict[str, Tier] = {GEMINI: GeminiTier(), LIVEKIT: LiveKitTier(), LOCAL: LocalTier()}
    if GPT in getattr(config, "LLM_CHAIN", ()):
        tiers[GPT] = GptTier()
    return tiers
