"""Pluggable, multi provider chat agent layer for the voice interaction backend.

Required pipeline: User Voice -> STT -> (Chat Agent -> Tool/API/KB -> Response) -> TTS.

This module keeps the prototype's core contract (a ChatAgent exposes one
method, respond(text_input) -> str) and adds:

    EchoAgent
        Unchanged from the prototype. Trivial agent, no dependencies, used
        to sanity check the pipeline with zero model calls.

    LlamaOllamaAgent
        Unchanged from the prototype. Talks to a locally running Ollama
        server. This is the local fallback every other path degrades to.

    GeminiAgent, GptAgent
        Real cloud LLM adapters (Google Gemini, OpenAI GPT). Each exposes
        the same respond() contract as every other agent; nothing else in
        the codebase needs to know which one is active.

    Orchestrator
        The new piece. When a cloud provider is configured and reachable,
        it runs a small multi agent collaboration: a tool/retrieval agent
        adds context (the same tool_callback hook the prototype stubbed
        out), and a composer agent turns it into the final spoken reply,
        streamed as it is generated. An optional reasoning agent can draft
        an analysis first (CLOUD_REASONING_STAGE, off by default since it
        delays the reply). All roles are served by the same configured
        cloud provider with different prompts, since standing up genuinely
        separate model backends per role is out of scope for this pass (see
        ARCHITECTURE.md). If the cloud provider is not configured, or a
        call to it fails (unreachable, unauthorized, over quota), the
        Orchestrator falls back to the local LlamaOllamaAgent handling the
        full response alone, with no multi agent attempt in that mode, and
        logs which mode served the reply so the frontend and a live demo
        can show it plainly.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Callable, Iterator, Optional

import requests

import config
from knowledge.prompts import rules_for, with_context


def _say(key: str, language: Optional[str], default: str) -> str:
    """Visitor facing text from backend/i18n in the request language; the English
    `default` if the lookup fails. Developer detail never goes through here."""
    try:
        import i18n  # type: ignore

        text = i18n.t(key, language)
        if isinstance(text, str) and text and text != key:
            return text
    except Exception:  # noqa: BLE001
        pass
    return default


def _local_unavailable(language: Optional[str], detail: str) -> str:
    """The reply when the local model cannot answer. The detailed English text goes
    to the log only; the visitor gets the translated generic sentence."""
    print(f"Local model: {detail}")
    return _say("llm.local_unavailable", language, "Sorry, I can't reach my language service right now. Please try again in a moment.")


def _lost_connection(language: Optional[str]) -> str:
    return " " + _say("llm.lost_connection", language, "Sorry, I lost my connection there. Could you ask that again?")


class ChatAgent(ABC):
    @abstractmethod
    def respond(self, text_input: str) -> str:
        """Takes user text, returns the agent's text response."""
        ...


class EchoAgent(ChatAgent):
    """Trivial agent that echoes the input back, clearly labeled. Proves
    the pluggability of the ChatAgent interface with zero dependencies."""

    def respond(self, text_input: str) -> str:
        start = time.perf_counter()
        response = f"Echo: {text_input}"
        elapsed = time.perf_counter() - start
        print(f"Agent latency: {elapsed:.2f}s")
        return response


# Ollama's /api/generate endpoint accepts an optional "system" field that
# overrides whatever system message the model's Modelfile defines, sent
# alongside "prompt" rather than mixed into it, so persona and behavior
# instructions never compete with the actual user text for the model's
# attention. Without this, llama3.2 answering a bare user string has no
# framing at all: it tends to over-explain, answer more than was asked, and
# add disclaimers, all of which reads badly once spoken aloud by tts.py
# rather than read as text on a screen.
_VOICE_ASSISTANT_SYSTEM_PROMPT = (
    "You are a concise voice assistant. Answer only what the user actually "
    "asked, nothing more and nothing less. Keep replies short and natural, "
    "since they will be spoken aloud rather than read as text. Do not add "
    "disclaimers, caveats, or commentary about being an AI. The user's "
    "message is a live speech to text transcript, so it may be missing "
    "punctuation or have imperfect grammar or phrasing; never comment on, "
    "correct, or apologize for that, just answer what was actually meant. "
    "Usually one to three sentences is enough."
)


# The language each reply must be written in, keyed by the language code
# picked on the frontend's first screen. The script is spelled out because
# a model asked for "Sinhala" alone sometimes answers in romanized
# Sinhala, which the Sinhala voice then reads badly. "guest" is how the
# user is addressed when no name is known, in that language, matching
# the spoken introduction (see _GREETINGS in server.py).
_REPLY_LANGUAGES = {
    "en": {"instruction": "Always reply in English.", "guest": "Guest"},
    "si": {
        "instruction": (
            "Always reply in Sinhala (සිංහල), written in Sinhala script, even if the "
            "user's message contains English words. Keep product names like SLT Mobitel as they are."
        ),
        "guest": "අමුත්තා",
    },
    "ta": {
        "instruction": (
            "Always reply in Tamil (தமிழ்), written in Tamil script, even if the "
            "user's message contains English words. Keep product names like SLT Mobitel as they are."
        ),
        "guest": "விருந்தினர்",
    },
}


def _with_knowledge(system_prompt: str, knowledge_context: Optional[str]) -> str:
    """Adds the SLT Mobitel grounding rules and, when retrieval found
    something, the knowledge base context (with its sources) to a system
    prompt. Without the rules the local model makes up plausible package
    names and prices (seen before this existed: "SLT 50GB, SLT 100GB" for a
    data package question). The wording lives in knowledge/prompts.py, so
    the evaluation in docs/RAG_EVAL.md measures exactly this prompt. On the
    local path this is plain context injection into one model, not multi
    agent collaboration (see ARCHITECTURE.md)."""
    return with_context(system_prompt, knowledge_context)


def _personalize(system_prompt: str, user_name: Optional[str], language: Optional[str] = None) -> str:
    """Adds who the assistant is talking to and which language to answer
    in: replies address the user by the name they gave (on the user
    screen, or by saying "my name is ..." during the conversation, see
    server.py), or as Guest when no name is known, and are written in the
    language picked on the first screen. Put in the system prompt rather
    than the user message for the same reason as the persona itself: it is
    framing, not part of what the user asked. No language (None) keeps the
    old behavior of answering in whatever language the user used."""
    reply_language = _REPLY_LANGUAGES.get(language)
    guest_word = reply_language["guest"] if reply_language else "Guest"
    if user_name:
        prompt = (
            f"{system_prompt} The user's name is {user_name}. Address them by name "
            "naturally, for example when greeting or thanking them, but not in every reply."
        )
    else:
        prompt = (
            f"{system_prompt} The user has not shared their name. If you address them "
            f"directly, call them {guest_word}."
        )
    if reply_language:
        prompt += " " + reply_language["instruction"]
    return prompt


# How long Ollama keeps the model loaded in memory after a request. Its
# default is 5 minutes, after which the next question pays several seconds
# to load the 2GB model from disk again; a demo with pauses between turns
# hit that regularly.
_OLLAMA_KEEP_ALIVE = "30m"

# Bounds on every local generation. A voice reply is one to three
# sentences, so a runaway answer (seen with the small model in Sinhala,
# where a single reply once kept Ollama busy until it failed) is cut at
# num_predict tokens; a low temperature keeps prices, codes and emails
# copied exactly instead of paraphrased (Ollama's default is 0.8); and a
# 2048 token context is plenty for the notes plus the question while
# costing less to prepare than the default 4096. warm_up() sends the same
# options, since a different num_ctx makes Ollama reload the model.
_OLLAMA_OPTIONS = {"temperature": 0.3, "num_predict": 220, "num_ctx": 2048}


class LlamaOllamaAgent(ChatAgent):
    """Chat agent backed by a local Ollama server. This is the local
    fallback every cloud path degrades to when a cloud provider is not
    configured or not reachable."""

    def __init__(
        self,
        model: str = None,
        host: str = None,
        tool_callback: Optional[Callable[[str], Optional[str]]] = None,
    ):
        self.model = model or config.LOCAL_LLM_MODEL
        self.host = (host or config.LOCAL_LLM_HOST).rstrip("/")
        self.tool_callback = tool_callback
        self.timeout_seconds = 60

    def warm_up(self) -> None:
        """Asks Ollama to load the model now (a generate request with no
        prompt only loads it), so the first real question does not wait
        for it. server.py calls this once at startup. Failure is only
        logged: the real request will report the problem properly."""
        start = time.perf_counter()
        try:
            requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "keep_alive": _OLLAMA_KEEP_ALIVE, "options": _OLLAMA_OPTIONS},
                timeout=self.timeout_seconds,
            ).raise_for_status()
            print(f"Agent warm up (local Llama): {time.perf_counter() - start:.2f}s")
        except requests.exceptions.RequestException as exc:
            print(f"Agent warm up (local Llama) failed: {exc}")

    def _system_prompt(self, text_input: str, user_name: Optional[str], language: Optional[str]) -> str:
        """The persona, who the user is, the reply language, the grounding
        rules and any retrieved knowledge, all in Ollama's system field.
        Retrieved context used to be prepended to the user prompt; in the
        system prompt it is framing the model answers from, rather than
        text it might read as part of the question."""
        tool_context = None
        if self.tool_callback is not None:
            try:
                tool_context = self.tool_callback(text_input)
            except Exception:
                tool_context = None
        return _with_knowledge(_personalize(_VOICE_ASSISTANT_SYSTEM_PROMPT, user_name, language), tool_context)

    def respond(self, text_input: str, user_name: Optional[str] = None, language: Optional[str] = None) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": text_input,
            "system": self._system_prompt(text_input, user_name, language),
            "stream": False,
            "keep_alive": _OLLAMA_KEEP_ALIVE,
            "options": _OLLAMA_OPTIONS,
        }

        start = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            elapsed = time.perf_counter() - start
            print(f"Agent latency (local Llama): {elapsed:.2f}s")
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(f"Agent latency (local Llama): {time.perf_counter() - start:.2f}s")
            return _local_unavailable(language, "could not reach the local Ollama server. Is it running?")
        except requests.exceptions.Timeout:
            print(f"Agent latency (local Llama): {time.perf_counter() - start:.2f}s")
            return _local_unavailable(language, "the local Ollama server took too long to respond.")
        except requests.exceptions.HTTPError:
            print(f"Agent latency (local Llama): {time.perf_counter() - start:.2f}s")
            if response.status_code == 404:
                return _local_unavailable(language, f"model '{self.model}' was not found on the Ollama server.")
            return _local_unavailable(language, f"Ollama server returned an HTTP error ({response.status_code}).")
        except requests.exceptions.RequestException as exc:
            print(f"Agent latency (local Llama): {time.perf_counter() - start:.2f}s")
            return _local_unavailable(language, f"could not get a response from Ollama ({exc}).")

        try:
            data = response.json()
        except ValueError:
            return _local_unavailable(language, "received an unreadable response from the Ollama server.")

        return data.get("response", "").strip()

    def respond_stream(
        self, text_input: str, user_name: Optional[str] = None, language: Optional[str] = None
    ) -> Iterator[str]:
        """Like respond(), but yields response text chunks as Ollama
        generates them (the API's stream mode, newline delimited JSON
        objects each carrying one token or short run of tokens) instead of
        blocking until the full reply is ready. Used by the WebSocket
        voice pipeline so the frontend can show the reply appearing as it
        is actually generated, not just after the fact. Never raises for a
        request level failure; on error it yields a single chunk with the
        same wording respond() would have returned, so callers that just
        concatenate yielded chunks behave the same either way."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": text_input,
            "system": self._system_prompt(text_input, user_name, language),
            "stream": True,
            "keep_alive": _OLLAMA_KEEP_ALIVE,
            "options": _OLLAMA_OPTIONS,
        }

        start = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds, stream=True)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(f"Agent latency (local Llama, streaming): {time.perf_counter() - start:.2f}s")
            yield _local_unavailable(language, "could not reach the local Ollama server. Is it running?")
            return
        except requests.exceptions.Timeout:
            print(f"Agent latency (local Llama, streaming): {time.perf_counter() - start:.2f}s")
            yield _local_unavailable(language, "the local Ollama server took too long to respond.")
            return
        except requests.exceptions.HTTPError:
            print(f"Agent latency (local Llama, streaming): {time.perf_counter() - start:.2f}s")
            if response.status_code == 404:
                yield _local_unavailable(language, f"model '{self.model}' was not found on the Ollama server.")
            else:
                yield _local_unavailable(language, f"Ollama server returned an HTTP error ({response.status_code}).")
            return
        except requests.exceptions.RequestException as exc:
            print(f"Agent latency (local Llama, streaming): {time.perf_counter() - start:.2f}s")
            yield _local_unavailable(language, f"could not get a response from Ollama ({exc}).")
            return

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break
        except requests.exceptions.RequestException as exc:
            yield "\n" + _local_unavailable(language, f"connection to Ollama dropped mid reply ({exc}).")
        finally:
            print(f"Agent latency (local Llama, streaming): {time.perf_counter() - start:.2f}s")


class CloudUnavailableError(Exception):
    """Raised by a cloud agent adapter when the call could not be
    completed (missing key, network failure, auth failure, quota). The
    Orchestrator catches this specifically to trigger local fallback,
    distinguishing "cloud is down, fall back" from a genuine bug."""


_gemini_client = None

# What each Gemini API error code actually means for this app, in plain
# words, so the log and /health say why the cloud path is not being used
# rather than only printing Google's raw error. Codes as observed from the
# live API with the google-genai SDK (errors.APIError.code).
_GEMINI_ERROR_REASONS = {
    400: "the request was rejected (often an invalid API key format or a bad parameter)",
    401: "the API key is invalid or was revoked; create a new one in AI Studio",
    402: "the project's prepaid credits are used up; top up billing in AI Studio",
    403: "the key is not allowed to use this API or model (API disabled, or key restrictions)",
    404: "the model named in GEMINI_MODEL does not exist or is not available to this key; pick one from the list models call",
    429: "the rate limit or quota was hit; wait, or raise the quota in AI Studio",
    500: "Google's service had an internal error; usually temporary",
    503: "Google's service is overloaded or unavailable; usually temporary",
}


# Gemini finish reasons that mean the model declined the question, not that
# the call failed.
_GEMINI_SAFETY_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}


def _raise_if_gemini_blocked(chunk) -> None:
    """A blocked prompt or reply is a valid answer, not an outage: raising
    SafetyBlocked lets the LLM chain end the turn instead of asking another
    model the same question. Imported here because llm_chain imports this
    module."""
    blocked = bool(getattr(getattr(chunk, "prompt_feedback", None), "block_reason", None))
    for candidate in getattr(chunk, "candidates", None) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None and str(getattr(reason, "name", reason)).split(".")[-1].upper() in _GEMINI_SAFETY_FINISH_REASONS:
            blocked = True
    if blocked:
        from llm_chain.errors import SafetyBlocked

        raise SafetyBlocked()


def _explain_gemini_error(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    reason = _GEMINI_ERROR_REASONS.get(code)
    if reason is None:
        # No HTTP code at all usually means the request never reached
        # Google: no internet, DNS, proxy or firewall.
        reason = "network problem reaching Google" if code is None else "unexpected error"
    return f"{code or 'no response'}: {reason} (Google said: {str(exc)[:200]})"


class GeminiAgent(ChatAgent):
    """Adapter for Google's Gemini API, using the current google-genai SDK
    (from google import genai; genai.Client). It replaced the deprecated
    google-generativeai package, which Google no longer updates. Only
    imported when actually used, so the rest of the codebase does not
    require it to be installed or configured to run locally."""

    def __init__(self, model: str = None, system_prompt: Optional[str] = None):
        self.model_name = model or config.GEMINI_MODEL
        self.system_prompt = system_prompt

    @staticmethod
    def _client():
        # One client for the whole process: it holds the HTTP connection,
        # so creating one per request would redo the connection setup.
        global _gemini_client
        if not config.GEMINI_API_KEY:
            raise CloudUnavailableError("GEMINI_API_KEY is not set.")
        if _gemini_client is None:
            from google import genai
            from google.genai import types

            _gemini_client = genai.Client(
                api_key=config.GEMINI_API_KEY,
                http_options=types.HttpOptions(timeout=10000),
            )
        return _gemini_client

    def respond(self, text_input: str) -> str:
        return "".join(self.respond_stream(text_input)).strip()

    def respond_stream(self, text_input: str) -> Iterator[str]:
        """Yields the reply as Gemini generates it
        (client.models.generate_content_stream, the streaming form of
        generate_content), so the voice pipeline can start speaking the
        first sentence before the last one exists. Any failure, before or
        during the stream, is raised as CloudUnavailableError."""
        start = time.perf_counter()
        try:
            from google.genai import types

            gen_kwargs = {
                "system_instruction": self.system_prompt,
                # No tools are passed, so the SDK's automatic function
                # calling has nothing to do; disabling it also stops
                # the warning it logs on every streamed call.
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            }
            if not config.CLOUD_REASONING_STAGE and "flash-lite" not in (self.model_name or ""):
                gen_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

            stream = self._client().models.generate_content_stream(
                model=self.model_name,
                contents=text_input,
                config=types.GenerateContentConfig(**gen_kwargs),
            )
            for chunk in stream:
                _raise_if_gemini_blocked(chunk)
                # chunk.text is None for a chunk with no text part (e.g. only
                # safety metadata).
                if chunk.text:
                    yield chunk.text
            print(f"Agent latency (Gemini, streaming): {time.perf_counter() - start:.2f}s")
        except CloudUnavailableError:
            raise
        except Exception as exc:
            if type(exc).__name__ == "SafetyBlocked":
                raise
            print(f"Agent latency (Gemini): {time.perf_counter() - start:.2f}s")
            raise CloudUnavailableError(f"Gemini call failed: {_explain_gemini_error(exc)}") from exc


class GptAgent(ChatAgent):
    """Adapter for OpenAI's GPT API."""

    def __init__(self, model: str = None, system_prompt: Optional[str] = None):
        self.model_name = model or config.OPENAI_MODEL
        self.system_prompt = system_prompt

    def respond(self, text_input: str) -> str:
        return "".join(self.respond_stream(text_input)).strip()

    def respond_stream(self, text_input: str) -> Iterator[str]:
        """Streaming counterpart of respond(), same contract as
        GeminiAgent.respond_stream()."""
        if not config.OPENAI_API_KEY:
            raise CloudUnavailableError("OPENAI_API_KEY is not set.")

        start = time.perf_counter()
        try:
            from openai import OpenAI

            client = OpenAI(api_key=config.OPENAI_API_KEY)
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": text_input})

            stream = client.chat.completions.create(model=self.model_name, messages=messages, stream=True)
            for event in stream:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
            print(f"Agent latency (GPT, streaming): {time.perf_counter() - start:.2f}s")
        except CloudUnavailableError:
            raise
        except Exception as exc:
            print(f"Agent latency (GPT): {time.perf_counter() - start:.2f}s")
            raise CloudUnavailableError(f"GPT call failed: {exc}") from exc


def _build_cloud_agent(role_prompt: Optional[str] = None) -> ChatAgent:
    """Instantiates the configured cloud provider's agent class for one
    collaboration role. Kept as a small factory so the Orchestrator does
    not need an if/elif per provider at every call site."""
    if config.CLOUD_LLM_PROVIDER == "livekit":
        # Imported here, like the other providers' SDKs: it needs livekit-api,
        # and it imports this module, so a top level import would be circular.
        from livekit_inference import LiveKitInferenceAgent

        return LiveKitInferenceAgent(system_prompt=role_prompt)
    if config.CLOUD_LLM_PROVIDER == "gpt":
        return GptAgent(system_prompt=role_prompt)
    return GeminiAgent(system_prompt=role_prompt)


_REASONING_ROLE_PROMPT = (
    "You are the reasoning stage of a voice assistant. In two or three short "
    "sentences, identify what the user is actually asking for and any context "
    "needed to answer it well. Do not answer the user directly; only produce "
    "the analysis."
)

_COMPOSER_ROLE_PROMPT = (
    "You are the response stage of a voice assistant. You will be given the "
    "user's message, optionally an internal analysis of what they need, and "
    "optional extra context. Reply directly to the user in natural spoken "
    "language, answering only what they actually asked, nothing more and "
    "nothing less. Keep it concise, usually one to three sentences, with no "
    "mention of any analysis or internal process, and no disclaimers or meta "
    "commentary about being an AI. The user's message is a live speech to text "
    "transcript and may have imperfect grammar or missing punctuation; never "
    "comment on or correct that, just answer what was meant."
)


_CLOUD_RETRY_AFTER_SECONDS = 300


class Orchestrator(ChatAgent):
    """Cloud first, local fallback chat agent.

    When the configured cloud provider is reachable, role prompts against
    that same provider stand in for separate agents:

        tool/retrieval agent -> looks up extra context via tool_callback
                              (the SLT Mobitel knowledge base retriever,
                              knowledge/retriever.py, plugged into the same
                              hook the prototype stubbed out) and folds
                              anything it returns into the conversation.
        composer agent    -> turns the user's message plus any tool
                              context into the final spoken reply, streamed
                              as it is generated.
        reasoning agent   -> optional (CLOUD_REASONING_STAGE=true in .env):
                              drafts a short analysis of what the user
                              needs before the composer runs. Off by
                              default because it is a whole extra model
                              round trip that has to finish before the
                              composer can even start, which roughly
                              doubled the time to the first spoken word,
                              for little gain on short spoken questions.

    If the cloud provider is not configured, or it fails before producing
    any text, the whole attempt is abandoned and LlamaOllamaAgent alone
    produces the full response, exactly as the prototype did, with no
    partial cloud/local mixing. This mirrors a real production fallback:
    half a cloud pipeline plus a local patch would be harder to reason
    about and debug than a clean either/or.
    """

    def __init__(
        self,
        tool_callback: Optional[Callable[[str], Optional[str]]] = None,
        external_agent_provider: Optional[Callable[[], Optional[ChatAgent]]] = None,
    ):
        self.tool_callback = tool_callback
        # Returns the external chat agent an admin has made active (see
        # admin/service.py), or None for this built in assistant. Called
        # once per turn, so the provider must be cheap (it is cached).
        self.external_agent_provider = external_agent_provider
        self.local_agent = LlamaOllamaAgent(tool_callback=tool_callback)
        self.last_mode: str = "unknown"
        self.last_expired = False  # the last reply was a "your session ended" notice
        self.last_timings: dict = {}  # tool_ms and llm_ms of the last hub grounded turn
        # Circuit breaker: after a cloud failure, go straight to the local
        # agent until this time instead of paying for another failing
        # cloud call (and its timeout) on every turn. A dead key or used up
        # quota rarely fixes itself within a conversation.
        self._cloud_retry_at = 0.0
        # Why the cloud path last failed, in plain words (see
        # _explain_gemini_error), or None once it has worked. Shown by
        # GET /health so the reason is visible without reading logs.
        self.last_cloud_error: Optional[str] = None
        self._chain = None

    def warm_up(self) -> None:
        """Called once at server startup so the first question does not pay
        for loading models: loads the local Llama model into Ollama and,
        if a cloud provider is configured, sends it one tiny request. That
        both loads its SDK (importing the Gemini SDK alone took several
        seconds here) and finds out right away if the key is not
        usable (bad model name, no credit left), in which case the circuit
        breaker is tripped now instead of on the user's first question."""
        self.local_agent.warm_up()
        if config.LLM_CHAIN:
            # Every tier is probed once now, so a dead key is found at startup
            # and its breaker is already open for the first question.
            self.chain.probe_all()
        elif config.cloud_llm_configured():
            start = time.perf_counter()
            try:
                _build_cloud_agent(role_prompt="Reply with one word.").respond("Hi")
                self.last_cloud_error = None
                print(f"Agent warm up (cloud): {time.perf_counter() - start:.2f}s")
            except CloudUnavailableError as exc:
                self.last_cloud_error = str(exc)
                self._cloud_retry_at = time.monotonic() + _CLOUD_RETRY_AFTER_SECONDS
                print(f"Agent warm up: cloud unavailable ({exc}); using the local agent for now.")

    def respond(
        self,
        text_input: str,
        user_name: Optional[str] = None,
        language: Optional[str] = None,
        access=None,
        status_cb=None,
    ) -> str:
        return "".join(self.respond_stream(text_input, user_name, language, access, status_cb)).strip()

    def respond_stream(
        self,
        text_input: str,
        user_name: Optional[str] = None,
        language: Optional[str] = None,
        access=None,
        status_cb=None,
    ) -> Iterator[str]:
        """The reply in pieces (see _respond_stream). With ANSWER_SOURCE=mcp_hub a
        sentence that carries a telephone number, web address or e-mail address
        that is neither in the hub's text for this turn nor an official SLT
        number is removed (tool_agent.scrub), because a model fills such gaps
        from memory; the official dial options are added instead."""
        holder: dict = {}
        stream = self._respond_stream(text_input, user_name, language, access, status_cb, holder)
        if config.ANSWER_SOURCE != "mcp_hub":
            yield from stream
            return
        yield from self._scrubbed(stream, holder, language)

    @staticmethod
    def _scrubbed(stream: Iterator[str], holder: dict, language: Optional[str]) -> Iterator[str]:
        first = next(stream, None)
        if first is None:
            return
        plan = holder.get("plan")
        if plan is None or plan.kind == "canned":
            yield first
            yield from stream
            return
        try:
            from tool_agent import scrub
            from tool_agent.messages import _dial, dial_text

            official = list(_dial().values())
            dial_line = dial_text("tool_agent.not_found", language)
            context = (plan.context or "") + "\n" + (plan.clause or "")
        except Exception as exc:  # noqa: BLE001  the guard must never cost the reply
            print(f"Orchestrator: output guard unavailable ({type(exc).__name__}).")
            yield first
            yield from stream
            return

        def pieces() -> Iterator[str]:
            yield first
            yield from stream

        yield from scrub.filter_stream(pieces(), context, official, dial_line)

    def _respond_stream(
        self,
        text_input: str,
        user_name: Optional[str],
        language: Optional[str],
        access,
        status_cb,
        holder: dict,
    ) -> Iterator[str]:
        """Yields the reply in pieces as it is generated, on both the cloud
        and the local path, so the voice pipeline can start speaking the
        first sentence while the rest is still being written.

        The cloud fallback decision is made on the first piece: if the
        cloud call fails before producing any text, the local agent
        answers instead. If it fails after some text has already been
        spoken, switching to a different model mid sentence would sound
        worse than admitting the problem, so a short apology is added to
        what was said instead.

        An active external agent (registered in the admin portal) answers
        first, and the same rule applies to it: a failure before any text
        falls back to the normal cloud/local path below.

        access is an access.AccessContext (None means guest). When access
        control is on, a guest asking for customer service gets the canned
        translated dial message and no model is called at all.

        With ANSWER_SOURCE=mcp_hub (the default) SLT facts come only from the
        hub: tool_agent.prepare fetches them first, a missing answer or a
        hub outage is a fixed translated reply with no model, and the
        old retrieval and an external agent are not used for SLT questions.
        status_cb(key) is told which short status line to show while the hub
        is being asked."""
        self.last_expired = False
        self.last_timings = {}
        blocked = self._blocked_reply(text_input, language, access)
        if blocked is not None:
            self.last_mode = "access policy (canned reply)"
            self.last_expired = bool(getattr(access, "session_expired", False))
            print(f"Orchestrator mode: {self.last_mode}")
            yield blocked
            return

        plan = self._hub_plan(text_input, user_name, language, access, status_cb)
        holder["plan"] = plan
        if plan is not None and plan.kind == "canned":
            self.last_mode = "SLT hub (fixed reply)"
            self.last_expired = bool(plan.expired)
            self.last_timings = {}
            print(f"Orchestrator mode: {self.last_mode}")
            yield plan.canned
            return

        external_agent = None if plan is not None else self._active_external_agent()
        if external_agent is not None:
            external_stream = external_agent.respond_stream(text_input, user_name, language)
            try:
                first_chunk = next(external_stream, "")
                if not first_chunk:
                    raise ExternalAgentError("the agent returned an empty reply")
            except ExternalAgentError as exc:
                print(
                    f"Orchestrator: external agent {external_agent.name} failed ({exc}), "
                    "falling back to the built in assistant."
                )
            else:
                self.last_mode = f"external agent ({external_agent.name})"
                print(f"Orchestrator mode: {self.last_mode}")
                yield first_chunk
                try:
                    yield from external_stream
                except ExternalAgentError as exc:
                    print(f"Orchestrator: external agent stream dropped mid reply ({exc}).")
                    yield _lost_connection(language)
                return

        if config.LLM_CHAIN:
            yield from self._stream_chain(text_input, user_name, language, access, plan)
            return

        # No chain: the older cloud then local path. Hub data travels in the
        # message, and a private turn (customer data) never goes to the cloud.
        suffix = ""
        private = False
        if plan is not None:
            text_input = self._with_plan(text_input, plan)
            suffix = plan.mode_suffix or ""
            private = bool(plan.private)

        cloud_paused_for = self._cloud_retry_at - time.monotonic()
        if private:
            print("Orchestrator: customer data in this turn, the cloud path is skipped.")
        elif config.cloud_llm_configured() and cloud_paused_for > 0:
            print(f"Orchestrator: cloud failed recently, skipping it for another {cloud_paused_for:.0f}s.")
        elif config.cloud_llm_configured():
            cloud_stream = self._stream_cloud(text_input, user_name, language)
            try:
                first_chunk = next(cloud_stream, "")
            except CloudUnavailableError as exc:
                self.last_cloud_error = str(exc)
                self._cloud_retry_at = time.monotonic() + _CLOUD_RETRY_AFTER_SECONDS
                print(f"Orchestrator: cloud path unavailable ({exc}), falling back to local agent alone.")
            else:
                self.last_cloud_error = None
                self.last_mode = f"cloud multi agent ({config.CLOUD_LLM_PROVIDER})" + suffix
                print(f"Orchestrator mode: {self.last_mode}")
                yield first_chunk
                try:
                    yield from cloud_stream
                except CloudUnavailableError as exc:
                    print(f"Orchestrator: cloud stream dropped mid reply ({exc}).")
                    yield _lost_connection(language)
                return
        else:
            print("Orchestrator: no cloud LLM provider configured, using local agent alone.")

        self.last_mode = "local single agent (Llama via Ollama)" + suffix
        print(f"Orchestrator mode: {self.last_mode}")
        yield from self.local_agent.respond_stream(text_input, user_name, language)

    @staticmethod
    def _blocked_reply(text_input: str, language: Optional[str], access) -> Optional[str]:
        """The canned dial message when the policy guard refuses this request,
        else None. A broken access module must not take the assistant down,
        but it also must not let a guest through: an error here blocks."""
        if not config.ACCESS_CONTROL_ENABLED:
            return None
        from access import GUEST, blocked_reply, evaluate

        try:
            decision = evaluate(text_input, language, access or GUEST)
        except Exception as exc:  # noqa: BLE001
            print(f"Orchestrator: access guard failed ({type(exc).__name__}), treating the request as blocked.")
            return blocked_reply(language)
        if decision.allowed:
            return None
        if getattr(access, "session_expired", False):
            # Their hub login ended: say so and offer to verify again.
            from access import expired_reply

            return expired_reply(language)
        return blocked_reply(language)

    @staticmethod
    def _hub_plan(text_input: str, user_name: Optional[str], language: Optional[str], access, status_cb=None):
        """The tool agent's plan (backend/tool_agent) when ANSWER_SOURCE is
        mcp_hub, else None. Blocking: respond_stream runs on a worker thread.
        It never raises; if it somehow does, the answer is the fixed
        "temporarily unavailable" text, never a model guess about SLT."""
        if config.ANSWER_SOURCE != "mcp_hub":
            return None
        try:
            import tool_agent

            return tool_agent.prepare(text_input, language, access, user_name, on_status=status_cb)
        except Exception as exc:  # noqa: BLE001
            print(f"Orchestrator: tool agent failed ({type(exc).__name__}), answering with the fixed unavailable text.")
            from tool_agent.messages import dial_text
            from tool_agent.plan import CANNED, TurnPlan

            return TurnPlan(CANNED, canned=dial_text("tool_agent.hub_unavailable", language))

    @staticmethod
    def _with_plan(text: str, plan) -> str:
        """The user's words plus the grounding clause and the hub data. The
        local model builds its own system prompt, so instructions and data
        travel in the message."""
        if plan.clause:
            text += f"\n\n{plan.clause}"
        if plan.kind == "grounded" and plan.context:
            text += f"\n\nInformation from the SLT systems (the only source for SLT facts):\n{plan.context}"
        return text

    @property
    def chain(self):
        """The LLM fallback chain (backend/llm_chain), built on first use
        because that package imports this module."""
        if self._chain is None:
            from llm_chain import LlmChain

            self._chain = LlmChain()
        return self._chain

    def _stream_chain(
        self, text_input: str, user_name: Optional[str], language: Optional[str], access, plan=None
    ) -> Iterator[str]:
        """One turn through Gemini, LiveKit Inference and local Llama in the
        order of LLM_CHAIN, each with its own breaker (llm_chain/chain.py)."""
        from access import GUEST, level_clause
        from llm_chain import Turn

        # Account data for a verified customer (hub tools, read only). A missing
        # sign in or an unreachable hub is answered with fixed translated text and
        # no model, so a model can never make up account details.
        outcome = None if plan is not None else self._customer_outcome(text_input, language, access)
        if outcome is not None and outcome.kind in ("reauth", "unavailable"):
            self.last_mode = "account data unavailable (fixed reply)"
            print(f"Orchestrator mode: {self.last_mode}")
            yield outcome.canned
            return

        tool_context = None
        if self.tool_callback is not None:
            try:
                tool_context = self.tool_callback(text_input)
            except Exception:
                tool_context = None
        system_prompt = (
            f"{_personalize(_COMPOSER_ROLE_PROMPT, user_name, language)}\n\n{rules_for(tool_context)}"
        )
        if config.ACCESS_CONTROL_ENABLED:
            system_prompt += f"\n\n{level_clause(access or GUEST, language)}"
        text = f"User said: {text_input}"
        if tool_context:
            text += f"\n\nKnowledge base excerpts (from the tool/retrieval stage):\n{tool_context}"
        private = False
        suffix = ""
        if plan is not None:
            text = self._with_plan(text, plan)
            private = bool(plan.private)
            suffix = plan.mode_suffix or ""
        if outcome is not None and outcome.kind == "data":
            from customer_service import system_clause

            # The instructions travel with the data in the message, because the local
            # model builds its own system prompt and ignores the turn's.
            text += (
                f"\n\n{system_clause(language, outcome.changes_requested)}"
                f"\n\nSLT account data for this customer, from the SLT system:\n{outcome.context}"
            )
            # Customer data stays on this machine unless the owner opened it up.
            private = not config.CUSTOMER_DATA_TO_CLOUD_LLM
            suffix = " + SLT account data"
        turn = Turn(
            text=text,
            system_prompt=system_prompt,
            user_name=user_name,
            language=language,
            tool_context=tool_context,
            private=private,
        )
        chain = self.chain
        first = True
        llm_started = time.perf_counter()
        for piece in chain.stream(turn):
            if first:
                first = False
                self.last_mode = chain.last_mode + suffix
                print(f"Orchestrator mode: {self.last_mode}")
            yield piece
        self.last_mode = chain.last_mode + suffix
        if plan is not None and plan.kind == "grounded":
            self.last_timings = {
                "tool_ms": int((plan.timings or {}).get("tool_ms", 0)),
                "llm_ms": int((time.perf_counter() - llm_started) * 1000),
            }
            print(f"Orchestrator timing: tool_ms={self.last_timings['tool_ms']} llm_ms={self.last_timings['llm_ms']}")

    @staticmethod
    def _customer_outcome(text_input: str, language: Optional[str], access):
        """Account data for a verified customer, or None. Blocking (a hub call
        with its own time budget); respond_stream already runs on a worker thread.
        A failure here must not cost the user their answer."""
        if not config.CUSTOMER_TOOLS_ENABLED or access is None or not getattr(access, "is_verified", False):
            return None
        if not config.mcp_hub_configured():
            # No hub, no account data: the assistant behaves exactly as it did before.
            return None
        try:
            from customer_service import fetch

            outcome = fetch(text_input, language, access)
        except Exception as exc:  # noqa: BLE001
            print(f"Orchestrator: customer data step failed ({type(exc).__name__}), answering without it.")
            return None
        return outcome if outcome.kind != "none" else None

    def _active_external_agent(self) -> Optional[ChatAgent]:
        """The external agent to try first, or None. A failing lookup (for
        example the database being unavailable) must not cost the user
        their answer, so it counts as none active."""
        if self.external_agent_provider is None:
            return None
        try:
            return self.external_agent_provider()
        except Exception as exc:
            print(f"Orchestrator: external agent lookup failed ({exc}), using the built in assistant.")
            return None

    def _stream_cloud(self, text_input: str, user_name: Optional[str], language: Optional[str]) -> Iterator[str]:
        analysis = None
        if config.CLOUD_REASONING_STAGE:
            analysis = _build_cloud_agent(role_prompt=_REASONING_ROLE_PROMPT).respond(text_input)

        tool_context = None
        if self.tool_callback is not None:
            try:
                tool_context = self.tool_callback(text_input)
            except Exception:
                tool_context = None

        composed_prompt = f"User said: {text_input}"
        if analysis:
            composed_prompt += f"\n\nInternal analysis: {analysis}"
        if tool_context:
            composed_prompt += f"\n\nKnowledge base excerpts (from the tool/retrieval stage):\n{tool_context}"

        # The grounding rules go in the composer's role prompt; the context
        # itself travels with the message, as the retrieval stage's output.
        composer_agent = _build_cloud_agent(
            role_prompt=f"{_personalize(_COMPOSER_ROLE_PROMPT, user_name, language)}\n\n{rules_for(tool_context)}"
        )
        yield from composer_agent.respond_stream(composed_prompt)


# ==========================================================================
# BEGIN external chat agents (R2, admin portal)
#
# One self contained section: ExternalAgentError and ExternalHttpAgent, an
# adapter that forwards the user's text to another chat agent over HTTP
# (configured in the admin portal, see backend/admin/service.py) and
# returns its text reply, which the voice pipeline then speaks exactly as
# it speaks any other agent's reply. Nothing above this section was
# changed for it.
# ==========================================================================

import re as _re
import threading as _threading
from typing import Iterable

import urllib3 as _urllib3

from url_safety import UnsafeUrlError, check_outbound_url, safe_session


class ExternalAgentError(Exception):
    """Raised by ExternalHttpAgent for every failure (refused URL, network
    error, timeout, non 2xx status, oversized or unreadable reply, missing
    response path), never a raw requests/urllib3 exception, so a caller
    can fall back to the built in assistant on this one exception type.
    The message is a plain sentence fit to show an admin."""


# Only these three names are placeholders. A single regex pass replaces
# them, so text the user said that happens to contain "{language}" is not
# substituted a second time.
_BODY_PLACEHOLDER = _re.compile(r"\{(text|user_name|language)\}")
_SSE_DONE = "[DONE]"
# A streamed reply may legitimately take longer overall than a single
# JSON reply; the per read timeout still applies between events.
_STREAM_TOTAL_TIMEOUT_FACTOR = 4
_MAX_CONNECT_TIMEOUT = 5.0
_READ_SIZE = 8192


def render_body_template(
    template: str, text: str, user_name: Optional[str] = None, language: Optional[str] = None
):
    """Fills {text}, {user_name} and {language} in a JSON body template
    and returns the parsed JSON value. Each value is JSON string escaped
    before it is inserted, so quotes, backslashes and newlines in what the
    user said cannot end the string early or add fields. Placeholders must
    therefore sit inside quotes ("{text}"). Missing values become empty
    strings."""
    values = {"text": text or "", "user_name": user_name or "", "language": language or ""}

    def substitute(match) -> str:
        return json.dumps(values[match.group(1)], ensure_ascii=False)[1:-1]

    rendered = _BODY_PLACEHOLDER.sub(substitute, template or "")
    try:
        return json.loads(rendered)
    except ValueError:
        raise ExternalAgentError(
            'The body template is not valid JSON once filled in. Placeholders must sit '
            'inside quotes, for example {"message": "{text}"}.'
        ) from None


def extract_response_path(data, path: Optional[str]):
    """Follows a dotted path such as "reply" or "choices.0.message.content"
    into parsed JSON; a numeric segment indexes a list (negative counts
    from the end). An empty path returns data itself."""
    if not path:
        return data
    current = data
    for part in path.split("."):
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                raise ExternalAgentError(
                    f"Response path '{path}': found a list where '{part}' was expected; use a number."
                ) from None
            if not -len(current) <= index < len(current):
                raise ExternalAgentError(f"Response path '{path}': list index {index} is out of range.")
            current = current[index]
        elif isinstance(current, dict):
            if part not in current:
                raise ExternalAgentError(f"The agent's reply has no '{part}' (response path '{path}').")
            current = current[part]
        else:
            raise ExternalAgentError(f"Response path '{path}': cannot look up '{part}' inside a plain value.")
    return current


def _reply_value_as_text(value, path: Optional[str]) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if value is None:
        raise ExternalAgentError(f"The agent's reply is empty at response path '{path}'.")
    raise ExternalAgentError(
        f"Response path '{path or '(whole reply)'}' points at a JSON object or list, not at text."
    )


def iter_sse_events(byte_chunks: Iterable[bytes]) -> Iterator[tuple[str, str]]:
    """Parses a Server Sent Events byte stream into (event name, data)
    pairs, following the WHATWG rules that matter here: "data:" lines of
    one event are joined with newlines, a blank line ends the event, lines
    starting with ":" are comments, and one space after the colon is
    dropped. A final event without its closing blank line is still
    delivered."""
    buffer = b""
    data_lines: list[str] = []
    event_name = "message"

    def dispatch():
        nonlocal data_lines, event_name
        if data_lines:
            yield event_name, "\n".join(data_lines)
        data_lines = []
        event_name = "message"

    for chunk in byte_chunks:
        buffer += chunk
        while b"\n" in buffer:
            raw_line, buffer = buffer.split(b"\n", 1)
            line = raw_line.rstrip(b"\r").decode("utf-8", errors="replace")
            if not line:
                yield from dispatch()
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event_name = value or "message"
    if buffer.strip():
        line = buffer.rstrip(b"\r").decode("utf-8", errors="replace")
        field, _, value = line.partition(":")
        if field == "data":
            data_lines.append(value[1:] if value.startswith(" ") else value)
    yield from dispatch()


def sse_text_pieces(events: Iterable[tuple[str, str]], response_path: Optional[str]) -> Iterator[str]:
    """Turns SSE events into reply text pieces. Each event's data is
    either plain text (used as is) or JSON, where response_path picks the
    piece (for example "choices.0.delta.content"); a JSON event without
    that path (a role only or keep alive event) is skipped. "[DONE]" ends
    the stream; an "error" event raises ExternalAgentError."""
    for event_name, data in events:
        if data.strip() == _SSE_DONE:
            return
        if event_name == "error":
            raise ExternalAgentError("The agent reported an error in its event stream.")
        try:
            parsed = json.loads(data)
        except ValueError:
            parsed = None
            piece = data
        else:
            if isinstance(parsed, (dict, list)):
                try:
                    piece = _reply_value_as_text(extract_response_path(parsed, response_path), response_path)
                except ExternalAgentError:
                    continue
            elif isinstance(parsed, str):
                piece = parsed
            else:
                piece = data
        if piece:
            yield piece


class ExternalHttpAgent(ChatAgent):
    """A chat agent reached over HTTP, configured in the admin portal.

    POST (default): the body is body_template (JSON text) with {text},
    {user_name} and {language} filled in, JSON escaped. GET: the same
    template is filled in and must be a flat JSON object; its fields are
    sent as query parameters instead of a body (for example
    {"q": "{text}", "lang": "{language}"} becomes ?q=...&lang=...).

    The reply is read from response_path in the JSON response. With
    streaming=True the agent is asked for Server Sent Events and each
    event's text is yielded as it arrives, so the voice pipeline can start
    speaking before the reply is complete; if the agent answers with plain
    JSON instead, that is parsed as a whole.

    Safety: the URL is checked against url_safety before every request and
    the connection is pinned to the checked address (DNS rebinding);
    redirects are not followed; the reply is capped at
    EXTERNAL_AGENT_MAX_RESPONSE_BYTES while it is read (Content-Length is
    not trusted); connect and read timeouts are strict. Every failure is
    raised as ExternalAgentError.
    """

    def __init__(
        self,
        name: str,
        url: str,
        method: str = "POST",
        auth_header_name: Optional[str] = None,
        auth_header_value: Optional[str] = None,
        body_template: str = '{"message": "{text}"}',
        response_path: str = "reply",
        timeout_seconds: Optional[float] = None,
        streaming: bool = False,
    ):
        self.name = name
        self.url = url
        self.method = (method or "POST").strip().upper()
        if self.method not in ("GET", "POST"):
            raise ExternalAgentError(f"Unsupported HTTP method '{method}'; use GET or POST.")
        self.auth_header_name = (auth_header_name or "").strip() or None
        self.auth_header_value = auth_header_value or None
        self.body_template = body_template or '{"message": "{text}"}'
        self.response_path = (response_path or "").strip()
        self.timeout_seconds = float(timeout_seconds or config.EXTERNAL_AGENT_DEFAULT_TIMEOUT)
        self.streaming = bool(streaming)
        # One HTTP session (kept alive connection) per thread: /chat and
        # the voice pipeline may call the same cached agent concurrently,
        # and requests.Session is not documented as thread safe.
        self._local = _threading.local()

    def __repr__(self) -> str:
        # Deliberately leaves out the auth header value.
        return f"ExternalHttpAgent(name={self.name!r}, url={self.url!r}, streaming={self.streaming})"

    def respond(self, text_input: str, user_name: Optional[str] = None, language: Optional[str] = None) -> str:
        return "".join(self.respond_stream(text_input, user_name, language)).strip()

    def respond_stream(
        self, text_input: str, user_name: Optional[str] = None, language: Optional[str] = None
    ) -> Iterator[str]:
        label = f"external agent {self.name}" + (", streaming" if self.streaming else "")
        start = time.perf_counter()
        try:
            yield from self._generate(text_input, user_name, language)
        finally:
            print(f"Agent latency ({label}): {time.perf_counter() - start:.2f}s")

    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            session = safe_session()
            self._local.session = session
        return session

    def _request_arguments(self, text_input: str, user_name: Optional[str], language: Optional[str]) -> dict:
        headers = {
            "Accept": "text/event-stream, application/json" if self.streaming else "application/json",
            "User-Agent": "SLT-Mobitel-Voice-Assistant/1.0",
        }
        if self.auth_header_name and self.auth_header_value:
            headers[self.auth_header_name] = self.auth_header_value
        body = render_body_template(self.body_template, text_input, user_name, language)
        if self.method == "GET":
            if not isinstance(body, dict):
                raise ExternalAgentError("For GET, the body template must be a JSON object; its fields become query parameters.")
            params = {
                key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                for key, value in body.items()
            }
            return {"params": params, "headers": headers}
        headers["Content-Type"] = "application/json; charset=utf-8"
        return {"data": json.dumps(body, ensure_ascii=False).encode("utf-8"), "headers": headers}

    def _generate(self, text_input: str, user_name: Optional[str], language: Optional[str]) -> Iterator[str]:
        arguments = self._request_arguments(text_input, user_name, language)
        try:
            check_outbound_url(self.url)
        except UnsafeUrlError as exc:
            raise ExternalAgentError(str(exc)) from None

        total_budget = self.timeout_seconds * (_STREAM_TOTAL_TIMEOUT_FACTOR if self.streaming else 1)
        deadline = time.monotonic() + total_budget
        try:
            response = self._session().request(
                self.method,
                self.url,
                timeout=(min(self.timeout_seconds, _MAX_CONNECT_TIMEOUT), self.timeout_seconds),
                allow_redirects=False,
                stream=True,
                **arguments,
            )
        except UnsafeUrlError as exc:
            raise ExternalAgentError(str(exc)) from None
        except requests.exceptions.Timeout:
            raise ExternalAgentError(f"The agent did not answer within {self.timeout_seconds:g} seconds.") from None
        except requests.exceptions.TooManyRedirects:
            raise ExternalAgentError("The agent answered with a redirect. Redirects are not followed; enter the final URL instead.") from None
        except requests.exceptions.SSLError:
            raise ExternalAgentError("The agent's HTTPS certificate could not be verified.") from None
        except requests.exceptions.ConnectionError:
            raise ExternalAgentError("Could not connect to the agent (connection refused or host unreachable).") from None
        except (requests.exceptions.RequestException, _urllib3.exceptions.HTTPError, OSError):
            raise ExternalAgentError("The request to the agent failed.") from None

        try:
            status = response.status_code
            if 300 <= status < 400:
                raise ExternalAgentError(
                    f"The agent answered with a redirect (HTTP {status}). Redirects are not followed; "
                    "enter the final URL instead."
                )
            if not 200 <= status < 300:
                reason = f" {str(response.reason)[:80]}" if response.reason else ""
                raise ExternalAgentError(f"The agent answered with HTTP {status}{reason}.")

            content_type = response.headers.get("Content-Type", "").lower()
            byte_chunks = self._read_capped(response, deadline, total_budget)
            if self.streaming and "text/event-stream" in content_type:
                produced = False
                for piece in sse_text_pieces(iter_sse_events(byte_chunks), self.response_path):
                    produced = True
                    yield piece
                if not produced:
                    raise ExternalAgentError("The agent's event stream ended without any reply text.")
            else:
                yield self._parse_whole(b"".join(byte_chunks), content_type)
        finally:
            response.close()

    def _read_capped(self, response, deadline: float, total_budget: float) -> Iterator[bytes]:
        """Yields the body as it arrives (read1 returns whatever is
        available, so SSE events are not held back waiting for a full
        buffer), stopping with ExternalAgentError once more than
        EXTERNAL_AGENT_MAX_RESPONSE_BYTES (after decompression) have come
        in or the total time budget is spent."""
        limit = config.EXTERNAL_AGENT_MAX_RESPONSE_BYTES
        total = 0
        while True:
            try:
                chunk = response.raw.read1(_READ_SIZE, decode_content=True)
            except (requests.exceptions.RequestException, _urllib3.exceptions.HTTPError, OSError) as exc:
                if isinstance(exc, (_urllib3.exceptions.ReadTimeoutError, TimeoutError)):
                    raise ExternalAgentError(
                        f"The agent stopped sending for more than {self.timeout_seconds:g} seconds."
                    ) from None
                raise ExternalAgentError("The connection to the agent dropped while reading its reply.") from None
            if not chunk:
                return
            total += len(chunk)
            if total > limit:
                raise ExternalAgentError(f"The agent's reply is larger than the {limit} byte limit.")
            if time.monotonic() > deadline:
                raise ExternalAgentError(f"The agent took longer than {total_budget:g} seconds in total.")
            yield chunk

    def _parse_whole(self, body: bytes, content_type: str) -> str:
        text = body.decode("utf-8", errors="replace")
        if not self.response_path and "json" not in content_type:
            reply = text.strip()
        else:
            try:
                data = json.loads(text)
            except ValueError:
                raise ExternalAgentError("The agent's reply is not valid JSON.") from None
            reply = _reply_value_as_text(extract_response_path(data, self.response_path), self.response_path).strip()
        if not reply:
            raise ExternalAgentError("The agent returned an empty reply.")
        return reply


# ==========================================================================
# END external chat agents
# ==========================================================================


def _run_interactive_test():
    """Standalone test mode: python agent.py

    Exercises both paths so each can be demonstrated independently:
    - the real cloud path, if a provider is configured
    - a forced local fallback, by talking to LlamaOllamaAgent directly
    Type 'quit' or 'exit' to stop.
    """
    orchestrator = Orchestrator()
    print("Orchestrator test mode. Type 'quit' or 'exit' to stop (Ctrl+C also works).")
    print(f"Cloud provider configured: {config.cloud_llm_configured()} ({config.CLOUD_LLM_PROVIDER})")
    print("Type 'force-local' to switch to a direct LlamaOllamaAgent call for the rest of the session.\n")

    force_local = False
    while True:
        try:
            text_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not text_input.strip():
            continue
        if text_input.strip().lower() in ("quit", "exit"):
            print("Exiting.")
            break
        if text_input.strip().lower() == "force-local":
            force_local = True
            print("Now forcing local only responses (no cloud attempt).")
            continue

        if force_local:
            reply = orchestrator.local_agent.respond(text_input)
        else:
            reply = orchestrator.respond(text_input)
        print(f"Agent [{orchestrator.last_mode if not force_local else 'local single agent (forced)'}]: {reply}")


if __name__ == "__main__":
    _run_interactive_test()
