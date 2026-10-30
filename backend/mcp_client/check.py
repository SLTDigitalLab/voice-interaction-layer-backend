"""Diagnose the connection to the SLT MCP hub.

    python -m backend.mcp_client.check            (from the repository root)
    cd backend; python -m mcp_client.check        (from backend/)

Prints what was configured (never the key, only whether one is set and how
long it is), whether DNS and TLS work, what each transport candidate
answered (SSE at /sse, streamable HTTP at /mcp and at the base URL, with
status codes and the response header names that matter), whether the key
was accepted, the tools the hub lists, and a one line verdict with what to
ask the hub administrator when it is not working.

Exit code 0 only when the verdict is "working".

Flags:
    --json             the same report as JSON
    --timeout SECONDS  per step timeout (default MCP_HUB_TIMEOUT_SECONDS)
    --call-read-only   also call, at most once each, the read only non
                       customer tools search_knowledgebase_tool,
                       compare_package and get_function_latencies. Nothing
                       else is ever called: the list is fixed in this file.

The real client (mcp_client.client.HubClient) does the tool listing, so the
verdict is about the code path the application itself uses.
"""

import argparse
import asyncio
import json
import socket
import ssl
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:  # `python -m mcp_client.check` from backend/ already has it
    sys.path.insert(0, str(_BACKEND))

try:
    import httpx2  # noqa: E402
except ModuleNotFoundError:
    import httpx as httpx2  # type: ignore[no-redef]  # noqa: E402

import config  # noqa: E402
from mcp_client.client import HubClient  # noqa: E402
from mcp_client.errors import HubError  # noqa: E402

# The only tools --call-read-only may call. A tool outside this set is never
# called, whatever else this module is asked.
READ_ONLY_TOOLS = ("search_knowledgebase_tool", "compare_package", "get_function_latencies")
SAFE_SEARCH_QUERY = "SLT Mobitel broadband packages"
_HEADERS_OF_INTEREST = ("mcp-session-id", "www-authenticate", "content-type", "allow", "server", "location")

VERDICTS = ("working", "unreachable", "wrong_path", "wrong_transport", "auth_rejected", "not_configured", "unknown")


class _Reporter:
    """Keeps the key out of everything that is printed."""

    def __init__(self, key: str):
        self.key = key

    def clean(self, text: str) -> str:
        return text.replace(self.key, "***") if self.key else text


# Network level steps


def split_url(url: str) -> tuple[str, str]:
    """(root, original) where root has no trailing /sse or /mcp."""
    root = url.strip().rstrip("/")
    for suffix in ("/sse", "/mcp"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return root, url


async def check_network(url: str, timeout: float) -> dict:
    """DNS, then a TCP connect, then (https) the TLS handshake."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    secure = parts.scheme == "https"
    port = parts.port or (443 if secure else 80)
    result: dict[str, Any] = {"host": host, "port": port, "dns": None, "tcp": None, "tls": None}
    loop = asyncio.get_running_loop()
    try:
        infos = await asyncio.wait_for(loop.getaddrinfo(host, port, type=socket.SOCK_STREAM), timeout)
        result["dns"] = {"ok": True, "addresses": sorted({i[4][0] for i in infos})[:4]}
    except asyncio.TimeoutError:
        result["dns"] = {"ok": False, "error": "timeout"}
        return result
    except OSError as exc:
        result["dns"] = {"ok": False, "error": f"{type(exc).__name__}"}
        return result
    try:
        context = ssl.create_default_context() if secure else None
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=context, server_hostname=host if secure else None), timeout
        )
        result["tcp"] = {"ok": True}
        if secure:
            ssl_object = writer.get_extra_info("ssl_object")
            result["tls"] = {"ok": True, "version": ssl_object.version() if ssl_object else None}
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), 1.0)
        except Exception:  # noqa: BLE001
            pass
    except asyncio.TimeoutError:
        result["tcp"] = {"ok": False, "error": "timeout"}
    except ssl.SSLError as exc:
        result["tcp"] = {"ok": True}
        result["tls"] = {"ok": False, "error": type(exc).__name__}
    except OSError as exc:
        result["tcp"] = {"ok": False, "error": type(exc).__name__}
    return result


def _header_names(response: httpx2.Response) -> dict:
    return {k: response.headers[k] for k in _HEADERS_OF_INTEREST if k in response.headers and k != "www-authenticate"} | (
        {"www-authenticate": response.headers["www-authenticate"][:80]} if "www-authenticate" in response.headers else {}
    )


async def _probe(client: httpx2.AsyncClient, method: str, url: str, headers: dict, body: Optional[dict], timeout: float) -> dict:
    """One request, status and headers only; the body is read only far enough
    to see the first SSE event name, then the connection is dropped."""
    out: dict[str, Any] = {"method": method, "url": url}
    try:

        async def run() -> None:
            async with client.stream(method, url, headers=headers, json=body) as response:
                out["status"] = response.status_code
                out["content_type"] = response.headers.get("content-type", "").split(";")[0]
                out["headers"] = _header_names(response)
                if response.status_code == 200 and out["content_type"] == "text/event-stream":
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            out["first_event"] = line[6:].strip()
                            break

        await asyncio.wait_for(run(), timeout)
    except asyncio.TimeoutError:
        out["error"] = "timeout"
    except httpx2.TimeoutException:
        out["error"] = "timeout"
    except httpx2.HTTPError as exc:
        out["error"] = type(exc).__name__
    return out


async def run_probes(root: str, header: str, key: str, timeout: float) -> dict:
    """The three transport candidates, plus the SSE path once without the key
    to learn whether the hub enforces authentication."""
    keyed = {header: key} if key else {}
    sse_headers = {**keyed, "Accept": "text/event-stream"}
    post_headers = {**keyed, "Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "hub-check", "version": "1"},
        },
    }
    async with httpx2.AsyncClient(timeout=httpx2.Timeout(timeout), follow_redirects=False) as client:
        sse = await _probe(client, "GET", root + "/sse", sse_headers, None, timeout)
        streamable_mcp = await _probe(client, "POST", root + "/mcp", post_headers, initialize, timeout)
        streamable_base = await _probe(client, "POST", root, post_headers, initialize, timeout)
        keyless = await _probe(client, "GET", root + "/sse", {"Accept": "text/event-stream"}, None, timeout)
    return {"sse": sse, "streamable_mcp": streamable_mcp, "streamable_base": streamable_base, "without_key": keyless}


def auth_result(probes: dict, hub_error: Optional[dict], key_set: bool) -> str:
    """accepted, rejected 401, rejected 403 or none (the hub asks for no key)."""
    keyed = [probes[n].get("status") for n in ("sse", "streamable_mcp", "streamable_base")]
    if hub_error and hub_error.get("reason") == "auth_rejected":
        return f"rejected {hub_error.get('status') or ''}".strip()
    for status in keyed:
        if status in (401, 403):
            return f"rejected {status}"
    keyless = probes["without_key"].get("status")
    answered = [s for s in keyed if s is not None and s < 400]
    if keyless in (401, 403):
        return "accepted" if (answered or hub_error is None) and key_set else "unknown"
    if keyless is not None and keyless < 400 and answered:
        return "none"
    return "unknown"


# Verdict


ASK = {
    "unreachable": (
        "Ask the hub administrator to confirm the hub is running and that this machine's network may reach "
        "it (DNS name, firewall, TLS certificate). Check your own internet connection and proxy first."
    ),
    "wrong_path": (
        "Ask the hub administrator for the exact URL of the MCP endpoint and the transport it speaks "
        "(SSE or streamable HTTP). None of /sse, /mcp or the base URL exists here."
    ),
    "wrong_transport": (
        "Ask the hub administrator which MCP transport the hub speaks (legacy SSE or streamable HTTP) and at "
        "which path. Something answers here, but not as an MCP SSE or streamable HTTP endpoint."
    ),
    "auth_rejected": (
        "Ask the hub administrator whether the API key is valid and active, which header carries it "
        "(MCP_HUB_AUTH_HEADER, here the one printed above) and whether this machine's address must be allowed."
    ),
    "not_configured": "Set MCP_HUB_API_KEY (and MCP_HUB_URL, MCP_HUB_ENABLED=true) in backend/.env, then run this again.",
    "unknown": (
        "Ask the hub administrator for the endpoint, transport and header; the answers seen above did not match "
        "any known pattern. Share this report (it contains no key)."
    ),
    "working": "",
}


def classify(report: dict) -> tuple[str, str, str]:
    """(verdict code, one line, what to ask). Pure, so tests can feed it."""
    net = report["network"]
    probes = report["probes"]
    hub = report["hub_client"]
    if not report["config"]["key_set"]:
        return "not_configured", "no API key is set (MCP_HUB_API_KEY is empty)", ASK["not_configured"]
    if not report["config"]["enabled"]:
        return "not_configured", "the hub is disabled (MCP_HUB_ENABLED is false)", ASK["not_configured"]
    if net["dns"] and not net["dns"]["ok"]:
        return "unreachable", f"unreachable: DNS lookup of {net['host']} failed ({net['dns']['error']})", ASK["unreachable"]
    if net["tcp"] and not net["tcp"]["ok"]:
        return "unreachable", f"unreachable: could not connect to {net['host']}:{net['port']} ({net['tcp']['error']})", ASK["unreachable"]
    if net["tls"] and not net["tls"]["ok"]:
        return "unreachable", f"unreachable: TLS handshake failed ({net['tls']['error']})", ASK["unreachable"]
    if hub["ok"]:
        return "working", f"working: {hub['transport']} transport, key accepted, {len(hub['tools'])} tools", ""
    if report["auth"].startswith("rejected"):
        return "auth_rejected", f"auth rejected: the hub answered {report['auth'].split()[-1]} to the key", ASK["auth_rejected"]
    keyed = [probes[n] for n in ("sse", "streamable_mcp", "streamable_base")]
    if all(p.get("error") == "timeout" for p in keyed):
        return "unreachable", "unreachable: every probe timed out", ASK["unreachable"]
    if any(p.get("error") for p in keyed) and not any("status" in p for p in keyed):
        return "unreachable", "unreachable: the hub did not answer any probe", ASK["unreachable"]
    statuses = [p.get("status") for p in keyed if "status" in p]
    if statuses and all(s == 404 for s in statuses):
        return "wrong_path", "wrong path: /sse, /mcp and the base URL all answered 404", ASK["wrong_path"]
    if any(s is not None and s >= 500 for s in statuses):
        return "unreachable", f"unreachable: the hub answers with a server error ({max(s for s in statuses if s)})", ASK["unreachable"]
    if any(s is not None and s != 404 and s not in (401, 403) for s in statuses):
        return (
            "wrong_transport",
            "wrong transport: a path answers but does not speak MCP over SSE or streamable HTTP",
            ASK["wrong_transport"],
        )
    return "unknown", f"not working: {hub.get('error') or 'no probe matched'}", ASK["unknown"]


# Hub client step


async def run_hub_client(url: str, header: str, key: str, timeout: float, enabled: bool) -> tuple[dict, Optional[HubClient]]:
    client = HubClient(url, header, key, timeout, retries=0, breaker_seconds=0, enabled=enabled)
    out: dict[str, Any] = {"ok": False, "transport": None, "tools": [], "error": None, "reason": None, "status": None}
    try:
        tools = await client.list_tools(refresh=True, budget=timeout * 2 + 1.0)
    except HubError as exc:
        out.update(error=f"{type(exc).__name__}: {client._scrub(str(exc))}", reason=exc.reason, status=exc.status)
        return out, client
    out["ok"] = True
    out["transport"] = client.status()["transport"]
    out["tools"] = [
        {"name": t.name, "description": (t.description.strip().splitlines() or [""])[0][:110], "schema": t.input_schema}
        for t in tools
    ]
    return out, client


async def call_read_only(client: HubClient, tools: list[dict]) -> list[dict]:
    """At most one call to each tool in READ_ONLY_TOOLS that the hub lists."""
    from mcp_client.client import HubTool

    results: list[dict] = []
    listed = {t["name"]: HubTool(t["name"], t["description"], t["schema"]) for t in tools}
    for name in READ_ONLY_TOOLS:
        tool = listed.get(name)
        if tool is None:
            results.append({"tool": name, "status": "not listed by the hub"})
            continue
        required = tool.input_schema.get("required") or []
        if name == "search_knowledgebase_tool":
            argument = tool.string_argument(("query", "question", "text"))
            if not argument:
                results.append({"tool": name, "status": "skipped: no single text argument found"})
                continue
            arguments = {argument: SAFE_SEARCH_QUERY}
        elif required:
            results.append({"tool": name, "status": f"skipped: needs arguments ({', '.join(required)}) the check cannot guess"})
            continue
        else:
            arguments = {}
        try:
            output = await client.call_tool(name, arguments, budget=30.0)
            results.append({"tool": name, "status": "ok", "arguments": list(arguments), "reply_start": " ".join(output.split())[:160]})
        except HubError as exc:
            results.append({"tool": name, "status": f"failed: {type(exc).__name__} ({exc.reason})"})
    return results


# Report


async def build_report(url: str, header: str, key: str, timeout: float, enabled: bool = True, read_only: bool = False) -> dict:
    root, _ = split_url(url)
    report: dict[str, Any] = {
        "config": {
            "base_url": root or url,
            "header": header,
            "key_set": bool(key),
            "key_length": len(key),
            "enabled": enabled,
            "timeout_seconds": timeout,
        }
    }
    report["network"] = await check_network(root, timeout) if root else {"host": "", "port": 0, "dns": {"ok": False, "error": "no url"}, "tcp": None, "tls": None}
    network_ok = bool(report["network"]["dns"] and report["network"]["dns"]["ok"] and report["network"]["tcp"] and report["network"]["tcp"]["ok"])
    empty = {"error": "skipped"}
    probes = {n: dict(empty) for n in ("sse", "streamable_mcp", "streamable_base", "without_key")}
    if network_ok and (not report["network"]["tls"] or report["network"]["tls"]["ok"]):
        probes = await run_probes(root, header, key, timeout)
    report["probes"] = probes
    hub_out: dict[str, Any] = {"ok": False, "transport": None, "tools": [], "error": "not tried", "reason": None, "status": None}
    client = None
    if key and enabled and probes["sse"].get("error") != "skipped":
        hub_out, client = await run_hub_client(url, header, key, timeout, enabled)
    report["hub_client"] = hub_out
    report["auth"] = auth_result(probes, hub_out if hub_out["error"] else None, bool(key)) if probes["sse"].get("error") != "skipped" else "unknown"
    if hub_out["ok"]:
        report["auth"] = "accepted"
    if read_only and client is not None and hub_out["ok"]:
        report["read_only_calls"] = await call_read_only(client, hub_out["tools"])
    code, line, ask = classify(report)
    report["verdict"] = {"code": code, "summary": line, "ask_the_administrator": ask}
    return report


def _probe_line(label: str, p: dict) -> str:
    if "status" in p:
        extra = ""
        if p.get("content_type"):
            extra += f" {p['content_type']}"
        if p.get("first_event"):
            extra += f" first event '{p['first_event']}'"
        names = ", ".join(f"{k}={v}" if k in ("mcp-session-id", "allow") else k for k, v in (p.get("headers") or {}).items() if k in ("mcp-session-id", "www-authenticate", "allow"))
        return f"  {label:<26} {p['method']} -> HTTP {p['status']}{extra}" + (f"  [{names}]" if names else "")
    return f"  {label:<26} {p['method']} -> {p.get('error', 'no answer')}"


def render_text(report: dict) -> str:
    c, n = report["config"], report["network"]
    lines = [
        "MCP hub check",
        f"  base URL        {c['base_url']}",
        f"  auth header     {c['header']}",
        f"  API key set     {'yes' if c['key_set'] else 'no'}" + (f" (length {c['key_length']})" if c["key_set"] else ""),
        f"  enabled         {c['enabled']}   timeout {c['timeout_seconds']}s",
        "",
        "Network",
    ]
    dns = n["dns"] or {}
    lines.append(f"  DNS             {'ok ' + ', '.join(dns.get('addresses', [])) if dns.get('ok') else 'FAILED ' + str(dns.get('error'))}")
    if n["tcp"] is not None:
        lines.append(f"  TCP             {'ok' if n['tcp']['ok'] else 'FAILED ' + str(n['tcp'].get('error'))}")
    if n["tls"] is not None:
        lines.append(f"  TLS             {'ok ' + str(n['tls'].get('version')) if n['tls']['ok'] else 'FAILED ' + str(n['tls'].get('error'))}")
    lines += ["", "Transports tried (with the key unless stated)"]
    pr = report["probes"]
    lines.append(_probe_line("SSE at /sse", pr["sse"]))
    lines.append(_probe_line("streamable HTTP at /mcp", pr["streamable_mcp"]))
    lines.append(_probe_line("streamable HTTP at base", pr["streamable_base"]))
    lines.append(_probe_line("SSE at /sse, no key", pr["without_key"]))
    hub = report["hub_client"]
    lines += ["", f"Auth            {report['auth']}", "", "HubClient (the application's own client)"]
    if hub["ok"]:
        lines.append(f"  transport       {hub['transport']}")
        lines.append(f"  tools           {len(hub['tools'])}")
        for t in hub["tools"]:
            lines.append(f"    {t['name']:<36} {t['description']}")
    else:
        lines.append(f"  failed          {hub['error']}" + (f" (reason {hub['reason']})" if hub["reason"] else ""))
    if "read_only_calls" in report:
        lines += ["", "Read only calls"]
        for r in report["read_only_calls"]:
            lines.append(f"  {r['tool']:<28} {r['status']}" + (f": {r['reply_start']}" if r.get("reply_start") else ""))
    v = report["verdict"]
    lines += ["", f"Verdict: {v['summary']}"]
    if v["ask_the_administrator"]:
        lines.append(f"What to do: {v['ask_the_administrator']}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="mcp_client.check", description="Diagnose the connection to the SLT MCP hub.")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--timeout", type=float, default=None, help="seconds per step (default MCP_HUB_TIMEOUT_SECONDS)")
    parser.add_argument("--call-read-only", action="store_true", help="also call the read only non customer tools once each")
    parser.add_argument("--url", default=None, help="override MCP_HUB_URL")
    args = parser.parse_args(argv)

    url = (args.url or config.MCP_HUB_URL or "").strip().rstrip("/")
    timeout = max(0.2, args.timeout if args.timeout is not None else config.MCP_HUB_TIMEOUT_SECONDS)
    key = config.MCP_HUB_API_KEY
    reporter = _Reporter(key)
    report = asyncio.run(
        build_report(url, config.MCP_HUB_AUTH_HEADER, key, timeout, enabled=config.MCP_HUB_ENABLED, read_only=args.call_read_only)
    )
    if args.json:
        for t in report["hub_client"]["tools"]:
            t.pop("schema", None)
        text = json.dumps(report, indent=2, ensure_ascii=False)
    else:
        text = render_text(report)
    print(reporter.clean(text))
    return 0 if report["verdict"]["code"] == "working" else 1


if __name__ == "__main__":
    raise SystemExit(main())
