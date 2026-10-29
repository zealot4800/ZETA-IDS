"""Shared local Ollama transport, retries and per-run request accounting."""
from __future__ import annotations

from collections import Counter, OrderedDict
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .evidence import digest


def answer(body: dict) -> dict:
    choice = body["choices"][0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise ValueError("incomplete_response")
    result = json.loads(choice["message"]["content"])
    if not isinstance(result, dict):
        raise ValueError("expected_json_object")
    return result


class Ollama:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.requests = Counter()
        self.tokens = Counter()
        self.cache = OrderedDict()
        self.cache_hits = 0

    def call(self, stage: str, body: dict, *, use_cache: bool = True) -> dict:
        options = self.cfg[stage]
        key = digest([stage, options["base_url"], options["endpoint"], body])
        if use_cache and key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        retries = options["request_retries"]
        for attempt in range(retries + 1):
            self.requests[stage] += 1
            request = Request(options["base_url"].rstrip("/") + options["endpoint"],
                              data=json.dumps(body, allow_nan=False).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(request, timeout=options["timeout_seconds"]) as response:
                    result = json.load(response)
                if not isinstance(result, dict) or "error" in result:
                    return {"body": None, "error": "provider_error_response"}
                usage = result.get("usage") or {}
                if not isinstance(usage, dict):
                    usage = {}
                for kind, field in (("input", "prompt_tokens"), ("output", "completion_tokens")):
                    value = usage.get(field, 0)
                    if type(value) is int and value >= 0:
                        self.tokens[kind] += value
                reply = {"body": result, "error": None}
                self.cache[key] = reply
                if len(self.cache) > 256:
                    self.cache.popitem(last=False)
                return reply
            except HTTPError as exc:
                error = f"provider_http_error_{exc.code}"
                retryable = exc.code in options["retry_http_statuses"]
                exc.close()
            except OSError:
                error, retryable = "connection_or_timeout", True
            except (ValueError, UnicodeError, TypeError):
                return {"body": None, "error": "malformed_response"}
            if not retryable or attempt == retries:
                return {"body": None, "error": error}
            time.sleep(options["retry_backoff_seconds"] * 2 ** attempt)


def request_body(options: dict, messages: list[dict], name: str, schema: dict) -> dict:
    return {"model": options["model"], "messages": messages,
            "temperature": options["temperature"], "max_tokens": options["max_tokens"],
            "reasoning_effort": options["reasoning_effort"], "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": name, "strict": True, "schema": schema}}}
