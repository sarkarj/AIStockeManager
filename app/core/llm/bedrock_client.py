from __future__ import annotations

import json
import os
import time
from typing import Any

from jsonschema import ValidationError, validate


def has_aws_credentials() -> bool:
    if os.getenv("AWS_ACCESS_KEY_ID", "").strip() and os.getenv("AWS_SECRET_ACCESS_KEY", "").strip():
        return True
    if os.getenv("AWS_PROFILE", "").strip():
        return True
    try:
        import boto3

        session = boto3.session.Session()
        credentials = session.get_credentials()
        return credentials is not None
    except Exception:
        return False


class BedrockLLMClient:
    def __init__(
        self,
        region: str,
        model_id: str,
        max_tokens: int = 800,
        temperature: float = 0.0,
        request_timeout_seconds: float | None = None,
    ):
        self.region = region
        self.model_id = model_id
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.request_timeout_seconds = self._normalize_timeout(request_timeout_seconds)
        self.last_invocation: dict[str, Any] = {}

        try:
            import boto3  # local import to avoid hard dependency at import time
            client_kwargs: dict[str, Any] = {"region_name": region}
            if self.request_timeout_seconds is not None:
                try:
                    from botocore.config import Config  # type: ignore

                    timeout = float(self.request_timeout_seconds)
                    client_kwargs["config"] = Config(
                        connect_timeout=timeout,
                        read_timeout=timeout,
                        retries={"max_attempts": 1, "mode": "standard"},
                    )
                except Exception:
                    pass

            self._runtime = boto3.client("bedrock-runtime", **client_kwargs)
        except Exception as exc:
            raise RuntimeError("BEDROCK_UNAVAILABLE") from exc

    def invoke_structured(self, prompt: str, json_schema: dict) -> dict:
        full_prompt = self._build_structured_prompt(prompt, json_schema)
        self.last_invocation = {}
        invoke_started = time.perf_counter()

        try:
            text, raw_usage = self._invoke_with_converse(full_prompt)
            transport = "converse"
        except Exception:
            try:
                text, raw_usage = self._invoke_with_invoke_model(full_prompt)
                transport = "invoke_model"
            except Exception as exc:
                raise RuntimeError("BEDROCK_UNAVAILABLE") from exc

        latency_ms = round((time.perf_counter() - invoke_started) * 1000.0, 2)
        usage = self._normalize_usage(raw_usage)
        self.last_invocation = {
            "transport": transport,
            "latency_ms": latency_ms,
            "usage": usage,
            "model_id": self.model_id,
            "region": self.region,
        }

        parsed = self._parse_json_object(text)

        try:
            validate(instance=parsed, schema=json_schema)
        except ValidationError as exc:
            raise ValueError(f"BEDROCK_SCHEMA_INVALID: {exc.message}") from exc

        return parsed

    def get_last_usage(self) -> dict[str, Any]:
        payload = self.last_invocation if isinstance(self.last_invocation, dict) else {}
        usage = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
        return {
            "model_id": str(payload.get("model_id", self.model_id)),
            "transport": str(payload.get("transport", "unknown")),
            "latency_ms": float(payload.get("latency_ms", 0.0) or 0.0),
            "usage": {
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
        }

    def _build_structured_prompt(self, prompt: str, json_schema: dict) -> str:
        schema_text = json.dumps(json_schema, ensure_ascii=True)
        return (
            "You are a strict JSON generator. Return one JSON object only. "
            "No markdown, no explanation, no code fences.\n"
            f"JSON schema:\n{schema_text}\n\n"
            f"Task:\n{prompt}\n"
        )

    def _invoke_with_converse(self, full_prompt: str) -> tuple[str, dict[str, Any]]:
        response = self._runtime.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": full_prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        texts = [item.get("text", "") for item in content if isinstance(item, dict)]
        text = "\n".join([t for t in texts if t]).strip()
        if not text:
            raise ValueError("Empty response from Bedrock converse API")
        return text, response.get("usage", {})

    def _invoke_with_invoke_model(self, full_prompt: str) -> tuple[str, dict[str, Any]]:
        body = {
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        response = self._runtime.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        raw_body = response.get("body")
        if hasattr(raw_body, "read"):
            payload_text = raw_body.read().decode("utf-8")
        else:
            payload_text = str(raw_body)

        try:
            payload_json = json.loads(payload_text)
        except json.JSONDecodeError:
            return payload_text, {}

        # Try common model response shapes.
        if isinstance(payload_json, dict):
            if isinstance(payload_json.get("content"), list):
                parts = payload_json["content"]
                texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
                text = "\n".join([t for t in texts if t]).strip()
                if text:
                    return text, payload_json.get("usage", {})
            if isinstance(payload_json.get("output"), dict):
                output_message = payload_json["output"].get("message", {})
                if isinstance(output_message, dict):
                    content = output_message.get("content", [])
                    texts = [p.get("text", "") for p in content if isinstance(p, dict)]
                    text = "\n".join([t for t in texts if t]).strip()
                    if text:
                        return text, payload_json.get("usage", {})
            if isinstance(payload_json.get("completion"), str):
                return payload_json["completion"], payload_json.get("usage", {})

        return payload_text, payload_json.get("usage", {}) if isinstance(payload_json, dict) else {}

    def _parse_json_object(self, text: str) -> dict:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()

        # Direct parse first.
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: extract first JSON object boundaries.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("BEDROCK_INVALID_JSON")

        candidate = stripped[start : end + 1]
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("BEDROCK_INVALID_JSON")
        return parsed

    def _normalize_usage(self, raw_usage: Any) -> dict[str, int]:
        if not isinstance(raw_usage, dict):
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        input_tokens = self._usage_int(raw_usage.get("inputTokens", raw_usage.get("input_tokens")))
        output_tokens = self._usage_int(raw_usage.get("outputTokens", raw_usage.get("output_tokens")))
        total_tokens = self._usage_int(raw_usage.get("totalTokens", raw_usage.get("total_tokens")))
        if total_tokens <= 0:
            total_tokens = max(0, input_tokens) + max(0, output_tokens)
        return {
            "input_tokens": max(0, input_tokens),
            "output_tokens": max(0, output_tokens),
            "total_tokens": max(0, total_tokens),
        }

    def _usage_int(self, value: Any) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    def _normalize_timeout(self, value: float | None) -> float | None:
        if value is None:
            return None
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            return None
        if timeout <= 0:
            return None
        return timeout
