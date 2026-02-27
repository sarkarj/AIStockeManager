#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
from typing import Any


def _invoke_ping(region: str, model_id: str) -> None:
    try:
        import boto3
        from botocore.config import Config
    except Exception as exc:  # pragma: no cover - import failure handled as FAIL
        raise RuntimeError(f"boto3 unavailable: {exc}") from exc

    runtime = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(
            connect_timeout=4,
            read_timeout=8,
            retries={"max_attempts": 1, "mode": "standard"},
        ),
    )

    prompt = "ping"

    try:
        response = runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 8, "temperature": 0.0},
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        text = "\n".join(item.get("text", "") for item in content if isinstance(item, dict)).strip()
        if text:
            return
        raise RuntimeError("empty response")
    except Exception as first_exc:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0.0,
        }
        try:
            response = runtime.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            payload: Any = response.get("body")
            if hasattr(payload, "read"):
                raw_text = payload.read().decode("utf-8", errors="ignore")
            else:
                raw_text = str(payload)
            if raw_text.strip():
                return
            raise RuntimeError("empty response")
        except Exception as second_exc:
            raise RuntimeError(
                f"{type(first_exc).__name__}: {first_exc}; {type(second_exc).__name__}: {second_exc}"
            ) from second_exc


def main() -> int:
    region = os.getenv("AWS_REGION", "").strip()
    model_id = os.getenv("BEDROCK_MODEL_ID", "").strip()
    if not model_id:
        model_id = os.getenv("BEDROCK_LLM_ID_CLAUDE", "").strip()
    if not model_id:
        model_id = os.getenv("BEDROCK_LLM_ID_OPENAI", "").strip()

    if not region or not model_id:
        print("SKIP: missing env")
        return 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke_ping, region, model_id)
            future.result(timeout=12)
        print("PASS: bedrock reachable")
        return 0
    except concurrent.futures.TimeoutError:
        print("FAIL: timeout")
        return 1
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
