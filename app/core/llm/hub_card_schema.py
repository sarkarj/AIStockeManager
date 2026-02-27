from __future__ import annotations

HUB_CARD_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["meta", "summary", "drivers", "conflicts", "watch", "evidence"],
    "properties": {
        "meta": {
            "type": "object",
            "additionalProperties": False,
            "required": ["ticker", "generated_at", "policy_id", "policy_version", "profile", "mode"],
            "properties": {
                "ticker": {"type": "string"},
                "generated_at": {"type": "string"},
                "policy_id": {"type": "string"},
                "policy_version": {"type": "string"},
                "profile": {"type": "string"},
                "mode": {"type": "string", "enum": ["FULL", "TECHNICAL_ONLY", "DEGRADED"]},
            },
        },
        "summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action_final", "confidence_cap", "one_liner"],
            "properties": {
                "action_final": {"type": "string", "enum": ["ACCUMULATE", "WAIT", "REDUCE"]},
                "confidence_cap": {"type": "number"},
                "one_liner": {"type": "string", "minLength": 10, "maxLength": 220},
            },
        },
        "drivers": {
            "type": "array",
            "minItems": 2,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "citations"],
                "properties": {
                    "text": {"type": "string", "minLength": 10, "maxLength": 220},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "conflicts": {
            "type": "array",
            "minItems": 0,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "citations"],
                "properties": {
                    "text": {"type": "string", "minLength": 10, "maxLength": 220},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "watch": {
            "type": "array",
            "minItems": 2,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "citations"],
                "properties": {
                    "text": {"type": "string", "minLength": 10, "maxLength": 220},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "evidence": {
            "type": "object",
            "additionalProperties": False,
            "required": ["used_ids"],
            "properties": {
                "used_ids": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 30,
                    "items": {"type": "string"},
                }
            },
        },
    },
}
