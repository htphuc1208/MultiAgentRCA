from __future__ import annotations

import json
from typing import Any

from app.openrca.schemas import OpenRCAPredictionOutput


FIELD_MAP = {
    "root_cause_occurrence_datetime": "root cause occurrence datetime",
    "root_cause_component": "root cause component",
    "root_cause_reason": "root cause reason",
}


def format_prediction(prediction: OpenRCAPredictionOutput | dict[str, Any]) -> str:
    """Format prediction as the JSON-like payload accepted by OpenRCA's evaluator."""

    if isinstance(prediction, dict):
        prediction = OpenRCAPredictionOutput.model_validate(prediction)

    payload: dict[str, dict[str, str]] = {}
    for index, item in enumerate(prediction.root_causes, start=1):
        item_payload: dict[str, str] = {}
        raw_item = item.model_dump()
        for source, target in FIELD_MAP.items():
            value = raw_item.get(source)
            if value:
                item_payload[target] = str(value)
        payload[str(index)] = item_payload
    return json.dumps(payload, ensure_ascii=False)
