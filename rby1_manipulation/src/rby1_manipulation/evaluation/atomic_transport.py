"""Aggregate prompt-compliance metrics from atomic-policy rollout JSONL files."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping


def _passed_atomic_task(record: Mapping) -> bool:
    validation = record.get("validation", {})
    if record.get("task_type") == "lift_basket":
        return bool(
            record.get("success")
            and validation.get("basket_grasp_held")
            and validation.get("preloaded_fruits_remain")
            and validation.get("terminal_hold_valid")
        )
    return bool(
        record.get("success")
        and validation.get("target_newly_inside")
        and validation.get("fully_released")
        and validation.get("non_target_unchanged")
        and not validation.get("non_target_inserted", False)
        and not validation.get("wrong_target_grasped", False)
        and validation.get("safe_retreat")
        and validation.get("returned_to_ready")
        and validation.get("wrist_camera_view_valid")
        and validation.get("terminal_hold_valid")
    )


def aggregate_metrics(records: Iterable[Mapping]) -> dict:
    rows = list(records)
    if not rows:
        raise ValueError("no evaluation records")
    placement = [row for row in rows if row.get("task_type") == "place_one"]
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        if row.get("sequence_group_id"):
            groups[str(row["sequence_group_id"])].append(row)

    def rate(predicate, selected=rows) -> float:
        return sum(bool(predicate(row)) for row in selected) / len(selected) if selected else 0.0

    per_prompt = {}
    prompts = sorted({str(row.get("canonical_prompt", row.get("prompt"))) for row in rows})
    for prompt in prompts:
        selected = [
            row for row in rows
            if str(row.get("canonical_prompt", row.get("prompt"))) == prompt
        ]
        per_prompt[prompt] = {
            "rollouts": len(selected),
            "atomic_success_rate": rate(_passed_atomic_task, selected),
        }

    failures = Counter(
        str(row.get("failure_reason")) for row in rows if not _passed_atomic_task(row)
    )
    return {
        "rollouts": len(rows),
        "placement_rollouts": len(placement),
        "prompt_compliance_rate": rate(_passed_atomic_task),
        "target_placement_rate": rate(
            lambda row: row.get("validation", {}).get("target_newly_inside", False),
            placement,
        ),
        "unnecessary_fruit_insertion_rate": rate(
            lambda row: row.get("validation", {}).get("non_target_inserted", False),
            placement,
        ),
        "wrong_target_grasp_rate": rate(
            lambda row: row.get("validation", {}).get("wrong_target_grasped", False),
            placement,
        ),
        "non_target_displacement_rate": rate(
            lambda row: not row.get("validation", {}).get("non_target_unchanged", True),
            placement,
        ),
        "complete_release_rate": rate(
            lambda row: row.get("validation", {}).get("fully_released", False),
            placement,
        ),
        "safe_retreat_rate": rate(
            lambda row: row.get("validation", {}).get("safe_retreat", False),
            placement,
        ),
        "terminal_hold_compliance_rate": rate(
            lambda row: row.get("validation", {}).get("terminal_hold_valid", False)
        ),
        "post_success_extra_close_rate": rate(
            lambda row: row.get("post_success_extra_close", False), placement
        ),
        "sequence_completion_rate": (
            sum(all(_passed_atomic_task(row) for row in group) for group in groups.values())
            / len(groups) if groups else 0.0
        ),
        "sequence_groups": len(groups),
        "failure_reasons": dict(sorted(failures.items())),
        "per_prompt": per_prompt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="one rollout JSON object per line")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    path = Path(args.results)
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics = aggregate_metrics(records)
    payload = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
