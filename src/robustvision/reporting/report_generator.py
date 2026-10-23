from datetime import datetime, timezone
from pathlib import Path


def generate_report(payload: dict, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = payload.get("metrics", {})
    output.write_text(
        f"# RobustVision AI Evaluation Report\n\nGenerated: {datetime.now(timezone.utc).isoformat()}\n\n## Provenance\n\n- Model: {payload.get('model', 'unknown')}\n- Device: {payload.get('device', 'unknown')}\n- Dataset/split: {payload.get('dataset', 'synthetic')}\n- Sample count: {metrics.get('samples', 'unknown')}\n\n## Metrics\n\n- Accuracy: {metrics.get('accuracy', 'N/A')}\n- Macro F1: {metrics.get('macro_f1', 'N/A')}\n\nResults reflect only this recorded run; they are not benchmark claims.\n",
        encoding="utf-8",
    )
    return output
