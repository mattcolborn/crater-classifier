"""
daily_run.py
Daily inference demonstration: predicts on the demo images and logs
results to the Supabase `daily_runs` table.

Run locally:
    export SUPABASE_URL="https://YOUR-PROJECT.supabase.co"
    export SUPABASE_SERVICE_KEY="your-service-role-key"
    poetry run python daily_run.py
"""

import os
from datetime import date
from pathlib import Path

import torch
from supabase import Client, create_client

from crater_classifier import config
from crater_classifier.model import build_model
from crater_classifier.predict import predict_image

DEMO_DIR = Path("demo_images")
MODEL_PATH = Path("models") / "crater_classifier.pth"


def true_label_from_filename(filename: str) -> str:
    """Derive the ground-truth label from a demo image filename."""
    return "no_craters" if filename.startswith("no_crater") else "craters"


def main() -> None:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    client: Client = create_client(url, key)

    checkpoint = torch.load(MODEL_PATH, map_location=config.DEVICE)
    class_to_idx: dict[str, int] = checkpoint["class_to_idx"]

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])

    rows: list[dict[str, object]] = []
    n_correct = 0
    today = date.today().isoformat()

    for image_path in sorted(DEMO_DIR.glob("*.png")):
        predicted, confidence = predict_image(str(image_path), model, class_to_idx)
        true_label = true_label_from_filename(image_path.name)
        correct = predicted == true_label
        n_correct += int(correct)
        rows.append(
            {
                "run_date": today,
                "filename": image_path.name,
                "true_label": true_label,
                "predicted_label": predicted,
                "confidence": round(confidence, 4),
                "correct": correct,
            }
        )

    client.table("daily_runs").insert(rows).execute()
    print(f"\nLogged {len(rows)} predictions — {n_correct}/{len(rows)} correct.")


if __name__ == "__main__":
    main()