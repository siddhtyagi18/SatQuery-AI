import hashlib
import numpy as np
from PIL import Image
from pathlib import Path
from app.services.model_inference import run_change_detection

def verify():
    repo_root = Path(__file__).resolve().parent.parent.parent
    before_path = repo_root / "public/demo/optical_before.jpg"
    after_path = repo_root / "public/demo/optical_after.jpg"

    assert before_path.exists()
    assert after_path.exists()

    res = run_change_detection(
        before_path=before_path,
        after_path=after_path,
        analysis_id="golden_verification",
        threshold=0.70,
    )

    mask_path = Path("data/results/golden_verification_changemap.png")
    assert mask_path.exists()

    with Image.open(mask_path) as img:
        mask = np.array(img.convert("L")) > 0
        total = mask.size
        changed = int(np.sum(mask))
        pct = round(changed / total * 100, 2)

    h = hashlib.sha256(mask_path.read_bytes()).hexdigest()

    print(f"Total Pixels: {total}")
    print(f"Changed Pixels: {changed}")
    print(f"Changed Percentage: {pct}%")
    print(f"Unchanged Percentage: {round(100 - pct, 2)}%")
    print(f"Threshold: {res.stats.get('threshold_used')}")
    print(f"Confidence: {res.stats.get('confidence')}")
    print(f"Mask SHA-256: {h}")

    assert changed == 71495, f"Expected 71495 changed pixels, got {changed}"
    assert total == 786432, f"Expected 786432 total pixels, got {total}"
    assert pct == 9.09, f"Expected 9.09%, got {pct}"
    assert h == "489e6fa5747b7c2c5f46c5416d047dc1def745ef9ba359601025f4cbac3638d2", f"Hash mismatch: {h}"
    assert res.stats.get("confidence") is None, f"Confidence must be None, got {res.stats.get('confidence')}"

    print("\n>>> GOLDEN BASELINE VERIFICATION: PASSED (100% BIT-FOR-BIT IDENTICAL)")

if __name__ == "__main__":
    verify()
