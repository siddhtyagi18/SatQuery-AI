"""
backend/scripts/audit_cold_warm_vqa.py
--------------------------------------
Cold vs Warm Change VQA latency audit using real SmolVLM + LoRA pipeline.
Measures:
- Model/adapter loading
- Imagery preprocessing
- Prompt tokenization
- Autoregressive generation
- Token decoding
- Total duration
"""
import time
from pathlib import Path
from PIL import Image

from app.services.vqa_service import get_vqa_service
from app.services.change_vqa import create_change_composite, build_change_vqa_prompt
from app.services.change_detection import _ensure_results_dir
from app.services.model_manager import get_model_manager

def run_audit():
    print("==================================================")
    print("SATQUERY-AI — CHANGE VQA COLD VS WARM LATENCY AUDIT")
    print("==================================================")

    repo_root = Path(__file__).resolve().parent.parent.parent
    img_a_path = repo_root / "public/demo/optical_before.jpg"
    img_b_path = repo_root / "public/demo/optical_after.jpg"

    assert img_a_path.exists(), f"Missing {img_a_path}"
    assert img_b_path.exists(), f"Missing {img_b_path}"

    results_dir = _ensure_results_dir()
    vqa_service = get_vqa_service()
    manager = get_model_manager()

    # Step 1: Prepare test reasoning image
    t_prep_0 = time.perf_counter()
    reasoning_img = create_change_composite(img_a=img_a_path, img_b=img_b_path, two_panel=True)
    reasoning_path = results_dir / "audit_test_reasoning.png"
    reasoning_img.save(reasoning_path)
    prep_time_ms = (time.perf_counter() - t_prep_0) * 1000
    print(f"Preparation (2-panel synthesis + disk save): {prep_time_ms:.1f} ms")

    prompt = build_change_vqa_prompt(query="Describe any significant land cover changes visible between T1 and T2.", date_a=None, date_b=None)

    # --- COLD RUN ---
    print("\n[RUN 1: COLD START (First invocation in process)]")
    # Reset cache to guarantee clean cold test
    manager._cache.clear()

    t_cold_0 = time.perf_counter()
    res_cold = vqa_service.run_real_or_fallback(
        query=prompt,
        mode="single_image",
        image_file_paths=[reasoning_path],
        tasks=["vqa"],
        tool_id="change_vqa",
        preferred_provider="local",
    )
    t_cold_total = time.perf_counter() - t_cold_0

    print(f"Cold Total Latency: {t_cold_total:.2f} s ({t_cold_total*1000:.1f} ms)")
    print(f"Answer: {res_cold.answer}")
    print(f"Evidence trace: {res_cold.evidence}")
    if res_cold.run_context and res_cold.run_context.inference_meta:
        print(f"Inference meta: {res_cold.run_context.inference_meta}")

    # --- WARM RUN ---
    print("\n[RUN 2: WARM START (Second invocation using same process / model in RAM)]")
    t_warm_0 = time.perf_counter()
    res_warm = vqa_service.run_real_or_fallback(
        query=prompt,
        mode="single_image",
        image_file_paths=[reasoning_path],
        tasks=["vqa"],
        tool_id="change_vqa",
        preferred_provider="local",
    )
    t_warm_total = time.perf_counter() - t_warm_0

    print(f"Warm Total Latency: {t_warm_total:.2f} s ({t_warm_total*1000:.1f} ms)")
    print(f"Answer: {res_warm.answer}")
    print(f"Evidence trace: {res_warm.evidence}")
    if res_warm.run_context and res_warm.run_context.inference_meta:
        print(f"Inference meta: {res_warm.run_context.inference_meta}")

    print("\n==================================================")
    print("AUDIT SUMMARY:")
    print(f"Cold Start: {t_cold_total:.2f}s")
    print(f"Warm Start: {t_warm_total:.2f}s")
    print(f"Latency Reduction: {((t_cold_total - t_warm_total) / t_cold_total)*100:.1f}%")
    print("==================================================")

if __name__ == "__main__":
    run_audit()
