import requests
import json
import time
import os

BASE_URL = "http://127.0.0.1:8000"

t1_path = r"C:\Users\Lenovo\Desktop\test_45.png"
t2_path = r"C:\Users\Lenovo\Desktop\test_45 (2).png"

assert os.path.exists(t1_path), f"File not found: {t1_path}"
assert os.path.exists(t2_path), f"File not found: {t2_path}"

# 1. Upload T1 and T2 once
print(f"Uploading T1: {t1_path}")
with open(t1_path, "rb") as f:
    r1 = requests.post(f"{BASE_URL}/api/upload", files={"file": ("test_45.png", f, "image/png")}, data={"role": "before"})
r1.raise_for_status()
t1_id = r1.json()["id"]
print(f"Uploaded T1 image ID: {t1_id}")

print(f"Uploading T2: {t2_path}")
with open(t2_path, "rb") as f:
    r2 = requests.post(f"{BASE_URL}/api/upload", files={"file": ("test_45 (2).png", f, "image/png")}, data={"role": "after"})
r2.raise_for_status()
t2_id = r2.json()["id"]
print(f"Uploaded T2 image ID: {t2_id}")

questions = [
    ("Q1", "What changes occurred between these two images?"),
    ("Q2", "What land-use changes occurred between these two dates?"),
    ("Q3", "Which areas appear to have changed from the first image to the second?"),
]

results = []

for q_label, query_text in questions:
    print(f"\n==========================================")
    print(f"Submitting {q_label}: {query_text}")
    print(f"==========================================")

    payload = {
        "mode": "bi_temporal",
        "imageIds": [t1_id, t2_id],
        "query": query_text
    }

    t_sub = time.time()
    r_sub = requests.post(f"{BASE_URL}/api/analysis", json=payload)
    r_sub.raise_for_status()
    analysis_id = r_sub.json()["analysisId"]
    print(f"Submitted analysis ID: {analysis_id}")

    # Poll for completion
    poll_count = 0
    while True:
        time.sleep(3)
        poll_count += 1
        elapsed = int(time.time() - t_sub)
        r_poll = requests.get(f"{BASE_URL}/api/analysis/{analysis_id}")
        data = r_poll.json()
        status = data.get("status")
        print(f"[{elapsed}s] Status: {status}")
        if status in ("completed", "failed"):
            break

    # Extract Change-VQA specialist step and stats
    exec_trace = data.get("executionTrace", {})
    steps = exec_trace.get("steps", [])
    step_6 = next((s for s in steps if s.get("id") == "step-6"), {})
    step_6_meta = step_6.get("meta", {})
    changed_pct = step_6_meta.get("changed_pixel_pct")

    # Change-VQA tool invocation
    invocations = data.get("toolInvocations", [])
    change_vqa_inv = next((inv for inv in invocations if inv.get("toolId") == "change_vqa"), {})

    # Extract answer text and stats
    answer_text = data.get("answerText", "")

    # Extract raw VLM answer and validation status from evidence or answer text
    # In our answer text:
    # **Qualitative Visual Interpretation (VLM):**
    # <vlm_output>
    # **VLM Interpretation Validation:** `STATUS` (reason)
    vlm_val_status = "UNKNOWN"
    if "VLM Interpretation Validation:** `ACCEPTED`" in answer_text:
        vlm_val_status = "ACCEPTED"
    elif "VLM Interpretation Validation:** `INSUFFICIENT_TEMPORAL_REASONING`" in answer_text:
        vlm_val_status = "INSUFFICIENT_TEMPORAL_REASONING"
    elif "VLM Interpretation Validation:** `REJECTED`" in answer_text:
        vlm_val_status = "REJECTED"

    # Extract the qualitative VLM section
    raw_vlm_str = None
    if "**Qualitative Visual Interpretation (VLM):**" in answer_text:
        part = answer_text.split("**Qualitative Visual Interpretation (VLM):**")[1]
        part = part.split("**VLM Interpretation Validation:**")[0].strip()
        raw_vlm_str = part

    res_entry = {
        "question_label": q_label,
        "query": query_text,
        "analysis_id": analysis_id,
        "status": status,
        "raw_vlm_output": raw_vlm_str,
        "validated_output": raw_vlm_str if vlm_val_status == "ACCEPTED" else f"[{vlm_val_status}] {raw_vlm_str}",
        "temporal_validation_status": vlm_val_status,
        "detector_changed_pct": changed_pct,
        "execution_mode": data.get("executionMode"),
        "is_mock": data.get("isMock"),
        "confidence": data.get("confidence"),
        "answer_text": answer_text,
        "change_vqa_invocation": change_vqa_inv,
        "evidence": data.get("evidence", []),
    }
    results.append(res_entry)
    print(f"Result for {q_label}:")
    print(f"  VLM Output: {raw_vlm_str}")
    print(f"  Validation Status: {vlm_val_status}")
    print(f"  Detector Changed %: {changed_pct}%")
    print(f"  isMock: {data.get('isMock')}, Confidence: {data.get('confidence')}")

# Save full results
with open(r"C:\Users\Lenovo\Desktop\SatQuery-AI\backend\step16d_audit_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll 3 questions tested successfully! Results saved to step16d_audit_results.json")
