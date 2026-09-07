import requests
import json
import time
import os

BASE_URL = "http://127.0.0.1:8000"

t1_path = r"C:\Users\Lenovo\Desktop\test_45.png"
t2_path = r"C:\Users\Lenovo\Desktop\test_45 (2).png"

assert os.path.exists(t1_path), f"File not found: {t1_path}"
assert os.path.exists(t2_path), f"File not found: {t2_path}"

print(f"Uploading T1: {t1_path} ({os.path.getsize(t1_path)} bytes)")
with open(t1_path, "rb") as f:
    r1 = requests.post(f"{BASE_URL}/api/upload", files={"file": ("test_45.png", f, "image/png")}, data={"role": "before"})
r1.raise_for_status()
t1_id = r1.json()["id"]
print(f"Uploaded T1 image ID: {t1_id}")

print(f"Uploading T2: {t2_path} ({os.path.getsize(t2_path)} bytes)")
with open(t2_path, "rb") as f:
    r2 = requests.post(f"{BASE_URL}/api/upload", files={"file": ("test_45 (2).png", f, "image/png")}, data={"role": "after"})
r2.raise_for_status()
t2_id = r2.json()["id"]
print(f"Uploaded T2 image ID: {t2_id}")

payload = {
    "mode": "bi_temporal",
    "imageIds": [t1_id, t2_id],
    "query": "What land-use changes occurred between these two dates?"
}

print("Submitting bi-temporal analysis...")
r_sub = requests.post(f"{BASE_URL}/api/analysis", json=payload)
r_sub.raise_for_status()
analysis_id = r_sub.json()["analysisId"]
print(f"Analysis submitted! ID: {analysis_id}")
print(f"Frontend URL: http://localhost:3000/analysis/{analysis_id}")

start_t = time.time()
while True:
    time.sleep(3)
    elapsed = int(time.time() - start_t)
    r_poll = requests.get(f"{BASE_URL}/api/analysis/{analysis_id}")
    data = r_poll.json()
    status = data.get("status")
    print(f"[{elapsed}s] Status: {status}")
    if status in ("completed", "failed"):
        break

print("\n--- ANALYSIS COMPLETED ---")
print(json.dumps(data, indent=2))

with open(r"C:\Users\Lenovo\Desktop\SatQuery-AI\backend\scratch_audit_step16c2_res.json", "w") as out_f:
    json.dump(data, out_f, indent=2)
print("Saved to scratch_audit_step16c2_res.json")
