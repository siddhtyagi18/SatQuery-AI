import requests
import json
import time
import hashlib
import os
from pathlib import Path

BASE = "http://127.0.0.1:8000"
p1 = Path("public/demo/optical_before.jpg")
p2 = Path("public/demo/optical_after.jpg")

print("--- 1. UPLOAD FLOW ---")
t0 = time.time()
with open(p1, "rb") as f1:
    up1 = requests.post(f"{BASE}/api/upload", files={"file": ("optical_before.jpg", f1, "image/jpeg")}, data={"role": "before"})
with open(p2, "rb") as f2:
    up2 = requests.post(f"{BASE}/api/upload", files={"file": ("optical_after.jpg", f2, "image/jpeg")}, data={"role": "after"})

t_upload = (time.time() - t0) * 1000
print(f"Upload 1 status: {up1.status_code}, Upload 2 status: {up2.status_code}, time: {t_upload:.1f}ms")
assert up1.status_code == 200, f"Upload 1 failed: {up1.text}"
assert up2.status_code == 200, f"Upload 2 failed: {up2.text}"

img1 = up1.json()
img2 = up2.json()

# Submit analysis
sub_payload = {
    "mode": "bi_temporal",
    "query": "Bi-temporal satellite change detection assessment",
    "imageIds": [img1["id"], img2["id"]]
}
sub_res = requests.post(f"{BASE}/api/analysis", json=sub_payload)
res_data = sub_res.json()
analysis_id = res_data.get("analysisId") or res_data.get("analysis_id")
print(f"Analysis created: {analysis_id}")

print("\n--- 2. COMPATIBILITY / METADATA ---")
print(f"  Img 1: dim={img1.get('metadata', {}).get('width')}x{img1.get('metadata', {}).get('height')}, format={img1.get('metadata', {}).get('fileFormat')}, modality={img1.get('metadata', {}).get('modality')}")
print(f"  Img 2: dim={img2.get('metadata', {}).get('width')}x{img2.get('metadata', {}).get('height')}, format={img2.get('metadata', {}).get('fileFormat')}, modality={img2.get('metadata', {}).get('modality')}")

print("\n--- 3. REAL CHANGE DETECTION RESULTS (FROM ANALYSIS) ---")
t_cd0 = time.time()
ana_res = requests.get(f"{BASE}/api/analysis/{analysis_id}")
t_cd = (time.time() - t_cd0) * 1000
assert ana_res.status_code == 200, f"Get analysis failed: {ana_res.text}"
full_ana = ana_res.json()
cm = full_ana.get("changeMap") or full_ana.get("change_map") or {}
print(f"Analysis retrieved in {t_cd:.1f}ms")
print(f"Overlay URL: {cm.get('overlayUrl') or cm.get('overlay_url')}")
print(f"Mask URL: {cm.get('maskUrl') or cm.get('mask_url')}")
print(f"Confidence: {full_ana.get('confidence')}")

# Extract stats
stats = full_ana.get("stats") or {}
if not stats and full_ana.get("geospatial_analytics"):
    stats = full_ana["geospatial_analytics"].get("global_statistics", {})

print("Change detection stats:", json.dumps({
    "changed_pixels": cm.get("changedPixels") or stats.get("changed_pixel_count"),
    "total_pixels": cm.get("totalPixels") or stats.get("total_pixel_count"),
    "changed_pct": cm.get("changedPixelPct") or stats.get("changed_pixel_pct"),
    "confidence": full_ana.get("confidence")
}, indent=2))

changed_px = cm.get("changedPixels") or stats.get("changed_pixel_count")
total_px = cm.get("totalPixels") or stats.get("total_pixel_count")
changed_pct = cm.get("changedPixelPct") or stats.get("changed_pixel_pct")

assert changed_px == 71495, f"Expected 71495 changed pixels, got {changed_px}"
assert total_px == 786432, f"Expected 786432 total pixels, got {total_px}"
assert changed_pct == 9.09, f"Expected 9.09%, got {changed_pct}"
assert full_ana.get("confidence") is None, f"Expected confidence None, got {full_ana.get('confidence')}"
print("GOLDEN REGRESSION NUMBERS MATCH EXACTLY: 71,495 / 786,432 (9.09%) | Confidence: None")

print("\n--- 4. GEO-SPATIAL CHANGE ANALYTICS ---")
t_geo0 = time.time()
geo_res = requests.get(f"{BASE}/api/analysis/{analysis_id}/change-analytics")
t_geo = (time.time() - t_geo0) * 1000
assert geo_res.status_code == 200, f"Get change-analytics failed: {geo_res.text}"
geo = geo_res.json()
print(f"Geo-spatial analytics retrieved in {t_geo:.1f}ms")
print(f"Hotspots count total: {geo.get('hotspots_count_total')}")
print(f"Hotspots count filtered: {geo.get('hotspots_count_filtered')}")
print(f"Density mean: {geo.get('change_density', {}).get('mean_change_density_pct')}%")
pa = geo.get("physical_area", {})
print(f"Physical area (km²): {pa.get('changed_area_km2')}")
print(f"Physical area formatted: {pa.get('formatted_summary')}")
assert pa.get("changed_area_km2") is None, "Physical area must be None without valid resolution"
assert "unavailable" in pa.get("formatted_summary", "").lower()

print("\n--- 5. INTERACTIVE ROI INVESTIGATION ---")
t_roi0 = time.time()
roi_payload = {
    "x1": 0.2,
    "y1": 0.2,
    "x2": 0.7,
    "y2": 0.7,
    "is_normalized": True,
    "run_vqa": False
}
roi_res = requests.post(f"{BASE}/api/analysis/{analysis_id}/roi-analysis", json=roi_payload)
t_roi = (time.time() - t_roi0) * 1000
print(f"ROI status: {roi_res.status_code}, time: {t_roi:.1f}ms")
assert roi_res.status_code == 200, f"ROI failed: {roi_res.text}"
roi_data = roi_res.json()
roi_stats = roi_data.get("statistics", {})
print("ROI Changed Pixels:", roi_stats.get("changed_pixels"))
print("ROI Changed Pct:", roi_stats.get("changed_percentage"))
print("ROI Relative Density Factor:", roi_data.get("global_comparison", {}).get("relative_density_factor"))
print("ROI Hotspots in Region:", roi_data.get("hotspots", {}).get("hotspots_count_total"))
print("ROI Physical Area Available:", roi_data.get("physical_area", {}).get("available"))
assert roi_data.get("physical_area", {}).get("available") is False, "ROI physical area must be False without resolution"
assert roi_data.get("vqa") is None, "ROI selection alone must NOT invoke VQA"

print("\n--- 6. ROI CHANGE VQA (EXPLICIT USER ACTION) ---")
t_vqa0 = time.time()
vqa_payload = {
    "x1": 0.2,
    "y1": 0.2,
    "x2": 0.7,
    "y2": 0.7,
    "is_normalized": True,
    "run_vqa": True,
    "vqa_query": "What primary structural or surface modifications occurred in this specific quadrant?"
}
vqa_res = requests.post(f"{BASE}/api/analysis/{analysis_id}/roi-analysis", json=vqa_payload)
t_vqa = (time.time() - t_vqa0) * 1000
print(f"ROI VQA status: {vqa_res.status_code}, time: {t_vqa:.1f}ms")
assert vqa_res.status_code == 200, f"ROI VQA failed: {vqa_res.text}"
vqa_data = vqa_res.json()
vqa_res_blob = vqa_data.get("vqa") or {}
print("ROI VQA answer snippet:", vqa_res_blob.get("answer", "")[:120])
print("ROI VQA confidence:", vqa_res_blob.get("confidence"))
assert vqa_res_blob.get("confidence") is None, "VQA confidence must be None"

print("\n--- 7. AI MISSION REPORT GENERATION ---")
t_rep0 = time.time()
rep_res = requests.post(f"{BASE}/api/analysis/{analysis_id}/report", json={"roi": vqa_data, "generate_pdf": True})
t_rep = (time.time() - t_rep0) * 1000
print(f"Report status: {rep_res.status_code}, time: {t_rep:.1f}ms")
assert rep_res.status_code == 200, f"Report failed: {rep_res.text}"
rep_data = rep_res.json()
rep_inner = rep_data.get("report") or {}
print("Executive Summary snippet:", rep_inner.get("executive_summary", "")[:120])
print("CD Confidence label:", rep_inner.get("change_detection", {}).get("confidence_label"))
print("CD Execution mode:", rep_inner.get("change_detection", {}).get("execution_mode"))
print("PDF URL:", rep_data.get("pdf_url"))
assert rep_inner.get("change_detection", {}).get("confidence_label") == "N/A — Uncalibrated"

print("\n--- 8. PDF VERIFICATION ---")
pdf_url = rep_data.get("pdf_url")
assert pdf_url, "PDF URL must not be empty"
pdf_res = requests.get(f"{BASE}{pdf_url}")
assert pdf_res.status_code == 200, f"PDF fetch failed: {pdf_res.status_code}"
assert pdf_res.content.startswith(b"%PDF"), "Content is not a valid PDF header"
print(f"PDF validated successfully: {len(pdf_res.content):,} bytes (starts with %PDF)")

print("\n============================================================")
print("AUDIT SUMMARY TIMINGS (WARM RUN):")
print(f"  Upload:                  {t_upload:.1f} ms")
print(f"  Change Detection (CPU):  {t_cd:.1f} ms")
print(f"  Geo-Spatial Analytics:   {t_geo:.1f} ms")
print(f"  ROI Calculation:         {t_roi:.1f} ms")
print(f"  ROI VQA:                 {t_vqa:.1f} ms")
print(f"  Mission Report Assembly: {t_rep:.1f} ms")
print("ALL LIVE AUDIT CHECKS PASSED WITH 100% INTEGRITY!")
print("============================================================")
