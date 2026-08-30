import pytest


def _upload_image(client, png_file, name, role):
    resp = client.post(
        "/api/upload",
        files={"file": (name, png_file, "image/png")},
        data={"role": role},
    )
    assert resp.status_code == 200
    return resp.json()["id"]


def test_submit_single_image_vqa_and_grounding(client, png_file):
    fid = _upload_image(client, png_file, "RESOURCESAT-2_LISS-IV_2024-03-15.png", "single")

    resp = client.post("/api/analysis", json={
        "mode": "single_image",
        "imageIds": [fid],
        "query": "What land cover types are visible and locate all buildings?",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "analysisId" in body
    aid = body["analysisId"]

    g = client.get(f"/api/analysis/{aid}")
    assert g.status_code == 200
    result = g.json()
    assert result["id"] == aid
    assert result["status"] == "completed"
    assert result["mode"] == "single_image"
    assert result["query"] == "What land cover types are visible and locate all buildings?"
    assert "vqa" in result["detectedTasks"]
    assert "grounding" in result["detectedTasks"]
    assert result["answerText"] is not None
    assert "[MOCK" in result["answerText"]
    assert result["confidence"] is not None
    assert 0 < result["confidence"] <= 1.0
    assert result["boundingBoxes"] is not None
    assert len(result["boundingBoxes"]) >= 3
    for b in result["boundingBoxes"]:
        assert set(b.keys()) == {"x", "y", "width", "height", "label", "confidence"}
    assert len(result["toolInvocations"]) >= 2
    assert len(result["images"]) == 1
    assert result["images"][0]["role"] == "single"
    assert result["task"] is not None
    assert len(result["selectedTools"]) >= 2
    assert result["evidence"] is not None
    assert any("[MOCK" in ev or "mock" in ev.lower() for ev in result["evidence"])


def test_submit_bi_temporal(client, png_file):
    f1 = _upload_image(client, png_file, "CARTOSAT-3_PAN_2022-01-10_T1.png", "before")
    f2 = _upload_image(client, png_file, "CARTOSAT-3_PAN_2024-01-08_T2.png", "after")

    resp = client.post("/api/analysis", json={
        "mode": "bi_temporal",
        "imageIds": [f1, f2],
        "query": "What changes occurred? Has urban expansion affected vegetation?",
    })
    assert resp.status_code == 200
    aid = resp.json()["analysisId"]

    g = client.get(f"/api/analysis/{aid}")
    assert g.status_code == 200
    result = g.json()
    assert "change_detection" in result["detectedTasks"] or "change_vqa" in result["detectedTasks"]
    assert result["changeMap"] is not None
    assert result["changeMap"]["overlayUrl"] is None
    assert "legend" in result["changeMap"]
    assert len(result["images"]) == 2


def test_submit_optical_sar(client, png_file):
    f1 = _upload_image(client, png_file, "optical_sample_2024-02-20.png", "optical")
    f2 = _upload_image(client, png_file, "RISAT-1A_SAR_C-band_VV_2024-02-22.png", "sar")

    resp = client.post("/api/analysis", json={
        "mode": "optical_sar",
        "imageIds": [f1, f2],
        "query": "Does SAR confirm optical detections? Find SAR-only features.",
    })
    assert resp.status_code == 200
    aid = resp.json()["analysisId"]

    g = client.get(f"/api/analysis/{aid}")
    assert g.status_code == 200
    result = g.json()
    assert "optical_sar_analyzer" in result["selectedTools"]
    assert len(result["images"]) == 2


def test_trace_endpoint(client, png_file):
    fid = _upload_image(client, png_file, "sample_2024-01-01.png", "single")
    aid = client.post("/api/analysis", json={
        "mode": "single_image", "imageIds": [fid],
        "query": "caption the image",
    }).json()["analysisId"]

    t = client.get(f"/api/analysis/{aid}/trace")
    assert t.status_code == 200
    trace = t.json()
    assert len(trace["steps"]) == 8
    expected = [
        "Query Received", "Input Validation", "Task Classification", "Tool Selection",
        "Parameters", "Processing", "Aggregation", "Completion",
    ]
    actual = [s["title"] for s in trace["steps"]]
    assert actual == expected
    for s in trace["steps"]:
        assert s["status"] == "done", f"Step {s['title']} status={s['status']}"
    assert trace["overallStatus"] == "completed"
    assert trace["totalElapsedMs"] is not None


def test_analysis_validation_wrong_count(client, png_file):
    fid = _upload_image(client, png_file, "sample.png", "single")
    resp = client.post("/api/analysis", json={
        "mode": "bi_temporal", "imageIds": [fid],
        "query": "something",
    })
    assert resp.status_code == 400


def test_analysis_validation_missing_image(client):
    resp = client.post("/api/analysis", json={
        "mode": "single_image", "imageIds": ["does-not-exist"],
        "query": "anything",
    })
    assert resp.status_code == 404


def test_list_history(client, png_file):
    fid = _upload_image(client, png_file, "sample_2024-01-01.png", "single")
    for _ in range(3):
        client.post("/api/analysis", json={
            "mode": "single_image", "imageIds": [fid], "query": "caption the image",
        })
    page = client.get("/api/analysis?pageSize=2&page=1").json()
    assert page["total"] >= 3
    assert len(page["items"]) == 2

    page2 = client.get("/api/analysis?pageSize=100&page=1").json()
    assert page2["total"] == len(page2["items"]) or page2["total"] >= 3

    by_mode = client.get("/api/analysis?mode=single_image").json()
    assert by_mode["total"] >= 3


def test_delete_analysis(client, png_file):
    fid = _upload_image(client, png_file, "sample.png", "single")
    aid = client.post("/api/analysis", json={
        "mode": "single_image", "imageIds": [fid], "query": "caption"
    }).json()["analysisId"]
    assert client.get(f"/api/analysis/{aid}").status_code == 200
    assert client.delete(f"/api/analysis/{aid}").status_code == 204
    assert client.get(f"/api/analysis/{aid}").status_code == 404
