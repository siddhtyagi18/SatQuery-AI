def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["docs"] == "/docs"


def test_list_tools(client):
    resp = client.get("/api/tools")
    assert resp.status_code == 200
    tools = resp.json()
    ids = [t["id"] for t in tools]
    for expected in ["rs_vqa", "rs_caption", "rs_grounding", "change_detector",
                     "change_vqa", "optical_sar_analyzer", "spatial_analyzer"]:
        assert expected in ids, f"Missing tool {expected}"
    for t in tools:
        assert set(t.keys()) >= {"id", "name", "taskTypes", "supportedModalities", "status", "version", "description"}


def test_list_benchmark(client):
    resp = client.get("/api/benchmark")
    assert resp.status_code == 200
    metrics = resp.json()
    assert len(metrics) >= 10
    for m in metrics:
        assert m["value"] is None
