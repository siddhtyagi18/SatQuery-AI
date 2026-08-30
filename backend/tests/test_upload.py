def test_upload_png(client, png_file, sample_png_name):
    resp = client.post(
        "/api/upload",
        files={"file": (sample_png_name, png_file, "image/png")},
        data={"role": "single"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "id" in body
    assert body["role"] == "single"
    meta = body["metadata"]
    assert meta["fileName"] == sample_png_name
    assert meta["fileFormat"] == "PNG"
    assert meta["fileSizeBytes"] > 0
    assert meta["widthPx"] == 64
    assert meta["heightPx"] == 48
    assert meta["bandCount"] == 3
    assert meta["modality"] in ("optical", "multispectral", "sar", "unknown")
    assert meta["acquisitionDate"] == "2024-03-15"
    assert body["previewUrl"] is not None


def test_upload_rejects_bad_extension(client):
    import io
    buf = io.BytesIO(b"abc")
    resp = client.post(
        "/api/upload",
        files={"file": ("bad.txt", buf, "text/plain")},
        data={"role": "single"},
    )
    assert resp.status_code == 400
    assert "Unsupported file extension" in resp.json()["detail"]


def test_upload_file_size_limit_error(client):
    import io
    from app.config import get_settings
    s = get_settings()
    huge = io.BytesIO(b"x" * (s.max_upload_bytes + 1))
    resp = client.post(
        "/api/upload",
        files={"file": ("huge.png", huge, "image/png")},
        data={"role": "single"},
    )
    assert resp.status_code == 400


def test_upload_sar_detects_modality(client, png_file):
    resp = client.post(
        "/api/upload",
        files={"file": ("RISAT-1A_SAR_C-band_VV_2024-02-22.png", png_file, "image/png")},
        data={"role": "sar"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["metadata"]["modality"] == "sar"
    assert body["metadata"]["modalityDetectionConfidence"] is not None
