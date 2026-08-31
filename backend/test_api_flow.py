# backend/test_api_flow.py
import requests
import time
from PIL import Image
import io

BASE_URL = "http://127.0.0.1:8000"

def test_full_flow():
    # 1. Health check
    res = requests.get(f"{BASE_URL}/health")
    print("1. Health check:", res.json())

    # 2. Tools list
    res = requests.get(f"{BASE_URL}/api/tools")
    print("2. Tools available:", len(res.json()))

    # 3. Create dummy test image
    img = Image.new("RGB", (400, 300), color=(40, 120, 200))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # 4. Upload image
    files = {"file": ("sample_optical.png", img_bytes, "image/png")}
    data = {"role": "single"}
    res = requests.post(f"{BASE_URL}/api/upload", files=files, data=data)
    upload_res = res.json()
    print("3. Uploaded image ID:", upload_res["id"], "Modality:", upload_res["metadata"]["modality"])
    image_id = upload_res["id"]

    # 5. Submit analysis
    payload = {
        "mode": "single_image",
        "imageIds": [image_id],
        "query": "Identify the primary water body and describe the land cover."
    }
    res = requests.post(f"{BASE_URL}/api/analysis", json=payload)
    submit_res = res.json()
    analysis_id = submit_res["analysisId"]
    print("4. Submitted analysis ID:", analysis_id)

    # 6. Poll for completion
    for i in range(15):
        time.sleep(0.5)
        res = requests.get(f"{BASE_URL}/api/analysis/{analysis_id}")
        data = res.json()
        status = data.get("status")
        print(f"   Polling status ({i+1}):", status)
        if status in ("completed", "failed"):
            print("5. Completed Result:")
            print("   Answer:", data.get("answerText"))
            print("   Confidence:", data.get("confidence"))
            print("   Detected tasks:", data.get("detectedTasks"))
            print("   Bounding boxes:", len(data.get("boundingBoxes") or []))
            print("   Execution steps:", len(data.get("executionTrace", {}).get("steps", [])))
            break

    # 7. List history
    res = requests.get(f"{BASE_URL}/api/analysis")
    hist = res.json()
    print("6. History count:", hist["total"])

if __name__ == "__main__":
    test_full_flow()
