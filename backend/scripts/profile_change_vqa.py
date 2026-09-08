import time
import sys
from pathlib import Path
from PIL import Image

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

print("=== PROFILING CHANGE VQA PIPELINE ===")
p1 = Path("public/demo/optical_before.jpg")
p2 = Path("public/demo/optical_after.jpg")
if not p1.exists():
    p1 = Path("../public/demo/optical_before.jpg")
    p2 = Path("../public/demo/optical_after.jpg")

from app.services.change_vqa import create_change_composite, build_change_vqa_prompt, validate_change_vqa_vlm_output
from app.services.preprocessing import preprocess_imagery_for_vqa
from app.services.vqa_service import get_vqa_service
from app.services.vqa_adapter import get_adapter_for_model, VQAInferenceInput
from app.config import get_settings

settings = get_settings()

# Stage 1: Composite creation
t0 = time.perf_counter()
reasoning_img = create_change_composite(p1, p2, two_panel=True)
evidence_img = create_change_composite(p1, p2, two_panel=False)
t_composite = (time.perf_counter() - t0) * 1000
print(f"1. Composite synthesis: {t_composite:.1f}ms | Reasoning img size: {reasoning_img.size}")

# Stage 2: Save to disk
t0 = time.perf_counter()
tmp_path = Path("backend/data/results/test_profile_reasoning.png")
tmp_path.parent.mkdir(parents=True, exist_ok=True)
reasoning_img.save(tmp_path, format="PNG")
t_save = (time.perf_counter() - t0) * 1000
print(f"2. Disk save: {t_save:.1f}ms")

# Stage 3: Preprocessing
t0 = time.perf_counter()
preproc = preprocess_imagery_for_vqa(tmp_path)
t_preproc = (time.perf_counter() - t0) * 1000
print(f"3. Imagery preprocessing (stretch/normalize): {t_preproc:.1f}ms | Preprocessed size: {preproc.rgb_image.size}")

# Stage 4: Model loading (Cold vs Warm)
vqa_svc = get_vqa_service()
model_id = settings.VQA_MODEL_ID
adapter = get_adapter_for_model(model_id)

t0 = time.perf_counter()
loaded = vqa_svc._manager.load(model_id, adapter.load)
t_model_load = (time.perf_counter() - t0) * 1000
print(f"4. ModelManager.load: {t_model_load:.1f}ms | Age: {loaded.age_sec:.1f}s | Device: {loaded.model_object.device}")

# Stage 5: Tokenization & Processor
prompt_text = build_change_vqa_prompt(query="Describe observed changes.", use_three_panel=False)
inf_input = VQAInferenceInput(
    rgb_image=preproc.rgb_image,
    query_text=prompt_text,
    max_new_tokens=settings.VQA_MAX_NEW_TOKENS,
    temperature=settings.VQA_TEMPERATURE,
)
t0 = time.perf_counter()
model_inputs = adapter.preprocess_input(inf_input, loaded)
t_tokenization = (time.perf_counter() - t0) * 1000
input_ids = model_inputs["input_ids"]
print(f"5. Processor preprocessing & tokenization: {t_tokenization:.1f}ms | input_ids shape: {input_ids.shape}")

# Stage 6: Generation (Inference)
import torch
gen_kwargs = {
    "max_new_tokens": min(inf_input.max_new_tokens, 128),
    "temperature": inf_input.temperature,
}
if gen_kwargs["temperature"] < 1e-3:
    gen_kwargs["do_sample"] = False
    gen_kwargs.pop("temperature", None)
else:
    gen_kwargs["do_sample"] = True

print(f"6. Starting model.generate with max_new_tokens={gen_kwargs.get('max_new_tokens')}, settings VQA_MAX_NEW_TOKENS={settings.VQA_MAX_NEW_TOKENS} ...")
t0 = time.perf_counter()
with torch.inference_mode():
    generated_ids = loaded.model_object.generate(**model_inputs, **gen_kwargs)
t_generate = (time.perf_counter() - t0) * 1000

input_len = input_ids.shape[-1]
new_tokens = generated_ids[:, input_len:]
num_new_tokens = int(new_tokens.shape[-1])
print(f"6. Generation complete: {t_generate:.1f}ms ({t_generate/1000:.2f}s) | Generated tokens: {num_new_tokens} | Rate: {t_generate/max(1, num_new_tokens):.1f}ms/token")

# Stage 7: Decoding
t0 = time.perf_counter()
answer = loaded.processor_object.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
t_decode = (time.perf_counter() - t0) * 1000
print(f"7. Decoding: {t_decode:.1f}ms")
print(f"Raw answer: {answer}")

# Stage 8: Warm Run (Model already loaded in memory)
print("\n--- WARM RUN TEST ---")
t0 = time.perf_counter()
model_inputs_warm = adapter.preprocess_input(inf_input, loaded)
with torch.inference_mode():
    generated_ids_warm = loaded.model_object.generate(**model_inputs_warm, **gen_kwargs)
new_tokens_warm = generated_ids_warm[:, input_len:]
answer_warm = loaded.processor_object.batch_decode(new_tokens_warm, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
t_warm_total = (time.perf_counter() - t0) * 1000
print(f"Warm full inference (tokenize + generate + decode): {t_warm_total:.1f}ms ({t_warm_total/1000:.2f}s)")
