import sys
import time
from pathlib import Path
import torch

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

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

reasoning_img = create_change_composite(p1, p2, two_panel=True)
tmp_path = Path("backend/data/results/test_opt_reasoning.png")
reasoning_img.save(tmp_path, format="PNG")
preproc = preprocess_imagery_for_vqa(tmp_path)

vqa_svc = get_vqa_service()
adapter = get_adapter_for_model(settings.VQA_MODEL_ID)
loaded = vqa_svc._manager.load(settings.VQA_MODEL_ID, adapter.load)

prompt_text = build_change_vqa_prompt(query="Describe observed changes.", use_three_panel=False)
inf_input = VQAInferenceInput(
    rgb_image=preproc.rgb_image,
    query_text=prompt_text,
    max_new_tokens=64,
    temperature=0.2,
)
model_inputs = adapter.preprocess_input(inf_input, loaded)
input_len = model_inputs["input_ids"].shape[-1]

print("--- Testing Generation with max_new_tokens=64, repetition_penalty=1.15, no_repeat_ngram_size=3 ---")
gen_kwargs = {
    "max_new_tokens": 64,
    "temperature": 0.2,
    "do_sample": True,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 3,
}

t0 = time.perf_counter()
with torch.inference_mode():
    generated_ids = loaded.model_object.generate(**model_inputs, **gen_kwargs)
t_gen = (time.perf_counter() - t0) * 1000

new_tokens = generated_ids[:, input_len:]
answer = loaded.processor_object.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()

val_res = validate_change_vqa_vlm_output(answer)

print(f"Time: {t_gen:.1f}ms ({t_gen/1000:.2f}s)")
print(f"Generated tokens: {int(new_tokens.shape[-1])}")
print(f"Answer: {answer!r}")
print(f"Validation: is_valid={val_res.is_valid}, status={val_res.status}, reason={val_res.reason}")
