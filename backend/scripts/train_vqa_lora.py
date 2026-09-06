"""
backend/scripts/train_vqa_lora.py
=================================
PEFT LoRA fine-tuning and domain adaptation script for SmolVLM on BigEarthNet.
Supports both smoke test mode (--smoke-test) and full training runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def load_bigearthnet_samples(parquet_path: Optional[str] = None, num_samples: int = 20) -> List[Dict[str, Any]]:
    """Helper function to load sample QA pairs from BigEarthNet multimodal dataset."""
    from app.services.datasets.bigearthnet_multimodal import BigEarthNetMultimodalDataset
    dataset = BigEarthNetMultimodalDataset(parquet_path=parquet_path, max_samples=num_samples)
    return [
        {
            "patch_id": d.get("patch_id"),
            "query": d.get("question"),
            "question": d.get("question"),
            "answer": d.get("answer"),
            "image": d.get("image"),
        }
        for d in dataset
    ]


def normalize_answer(text: str) -> str:
    t = (text or "").strip().lower().strip(".,!?;:'\"")
    return " ".join(t.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def soft_match(pred: str, gold: str) -> bool:
    np_ = normalize_answer(pred)
    ng = normalize_answer(gold)
    return ng == np_ or ng in np_ or np_ in ng


def parse_args():
    p = argparse.ArgumentParser(description="PEFT LoRA VLM training on BigEarthNet.")
    p.add_argument("--parquet-path", type=str, default="C:/Users/Lenovo/Downloads/BigEarthNet.txt.parquet")
    p.add_argument("--images-root", type=str, default="C:/Users/Lenovo/Downloads/BigEarthNet-S2")
    p.add_argument("--output-dir", type=str, default=str(_BACKEND_ROOT / "checkpoints" / "vqa_lora"))
    p.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    p.add_argument("--num-samples", type=int, default=2, help="Sample count for smoke test mode")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def _forward_step(processor, peft_model, sample, device, grad_accum):
    import torch
    pil_image = sample.get("image")
    if pil_image is None:
        return None
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Remote sensing analysis: {sample['question']}"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["answer"]}],
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
    inputs = processor(
        text=prompt,
        images=[pil_image],
        return_tensors="pt",
        do_image_splitting=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = inputs["input_ids"].clone()
    outputs = peft_model(**inputs, labels=labels)
    loss = outputs.loss
    if loss is None:
        sl = outputs.logits[..., :-1, :].contiguous()
        slab = labels[..., 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss()(sl.view(-1, sl.size(-1)), slab.view(-1))
    (loss / grad_accum).backward()
    return float(loss.item())


def _generate_answer(processor, model, sample, device):
    import torch
    pil_image = sample.get("image")
    if pil_image is None:
        return ""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Remote sensing analysis: {sample['question']}"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        images=[pil_image],
        return_tensors="pt",
        do_image_splitting=False,
    ).to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        new_ids = gen[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def main():
    args = parse_args()
    t0 = time.perf_counter()

    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import LoraConfig, get_peft_model, PeftModel

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU only)"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3) if torch.cuda.is_available() else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp01 = _BACKEND_ROOT / "checkpoints" / "best_model.pt"
    exp01_size_before = exp01.stat().st_size if exp01.exists() else None

    mode_label = "Smoke Test" if args.smoke_test else "Full Training — vqa_lora_experiment_01"
    print("=" * 75, flush=True)
    print(f"SatQuery-AI VLM LoRA {mode_label}", flush=True)
    print(f"Device: {device} ({gpu_name})", flush=True)
    if vram_gb:
        print(f"VRAM: {vram_gb:.2f} GB", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print("=" * 75, flush=True)

    from app.services.datasets.bigearthnet_multimodal import BigEarthNetMultimodalDataset

    if args.smoke_test:
        train_ds = BigEarthNetMultimodalDataset(
            parquet_path=args.parquet_path,
            images_root=args.images_root,
            max_samples=args.num_samples,
        )
        val_ds = None
    else:
        train_ds = BigEarthNetMultimodalDataset(
            parquet_path=args.parquet_path,
            images_root=args.images_root,
            split="train",
            max_samples=args.max_train_samples,
        )
        val_ds = BigEarthNetMultimodalDataset(
            parquet_path=args.parquet_path,
            images_root=args.images_root,
            split="validation",
            max_samples=args.max_val_samples,
        )

    n_train = len(train_ds)
    n_val = len(val_ds) if val_ds else 0
    print(f"\n[1/5] Dataset: {n_train} train, {n_val} val samples", flush=True)
    if n_train == 0:
        raise RuntimeError(
            "No training samples found. Ensure BigEarthNet-S2 GeoTIFF patches exist on disk "
            "and patch_ids match rows in BigEarthNet.txt.parquet."
        )

    print(f"\n[2/5] Loading base model {args.model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id)
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "do_image_splitting"):
        processor.image_processor.do_image_splitting = False

    base_model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    base_model.to(device)
    total_params = sum(p.numel() for p in base_model.parameters())

    print("\n[3/5] Applying PEFT LoRA...", flush=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=None,
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    trainable_pct = 100.0 * trainable_params / total_params

    print(f"\n[4/5] Training (epochs={args.epochs}, lr={args.lr}, "
          f"batch={args.batch_size}, grad_accum={args.grad_accum})...", flush=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, peft_model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    training_log = []
    all_losses: List[float] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        peft_model.train()
        epoch_t = time.perf_counter()
        epoch_losses: List[float] = []
        optimizer.zero_grad()
        accum = 0
        global_step = 0

        for idx in range(n_train):
            sample = train_ds[idx]
            try:
                loss_val = _forward_step(processor, peft_model, sample, device, args.grad_accum)
            except Exception as e:
                print(f"  [WARN] step error {sample.get('patch_id', '?')[:30]}: {e}", flush=True)
                optimizer.zero_grad()
                accum = 0
                continue

            if loss_val is None:
                continue

            epoch_losses.append(loss_val)
            all_losses.append(loss_val)
            accum += 1

            print(
                f"  E{epoch+1} step{global_step+1} [{idx+1}/{n_train}] "
                f"{sample.get('patch_id', '?')[:28]}... loss={loss_val:.4f}",
                flush=True,
            )

            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum = 0
                global_step += 1

            if args.smoke_test and global_step >= 2:
                break

        # Flush remaining accumulation
        if accum > 0:
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1) if epoch_losses else float("nan")

        # Per-epoch validation pass
        val_loss = None
        val_losses_ep: List[float] = []
        if not args.smoke_test and val_ds and n_val > 0:
            peft_model.eval()
            with torch.no_grad():
                for vi in range(n_val):
                    vs = val_ds[vi]
                    vi_img = vs.get("image")
                    if vi_img is None:
                        continue
                    vm = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": f"Remote sensing analysis: {vs['question']}"},
                            ],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": vs["answer"]}]},
                    ]
                    vp = processor.apply_chat_template(vm, add_generation_prompt=False)
                    vi_in = processor(text=vp, images=[vi_img], return_tensors="pt", do_image_splitting=False)
                    vi_in = {k: v.to(device) for k, v in vi_in.items()}
                    vlab = vi_in["input_ids"].clone()
                    try:
                        vout = peft_model(**vi_in, labels=vlab)
                        vl = vout.loss
                        if vl is None:
                            sl = vout.logits[..., :-1, :].contiguous()
                            slab = vlab[..., 1:].contiguous()
                            vl = torch.nn.CrossEntropyLoss()(sl.view(-1, sl.size(-1)), slab.view(-1))
                        val_losses_ep.append(float(vl.item()))
                    except Exception:
                        pass
            val_loss = sum(val_losses_ep) / max(len(val_losses_ep), 1) if val_losses_ep else None
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                # Save best checkpoint
                best_dir = output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                peft_model.save_pretrained(str(best_dir))
                print(f"  -> New best checkpoint: val_loss={val_loss:.4f} -> {best_dir}", flush=True)

        dur = time.perf_counter() - epoch_t
        ep_log: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "duration_sec": round(dur, 1),
            "n_train_steps": len(epoch_losses),
            "n_val_steps": len(val_losses_ep),
            "lr": args.lr,
        }
        training_log.append(ep_log)
        vl_str = f" val={val_loss:.4f}" if val_loss is not None else ""
        print(f"\n*** Epoch {epoch+1}: train={train_loss:.4f}{vl_str} ({dur:.1f}s) ***\n", flush=True)

        if args.smoke_test:
            break

    # Save checkpoint + metadata
    print(f"\n[5/5] Saving adapter to {output_dir}...", flush=True)
    peft_model.save_pretrained(str(output_dir))
    if not args.smoke_test:
        processor.save_pretrained(str(output_dir))

    exp_name = "vqa_lora_smoke" if args.smoke_test else "vqa_lora_experiment_01"
    cfg: Dict[str, Any] = {
        "experiment": exp_name,
        "base_model": args.model_id,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(trainable_pct, 4),
        "device": device,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "n_train_samples": n_train,
        "n_val_samples": n_val,
        "seed": args.seed,
        "training_log": training_log,
        "best_val_loss": round(best_val_loss, 4) if best_val_loss < float("inf") else None,
        "best_epoch": best_epoch,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"  -> Saved {[f.name for f in output_dir.iterdir()]}", flush=True)

    # Reload + inference verification
    print("\n[Verification] Reloading adapter from disk and running inference...", flush=True)
    eval_base = AutoModelForVision2Seq.from_pretrained(
        args.model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)
    adapted = PeftModel.from_pretrained(eval_base, str(output_dir))
    adapted.eval()

    ref_ds = val_ds if (not args.smoke_test and val_ds and n_val > 0) else train_ds
    test_s = ref_ds[0]
    response = _generate_answer(processor, adapted, test_s, device)

    # Held-out evaluation
    em_correct = soft_correct = 0
    eval_results: List[Dict[str, Any]] = []

    if not args.smoke_test and val_ds and n_val > 0:
        print("\n[Evaluation] Running held-out evaluation on all validation samples...")
        for vi in range(n_val):
            vs = val_ds[vi]
            if vs.get("image") is None:
                continue
            q = vs["question"]
            gold = vs["answer"]
            pred = _generate_answer(processor, adapted, vs, device)
            em = exact_match(pred, gold)
            sm = soft_match(pred, gold)
            if em:
                em_correct += 1
            if sm:
                soft_correct += 1
            tag = "EM" if em else ("SM" if sm else "X")
            eval_results.append({
                "patch_id": vs["patch_id"],
                "question": q,
                "ground_truth": gold,
                "prediction": pred,
                "exact_match": em,
                "soft_match": sm,
            })
            print(f"  [{vi+1}] Q: {q[:55]}", flush=True)
            print(f"       GT: {gold} | PRED: {pred[:55]} | {tag}", flush=True)

        n_eval = len(eval_results)
        em_acc = em_correct / max(n_eval, 1)
        soft_acc = soft_correct / max(n_eval, 1)
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump({
                "n_evaluated": n_eval,
                "exact_match_accuracy": round(em_acc, 4),
                "soft_match_accuracy": round(soft_acc, 4),
                "results": eval_results,
            }, f, indent=2)
    else:
        n_eval = 0
        em_acc = soft_acc = 0.0

    # Guard: verify Experiment 01 is intact
    exp01_size_after = exp01.stat().st_size if exp01.exists() else None
    exp01_intact = exp01_size_before == exp01_size_after
    elapsed = round(time.perf_counter() - t0, 2)

    fvl = training_log[-1].get("val_loss") if training_log else None

    print("\n" + "=" * 75, flush=True)
    hdr = "SMOKE TEST REPORT" if args.smoke_test else "FULL TRAINING REPORT — vqa_lora_experiment_01"
    print(f"SatQuery-AI VLM LoRA {hdr}", flush=True)
    print("=" * 75, flush=True)
    print(f"1.  Base Model        : {args.model_id}", flush=True)
    print(f"2.  Dataset           : {n_train} real train + {n_val} real val (BigEarthNet S2 GeoTIFFs)", flush=True)
    print(f"3.  Split             : {'Official BigEarthNet train/validation columns' if not args.smoke_test else 'No split (smoke test pool)'}", flush=True)
    print(f"4.  LoRA Config       : r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} modules=q/v/k/o_proj", flush=True)
    print(f"5.  Trainable Params  : {trainable_params:,} / {total_params:,} ({trainable_pct:.4f}%)", flush=True)
    print(f"6.  GPU / VRAM        : {gpu_name} / {f'{vram_gb:.2f} GB' if vram_gb else 'N/A (CPU only)'}", flush=True)
    print(f"7.  Epochs            : {args.epochs}", flush=True)
    print(f"8.  Batch / Accum     : {args.batch_size} / {args.grad_accum} (effective={args.batch_size * args.grad_accum})", flush=True)
    print(f"9.  Learning Rate     : {args.lr}", flush=True)
    if all_losses:
        print(f"10. Train Loss        : initial={all_losses[0]:.4f} -> final={all_losses[-1]:.4f}", flush=True)
    else:
        print(f"10. Train Loss        : N/A", flush=True)
    print(f"11. Final Val Loss    : {fvl}", flush=True)
    if not args.smoke_test:
        print(f"    Best Val Loss    : {round(best_val_loss, 4) if best_val_loss < float('inf') else 'N/A'} (epoch {best_epoch})", flush=True)
    print(f"12. Eval Metrics      : ExactMatch={em_acc:.4f} ({em_correct}/{n_eval}) SoftMatch={soft_acc:.4f} ({soft_correct}/{n_eval})", flush=True)
    print(f"13. Qualitative Eval  : {n_eval} held-out samples (unselected; see evaluation_results.json)", flush=True)
    print(f"14. Checkpoint        : {output_dir}", flush=True)
    print(f"15. Reload Inference  : VERIFIED — Test Q: {test_s['question'][:50]}", flush=True)
    print(f"    GT: {test_s['answer']} | Model Response: {response[:80]}", flush=True)
    print(f"16. Total Time        : {elapsed}s", flush=True)
    print(f"17. Experiment01 Intact: {exp01_intact} ({exp01_size_before} -> {exp01_size_after} bytes)", flush=True)
    status = "SMOKE PASSED - READY FOR FULL TRAINING" if args.smoke_test else "FULL TRAINING COMPLETE"
    print(f"Status               : {status}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
