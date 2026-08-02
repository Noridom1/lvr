# FrozenLake LVR on one Colab A100

For the guided setup, open
[the Colab notebook](https://colab.research.google.com/github/Noridom1/lvr/blob/main/notebooks/frozenlake_lvr_colab.ipynb).
It downloads and verifies the public FrozenLake archive, restores the expected
directory layout, and exposes separate smoke-test, pilot, full-training,
resume, validation, and final-test gates.

This handoff targets the measured runtime:

- NVIDIA A100-SXM4-40GB;
- 83.5 GiB system RAM;
- about 189 GiB free local disk.

Run every command from the repository root. Keep training data at
`training_samples/frozenlake`, or override `IMAGE_FOLDER` when launching.

## 1. Install the pinned environment

Colab package versions change over time. The custom forward targets
Transformers 4.54.0 from `requirements.txt`; `environment.yaml` has the older
4.51.3 model layout and must not be used for this FrozenLake path.

```bash
pip install -r requirements.txt
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

Restart the Colab runtime after changing Torch/Transformers. If FlashAttention
cannot be installed, skip its install and use the SDPA commands below.

## 2. Check environment, paths, and manifests

With FlashAttention:

```bash
python scripts/check_frozenlake_colab.py
```

Without FlashAttention:

```bash
python scripts/check_frozenlake_colab.py --attention sdpa
```

The result must have `status: ok`. Then validate all formatted records and
referenced images:

```bash
python scripts/validate_frozenlake_manifests.py
```

The expected totals are 4,000 samples, 12,806 transitions, and 16,806 images.

## 3. Run the one-batch forward gate

With FlashAttention:

```bash
python scripts/smoke_test_frozenlake_lvr.py
```

Without FlashAttention:

```bash
python scripts/smoke_test_frozenlake_lvr.py --attention sdpa
```

Do not start training unless the result has `status: ok`, the latent and target
counts match, and all three losses are finite.

## 4. Run a short backward/update pilot

This executes 10 optimizer updates but avoids a large resumable optimizer
checkpoint. It validates DeepSpeed, CPU offload, backward, and optimizer update:

```bash
OUTPUT_DIR=/content/frozenlake_pilot \
MAX_STEPS=10 \
SAVE_STEPS=1000 \
EVAL_STEPS=1000 \
bash scripts/finetune_lvr_frozenlake_3b_colab.sh
```

For the SDPA fallback, add `DISABLE_FLASH_ATTN2=True` before the command.

Watch GPU memory from another cell:

```bash
watch -n 2 nvidia-smi
```

```python
import psutil

memory = psutil.virtual_memory()
print(f"RAM used: {(memory.total - memory.available) / 1024**3:.1f} GiB")
print(f"RAM free: {memory.available / 1024**3:.1f} GiB")
```

## 5. Run the full job

The Colab launcher uses one GPU, micro-batch 1, accumulation 16, ZeRO-3 CPU
offload, and keeps one resumable checkpoint:

```bash
OUTPUT_DIR=/content/frozenlake_checkpoints/qwen2.5-vl-3b \
bash scripts/finetune_lvr_frozenlake_3b_colab.sh
```

To use SDPA:

```bash
DISABLE_FLASH_ATTN2=True \
OUTPUT_DIR=/content/frozenlake_checkpoints/qwen2.5-vl-3b \
bash scripts/finetune_lvr_frozenlake_3b_colab.sh
```

The full default is five epochs. `MAX_STEPS=-1` means epochs determine the end
of training. Local `/content` storage is faster than mounted Google Drive, but
it disappears when the runtime is recycled; copy the retained checkpoint and
final model to persistent storage.

## 6. Resume after interruption

Copy the retained checkpoint back to local disk, then pass the exact
`checkpoint-N` directory:

```bash
OUTPUT_DIR=/content/frozenlake_checkpoints/qwen2.5-vl-3b \
CHECKPOINT_NAME=/content/frozenlake_checkpoints/qwen2.5-vl-3b/checkpoint-250 \
bash scripts/finetune_lvr_frozenlake_3b_colab.sh
```

The trainer loads model weights plus optimizer/scheduler state. The processor
is loaded from the checkpoint so the learned LVR token ids remain identical.

## 7. Validate before touching the test set

First evaluate 20 validation samples and tune only the latent-end threshold:

```bash
python evaluation/evaluate_frozenlake_lvr.py \
  --checkpoint /content/frozenlake_checkpoints/qwen2.5-vl-3b \
  --data-path data/frozenlake/validation.jsonl \
  --max-samples 20
```

Then evaluate all validation samples. Use the test split once after selecting
the threshold:

```bash
python evaluation/evaluate_frozenlake_lvr.py \
  --checkpoint /content/frozenlake_checkpoints/qwen2.5-vl-3b \
  --data-path data/frozenlake/test.jsonl
```
