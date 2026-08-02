# LVR Codebase Guide

Implementation of **Latent Visual Reasoning (LVR)** — ICLR 2026. Instead of reasoning in text space ("Think about Images") or via external tools ("Think with Images"), LVR enables the LLM to reason directly in visual embedding space by generating hidden states that reconstruct query-relevant visual tokens.

---

## Directory Structure

```
lvr/
├── src/
│   ├── constants.py                    # Special token definitions
│   ├── params.py                       # Training hyperparameters (SFT + RL)
│   ├── params_vanilla.py               # Vanilla (non-LVR) training params
│   ├── utils.py                        # General utilities
│   ├── lvr_utils.py                    # Bounding box → visual token index mapping
│   ├── merge_lora_weights.py           # LoRA weight merging
│   ├── s3_checkpoints_lvr.py           # OCI/S3 checkpoint management
│   │
│   ├── model/
│   │   ├── qwen_lvr_model.py           # Main model class + 3 decoding strategies
│   │   └── lvr_heads.py                # LVR projection heads (Simple MLP + GLU)
│   │
│   ├── dataset/
│   │   ├── data_utils.py               # Image processing, LVR token replacement
│   │   ├── lvr_sft_dataset.py          # Stage 1 SFT dataset (with bbox annotations)
│   │   ├── lvr_sft_dataset_packed.py   # Stage 1 with efficient data packing
│   │   ├── grpo_dataset.py             # Stage 2 RL dataset (plain QA pairs)
│   │   ├── sft_dataset.py              # Standard SFT (non-LVR baseline)
│   │   └── dpo_dataset.py              # DPO dataset
│   │
│   ├── train/
│   │   ├── train_sft.py                # Stage 1 entry point
│   │   ├── train_grpo.py               # Stage 2 entry point
│   │   ├── train_lvr.py                # Alternative LVR training entry
│   │   ├── train_dpo.py                # DPO training entry
│   │   ├── reward_funcs.py             # RL reward functions
│   │   ├── monkey_patch_forward.py            # Base forward patch
│   │   ├── monkey_patch_forward_lvr.py        # Stage 1 forward with MSE loss
│   │   ├── monkey_patch_forward_lvr_rl.py     # Stage 2 forward with hidden state replay
│   │   ├── monkey_patch_dataloader.py         # DataLoader patches
│   │   ├── monkey_patch_patch_emb.py          # Vision encoder NaN fix
│   │   ├── train_utils.py              # Shared training utilities
│   │   └── helper_functions.py         # Misc helper functions
│   │
│   └── trainer/
│       ├── sft_trainer.py              # Stage 1 HuggingFace Trainer subclass
│       ├── lvr_trainer.py              # LVR-specific trainer
│       ├── grpo_trainer.py             # Stage 2 GRPO RL trainer
│       └── dpo_trainer.py              # DPO trainer
│
├── evaluation/
│   └── evaluation.py                   # Benchmark evaluation
│
└── scripts/
    ├── zero2.json                      # DeepSpeed ZeRO-2 config
    ├── zero2_offload.json
    ├── zero3.json                      # DeepSpeed ZeRO-3 config
    └── zero3_offload.json
```

---

## Core Concept

The key innovation is treating LLM hidden states as visual reconstructions. The standard Vision–Projector–LLM pipeline maps images into a joint semantic space with text. LVR exploits this shared space:

1. **Input**: Image tokens and question are embedded into a joint space `V_T` (visual) and `T` (text).
2. **LVR Mode**: When `<|lvr_start|>` is generated, the model no longer predicts discrete tokens. Instead, it generates **hidden states** that are fed back as input embeddings.
3. **Reconstruction Target**: These hidden states are trained to match the actual visual embeddings of query-relevant image patches (identified by bounding boxes in Stage 1).
4. **Output**: After `<|lvr_end|>`, the model resumes standard text generation, now conditioned on the reconstructed visual context.

---

## Special Tokens (`src/constants.py`)

```python
LVR_START_TOKEN = "<|lvr_start|>"       # Enters latent reasoning mode
LVR_END_TOKEN   = "<|lvr_end|>"         # Exits latent reasoning mode
LVR_TOKEN       = "<|lvr|>"             # Placeholder for each latent step
LVR_LATENT_END_TOKEN = "<|lvr_latent_end|>"  # Learnable end signal (Latent End strategy)
LVR_PLACEHOLDER = "<lvr>"              # In raw data before tokenization
```

---

## Training: Two Stages

### Stage 1 — Supervised Fine-Tuning (SFT)

**Goal**: Teach the model to reconstruct visual semantics for a given ROI via hidden state supervision.

**Entry point**: `src/train/train_sft.py`  
**Dataset**: `src/dataset/lvr_sft_dataset.py` (or `_packed.py`)  
**Forward patch**: `src/train/monkey_patch_forward_lvr.py`  
**Trainer**: `src/trainer/sft_trainer.py`  
**Training data**: VISUAL COT (438k QA pairs with bounding box annotations)

#### Data Format

```json
{
  "image": ["path/to/image.jpg"],
  "conversations": [
    {"from": "human", "value": "<image>\nWhat color is the hat?\nProvide a short response."},
    {"from": "gpt",   "value": "<lvr>\n<answer>Navy</answer>"}
  ],
  "bboxes": [[0.382, 0.456, 0.718, 0.656]]
}
```

The `<lvr>` placeholder in the response is replaced by `data_utils.py:replace_lvr_tokens()` with:
```
<|lvr_start|><|lvr|><|lvr|>...<|lvr|><|lvr_end|>
```
where the number of `<|lvr|>` tokens equals the number of visual patches in the bounding box.

#### Bounding Box → Token Indices (`src/lvr_utils.py`)

1. Create a binary RGB mask image from the bounding box coordinates.
2. Pass the mask through Qwen2.5-VL's image processor (same pipeline as the actual image).
3. Extract indices of non-zero positions → these are the flattened patch indices for the ROI.
4. Use these indices to select visual embeddings from the encoded image.

#### Forward Pass (Stage 1)

`monkey_patch_forward_lvr.py` replaces Qwen2.5-VL's forward method:

```
Input: image + question + <|lvr_start|><|lvr|>×N<|lvr_end|><answer>...

1. Encode full image → visual embeddings V_T (in joint space)
2. Find positions of <|lvr|> tokens in input_ids
3. Replace those embedding slots with the actual visual embeddings for the ROI patches
4. Run full forward pass through LLM
5. Collect hidden states at the <|lvr|> positions (these are the model's "predictions")
6. Compute MSE loss between predicted hidden states and target visual embeddings
7. Compute CE loss on <answer>...</answer> tokens
8. Total loss = CE_loss + λ_LVR × MSE_loss
```

#### Loss Functions

| Symbol | Formula | Config key |
|--------|---------|------------|
| L_LVR  | `(1/T_v) Σ ||h_t - v_t||²` | `loss_lvr_fct = "mse"` |
| L_NTP  | `-(1/T_y) Σ log p(y_t \| ...)` | standard CE |
| L      | `L_NTP + λ_LVR · L_LVR` | `loss_lvr_lambda = 1e-1` |

Alternative loss functions available: `"mae"` (L1), `"cosine"`.

#### Optional LVR Heads (`src/model/lvr_heads.py`)

By default (`lvr_head=False`), raw hidden states are directly compared to visual embeddings — this works because both live in the joint semantic space.

If `lvr_head=True`, a projection head maps hidden states before comparison:

- **Simple** (`lvr_head_type="simple"`): LayerNorm → Linear → GELU → Linear
- **GLU** (`lvr_head_type="glu"`): gated linear unit with 3× intermediate expansion

Ablation (Table 3) shows the standard no-head approach outperforms both heads.

---

### Stage 2 — Reinforcement Learning (GRPOlatent)

**Goal**: Allow the model to self-evolve the latent reasoning process, guided only by output correctness (no bounding box supervision).

**Entry point**: `src/train/train_grpo.py`  
**Dataset**: `src/dataset/grpo_dataset.py`  
**Forward patch**: `src/train/monkey_patch_forward_lvr_rl.py`  
**Trainer**: `src/trainer/grpo_trainer.py`  
**Training data**: ViRL (plain image–question–answer, no bounding boxes)

#### Data Format

```json
{
  "image": "ViRL39K/image.png",
  "conversations": [
    {"from": "human", "value": "What is the value of x?"},
    {"from": "gpt",   "value": "<answer>4</answer>"}
  ]
}
```

No `<lvr>` tokens in RL data — the model must learn to generate `<|lvr_start|>` on its own.

#### The GRPOlatent Problem

Standard GRPO computes per-token importance ratios:
```
r_{i,t}(θ) = π_θ(y_{i,t} | context) / π_{θ_old}(y_{i,t} | context)
```

But the "context" includes latent reasoning hidden states that have **no token distribution** — they are continuous vectors, not sampled from a vocabulary. Direct application of GRPO is impossible.

#### GRPOlatent Solution: Hidden State Replay

During the **rollout phase**: generate completions and **record** all LVR hidden states:
```python
h̃_i^latent = {h_{i,1}^latent, ..., h_{i,L}^latent}
```

During the **policy gradient phase**: perform teacher-forcing, but **patch in the recorded hidden states** at LVR positions instead of re-generating them:
```python
inputs_embeds = torch.where(
    lvr_mask.unsqueeze(-1),   # at LVR positions?
    lvr_states,                # use recorded h̃^latent
    token_embeddings           # else use normal embeddings
)
```

This gives consistent conditional log-probabilities for the text tokens, making the importance ratio well-defined. The GRPO loss is computed **only over text tokens**, not latent steps.

#### Reward Functions (`src/train/reward_funcs.py`)

| Reward | Value | Condition |
|--------|-------|-----------|
| Format | 1.0   | Response contains both `<\|lvr_start\|>` and `<\|lvr_end\|>` |
| Format | 0.0   | Missing either special token |
| Accuracy | 1.0 | Answer matches ground truth (symbolic verify or string match) |
| Accuracy | 0.0 | Wrong answer |

The format reward is crucial — without it, the model collapses to pure text generation and skips LVR.

#### Group-Normalized Reward (GRPO)

```
R̃_i = (R(y_i) - mean(R(y_1..G))) / std(R(y_1..G))
Â_{i,t} = R̃_i   for all t
```

---

## Decoding Strategies (`src/model/qwen_lvr_model.py`)

All strategies share the same loop structure: track `lvr_mode_switch` (bool tensor, one per batch), enter LVR mode on `<|lvr_start|>`, exit on stopping criterion.

### 1. Fixed Token (`decoding_strategy="steps"`) — Best Performance

```
Config: lvr_steps = 4 | 8 | 16
```

Generates exactly `lvr_steps` hidden states, then forces exit. Uses a per-instance countdown:
```python
lvr_remaining_steps -= lvr_mode_switch.long()   # decrement while in mode
lvr_mode_switch = ... & (lvr_remaining_steps > 0)  # exit at 0
```

### 2. Latent End Token (`decoding_strategy="latent"`) — Unstable

```
Config: lvr_end_threshold = 0.02, criterion = "mse" | "mae" | "cosine"
```

A learnable parameter `lvr_latent_end_emb` is initialized as a unit vector scaled by `√d`. At each step, compute distance from the current hidden state to this embedding; exit when distance falls below threshold.

In practice this is unreliable — cosine/L1/L2 distances all failed to find a stable threshold.

### 3. Mode Switching Loss (`decoding_strategy=None`) — Failed

During SFT, an auxiliary BCE loss trains the LM head to predict `<|lvr_end|>` at the final LVR position. Inference exits when the model naturally generates `<|lvr_end|>`. 

In practice this collapses to 0 LVR steps (model learns to immediately generate end token).

---

## Model Architecture (`src/model/qwen_lvr_model.py`)

Base: **Qwen2.5-VL** (3B or 7B). The class `QwenWithLVR` extends `Qwen2_5_VLForConditionalGeneration`.

Key additions:
- `lvr_latent_end_emb`: trainable `nn.Parameter` for Latent End strategy
- `_lvr_decoding_by_steps()`: Fixed Token generation loop
- `_lvr_decoding_with_latentend()`: Latent End generation loop
- `generate()` override: dispatches to correct decoding strategy

During training, both the visual encoder and the multimodal projector are **frozen**. Only the LLM backbone is updated. This reflects the hypothesis that the joint semantic space is already well-aligned.

---

## Key Training Hyperparameters (`src/params.py`)

### Stage 1 (SFT)
```python
loss_lvr_fct      = "mse"      # Reconstruction loss type
loss_lvr_lambda   = 1e-1       # Weight for LVR reconstruction loss
lvr_head          = False       # Use projection head (False = best)
lvr_head_type     = "simple"   # "simple" | "glu"
enable_data_packing = False    # Pack short instances for efficiency
learning_rate     = 1e-5
steps             = 2500        # ~40h on 4× AMD MI250 for 7B
```

### Stage 2 (RL)
```python
beta              = 0.04        # KL divergence coefficient
temperature       = 0.9         # Sampling temperature
num_generations   = 8           # Rollouts per input
max_completion_length = 256
decoding_strategy = "steps"
lvr_steps         = 16
learning_rate     = 1e-5
steps             = 1500        # ~20h on 4× AMD MI250 for 3B
```

---

## Data Processing (`src/dataset/data_utils.py`)

### `replace_lvr_tokens()`

Converts the raw `<lvr>` placeholder in responses into the actual token sequence:

```
Input:  "<lvr>\n<answer>Navy</answer>"
Output: "<|lvr_start|><|lvr|><|lvr|>...<|lvr|><|lvr_end|>\n<answer>Navy</answer>"
```

The number of `<|lvr|>` tokens is determined by how many visual patches fall within the annotated bounding box.

### Data Packing (`lvr_sft_dataset_packed.py`)

Addresses the imbalanced instance lengths (variable image resolution + variable LVR token count):
- Short instances are packed together within a single sequence up to `max_packed_tokens`
- Long instances are grouped in smaller batches
- Average effective batch size: ~3.2 instances per device

---

## Evaluation Benchmarks

| Benchmark | Task |
|-----------|------|
| V* Bench (D.A.) | Fine-grained visual detail search |
| V* Bench (R.P.) | Relative spatial reasoning |
| MMVP | Perception robustness under subtle perturbations |
| BLINK: Counting | Object enumeration |
| BLINK: JigSaw | Image reconstruction from fragments |
| BLINK: Relative Reflectance | Pixel-level albedo comparison |
| BLINK: Spatial Relation | Object position understanding |

Evaluation uses `lmms-eval` for standardized metrics.

---

## Infrastructure

### Multi-GPU Training
- DeepSpeed ZeRO-2 (`scripts/zero2.json`) — gradients + optimizer states sharded
- DeepSpeed ZeRO-3 (`scripts/zero3.json`) — additionally shards parameters
- `*_offload.json` variants add CPU offloading for larger models

### NaN Stability (`monkey_patch_patch_emb.py`)
Qwen2.5-VL's 3D convolution can produce NaN at high resolutions. Fixed by:
1. Converting patch embedding to fp32 before the convolution
2. Applying a NaN sanitizer hook during training

### Checkpoint Management (`s3_checkpoints_lvr.py`)
- OCI/S3 cloud checkpointing with automatic synchronization across distributed ranks
- Supports resume from checkpoint

---

## Results Summary (7B model, Table 1)

| Method | V* | V* D.A. | V* R.P. | MMVP |
|--------|----|---------|---------|------|
| Qwen2.5-VL (base) | 78.5 | 81.7 | 73.7 | 66.7 |
| LVR (4 steps) | 81.2 | 84.4 | 76.3 | **72.0** |
| LVR (8 steps) | **81.7** | **84.4** | 77.6 | 71.7 |
| LVR (16 steps) | 80.6 | 81.7 | **79.0** | 71.7 |

RL further boosts 3B results by ~1-2% across benchmarks (Table 2).

---

## Common Gotchas

1. **Stage 2 requires `transformers>=4.54.0`** due to abstract model API changes. The `requirements.txt` pins this.

2. **LVR tokens must not be "special" in Stage 2**: The model needs to be able to *generate* `<|lvr_start|>` freely. If registered as special tokens they may be blocked from being output.

3. **Format reward is non-negotiable in RL**: Removing the `<|lvr_start|>/<|lvr_end|>` format reward causes complete collapse to text-only responses.

4. **Visual encoder is always frozen**: Both stages freeze the vision tower and projector. Only the LLM is trained. This is a deliberate design choice — the joint semantic space is assumed fixed.

5. **The SFT stage acts as teacher-forcing**: The reconstruction target is the actual visual embedding for the ROI, injected directly into the model during the forward pass. The model's hidden states are then compared to these injected embeddings via MSE.
