# Forward Pass Monkey Patching

The original `Qwen2_5_VLForConditionalGeneration.forward` from HuggingFace knows nothing about LVR tokens, MSE reconstruction loss, or hidden-state replay. Both training stages replace this method on the **class** before any model instance is created, so every instance inherits the patched behavior automatically.

```python
# The one-line change that both dispatchers ultimately execute:
transformers.models.qwen2_5_vl.modeling_qwen2_5_vl \
    .Qwen2_5_VLForConditionalGeneration.forward = <patched_function>
```

---

## Stage 1 — SFT Forward

**Dispatcher**: [monkey_patch_forward_lvr.py:34](../src/train/monkey_patch_forward_lvr.py#L34)  
**Main forward function**: [monkey_patch_forward_lvr.py:118](../src/train/monkey_patch_forward_lvr.py#L118) — `qwen2_5_mixed_modality_forward_lvr`

The dispatcher selects from several variants depending on flags (`lvr_head`, `mode_switch_loss`, `latent_end_token`). The default production path — no head, no auxiliary losses — installs `qwen2_5_mixed_modality_forward_lvr` at line 67.

### What it changes vs. the original

The original forward does:
1. Embed `input_ids` → `inputs_embeds`
2. If `pixel_values` is given, encode image and scatter visual embeddings into `<|image_pad|>` positions
3. Run LLM
4. Compute CE loss on all unmasked tokens

The patched forward does all of that, plus the following additions:

---

#### Addition 1 — Recurrent hidden-state feedback (inference only)

[monkey_patch_forward_lvr.py:161](../src/train/monkey_patch_forward_lvr.py#L161)

When the decoding loop is in LVR mode (`last_position_hidden_state is not None`), the last embedding slot is replaced with the **previous step's hidden state** before the LLM sees it:

```python
inputs_embeds[lvr_mode_switch, -1, :] = last_position_hidden_state[lvr_mode_switch]
```

This is what makes the latent reasoning loop recurrent: each step feeds its own output back as the next input, rather than using a discrete token embedding.

---

#### Addition 2 — Dummy image for DeepSpeed stability (training only)

[monkey_patch_forward_lvr.py:169](../src/train/monkey_patch_forward_lvr.py#L169)

In distributed training, some batches may be text-only (no image). DeepSpeed ZeRO requires all parameters to participate in every forward pass, but if `pixel_values is None`, the vision encoder is never called and its gradients are skipped — causing a hang. The fix runs a dummy `28×28` black image through the vision encoder and adds `image_embeds.mean() * 0` to the embeddings. The value is exactly zero, so it has no effect on the output, but all vision encoder parameters participate in the computation graph.

---

#### Addition 3 — Inject ROI visual embeddings at `<|lvr|>` positions (training only)

[monkey_patch_forward_lvr.py:211](../src/train/monkey_patch_forward_lvr.py#L211)

This is the core of Stage 1 teacher-forcing. After scattering image embeddings into `<|image_pad|>` slots, it also replaces the `<|lvr|>` token embedding slots with the actual visual embeddings from the bounding-box ROI:

```python
# lvr_tokens = list of patch indices (from bbox → patch index mapping)
selected_lvr_embeds = image_embeds[global_lvr_token_indices]   # (L_total, H)
inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
```

The key idea: the model does not need to "figure out" what visual content goes at LVR positions — the ground-truth visual embeddings are injected as inputs. The model's job is to produce hidden states at those positions that **match** those injected embeddings.

See [monkey_patch_forward_lvr.py:216–244](../src/train/monkey_patch_forward_lvr.py#L216) for the full index offset calculation that converts per-image local patch indices into global indices into the concatenated `image_embeds` tensor.

---

#### Addition 4 — Dual loss: MSE + CE

[monkey_patch_forward_lvr.py:325](../src/train/monkey_patch_forward_lvr.py#L325) (CE) and [monkey_patch_forward_lvr.py:334](../src/train/monkey_patch_forward_lvr.py#L334) (LVR)

After the LLM runs, two losses are computed:

**CE loss** — standard cross-entropy on answer tokens. `<|lvr|>` positions are explicitly excluded by masking them with `IGNORE_INDEX`:
```python
shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)
loss_ce = CrossEntropyLoss()(shift_logits, shift_labels)
```

**LVR (MSE) loss** — the hidden states at the position *before* each `<|lvr|>` token (i.e., the hidden state that "predicted" that LVR position) are compared against the injected visual embeddings:
```python
seq_positions_start = seq_positions - 1   # shifts to the predicting position
selected_hidden_states = hidden_states[batch_indices, seq_positions_start]
loss_lvr = MSELoss()(selected_hidden_states, selected_lvr_embeds)
```

The trainer combines them: `loss = loss_ce + λ * loss_lvr` where `λ = loss_lvr_lambda` (default `0.1`).

Both losses are returned in the custom `Qwen2_5_VLCausalLMOutputWithPast` dataclass defined at [monkey_patch_forward_lvr.py:75](../src/train/monkey_patch_forward_lvr.py#L75), which extends the standard output with `loss_lvr`, `loss_ce`, `loss_mode_switch`, and `last_position_hidden_state` fields.

---

### Stage 1 Forward — Full Data Flow

```
input_ids + pixel_values + lvr_tokens (bbox patch indices)
        │
        ▼
embed input_ids → inputs_embeds
        │
        ▼ (if pixel_values)
encode image → image_embeds
scatter image_embeds into <|image_pad|> slots of inputs_embeds
        │
        ▼ (if lvr_tokens, training only)
select ROI patches from image_embeds using bbox indices
overwrite <|lvr|> slots in inputs_embeds with selected_lvr_embeds
        │
        ▼
run LLM → hidden_states, logits
        │
        ├── CE loss on answer tokens (excl. <|lvr|> positions)
        │
        └── MSE loss: hidden_states at (lvr_pos - 1) vs. selected_lvr_embeds
```

---

## Stage 2 — RL (GRPO) Forward

**Dispatcher**: [monkey_patch_forward_lvr_rl.py:34](../src/train/monkey_patch_forward_lvr_rl.py#L34)  
**Main forward function**: [monkey_patch_forward_lvr_rl.py:90](../src/train/monkey_patch_forward_lvr_rl.py#L90) — `qwen2_5_mixed_modality_forward_lvr_grpo`

Stage 2 has no bounding-box supervision. The model must learn to generate `<|lvr_start|>` on its own and is rewarded only for correct answers. This creates the GRPOlatent problem: GRPO needs per-token log-probabilities, but LVR positions are **continuous hidden states with no token distribution**.

The solution is hidden-state replay, implemented in this forward function.

### New parameters (not present in Stage 1)

| Parameter | Type | Purpose |
|---|---|---|
| `lvr_mask` | `FloatTensor [B, C]` | Boolean mask over completion positions — True where LVR steps are |
| `lvr_states` | `FloatTensor [B, C, H]` | Hidden states recorded during rollout, one per LVR position |
| `prompt_length` | `int` | Length of the prompt portion (so completion can be sliced out) |

These are only passed during the **policy gradient update step**, not during rollout generation.

### What it changes vs. the original

---

#### Addition 1 — Recurrent hidden-state feedback (inference, same as Stage 1)

[monkey_patch_forward_lvr_rl.py:129](../src/train/monkey_patch_forward_lvr_rl.py#L129)

Identical to Stage 1: replaces the last embedding slot with `last_position_hidden_state` when in LVR decoding mode.

---

#### Addition 2 — Hidden-state replay (training / policy gradient only)

[monkey_patch_forward_lvr_rl.py:135](../src/train/monkey_patch_forward_lvr_rl.py#L135)

This is the core of GRPOlatent. After building `inputs_embeds` from token embeddings, it replaces the LVR positions in the **completion** with the hidden states that were recorded during the rollout:

```python
comp_embeds = inputs_embeds[:, prompt_length:, :]     # slice: completion only (B, C, H)
comp_embeds = torch.where(
    lvr_mask.unsqueeze(-1),   # (B, C, 1) — True at LVR positions
    lvr_states,               # (B, C, H) — hidden states from rollout
    comp_embeds               # (B, C, H) — normal token embeddings elsewhere
)
inputs_embeds = torch.cat([inputs_embeds[:, :prompt_length, :], comp_embeds], dim=1)
```

After this, `inputs_embeds` has:
- Normal token embeddings at all prompt positions
- Normal token embeddings at all text token positions within the completion
- **Rollout hidden states** at all LVR positions within the completion

The LLM runs a single forward pass over this combined sequence. The log-probabilities of the **text tokens** (answer, `<|lvr_start|>`, `<|lvr_end|>`) are now well-defined and can be used for the GRPO importance ratio. The LVR positions themselves are excluded from the loss.

---

#### Addition 3 — CE loss only, no LVR loss

[monkey_patch_forward_lvr_rl.py:230](../src/train/monkey_patch_forward_lvr_rl.py#L230)

```python
loss_lvr = None   # no reconstruction target in Stage 2
```

There are no bounding boxes and no visual embedding targets in the RL data. Only `loss_ce` is computed (same exclusion of `<|lvr|>` positions as Stage 1). The GRPO trainer wraps this CE loss with the group-normalized advantage to form the final policy gradient loss.

---

### Stage 2 Forward — Full Data Flow

```
                 ┌── ROLLOUT PHASE ──────────────────────────────────────────┐
                 │  generate completions autoregressively                     │
                 │  at each LVR step: record hidden_state → lvr_states       │
                 └───────────────────────────────────────────────────────────┘
                                           │
                                    store lvr_states
                                           │
                 ┌── POLICY GRADIENT PHASE ──────────────────────────────────┐
                 │                                                            │
input_ids + pixel_values                                                      │
        │                                                                     │
        ▼                                                                     │
embed input_ids → inputs_embeds                                               │
        │                                                                     │
        ▼ (if pixel_values)                                                   │
encode image → image_embeds                                                   │
scatter image_embeds into <|image_pad|> slots                                 │
        │                                                                     │
        ▼ (if lvr_states + lvr_mask — policy gradient step only)              │
slice out completion: comp_embeds = inputs_embeds[:, prompt_length:, :]       │
patch LVR positions: comp_embeds[lvr_mask] = lvr_states                       │
reassemble: inputs_embeds = [prompt_embeds | patched_comp_embeds]             │
        │                                                                     │
        ▼                                                                     │
run LLM → hidden_states, logits                                               │
        │                                                                     │
        └── CE loss on text tokens only (excl. <|lvr|> positions)            │
                                                                              │
                 └───────────────────────────────────────────────────────────┘
                 GRPO trainer uses CE log-probs + group-normalized reward
                 to compute importance ratio and policy gradient update
```

---

## Key Differences Between the Two Patched Forwards

| | Stage 1 SFT | Stage 2 RL |
|---|---|---|
| **File** | [monkey_patch_forward_lvr.py](../src/train/monkey_patch_forward_lvr.py) | [monkey_patch_forward_lvr_rl.py](../src/train/monkey_patch_forward_lvr_rl.py) |
| **Inject visual embeds at `<lvr>` positions** | Yes — from bbox ROI | No |
| **Hidden-state replay** | No | Yes — from rollout recording |
| **LVR loss (MSE)** | Yes | No |
| **CE loss** | Yes (excl. `<lvr>` positions) | Yes (excl. `<lvr>` positions) |
| **Recurrent inference loop** | Yes | Yes |
| **Dummy image (DeepSpeed)** | Yes | No |
| **Transformers version** | `>=4.54.0` (uses `self.model.language_model`) | `<4.54.0` (uses `self.model`) |

> **Note on the API difference**: The two files were written against different `transformers` versions. Stage 1 calls `self.model.language_model(...)` to invoke the LLM backbone; Stage 2 calls `self.model(...)`. The comment in [monkey_patch_forward_lvr_rl.py:87](../src/train/monkey_patch_forward_lvr_rl.py#L87) acknowledges this and flags it for cleanup before final release.
