# FrozenLake LVR training

This adaptation trains Qwen2.5-VL-3B with one visible initial state, ordered
successor-state images as latent targets, and an action-only final answer.

## Data semantics

For trajectory frames `frame_000 ... frame_N` and actions `a_0 ... a_(N-1)`:

- `frame_000` is the only image in the user prompt.
- `frame_(i+1)` supplies the visual embedding target for action `a_i`.
- The final goal frame is included as the last target.
- No textual chain of thought is supervised.

Each 256x256 target frame produces approximately 81 Qwen merged visual tokens.
The dataset therefore averages about 259 latent targets per sample. This direct
full-patch baseline avoids introducing an untrained resampler.

## Loss

The trainer uses:

```text
L = L_action_CE + 0.1 * L_transition_cosine + 0.1 * L_latent_end_MSE
```

The vision tower and merger are frozen so both the input and target visual
representations stay stable. The language model and learned latent-end vector
are optimized.

## GPU handoff

Install the pinned environment from `requirements.txt`, then run the non-mutating
one-sample forward check:

```bash
python scripts/smoke_test_frozenlake_lvr.py
```

It must report `status: ok`, equal latent/target counts, and finite CE, latent,
and mode-switch losses before training is launched.

Start the default two-GPU ZeRO-3 CPU-offload job:

```bash
bash scripts/finetune_lvr_frozenlake_3b.sh
```

Paths and GPU count can be overridden without editing the launcher:

```bash
NUM_GPUS=4 OUTPUT_DIR=/checkpoints/frozenlake \
  bash scripts/finetune_lvr_frozenlake_3b.sh
```

The default is full-parameter training. The public LVR code declares LoRA
arguments but does not attach PEFT adapters, so this entry point rejects
`--lora_enable True` rather than silently running an unintended full tune.

## Evaluation

First check a few held-out samples:

```bash
python evaluation/evaluate_frozenlake_lvr.py \
  --checkpoint /checkpoints/frozenlake/checkpoint-100 \
  --max-samples 20
```

Then remove `--max-samples` for all 200 test trajectories. The evaluator reports:

- exact agreement with the recorded action sequence;
- valid action-only answer format;
- whether the route reaches the goal without hitting a hole or leaving the board;
- whether the successful route is also shortest.

The latent decoder uses learned MSE stopping with a 2,048-step safety cap. Tune
`--lvr-end-threshold` on the validation split only; keep the test split untouched
until the threshold is selected.

To bypass learned mode switching and use the training-set average of 259 latent
tokens, select fixed decoding:

```bash
python evaluation/evaluate_frozenlake_lvr.py \
  --checkpoint /checkpoints/frozenlake/checkpoint-100 \
  --data-path data/frozenlake/test.jsonl \
  --sample-index 0 \
  --decoding-strategy fixed \
  --fixed-lvr-steps 259
```

At the fixed boundary the decoder inserts `<|lvr_latent_end|>` before returning
to ordinary text generation, matching the boundary context used during
training. The Colab notebook's fixed-length sample cell can run the same path
against either the train or test split.
