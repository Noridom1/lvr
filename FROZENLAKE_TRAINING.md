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
L = L_action_CE + 0.1 * L_transition_MSE
```

The vision tower and merger are frozen so both the input and target visual
representations stay stable. The standard multi-GPU launcher optimizes the
language model; the one-A100 Colab launcher uses LoRA as a constrained variant.

## GPU handoff

Install the pinned environment from `requirements.txt`, then run the non-mutating
one-sample forward check:

```bash
python scripts/smoke_test_frozenlake_lvr.py
```

It must report `status: ok`, equal latent/target counts, finite CE and MSE
reconstruction losses, and a null mode-switch loss before training is launched.

Start the default two-GPU ZeRO-3 CPU-offload job:

```bash
bash scripts/finetune_lvr_frozenlake_3b.sh
```

Paths and GPU count can be overridden without editing the launcher:

```bash
NUM_GPUS=4 OUTPUT_DIR=/checkpoints/frozenlake \
  bash scripts/finetune_lvr_frozenlake_3b.sh
```

The default multi-GPU job is full-parameter language-model training, matching
the paper. The Colab wrapper explicitly enables LoRA to fit a single A100; treat
those results as a resource-constrained variant rather than a paper replication.

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

The latent decoder always exits after a fixed budget and forcibly emits
`<|lvr_end|>`. Select a budget on validation with:

```bash
python evaluation/evaluate_frozenlake_lvr.py \
  --checkpoint /checkpoints/frozenlake \
  --data-path data/frozenlake/validation.jsonl \
  --sweep-lvr-steps 4 8 16 32 64 128 256 512
```

The sweep selects by goal success, shortest-path success, exact match, and then
the smaller budget. Pass the selected value through `--lvr-steps` for the single
final test evaluation.
