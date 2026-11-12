# weightmap

**Can a larger language model's representations be mapped into a smaller model, without training, so the small model gains abilities only the large one has?**

`weightmap` is a small library for building and testing that kind of map, plus the experiments that ran it. It finds the layer correspondence between two causal LMs, fits closed-form linear maps between their hidden states, and lets you translate, steer, transplant, or fold edits into the small model's weights. No gradient descent at any point.

> Naming note: despite the name, the map lives in **activation (representation) space**: hidden states at aligned tokens. Comparing raw weight spaces across different widths isn't meaningful, because of the dimension mismatch and the permutation and rotation symmetries. The one true weight edit is `fold()`, which writes a steering vector into the small model's weights.

## Hypothesis

This was inspired by [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) (MIT, Nov 2026), in which a frontier LLM edits small networks' weights directly.

| | Claim |
|---|---|
| **H1** | Models of different sizes occupy related *sets* (regions) of a shared space. They aren't identical points, but they are structurally related. |
| **H2** | There is a map **F** between the small model's space (S) and the large model's (L). |
| **H3** | S is effectively *contained* in L. |
| **H4** | F carries capability-relevant information *from L*: it can steer or edit S toward problems only L solves, **without training**. |
| **H5** | It is recursive: tiny → small → large compose. |

## Conclusion so far (experiment 1: Qwen2.5-0.5B ↔ 1.5B, MATH-500 + NQ-Open)

| | Verdict | Evidence |
|---|---|---|
| Much of L's advantage is latent in S | **Supported** | 62% of MATH problems only L solves are solved by S at least once in 8 samples |
| H1: related sets | **Supported** | Layer correspondence by CKA is monotone, CKA ≈ 0.92 |
| H2: a map exists | **Supported** | A linear map predicts **~66%** of S's hidden-state variance from L's on held-out texts¹ (shuffled control −0.15; S's own adjacent layer: 0.86–0.91). It holds across model families too: SmolLM2-360M ← Qwen-1.5B gives 0.66–0.68. S still works when its states are replaced by mapped L states |
| H3: S inside L | Weak to moderate | R² L→S beats S→L by 0.04 within Qwen and 0.07–0.12 across families (SmolLM2 ← Qwen) |
| H4: the map transfers ability | **Not supported** | L-derived edits: MATH −0.9 points [95% CI −7, +5], NQ −3.7 [−6, −1]. 0/9 "absent" problems fixed |
| f is linear | **Yes** | Adding non-linear (random-feature) terms doesn't improve the map (−0.03) |
| S looks like a *pruned* L | **Supported** | Picking 896 of L's 1,536 coordinates predicts S as well as the full map (0.641 vs 0.638); random picks lose 0.06 |
| A map between the *weights* (embedding tables) | Weak | R² 0.31 within Qwen, 0.42 SmolLM2←Qwen, about half the hidden-state map |
| H1 refined: convergence tracks capability | **Supported** | Across 5 small models from 4 families, the more capable the small model, the more of Qwen-1.5B it captures (ρ = 0.9) |
| The map as a *diagnostic* tool | **Works, no advantage** | Qwen's "looping" direction, mapped into SmolLM2-135M, cuts its loops 21% → 10% (random: 37%). S's own direction does as well (6.5%), and accuracy doesn't change. A failure direction transfers, but no new ability does |
| H5: recursion | **Supported for compression** | 1.5B → 0.5B → 360M → 135M: chaining the maps keeps 95.5% of a direct map, and errors don't compound. The "small → large adds ability" direction is still unsupported |

¹ Measured on answer text with whole texts held out. Experiment 1 originally reported 0.85 because it fitted and scored on chat-formatted text, where the system prompt and template tokens repeat in every example and are trivially easy to map. Holding out whole texts barely changes this (0.85 → 0.83). Removing the boilerplate is what matters (0.83 → 0.66). See *Measurement check* in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#56-measurement-check-text-level-holdout-template-tokens-layer-pairing).

**In one line:** a small model looks like a compressed, partly *pruned* view of a large one. About two-thirds of it is a linear readout of a subset of the large model's coordinates, even across model families. But the map is a lossy projection, so pushing L's knowledge through it doesn't make S smarter. The extra ability is exactly what the projection discards. This matches the blog's finding that edits unlock existing abilities but don't add new ones.

![main result](docs/figures/fig4_main_result.png)

![recursion](docs/figures/fig7_recursion.png)

Full details, every table, the figures, and the threats to validity are in **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

## Install

```bash
uv sync                      # library only (torch, transformers, numpy)
uv sync --extra experiments  # + datasets, math-verify, matplotlib for scripts/
```

## Usage: finding and using the L ↔ S map

The workflow has five steps: **discover** the map, **check** that it is real, **translate** between the models, **test** whether L helps S on your task, and **keep** what works.

### 1. Discover the map

```python
from weightmap import WeightMap

# hub ids, local paths, loaded models, or (model, tokenizer) pairs
wm = WeightMap(large="Qwen/Qwen2.5-1.5B-Instruct", small="Qwen/Qwen2.5-0.5B-Instruct")

# any text both models can read; in-domain text gives better maps
wm.fit(texts, layers=[10, 14])
```

`fit` does two passes:
1. It runs both models on the same texts, computes linear CKA between **every** small layer and **every** large layer, and pairs each small layer with its most similar large layer (`wm.layer_match`, `wm.cka`). Pass `pair_by="r2"` to pair by held-out predictability instead. It is slower and, so far, gives the same map quality.
2. For the requested layers, it fits closed-form ridge maps **L→S** and **S→L** on hidden states at aligned tokens, and scores them on **held-out texts** (`holdout_by="text"`, the default; `"token"` is available but optimistic).

Nothing is trained.

### 2. Check that the map is real

```python
for row in wm.report():
    print(row)
# {'small_layer': 10, 'large_layer': 13, 'cka': 0.913, 'r2_large_to_small': 0.664,
#  'r2_small_to_large': 0.621, 'r2_shuffled_control': -0.154, 'r2_small_prev_layer': 0.914,
#  'holdout_by': 'text', 'pair_by': 'cka', ...}
```

(Real output: Qwen2.5-0.5B ↔ 1.5B, fitted on 520 MATH/NQ answer texts. Fit on content text rather than chat-templated prompts: repeated boilerplate tokens inflate R².)

| Field | Meaning | What to look for |
|---|---|---|
| `large_layer` | L layer that best matches this S layer (CKA) | Rises with `small_layer`, meaning the models process in the same order |
| `cka` | Linear CKA between the paired layers | High (> 0.8), but inflated by token identity, so don't rely on it alone |
| `r2_large_to_small` | Held-out R² predicting S's states from L's | The main evidence: how much of S is linearly recoverable from L |
| `r2_small_to_large` | The reverse | If clearly lower than L→S, S looks "contained in" L |
| `r2_shuffled_control` | The same fit with token pairing broken | Should be ≤ 0. If it isn't, the map isn't using token correspondence |
| `r2_small_prev_layer` | S's layer ℓ−1 predicting layer ℓ | A ceiling for "as related as adjacent layers of one model" |

To see the full layer correspondence:

```python
import matplotlib.pyplot as plt
plt.imshow(wm.cka.T, origin="lower", aspect="auto"); plt.xlabel("small layer"); plt.ylabel("large layer")
```

### 3. Translate between the models

```python
h_small_hat = wm.to_small(h_large, layer=10)   # L's states (at L layer wm.layer_match[10]) -> S space
h_large_hat = wm.to_large(h_small, layer=10)   # S -> L
d_small     = wm.transport(d_large, layer=10)  # a *direction* (difference vector): linear part only
```

### 4. Test whether L helps S on your task

This is the experiment-1 protocol in a few lines. Use problems that **L gets right and S gets wrong**. Build a "correct vs. wrong" direction in **L's** space, transport it into S, and compare it with S's own direction and a random one.

```python
import torch
# prompts / l_correct / s_wrong / test_prompts come from your task; accuracy() is your grader
pos = [p + l_answer for p, l_answer in zip(prompts, l_correct)]   # L's correct answers
neg = [p + s_answer for p, s_answer in zip(prompts, s_wrong)]     # S's wrong answers
start = [len(p) for p in prompts]                                 # average over the answer only

lay = 10
d_L = wm.transport(wm.contrast_direction(pos, neg, wm.layer_match[lay], which="large", start_chars=start), lay)
d_S = wm.contrast_direction(pos, neg, lay, which="small", start_chars=start)
scale = d_S.norm()                                                 # same size for every method -> fair

for name, d in {"L->S": d_L, "S-native": d_S, "random": torch.randn_like(d_S)}.items():
    outs = wm.generate(test_prompts, steer=(d / d.norm() * scale * 2, lay))
    print(name, accuracy(outs))

# per-problem alternative: mix L's mapped prompt states straight into S (needs L at inference time)
outs = wm.generate(test_prompts, transplant_layer=14, beta=0.5)
```

If `L->S` beats `S-native` and `random` on held-out problems, the map carries information from L that S lacks. In experiment 1 it didn't (see the conclusion above).

### 5. Keep what works

```python
handle = wm.fold(d_L / d_L.norm() * scale * 2, layer=lay)   # write the steering vector into S's weights
wm.small.save_pretrained("small-edited")                    # an ordinary HF checkpoint
handle.undo()                                               # or restore the original weights

wm.save("map.pt")                                           # reuse a fitted map later
wm = WeightMap.load("map.pt", large="Qwen/Qwen2.5-1.5B-Instruct", small="Qwen/Qwen2.5-0.5B-Instruct")
```

### Different tokenizers (cross-family)

```python
wm = WeightMap(large="Qwen/Qwen2.5-1.5B-Instruct", small="HuggingFaceTB/SmolLM2-360M")
wm.shared_vocab   # False: tokens are aligned automatically
wm.fit(texts)
```

Tokens are paired where both tokenizers have a token **ending at the same character**, which is exactly where both causal models have read the same text prefix. With identical tokenizers, every position is paired. Both tokenizers must be "fast" tokenizers, since character offsets are required.

**Supported models:** HF decoder-only causal LMs with a `layers` / `h` / `blocks` module list. `fold()` needs a Llama/Qwen-style `mlp.down_proj`. On Apple Silicon, cap GPU memory with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3`.

## Layout

```
weightmap/              the library
  core.py               WeightMap: fit / report / to_small / transport / steer / generate / fold / save
  align.py              cross-tokenizer alignment by shared character boundaries
  linalg.py             linear CKA, closed-form ridge, R^2
  models.py             model resolution, hidden states, layer hooks
tests/                  pytest suite (tiny random models, CPU, ~5 s)
scripts/                one-off pipelines and checks
  exp1_qwen_0.5b_1.5b/  experiment 1: stages s1–s5, figures, cached results
  validate_library.py   library vs. experiment 1 + cross-tokenizer sanity run
docs/
  EXPERIMENTS.md        full write-up
  figures/              paper figures (PDF + PNG, color-blind safe)
```

## Tests

```bash
uv run pytest -q
```

## Status

Research code. Experiment 1 is small: about 22 target problems per task, one model family, one simple kind of edit. See *Threats to validity* in the write-up. Next steps planned there: targeted failure-direction edits (the blog's method), recursion with Qwen2.5-3B, and cross-family maps.
