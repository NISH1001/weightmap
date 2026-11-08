"""Tests for the weightmap library, using tiny randomly-initialised Qwen2 models on CPU (fp32).

Only the (cached) Qwen tokenizer is needed; no model weights are downloaded.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from weightmap import WeightMap, align, linalg

TOK_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TEXTS = [
    "The quick brown fox jumps over the lazy dog while the cat watches from the window sill.",
    "Solve for x: 3x + 7 = 22. Subtract seven from both sides, then divide by three to get five.",
    "Paris is the capital of France, and Berlin is the capital of Germany in central Europe.",
    "In 1969 the first humans landed on the moon, an event watched by millions around the world.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))  # prints five on the console",
    "Photosynthesis converts light energy into chemical energy stored inside glucose molecules.",
    "The derivative of x squared is two x, and the integral of two x is x squared plus a constant.",
    "She opened the old book and found a pressed flower between the yellowed and brittle pages.",
] * 3


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOK_NAME)


def tiny(tok, hidden, layers, seed):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def small(tok):
    return tiny(tok, 64, 3, seed=0)


@pytest.fixture(scope="module")
def large(tok):
    return tiny(tok, 96, 4, seed=1)


FIT = dict(positions_per_text=24, batch_size=4, max_length=64, cka_texts=12)


def test_self_map_is_near_perfect(tok, small):
    """Mapping a model onto itself must recover identity: R^2 ~ 1, diagonal layer match,
    and the shuffled-pairing control must be far worse."""
    wm = WeightMap(large=(small, tok), small=(small, tok), device="cpu").fit(TEXTS, **FIT)
    for s, l in wm.layer_match.items():
        assert l == s
    for row in wm.report():
        assert row["r2_large_to_small"] > 0.99
        assert row["r2_shuffled_control"] < row["r2_large_to_small"] - 0.3


def test_shapes_with_different_widths(tok, small, large):
    wm = WeightMap(large=(large, tok), small=(small, tok), device="cpu").fit(TEXTS, **FIT)
    assert wm.cka.shape == (2, 3)            # final (normed) layer excluded on both sides
    s = wm.mapped_layers[0]
    hL = torch.randn(5, 96)
    assert wm.to_small(hL, s).shape == (5, 64)
    assert wm.to_large(torch.randn(5, 64), s).shape == (5, 96)
    assert wm.transport(torch.randn(96), s).shape == (64,)
    keys = {"small_layer", "large_layer", "cka", "r2_large_to_small", "r2_small_to_large",
            "r2_shuffled_control", "r2_small_prev_layer"}
    assert keys <= set(wm.report()[0])


def test_steer_zero_is_noop_and_nonzero_changes(tok, small):
    wm = WeightMap(large=(small, tok), small=(small, tok), device="cpu")
    enc = tok(["hello world, this is a test"], return_tensors="pt")
    with torch.no_grad():
        base = small(**enc).logits
        with wm.steer(torch.zeros(64), layer=1):
            z = small(**enc).logits
        with wm.steer(torch.randn(64) * 5, layer=1):
            r = small(**enc).logits
        after = small(**enc).logits
    assert torch.equal(base, z)
    assert not torch.allclose(base, r)
    assert torch.equal(base, after)          # hook removed on exit


def test_fold_matches_steer_and_undo_restores(tok, small):
    wm = WeightMap(large=(small, tok), small=(small, tok), device="cpu")
    enc = tok(["fold this vector into the weights"], return_tensors="pt")
    v = torch.randn(64)
    with torch.no_grad():
        base = small(**enc).logits
        with wm.steer(v, layer=1):
            ref = small(**enc).logits
        handle = wm.fold(v, layer=1)
        got = small(**enc).logits
        handle.undo()
        restored = small(**enc).logits
    assert (ref - got).abs().max() < 1e-4
    assert torch.equal(base, restored)


def test_transplant_beta_zero_equals_plain_generation(tok, small, large):
    wm = WeightMap(large=(large, tok), small=(small, tok), device="cpu").fit(TEXTS, **FIT)
    prompts = ["The capital of France is", "Two plus two equals"]
    plain = wm.generate(prompts, max_new_tokens=6)
    t0 = wm.generate(prompts, max_new_tokens=6, transplant_layer=wm.mapped_layers[0], beta=0.0)
    t1 = wm.generate(prompts, max_new_tokens=6, transplant_layer=wm.mapped_layers[0], beta=1.0)
    assert plain == t0
    assert len(t1) == 2


def test_contrast_direction_shape_and_sign(tok, small):
    wm = WeightMap(large=(small, tok), small=(small, tok), device="cpu")
    pos, neg = TEXTS[:4], TEXTS[4:8]
    d = wm.contrast_direction(pos, neg, layer=1, which="small")
    d_rev = wm.contrast_direction(neg, pos, layer=1, which="small")
    assert d.shape == (64,)
    assert torch.allclose(d, -d_rev, atol=1e-5)


def test_save_load_roundtrip(tok, small, large, tmp_path):
    wm = WeightMap(large=(large, tok), small=(small, tok), device="cpu").fit(TEXTS, **FIT)
    p = tmp_path / "map.pt"
    wm.save(p)
    wm2 = WeightMap.load(p, large=(large, tok), small=(small, tok), device="cpu")
    s = wm.mapped_layers[0]
    h = torch.randn(3, 96)
    assert torch.allclose(wm.to_small(h, s), wm2.to_small(h, s))
    assert wm2.layer_match == wm.layer_match


@pytest.fixture(scope="module")
def gpt2_tok():
    return AutoTokenizer.from_pretrained("gpt2")


def test_alignment_same_tokenizer_pairs_every_position(tok):
    texts = ["Paris is the capital of France.", "def f(x): return x * 2"]
    enc, offs = align.tokenize(tok, texts)
    for j in range(len(texts)):
        m = enc["attention_mask"][j].bool()
        pairs = align.boundary_pairs(offs[j], m, offs[j], m)
        n_real = int(m.sum()) - 1                      # first token excluded
        assert len(pairs) == n_real
        assert all(a == b for a, b in pairs)


def test_alignment_cross_tokenizer_pairs_share_prefix(tok, gpt2_tok):
    """Every aligned pair must end at the same character, i.e. both models read the same prefix."""
    text = "Photosynthesis converts light energy into chemical energy, naturally."
    enc_a, offs_a = align.tokenize(tok, [text])
    enc_b, offs_b = align.tokenize(gpt2_tok, [text])
    pairs = align.boundary_pairs(offs_a[0], enc_a["attention_mask"][0].bool(),
                                 offs_b[0], enc_b["attention_mask"][0].bool())
    assert len(pairs) >= 5
    for ia, ib in pairs:
        assert int(offs_a[0][ia, 1]) == int(offs_b[0][ib, 1])
    # the decoded prefixes up to each aligned token are the same text
    ia, ib = pairs[-1]
    pa = tok.decode(enc_a["input_ids"][0][: ia + 1], skip_special_tokens=True)
    pb = gpt2_tok.decode(enc_b["input_ids"][0][: ib + 1], skip_special_tokens=True)
    assert pa.strip() == pb.strip()


def test_cross_tokenizer_map_fits_and_generates(tok, gpt2_tok, small):
    large_gpt2 = tiny(gpt2_tok, 96, 4, seed=2)
    wm = WeightMap(large=(large_gpt2, gpt2_tok), small=(small, tok), device="cpu").fit(TEXTS, **FIT)
    assert not wm.shared_vocab
    rows = wm.report()
    assert rows and all(r["n_tokens"] > 50 for r in rows)
    out = wm.generate(["The capital of France is"], max_new_tokens=4,
                      transplant_layer=wm.mapped_layers[0], beta=0.5)
    assert len(out) == 1


def test_contrast_direction_start_chars(tok, small):
    """start_chars restricts averaging to the completion after a shared prompt."""
    wm = WeightMap(large=(small, tok), small=(small, tok), device="cpu")
    prompt = "Question: what is two plus two? Answer:"
    good, bad = [prompt + " four, clearly."], [prompt + " seven, obviously."]
    d = wm.contrast_direction(good, bad, layer=1, start_chars=[len(prompt)])
    d_all = wm.contrast_direction(good, bad, layer=1)
    assert d.shape == (64,) and not torch.allclose(d, d_all)


def test_unfitted_map_raises(tok, small):
    wm = WeightMap(large=(small, tok), small=(small, tok), device="cpu")
    with pytest.raises(RuntimeError, match="fit"):
        wm.to_small(torch.randn(2, 64), 0)


def test_text_holdout_never_splits_a_text():
    groups = torch.tensor([0] * 5 + [1] * 7 + [2] * 3 + [3] * 9 + [4] * 4)
    tr, te = linalg.split(len(groups), 0.4, seed=0, groups=groups)
    assert set(groups[tr].tolist()).isdisjoint(groups[te].tolist())
    assert len(tr) + len(te) == len(groups) and len(te) > 0


@pytest.mark.parametrize("pair_by", ["r2", "cka"])
def test_self_map_pairs_diagonally_under_both_pairings(tok, small, pair_by):
    wm = WeightMap(large=(small, tok), small=(small, tok), device="cpu").fit(TEXTS, pair_by=pair_by, **FIT)
    assert all(l == s for s, l in wm.layer_match.items())
    assert all(r["holdout_by"] == "text" and r["pair_by"] == pair_by for r in wm.report())


def test_token_holdout_option_still_available(tok, small, large):
    wm = WeightMap(large=(large, tok), small=(small, tok), device="cpu").fit(TEXTS, holdout_by="token", **FIT)
    assert all(r["holdout_by"] == "token" for r in wm.report())
