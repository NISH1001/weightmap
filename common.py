"""Shared utilities: models, tasks (MATH reasoning, NQ-Open facts), grading, batched generation, hooks.

Nothing in this project trains or fine-tunes a model. Every step is forward passes, sampling,
or closed-form linear algebra.
"""
import json, random, re, string, time
from contextlib import contextmanager
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16
S_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # small model S (24 layers, d=896)
L_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # large model L (28 layers, d=1536)
ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def load(name):
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE).to(DEV).eval()
    return tok, model


# ---------------------------------------------------------------- tasks
def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if depth == 0:
            return s[j + 1:k]
    return None


MATH_SYS = "Please reason step by step, and put your final answer within \\boxed{}."
NQ_SYS = "Answer the question with a short factual answer (a few words). Do not explain."
MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]


def task_data(task, split, n=None, seed=0):
    if task == "math" and split == "test":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [{"q": r["problem"], "gold": r["answer"]} for r in ds]
    elif task == "math" and split == "train":
        rows = []
        for c in MATH_CONFIGS:
            for r in load_dataset("EleutherAI/hendrycks_math", c, split="train"):
                g = last_boxed(r["solution"])
                if g is not None:
                    rows.append({"q": r["problem"], "gold": g})
    elif task == "nq":
        ds = load_dataset("google-research-datasets/nq_open", split="validation" if split == "test" else "train")
        rows = [{"q": r["question"].strip() + "?", "gold": r["answer"]} for r in ds]
    else:
        raise ValueError(task)
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    for i, r in enumerate(rows):
        r["id"] = i
    return rows


# small, ~15-minute configuration
N = {("math", "train"): 120, ("math", "test"): 150, ("nq", "train"): 400, ("nq", "test"): 400}
S_LAYERS = [10, 14]


def split_data(task, split):
    return task_data(task, split, N[(task, split)])


def max_new(task):
    return 512 if task == "math" else 24


def prompt_text(tok, task, q):
    sysm = MATH_SYS if task == "math" else NQ_SYS
    msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def correct(task, text, gold):
    if task == "math":
        from math_verify import parse, verify
        b = last_boxed(text)
        if b is None:
            return False
        try:
            return bool(verify(parse("$" + gold + "$"), parse("$" + b + "$")))
        except Exception:
            return _norm(b) == _norm(gold)
    pred = _norm(text.split("\n")[0])
    return any(_norm(g) and (_norm(g) in pred) for g in gold)


# ---------------------------------------------------------------- generation
@torch.no_grad()
def generate(tok, model, prompts, max_new=512, sample=False, temp=0.8, bs=32, n=1, prefill_fn=None):
    """Batched generation. Returns list (len prompts) of lists (len n) of completions.
    prefill_fn(enc_chunk) -> context manager applied around generation for that chunk (used for transplant)."""
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True).to(DEV)
        kw = dict(do_sample=True, temperature=temp, top_p=0.95) if sample else dict(do_sample=False)
        ctx = prefill_fn(enc) if prefill_fn else _null()
        with ctx:
            g = model.generate(**enc, max_new_tokens=max_new, num_return_sequences=n,
                               pad_token_id=tok.pad_token_id, **kw)
        texts = tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        outs += [texts[j * n:(j + 1) * n] for j in range(len(chunk))]
        del g, enc
        if DEV == "mps":
            torch.mps.empty_cache()
    return outs


@contextmanager
def _null():
    yield


@contextmanager
def hook_layer(model, layer, fn):
    """Replace the residual-stream output h of decoder layer `layer` by fn(h)."""
    def hook(_m, _i, out):
        if isinstance(out, tuple):
            return (fn(out[0]),) + tuple(out[1:])
        return fn(out)
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def add_vec(model, layer, vec):
    return hook_layer(model, layer, lambda h: h + vec.to(h.dtype))


@torch.no_grad()
def hidden_states(model, enc):
    """All residual-stream states: list over layers (index l = output of decoder layer l)."""
    # call the base transformer (no LM head): avoids materialising [B, T, 152k] logits
    out = model.model(**enc, output_hidden_states=True)
    return out.hidden_states[1:]


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1))


def read(name):
    return json.loads((OUT / name).read_text())


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
