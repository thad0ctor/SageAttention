"""End-to-end demo: run a real GQA model with an NVFP4 KV cache during decode.

PREFILL runs on the model's normal bf16 attention. We monkeypatch the Qwen2
attention forward so that during DECODE (seqlen_q == 1) each layer routes through
its per-layer ``NVFP4KVCache`` and ``nvfp4_flash_decode_prequant`` instead of the
standard cache + SDPA. The fp4 cache stores POST-RoPE K and V in NVFP4.

Reports: (a) coherent generated text; (b) token-agreement vs the bf16 baseline;
(c) KV-cache memory fp4 vs bf16; (d) per-step decode latency fp4 vs bf16 at long
context.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<worktree> python example/nvfp4_decode_demo.py
"""

from __future__ import annotations

import math
import time
import types

import torch

import sageattention.nvfp4.flash as _f
print("sageattention.nvfp4.flash.__file__:", _f.__file__)
from sageattention.nvfp4 import NVFP4KVCache, nvfp4_flash_decode_prequant

import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Monkeypatched attention forward. During decode, route through the fp4 cache.
# ---------------------------------------------------------------------------
_FP4_ENABLED = False  # toggled on once prefill has seeded the caches
_HYBRID = False        # if True, route decode through cache.hybrid_decode (recent bf16 + fp4 old)
RECENT_WINDOW = 128    # recent-window size for hybrid mode


def make_fp4_forward(orig_forward):
    """Build a Qwen2Attention.forward that routes decode through the FP4 KV cache."""
    apply_rope = qwen2_mod.apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        seqlen_q = input_shape[-1]

        # Decode step through the fp4 cache (only after prefill seeded it).
        if _FP4_ENABLED and seqlen_q == 1 and hasattr(self, "_fp4_cache"):
            hidden_shape = (*input_shape, -1, self.head_dim)
            q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rope(q, k, cos, sin)  # post-RoPE Q, K

            cache: NVFP4KVCache = self._fp4_cache
            cache.append(k, v)  # store post-RoPE K and V token in fp4

            g = self.config.num_attention_heads // self.config.num_key_value_heads
            if _HYBRID:
                attn = cache.hybrid_decode(q, self.scaling, g)  # recent bf16 + fp4 old
            else:
                attn = nvfp4_flash_decode_prequant(
                    q, cache.get_packed(), self.scaling, num_key_value_groups=g
                )  # [Z, H, 1, D]
            attn = attn.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn), None

        # Prefill (or fp4 disabled): standard path.
        return orig_forward(self, hidden_states, position_embeddings,
                            attention_mask, past_key_values, **kwargs)

    return forward


def install_patch(model):
    """Swap every Qwen2 attention layer's forward for the FP4-cache version."""
    layers = model.model.layers
    orig = qwen2_mod.Qwen2Attention.forward
    for layer in layers:
        attn = layer.self_attn
        attn.forward = types.MethodType(make_fp4_forward(orig), attn)
    return orig


# ---------------------------------------------------------------------------
# Seed each layer's fp4 cache from a normal bf16 prefill (post-RoPE K + V).
# We grab post-RoPE K/V via forward hooks on the standard prefill.
# ---------------------------------------------------------------------------
def prefill_and_seed(model, input_ids, max_seq_len):
    """Run a normal prefill; capture each layer's post-RoPE K/V; build fp4 caches."""
    cfg = model.config
    z = input_ids.shape[0]
    hk = cfg.num_key_value_heads
    d = cfg.hidden_size // cfg.num_attention_heads
    captured = {}
    apply_rope = qwen2_mod.apply_rotary_pos_emb

    handles = []

    def mk_hook(idx):
        def hook(mod, args, kwargs):
            # recompute post-RoPE K and the V projection from the inputs.
            hs = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings")
            if pe is None:
                pe = args[1]
            ishape = hs.shape[:-1]
            hshape = (*ishape, -1, mod.head_dim)
            k = mod.k_proj(hs).view(hshape).transpose(1, 2)
            v = mod.v_proj(hs).view(hshape).transpose(1, 2)
            cos, sin = pe
            # apply_rope needs a real q; build it from the hidden states
            q = mod.q_proj(hs).view((*ishape, -1, mod.head_dim)).transpose(1, 2)
            _, k = apply_rope(q, k, cos, sin)
            captured[idx] = (k.detach(), v.detach())
        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(mk_hook(i), with_kwargs=True))

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    for h in handles:
        h.remove()

    # build a fp4 cache per layer from the captured prefill K/V.
    rw = RECENT_WINDOW if _HYBRID else 0
    for i, layer in enumerate(model.model.layers):
        k, v = captured[i]  # [Z, Hk, S, D]
        cache = NVFP4KVCache(z, hk, d, max_seq_len=max_seq_len, device=DEVICE,
                             dtype=DTYPE, recent_window=rw)
        cache.prefill(k, v)
        layer.self_attn._fp4_cache = cache
    return out, captured


# ---------------------------------------------------------------------------
# fp4-cache greedy decode (manual loop driving the patched model).
# ---------------------------------------------------------------------------
def generate_fp4(model, input_ids, n_new, position_ids_start):
    """Greedily decode n_new tokens through the FP4 KV-cache path."""
    global _FP4_ENABLED
    out, _ = prefill_and_seed(model, input_ids, max_seq_len=input_ids.shape[1] + n_new + 4)
    next_logits = out.logits[:, -1, :]
    gen = []
    _FP4_ENABLED = True
    pos = position_ids_start
    try:
        for _ in range(n_new):
            tok = next_logits.argmax(-1, keepdim=True)  # [Z,1]
            gen.append(tok)
            position_ids = torch.full_like(tok, pos)
            with torch.no_grad():
                step = model(input_ids=tok, position_ids=position_ids, use_cache=False)
            next_logits = step.logits[:, -1, :]
            pos += 1
    finally:
        _FP4_ENABLED = False
    return torch.cat(gen, dim=1)


def generate_bf16(model, input_ids, n_new):
    """Baseline greedy decode with the standard HF cache (fp4 disabled)."""
    global _FP4_ENABLED
    _FP4_ENABLED = False
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, max_new_tokens=n_new, do_sample=False,
            num_beams=1, use_cache=True,
        )
    return out[:, input_ids.shape[1]:]


def fp4_logits_stepwise(model, input_ids, n_new, ref_tokens, pos_start):
    """Run fp4 decode but FEED the bf16 baseline tokens (teacher forcing) so we can
    measure per-step next-token logit cos and top-1 agreement on the SAME context."""
    global _FP4_ENABLED
    out, _ = prefill_and_seed(model, input_ids, max_seq_len=input_ids.shape[1] + n_new + 4)
    logits = [out.logits[:, -1, :]]
    _FP4_ENABLED = True
    pos = pos_start
    try:
        for t in range(n_new - 1):
            tok = ref_tokens[:, t : t + 1]  # feed baseline token
            position_ids = torch.full_like(tok, pos)
            with torch.no_grad():
                step = model(input_ids=tok, position_ids=position_ids, use_cache=False)
            logits.append(step.logits[:, -1, :])
            pos += 1
    finally:
        _FP4_ENABLED = False
    return torch.stack(logits, dim=1)  # [Z, n_new, V]


def bf16_logits_stepwise(model, input_ids, n_new):
    """Teacher-forced bf16 reference: per-step logits with the stock cache."""
    global _FP4_ENABLED
    _FP4_ENABLED = False
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, max_new_tokens=n_new, do_sample=False, num_beams=1,
            use_cache=True, return_dict_in_generate=True, output_scores=True,
        )
    toks = out.sequences[:, input_ids.shape[1]:]
    scores = torch.stack(out.scores, dim=1)  # [Z, n_new, V]
    return toks, scores


def main():
    """Run the FP4 vs bf16 decode demo: coherence, token agreement, mem, latency."""
    print("device cap:", torch.cuda.get_device_capability(0),
          "name:", torch.cuda.get_device_name(0))
    assert torch.cuda.get_device_capability(0) == (12, 0), "need sm_120"

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, attn_implementation="eager"
    ).to(DEVICE).eval()
    cfg = model.config
    print(f"model: {MODEL_ID}  head_dim={cfg.hidden_size//cfg.num_attention_heads} "
          f"H={cfg.num_attention_heads} Hk={cfg.num_key_value_heads} "
          f"g={cfg.num_attention_heads//cfg.num_key_value_heads} layers={cfg.num_hidden_layers}")
    assert cfg.hidden_size // cfg.num_attention_heads == 128, "demo expects head_dim 128"

    install_patch(model)

    # ----- prompt -----
    messages = [{"role": "user", "content": "Explain what a KV cache is in a transformer, in 3 short sentences."}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    S = input_ids.shape[1]
    n_new = 96
    print(f"\nprompt tokens: {S}, generating {n_new} new tokens (greedy)\n")

    global _HYBRID

    # ----- (b) bf16 baseline (reference) -----
    bf16_tokens, bf16_scores = bf16_logits_stepwise(model, input_ids, n_new)
    bf16_text = tok.decode(bf16_tokens[0], skip_special_tokens=True)

    def report(label, gen_tokens, scores):
        """Print generated text + per-step top-token agreement vs the bf16 reference."""
        text = tok.decode(gen_tokens[0], skip_special_tokens=True)
        print("\n" + "=" * 70)
        print(f"{label} GENERATED TEXT:")
        print("=" * 70)
        print(text)
        n = min(gen_tokens.shape[1], bf16_tokens.shape[1])
        agree = (gen_tokens[:, :n] == bf16_tokens[:, :n]).float().mean().item()
        eq = (gen_tokens[0, :n] == bf16_tokens[0, :n])
        div = int((~eq).nonzero()[0].item()) if (~eq).any() else n
        m = min(scores.shape[1], bf16_scores.shape[1])
        fs, bs = scores[:, :m].float(), bf16_scores[:, :m].float()
        cos_step = torch.nn.functional.cosine_similarity(fs, bs, dim=-1).mean().item()
        top1 = (fs.argmax(-1) == bs.argmax(-1)).float().mean().item()
        print(f"  free-running top-1 token agreement vs bf16: {agree*100:.1f}% over {n} "
              f"steps; first divergence at step {div}")
        print(f"  teacher-forced (same context) per-step logit cos: {cos_step:.4f}; "
              f"top-1 agreement: {top1*100:.1f}% over {m} steps")

    # ----- (a) PURE-FP4 KV cache -----
    _HYBRID = False
    fp4_tokens = generate_fp4(model, input_ids, n_new, position_ids_start=S)
    fp4_scores = fp4_logits_stepwise(model, input_ids, n_new, bf16_tokens, pos_start=S)
    report("PURE-FP4 KV CACHE (all keys/values in NVFP4)", fp4_tokens, fp4_scores)

    # ----- (a') HYBRID KV cache (recent-window bf16 + fp4 old) -----
    _HYBRID = True
    hyb_tokens = generate_fp4(model, input_ids, n_new, position_ids_start=S)
    hyb_scores = fp4_logits_stepwise(model, input_ids, n_new, bf16_tokens, pos_start=S)
    report(f"HYBRID KV CACHE (recent {RECENT_WINDOW} bf16 + older NVFP4)",
           hyb_tokens, hyb_scores)
    _HYBRID = False

    print("\n" + "=" * 70)
    print("BF16 BASELINE GENERATED TEXT (reference):")
    print("=" * 70)
    print(bf16_text)

    # ----- (c) KV-cache memory -----
    cache0: NVFP4KVCache = model.model.layers[0].self_attn._fp4_cache
    per_layer_fp4 = cache0.fp4_bytes()
    per_layer_bf16 = cache0.bf16_bytes()
    L = cfg.num_hidden_layers
    print("\n" + "=" * 70)
    print("KV-CACHE MEMORY (post-generation length "
          f"{cache0.length}, {L} layers):")
    print("=" * 70)
    per_layer_hyb = cache0.hybrid_bytes()
    print(f"  pure-fp4 total: {per_layer_fp4*L/1e6:8.3f} MB  ({per_layer_fp4/1e3:.1f} KB/layer)  "
          f"-> {per_layer_bf16/per_layer_fp4:.2f}x vs bf16")
    print(f"  hybrid   total: {per_layer_hyb*L/1e6:8.3f} MB  ({per_layer_hyb/1e3:.1f} KB/layer)  "
          f"-> {per_layer_bf16/per_layer_hyb:.2f}x vs bf16  (incl. recent-{RECENT_WINDOW} bf16 ring)")
    print(f"  bf16     total: {per_layer_bf16*L/1e6:8.3f} MB  ({per_layer_bf16/1e3:.1f} KB/layer)")
    print("  NOTE: ratio grows with context; the bf16 ring is a fixed add-on, so "
          "hybrid -> ~3.5x at long context (see latency section).")

    # ----- (d) decode latency at long context -----
    bench_latency(model, tok, cfg)


def bench_latency(model, tok, cfg, ctx_len=3072, n_steps=40):
    """Benchmark per-step decode latency: FP4 cache vs HF DynamicCache."""
    print("\n" + "=" * 70)
    print(f"DECODE LATENCY @ context {ctx_len} ({n_steps} timed steps):")
    print("=" * 70)
    global _FP4_ENABLED

    # build a long prompt by repeating text.
    base = "The quick brown fox jumps over the lazy dog. "
    ids = tok(base * 400, return_tensors="pt").input_ids[:, :ctx_len].to(DEVICE)
    S = ids.shape[1]

    # --- fp4 path ---
    _FP4_ENABLED = False
    out, _ = prefill_and_seed(model, ids, max_seq_len=S + n_steps + 16)
    nl = out.logits[:, -1, :]
    _FP4_ENABLED = True
    pos = S

    def fp4_step(next_logits, pos):
        t = next_logits.argmax(-1, keepdim=True)
        pid = torch.full_like(t, pos)
        with torch.no_grad():
            s = model(input_ids=t, position_ids=pid, use_cache=False)
        return s.logits[:, -1, :]

    for _ in range(5):  # warmup
        nl = fp4_step(nl, pos); pos += 1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        nl = fp4_step(nl, pos); pos += 1
    torch.cuda.synchronize()
    fp4_ms = (time.perf_counter() - t0) / n_steps * 1e3
    _FP4_ENABLED = False

    # --- bf16 baseline (standard HF cache) ---
    from transformers import DynamicCache
    with torch.no_grad():
        pc = DynamicCache()
        o = model(input_ids=ids, use_cache=True, past_key_values=pc)
        nl = o.logits[:, -1, :]

        def bf_step(next_logits, pc):
            t = next_logits.argmax(-1, keepdim=True)
            s = model(input_ids=t, use_cache=True, past_key_values=pc)
            return s.logits[:, -1, :]

        for _ in range(5):
            nl = bf_step(nl, pc)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            nl = bf_step(nl, pc)
        torch.cuda.synchronize()
        bf_ms = (time.perf_counter() - t0) / n_steps * 1e3

    print(f"  fp4-cache decode:  {fp4_ms:.3f} ms/step")
    print(f"  bf16 baseline:     {bf_ms:.3f} ms/step")
    print(f"  speedup: {bf_ms/fp4_ms:.2f}x")

    # memory at this (long) context — the regime where the fp4 win is realized.
    c0 = model.model.layers[0].self_attn._fp4_cache
    L = cfg.num_hidden_layers
    print(f"  KV memory @ {S} ctx: fp4 {c0.fp4_bytes()*L/1e6:.1f} MB vs "
          f"bf16 {c0.bf16_bytes()*L/1e6:.1f} MB  ({c0.bf16_bytes()/c0.fp4_bytes():.2f}x)")


if __name__ == "__main__":
    main()
