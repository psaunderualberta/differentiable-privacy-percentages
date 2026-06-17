#!/usr/bin/env python3
"""Generate presentation.html (a self-contained PptxGenJS deck) for the
differentiable-privacy-percentages 45-min overview talk.

Run:  python build_presentation.py
Then: open presentation.html in a browser; it downloads talk.pptx.
"""

import base64
import json
import pathlib
import struct

# project root
ROOT = pathlib.Path(__file__).parent.parent.parent
RES = ROOT / "src" / "cache" / "results"
TL = RES / "psaunder__TestLadderSweep" / "plots" / "sgd"
NM = RES / "psaunder__NoMomentumLadderSweep" / "plots" / "sgd"

# JS key -> figure path
FIGS = {
    "TL_MAIN": TL / "t_sweep_main.png",
    "TL_DELTA": TL / "t_sweep_delta_vs_constant.png",
    "TL_SIGMA": TL / "sigma_shape.png",
    "TL_CLIP": TL / "clip_shape.png",
    "NM_DELTA": NM / "t_sweep_delta_vs_constant.png",
    "NM_SIGMA": NM / "sigma_shape.png",
    "ARCH": TL / "ladders" / "overall" / "arch_forest_abs.png",
}


def png_dims(d: bytes):
    w, h = struct.unpack(">II", d[16:24])
    return w, h


def fig_map_js() -> str:
    """Build a JS object literal: KEY -> {d: dataURI, ar: width/height}."""
    entries = {}
    for key, path in FIGS.items():
        if not path.exists():
            raise SystemExit(f"missing figure: {path}")
        raw = path.read_bytes()
        w, h = png_dims(raw)
        b = base64.b64encode(raw).decode("ascii")
        entries[key] = {"d": f"image/png;base64,{b}", "ar": round(w / h, 5)}
    return json.dumps(entries)


JS = r"""
const C = { ink:"1F2933", accent:"2F6F4E", accent2:"C2553B", accent3:"3B6FA0",
            tint:"F4F1EA", muted:"7B8794", line:"DDDDDD", white:"FFFFFF" };
const TALK = "Learning Noise & Clipping Schedules for DP-SGD";
const FIG = __FIGMAP__;

// rt(): turn a string with _x / ^x / _{..} / ^{..} markup into PptxGenJS text
// runs with real subscript/superscript (PptxGenJS has no LaTeX/OMML import).
function rt(str, opt){
  opt = opt || {};
  const runs = []; let i = 0, buf = "";
  const flush = () => { if(buf){ runs.push({ text:buf, options:Object.assign({},opt) }); buf=""; } };
  while(i < str.length){
    const c = str[i];
    if(c === "_" || c === "^"){
      const sup = c === "^"; flush();
      let seg;
      if(str[i+1] === "{"){ const e = str.indexOf("}", i+2); seg = str.slice(i+2, e); i = e+1; }
      else { seg = str[i+1]; i += 2; }
      runs.push({ text:seg, options:Object.assign({}, opt, sup?{superscript:true}:{subscript:true}) });
    } else { buf += c; i++; }
  }
  flush();
  return runs;
}
// join run-arrays / runs, inserting a hard line break after each line
function eqlines(lines){
  const out = [];
  lines.forEach((ln, k) => {
    const arr = Array.isArray(ln) ? ln : [ln];
    arr.forEach(r => out.push(r));
    out.push({ text:"\n", options:{} });
  });
  return out;
}
// superscript citation marker, e.g. cite("1,3")
function cite(n, color){ return { text:"["+n+"]", options:{ superscript:true, color:color||C.muted, bold:false } }; }

function defineMasters(pptx){
  pptx.defineSlideMaster({
    title:"CONTENT", background:{color:C.white},
    objects:[
      { rect:{ x:0,y:0,w:"100%",h:0.12, fill:{color:C.accent} } },
      { text:{ text:TALK, options:{ x:0.5,y:7.06,w:10,h:0.3, fontSize:9, color:C.muted } } },
    ],
    slideNumber:{ x:12.7,y:7.06, fontSize:9, color:C.muted },
  });
  pptx.defineSlideMaster({ title:"SECTION", background:{color:C.ink},
    objects:[ { rect:{ x:0,y:0,w:0.18,h:"100%", fill:{color:C.accent} } } ] });
  pptx.defineSlideMaster({ title:"TITLE", background:{color:C.ink},
    objects:[ { rect:{ x:0,y:6.9,w:"100%",h:0.6, fill:{color:C.accent} } } ] });
}

function head(s, headline, sub){
  s.addText(headline, { x:0.5,y:0.32,w:12.3,h:0.95, fontSize:26, bold:true, color:C.ink, valign:"top" });
  s.addShape(pptx.ShapeType.line, { x:0.55,y:1.28,w:2.6,h:0, line:{color:C.accent,width:2.5} });
  if(sub) s.addText(sub, { x:0.5,y:1.32,w:12.3,h:0.4, fontSize:13, italic:true, color:C.muted });
}
function bullets(s, items, opt){
  opt = opt||{};
  s.addText(items.map(t => (typeof t==="string"
        ? { text:t, options:{ bullet:{indent:14} } }
        : { text:t.t, options:{ bullet: t.sub?{indent:24}:{indent:14}, indentLevel: t.sub?1:0,
                                color: t.c||C.ink, bold: t.b||false } })),
    Object.assign({ x:0.55,y:1.7,w:6.0,h:4.9, fontSize:15, color:C.ink, lineSpacingMultiple:1.25, valign:"top" }, opt));
}
// Fit figure KEY into bounding box {x,y,w,h} preserving native aspect ratio,
// centered. Explicit w/h (not just sizing) so no renderer can squish it.
function fig(s, key, box){
  const m = FIG[key];
  let w = box.w, h = w / m.ar;
  if(h > box.h){ h = box.h; w = h * m.ar; }
  const x = box.x + (box.w - w)/2;
  const y = box.y + (box.h - h)/2;
  s.addImage({ data:m.d, x:x, y:y, w:w, h:h, sizing:{ type:"contain", w:w, h:h } });
}
function caption(s, t, o){ o=o||{};
  s.addText(t, { x:o.x!=null?o.x:6.55, y:o.y!=null?o.y:6.5, w:o.w||6.3, h:0.5,
    fontSize:11, italic:true, color:C.muted, align:"center" }); }

function sectionSlide(pptx, kicker, title){
  const s = pptx.addSlide({ masterName:"SECTION" });
  s.addText(kicker, { x:0.85,y:2.55,w:11.5,h:0.5, fontSize:15, color:C.accent, charSpacing:3 });
  s.addText(title, { x:0.8,y:3.0,w:11.7,h:1.6, fontSize:40, bold:true, color:C.white });
  s.addShape(pptx.ShapeType.line, { x:0.85,y:4.5,w:2.5,h:0, line:{color:C.accent,width:3} });
  return s;
}

function buildSlides(pptx){
  // ---------- TITLE ----------
  let s = pptx.addSlide({ masterName:"TITLE" });
  s.addText("Differentiable Privacy Percentages", { x:0.8,y:2.1,w:11.7,h:1.0, fontSize:40, bold:true, color:C.white });
  s.addText("Learning where to spend the privacy budget in DP-SGD", { x:0.8,y:3.25,w:11.7,h:0.7, fontSize:21, color:"CBD2D9" });
  s.addText([{text:"Paul Saunders",options:{bold:true}},{text:"   ·   University of Alberta   ·   Lab overview",options:{color:"9AA5B1"}}],
            { x:0.8,y:4.5,w:11.7,h:0.5, fontSize:15, color:C.white });
  s.addText("Learn per-step noise (σ) and clipping (C) schedules by differentiating through training, under a fixed Gaussian-DP budget.",
            { x:0.8,y:5.2,w:10.5,h:0.9, fontSize:13, italic:true, color:"9AA5B1" });
  s.addNotes("Title. One-line hook: instead of a fixed noise level, learn how to allocate the privacy budget across training. 45-minute overview: motivation, method (with optional deep dives), then results from two sweeps.");

  // ---------- AGENDA ----------
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Roadmap");
  bullets(s, [
    {t:"1 · Motivation", b:true, c:C.accent}, {t:"Why private learning, and the fixed-knob assumption in DP-SGD", sub:true, c:C.muted},
    {t:"2 · Method", b:true, c:C.accent}, {t:"Schedules as a bilevel optimization problem under a Gaussian-DP budget", sub:true, c:C.muted},
    {t:"3 · Experimental setup", b:true, c:C.accent}, {t:"Two datasets, four budgets, a T-sweep and architecture ladders", sub:true, c:C.muted},
    {t:"4 · Results", b:true, c:C.accent}, {t:"What is learned, how much it helps, and when", sub:true, c:C.muted},
    {t:"5 · Discussion & conclusion", b:true, c:C.accent}, {t:"When it helps, limitations, next steps", sub:true, c:C.muted},
  ], {w:11.5});
  s.addText([{text:"Method section contains 3 optional deep-dive slides (privacy accounting) — prunable for time.",options:{italic:true}}],
            { x:0.55,y:6.4,w:11.5,h:0.4, fontSize:12, color:C.muted });
  s.addNotes("Roadmap. Flag that the privacy-accounting deep dives can be skipped if running short.");

  // ================= SECTION 1: MOTIVATION =================
  sectionSlide(pptx, "PART 1", "Motivation").addNotes("Set up the why before the how.");

  // S: private ML leaks
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Models memorize their training data — and that leaks");
  bullets(s, [
    "Neural nets trained on sensitive data (medical, financial, text) can memorize individual records.",
    "Membership-inference & extraction attacks recover whether — and sometimes what — a person contributed. [8]",
    {t:"Differential Privacy (DP) gives a formal guarantee: [2]", b:true},
    {t:"any single record changes the output distribution by a bounded amount (ε, δ).", sub:true},
    "DP-SGD is the standard way to train deep nets with a DP guarantee. [1]",
  ], {w:6.0});
  s.addShape(pptx.ShapeType.roundRect, { x:7.0,y:2.1,w:5.6,h:3.6, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"The DP promise\n",options:{bold:true,fontSize:18,color:C.ink}},
    {text:"“Whether or not your data was used,\nwhat anyone can learn about you\nis almost the same.”\n\n",options:{fontSize:15,italic:true,color:C.ink}},
    {text:"Smaller ε  ⇒  stronger privacy  ⇒  more noise",options:{fontSize:14,bold:true,color:C.accent2}},
  ], { x:7.3,y:2.4,w:5.0,h:3.0, valign:"middle", align:"center" });
  s.addNotes("Privacy is not just policy — models genuinely memorize. DP is the formal antidote; DP-SGD is how we get it in deep learning. Lower epsilon = more noise = the central tension.");

  // S: DP-SGD recap
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "DP-SGD: two knobs turn SGD into a private algorithm");
  bullets(s, [
    {t:"Each step does two extra things:", b:true},
    {t:"Clip every per-example gradient to norm C  (bound one person's influence).", sub:true},
    {t:"Add Gaussian noise of scale σ  (mask that influence).", sub:true},
    "A privacy accountant tracks the total (ε, δ) spent over all T steps.",
    {t:"σ and C are the levers that trade accuracy against privacy.", b:true, c:C.accent2},
  ], {w:6.1});
  s.addShape(pptx.ShapeType.roundRect, { x:7.0,y:1.9,w:5.6,h:4.3, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"One DP-SGD step ",options:{bold:true,fontSize:16,color:C.ink}}, cite("1"),
    {text:"\n\n",options:{}},
    ...rt("g_{i} = ∇ loss(x_{i})", {fontSize:15,color:C.ink}), {text:"\n",options:{}},
    ...rt("ḡ_{i} = g_{i}·min(1, C/‖g_{i}‖)", {fontSize:15,color:C.accent3}),
    {text:"  ← clip\n",options:{fontSize:12,italic:true,color:C.muted}},
    ...rt("g̃ = (Σ ḡ_{i} + 𝒩(0, σ^{2}C^{2}I))/B", {fontSize:15,color:C.accent2}),
    {text:"  ← noise\n",options:{fontSize:12,italic:true,color:C.muted}},
    ...rt("θ ← θ − η·g̃", {fontSize:15,color:C.ink}), {text:"\n",options:{}},
  ], { x:7.35,y:2.2,w:5.0,h:3.7, valign:"top" });
  s.addNotes("Recap for the mixed audience. Two knobs: clip C bounds sensitivity, noise sigma masks it. Accountant converts the per-step noise into a total epsilon. Everything in the talk is about choosing sigma and C.");

  // S: fixed-knob problem
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Standard DP-SGD holds both knobs constant — why assume that?");
  bullets(s, [
    {t:"Convention: pick one σ and one C, hold them fixed for all T steps.", b:true},
    "But training is non-stationary — early steps are large & noisy, late steps are fine-tuning.",
    "Intuition says the budget could be spent unevenly: more noise where it's cheap, less where it hurts.",
    {t:"The schedule itself is a hyperparameter we never optimize.", b:true, c:C.accent2},
    {t:"Question: if we could shape σ(t) and C(t) over training — under the same (ε, δ) — would it help?", b:true, c:C.accent},
  ], {w:7.4});
  s.addShape(pptx.ShapeType.roundRect, { x:8.5,y:2.2,w:4.2,h:3.4, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"Same budget,\ndifferent allocation\n\n",options:{bold:true,fontSize:16,color:C.ink,align:"center"}},
    {text:"flat  ▁▁▁▁▁▁▁\n",options:{fontSize:16,color:C.muted,align:"center"}},
    {text:"vs\n",options:{fontSize:13,italic:true,color:C.muted,align:"center"}},
    {text:"shaped ▇▃▂▂▂▃▅",options:{fontSize:16,color:C.accent,align:"center",bold:true}},
  ], { x:8.7,y:2.5,w:3.8,h:2.8, valign:"middle" });
  s.addNotes("The gap: the noise/clip schedule is itself an un-tuned hyperparameter. Training is non-stationary, so a flat schedule is unlikely optimal. The thesis question lands here.");

  // S: thesis statement
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "This work: learn the schedule by differentiating through training");
  bullets(s, [
    {t:"Treat the per-step schedule (σ₁..σ_T, C₁..C_T) as learnable parameters.", b:true},
    "Differentiate through the entire DP-SGD run to get gradients of final accuracy w.r.t. the schedule.",
    "After each update, project back onto the exact Gaussian-DP budget — privacy is never violated.",
    {t:"Contributions:", b:true, c:C.accent},
    {t:"A differentiable, budget-constrained outer loop for DP-SGD schedules.", sub:true},
    {t:"A JAX projection onto the GDP constraint set (bisection + Newton).", sub:true},
    {t:"Evidence across budgets, training lengths, and architectures.", sub:true},
  ], {w:11.8});
  s.addNotes("Thesis statement. Three contributions: the differentiable constrained outer loop, the projection, and the empirical study. Privacy is guaranteed by construction via the projection.");

  // ================= SECTION 2: METHOD =================
  sectionSlide(pptx, "PART 2", "Method: Learning the Schedule").addNotes("Now the how. Three of these slides are optional deep dives on the privacy math.");

  // S: bilevel framing
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "A bilevel optimization problem", "Outer problem shapes the schedule; inner problem is DP-SGD itself");
  bullets(s, [
    {t:"Inner problem (lower level):", b:true, c:C.accent3},
    {t:"given a schedule φ, run DP-SGD → trained weights θ*(φ).", sub:true},
    {t:"Outer problem (upper level):", b:true, c:C.accent2},
    {t:"choose φ to minimize validation loss of θ*(φ), subject to the DP budget. [4,5]", sub:true},
    "We solve the outer problem by gradient descent — backpropagating through the inner solve.",
    {t:"Feasible set Φ = all schedules that exactly spend (ε, δ).", b:true},
  ], {w:6.0});
  s.addShape(pptx.ShapeType.roundRect, { x:6.9,y:1.95,w:5.8,h:4.3, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    ...rt("min  L_{val}( θ*(φ) )", {fontSize:17,color:C.accent2,bold:true}), {text:"\n",options:{}},
    {text:" φ∈Φ\n\n",options:{fontSize:13,color:C.muted}},
    {text:"s.t.  θ*(φ) = DP-SGD(φ)\n",options:{fontSize:15,color:C.accent3}},
    {text:"      φ spends exactly (ε, δ)\n\n",options:{fontSize:15,color:C.ink}},
    {text:"Outer update:\n",options:{fontSize:13,italic:true,color:C.muted}},
    ...rt("φ ← Proj_{Φ}( φ − α ∇_{φ} L_{val} )", {fontSize:15,color:C.ink,bold:true}),
  ], { x:7.25,y:2.3,w:5.1,h:3.6, valign:"top" });
  s.addNotes("Bilevel framing (not RL): inner level is the DP-SGD training run, outer level shapes the schedule to minimize val loss. We descend the outer objective by differentiating through the inner solve, projecting back to the feasible budget set each step. Cite Maclaurin 2015 / Franceschi 2018 for differentiating through training.");

  // S: GDP accounting [DEEP 1]
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Deep dive ① — Budget as a single Gaussian-DP parameter μ", "Optional: skippable for time");
  bullets(s, [
    "Gaussian-DP (GDP) summarizes a mechanism by one number μ: indistinguishable up to a 𝒩(0,1) vs 𝒩(μ,1) test. [3]",
    {t:"At startup we convert the target (ε, δ) → total μ once (Brent's method, outside JAX).", b:true},
    "Composition over T steps: the per-step μ(t) combine in quadrature → total μ.",
    {t:"So the schedule lives in 'μ-space': allocate a non-negative per-step μ(t) budget across the T steps.", b:true, c:C.accent},
    {t:"Then  σ(t) = C(t) / μ(t)  — noise and clip both fall out of the μ allocation.", sub:true, c:C.muted},
  ], {w:6.2});
  s.addShape(pptx.ShapeType.roundRect, { x:7.1,y:2.0,w:5.5,h:4.0, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"From budget to schedule\n\n",options:{bold:true,fontSize:16,color:C.ink}},
    ...rt("(ε, δ)  ──Brent──▶  μ_{total}", {fontSize:14,color:C.accent3}), {text:"\n\n",options:{}},
    ...rt("weights w_{t}  ──▶  μ_{t}  (per step)", {fontSize:14,color:C.ink}), {text:"\n\n",options:{}},
    ...rt("σ_{t} = C_{t} / μ_{t}", {fontSize:15,color:C.accent2,bold:true}), {text:"\n\n",options:{}},
    ...rt("Σ_{t} (subsampled) μ_{t}^{2}  =  μ_{total}^{2}", {fontSize:14,italic:true,color:C.muted}),
  ], { x:7.4,y:2.35,w:4.9,h:3.3, valign:"top" });
  s.addNotes("Deep dive 1. GDP collapses the budget to one number mu. We work in mu-space: distribute per-step mu across T steps. Sigma and clip are both derived from that allocation. Skippable.");

  // S: constraint & projection [DEEP 2]
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Deep dive ② — The feasible set is a curved constraint surface", "Optional: skippable for time");
  bullets(s, [
    "We parameterize the allocation by free weights w(t) ∈ ℝ — unconstrained, so gradient steps are easy.",
    {t:"Valid schedules must satisfy a single nonlinear equality (below):", b:true},
    {t:"subsampling rate p, budget μ — a curved surface, not a box.", sub:true, c:C.muted},
    "A raw gradient step leaves this surface → the schedule would over- or under-spend the budget.",
    {t:"Projection puts us back on the surface after every outer step — privacy holds exactly.", b:true, c:C.accent},
  ], {w:7.7});
  // formatted constraint equation (real sub/superscripts), highlighted strip
  s.addShape(pptx.ShapeType.roundRect, { x:0.55,y:5.55,w:7.7,h:0.75, fill:{color:C.tint}, line:{color:C.accent,width:1.5} });
  s.addText(rt("Σ_{t} exp(w_{t}^{2})  =  (μ / p)^{2}  +  T",
            {fontSize:19,color:C.accent2,bold:true}),
            { x:0.7,y:5.6,w:7.4,h:0.65, valign:"middle", align:"center" });
  s.addShape(pptx.ShapeType.roundRect, { x:8.7,y:2.1,w:4.0,h:3.2, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"gradient step\n",options:{fontSize:13,italic:true,color:C.muted,align:"center"}},
    {text:"●",options:{fontSize:20,color:C.ink,align:"center"}},
    {text:" ↘ leaves budget\n",options:{fontSize:13,color:C.accent2,align:"center"}},
    {text:"  ◌\n",options:{fontSize:18,color:C.muted,align:"center"}},
    {text:"↑ project back\n",options:{fontSize:13,color:C.accent,align:"center"}},
    {text:"●  on-budget\n",options:{fontSize:20,color:C.accent,align:"center"}},
  ], { x:8.9,y:2.5,w:3.6,h:2.9, valign:"middle" });
  s.addNotes("Deep dive 2. The learnable weights are unconstrained; valid schedules satisfy a nonlinear equality (sum of exp(w^2) equals a budget-determined constant). Gradient steps leave the surface; projection restores feasibility. This is what guarantees privacy.");

  // S: projection algorithm [DEEP 3]
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Deep dive ③ — Projection: nested bisection + Newton, all in JAX", "Optional: skippable for time");
  bullets(s, [
    "Project raw weights onto Σ exp(w_t²) = const by finding a single Lagrange-multiplier scalar.",
    {t:"Outer bisection on the multiplier; inner Newton solves each coordinate.", b:true},
    "Implemented entirely with jax.lax.while_loop → JIT-compatible and differentiable-friendly.",
    {t:"Runs every outer step, fully on-device — no host round-trips, no SciPy in the hot loop.", b:true, c:C.accent},
    {t:"Net effect: the outer optimizer can take any step it likes; projection keeps it exactly on-budget.", b:true, c:C.accent2},
  ], {w:11.8});
  s.addText([{text:"Detail: src/privacy/gdp_privacy.py — project_weights()",options:{italic:true}}],
            { x:0.55,y:6.3,w:11.5,h:0.4, fontSize:12, color:C.muted });
  s.addNotes("Deep dive 3. The projection itself: a Lagrange-multiplier root-find solved by nested bisection/Newton, written in lax.while_loop so it JIT-compiles and runs on the accelerator every step. Most prunable slide of the three.");

  // S: differentiate through DP-SGD
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Differentiating through the whole DP-SGD run");
  bullets(s, [
    "The inner DP-SGD loop is a jax.lax.scan over T steps — clip, add noise, optax update.",
    {t:"We backpropagate through all T steps to get ∇_φ (final val loss).", b:true},
    {t:"jax.checkpoint (rematerialization) keeps memory tractable for T up to ~7000 steps.", sub:true, c:C.muted},
    "Each outer step samples a fresh random network and fresh noise → low-variance gradient estimate.",
    {t:"shard_map parallelizes a batch of independent inner runs across GPUs.", b:true, c:C.accent},
    "eqx.filter_jit, eqx.filter_value_and_grad; NaN-guarded optimizer (clip + zero-nans + SGD).",
  ], {w:7.5});
  s.addShape(pptx.ShapeType.roundRect, { x:8.6,y:2.0,w:4.1,h:4.0, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"Inner loop (scan over T)\n\n",options:{bold:true,fontSize:14,color:C.ink}},
    {text:"for t in 1..T:\n",options:{fontSize:13,color:C.muted}},
    {text:"  sample minibatch\n",options:{fontSize:13,color:C.ink}},
    ...rt("  clip to C_{t}", {fontSize:13,color:C.ink}), {text:"\n",options:{}},
    ...rt("  add 𝒩(0, σ_{t}^{2})", {fontSize:13,color:C.ink}), {text:"\n",options:{}},
    {text:"  optax update θ\n\n",options:{fontSize:13,color:C.ink}},
    ...rt("return L_{val}(θ_{T})", {fontSize:13,color:C.accent2,bold:true}), {text:"\n",options:{}},
    {text:"────────────\n",options:{fontSize:12,color:C.line}},
    ...rt("∇_{φ} via reverse-mode", {fontSize:13,italic:true,color:C.accent}),
    {text:"\n+ checkpointing",options:{fontSize:13,italic:true,color:C.accent}},
  ], { x:8.85,y:2.3,w:3.6,h:3.5, valign:"top" });
  s.addNotes("The inner run is a scan; we differentiate through all T steps with checkpointing to control memory. Fresh network + fresh noise each outer step reduces gradient variance; shard_map fans the batch across GPUs.");

  // S: full algorithm
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "The outer loop, end to end");
  s.addText([
    {text:"repeat for each outer step:\n\n",options:{bold:true,fontSize:16,color:C.ink}},
    {text:"1.  ",options:{fontSize:15,bold:true,color:C.accent}},{text:"Read the current schedule φ → per-step σ_t, C_t\n",options:{fontSize:15,color:C.ink}},
    {text:"2.  ",options:{fontSize:15,bold:true,color:C.accent}},{text:"Run DP-SGD (scan over T) on a fresh network → val loss\n",options:{fontSize:15,color:C.ink}},
    {text:"3.  ",options:{fontSize:15,bold:true,color:C.accent}},
    ...rt("Backprop through the run → ∇_{φ} val loss  (vmap'd, sharded over GPUs)", {fontSize:15,color:C.ink}),
    {text:"\n",options:{}},
    {text:"4.  ",options:{fontSize:15,bold:true,color:C.accent}},{text:"Gradient step on φ\n",options:{fontSize:15,color:C.ink}},
    {text:"5.  ",options:{fontSize:15,bold:true,color:C.accent2}},{text:"Project φ back onto the (ε, δ) budget surface\n",options:{fontSize:15,color:C.accent2,bold:true}},
  ], { x:0.7,y:1.7,w:8.2,h:4.5, valign:"top", lineSpacingMultiple:1.15 });
  s.addShape(pptx.ShapeType.roundRect, { x:9.1,y:1.8,w:3.6,h:4.3, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"Invariant\n\n",options:{bold:true,fontSize:15,color:C.ink,align:"center"}},
    {text:"After every step,\nthe schedule spends\n",options:{fontSize:14,color:C.ink,align:"center"}},
    {text:"exactly (ε, δ).\n\n",options:{fontSize:15,bold:true,color:C.accent2,align:"center"}},
    {text:"Accuracy is optimized;\nprivacy is fixed.",options:{fontSize:14,italic:true,color:C.accent,align:"center"}},
  ], { x:9.3,y:2.4,w:3.2,h:3.2, valign:"middle" });
  s.addNotes("Tie it together. The five-step loop; step 5 (projection) is the privacy guarantee. We never trade privacy for accuracy — privacy is a hard constraint, accuracy is the objective.");

  // S: baselines
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "What we compare against");
  s.addTable([
    [{text:"Method",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Schedule for σ and C",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Role",options:{bold:true,fill:{color:C.ink},color:C.white}}],
    [{text:"Learned",options:{bold:true,color:C.accent}}, "Optimized per-step via our bilevel loop", "Ours"],
    ["Constant", "Single σ, single C held flat (textbook DP-SGD) [1]", "Main baseline"],
    ["Dynamic-DPSGD", "Hand-designed dynamic schedule from prior work [6]", "Strong baseline"],
    ["Median-Clip", "Clip from a running gradient-norm median (adaptive clipping) [7]", "Adaptive-clip baseline"],
  ], { x:0.55,y:1.8,w:12.2, fontSize:14, rowH:0.62, valign:"middle",
       border:{type:"solid",color:C.line,pt:1}, color:C.ink, fill:{color:C.white} });
  s.addText("All methods are held to the same (ε, δ) budget, dataset, architecture, and step count — only the schedule differs.",
            { x:0.55,y:5.6,w:12.2,h:0.5, fontSize:14, italic:true, color:C.muted });
  s.addNotes("Four methods, same budget and setup. Constant is the textbook baseline we most care about beating; Dynamic-DPSGD and Median-Clip are stronger/adaptive baselines from the literature.");

  // ================= SECTION 3: SETUP =================
  sectionSlide(pptx, "PART 3", "Experimental Setup").addNotes("Quick — what we ran.");

  // S: the grid
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "A budget × length × architecture grid");
  bullets(s, [
    {t:"Datasets:", b:true}, {t:"MNIST and Fashion-MNIST.", sub:true},
    {t:"Privacy budgets:", b:true}, {t:"ε ∈ {1, 3, 5, 8},  δ = 1e-7.", sub:true},
    {t:"Training length sweep:", b:true}, {t:"T ∈ {1500, 2000, 3000, 5000, 7000} inner steps.", sub:true},
    {t:"Architecture ladders (at ε = 8):", b:true}, {t:"MLP width, MLP depth, CNN width.", sub:true},
    {t:"3 seeds per cell; we report test accuracy (mean ± std).", b:true, c:C.accent},
  ], {w:6.3});
  s.addShape(pptx.ShapeType.roundRect, { x:7.1,y:2.0,w:5.5,h:4.0, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"Two questions\n\n",options:{bold:true,fontSize:16,color:C.ink}},
    {text:"Q1 — Length:\n",options:{fontSize:14,bold:true,color:C.accent3}},
    {text:"does the benefit hold as training gets longer?\n\n",options:{fontSize:14,color:C.ink}},
    {text:"Q2 — Architecture:\n",options:{fontSize:14,bold:true,color:C.accent2}},
    {text:"does it transfer across model sizes & types?",options:{fontSize:14,color:C.ink}},
  ], { x:7.4,y:2.35,w:4.9,h:3.3, valign:"top" });
  s.addNotes("The grid. MNIST-family datasets, four budgets, a length sweep, and architecture ladders at epsilon 8. Two questions: does it hold with longer training, and does it transfer across architectures.");

  // S: two regimes
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Two optimizer regimes — momentum matters a lot");
  bullets(s, [
    "We ran the entire grid twice for the inner optimizer:",
    {t:"With momentum  (TestLadderSweep)", b:true, c:C.accent3},
    {t:"Constant DP-SGD is already strong here — momentum smooths noisy updates.", sub:true, c:C.muted},
    {t:"Without momentum  (NoMomentumLadderSweep)", b:true, c:C.accent2},
    {t:"Plain SGD inner loop — more exposed to how the budget is spent.", sub:true, c:C.muted},
    {t:"Pre-view: the learned schedule helps in both — but far more without momentum.", b:true, c:C.accent},
  ], {w:11.8});
  s.addNotes("Important framing for the results: two regimes. With momentum the constant baseline is hard to beat; without momentum the schedule has much more room to help. Keep this distinction in mind through the results.");

  // ================= SECTION 4: RESULTS =================
  sectionSlide(pptx, "PART 4", "Results").addNotes("Figure-forward from here.");

  // R: headline main
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Learned matches or beats every baseline as training scales", "With momentum · test accuracy vs T");
  fig(s, "TL_MAIN", {x:3.3,y:1.6,w:9.4,h:4.4});
  caption(s, "Rows: Fashion-MNIST (top), MNIST (bottom). Columns: ε = 1, 3, 5, 8.", {x:3.3,y:6.35,w:9.4});
  s.addText([
    {text:"Read-off\n\n",options:{bold:true,fontSize:15,color:C.ink}},
    {text:"Blue (Learned) sits at or above all baselines for ε ≥ 3.\n\n",options:{fontSize:13,color:C.ink}},
    {text:"Median-Clip (green) lags badly — adaptive clip alone is not enough.\n\n",options:{fontSize:13,color:C.muted}},
    {text:"At ε = 1 the curves bunch — noise-dominated regime.",options:{fontSize:13,italic:true,color:C.accent2}},
  ], { x:0.5,y:1.7,w:2.7,h:4.4, valign:"top" });
  s.addNotes("Headline figure (with momentum). Learned at or above all baselines for eps>=3. Median-clip is clearly worst. At eps=1 everything bunches — too noisy for the schedule to matter much. This motivates looking at the improvement-over-constant view next.");

  // R: delta vs constant (momentum)
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Where it helps: gains grow with budget, fade at ε = 1", "With momentum · (Learned − Constant) accuracy");
  fig(s, "TL_DELTA", {x:3.3,y:1.6,w:9.4,h:4.4});
  caption(s, "Above zero = Learned beats Constant. Band = ±1 std over seeds.", {x:3.3,y:6.35,w:9.4});
  s.addText([
    {text:"Read-off\n\n",options:{bold:true,fontSize:15,color:C.ink}},
    {text:"ε = 3,5,8: consistently positive (≈ +0.5 to +1 pt).\n\n",options:{fontSize:13,color:C.accent}},
    {text:"ε = 1, Fashion: dips below zero at large T.\n\n",options:{fontSize:13,color:C.accent2}},
    {text:"With momentum, the constant baseline is genuinely hard to beat at tight budgets.",options:{fontSize:13,italic:true,color:C.muted}},
  ], { x:0.5,y:1.7,w:2.7,h:4.4, valign:"top" });
  s.addNotes("Improvement over constant, with momentum. Clear positive gains at eps>=3; at eps=1 it can go slightly negative on Fashion. Honest: with momentum the schedule is not a free lunch at the tightest budget.");

  // R: learned sigma shape
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "What does it learn? A front-loaded noise spike, then a low plateau", "Learned σ(t) across all runs");
  fig(s, "TL_SIGMA", {x:3.2,y:1.7,w:9.5,h:4.0});
  caption(s, "x: normalized step t/T. Colored by ε. Each line = one run.", {x:3.2,y:6.05,w:9.5});
  s.addText([
    {text:"Pattern\n\n",options:{bold:true,fontSize:15,color:C.ink}},
    {text:"Sharp σ spike at t ≈ 0 — spend noise early when gradients are large and uninformative.\n\n",options:{fontSize:13,color:C.ink}},
    {text:"Then drop to a low plateau and gently decay toward the end.\n\n",options:{fontSize:13,color:C.ink}},
    {text:"Remarkably consistent across ε and dataset.",options:{fontSize:13,italic:true,color:C.accent}},
  ], { x:0.5,y:1.7,w:2.6,h:4.3, valign:"top" });
  s.addNotes("The learned sigma shape. A consistent story: big noise spike at the very start, then a low plateau decaying to the end. Interpretation: early gradients are large/uninformative, so noise is 'cheap' there; protect the budget for the productive middle/late training.");

  // R: learned clip shape
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "...and a mid-training clipping hump", "Learned C(t) across all runs");
  fig(s, "TL_CLIP", {x:3.2,y:1.7,w:9.5,h:4.0});
  caption(s, "Clip norm rises into mid-training, then relaxes near the end.", {x:3.2,y:6.05,w:9.5});
  s.addText([
    {text:"Pattern\n\n",options:{bold:true,fontSize:15,color:C.ink}},
    {text:"Clip C(t) is low early, peaks mid-training, eases at the end.\n\n",options:{fontSize:13,color:C.ink}},
    {text:"Higher ε ⇒ taller hump (more signal allowed through).\n\n",options:{fontSize:13,color:C.accent2}},
    {text:"Opposite phase to σ — let signal through exactly when noise is lowest.",options:{fontSize:13,italic:true,color:C.accent}},
  ], { x:0.5,y:1.7,w:2.6,h:4.3, valign:"top" });
  s.addNotes("The clip shape: a mid-training hump, anti-phase to the noise spike. Together: noisy and tightly-clipped early, then open up the clip and quiet the noise through the productive middle. Taller hump at larger epsilon.");

  // R: no-momentum table
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Without momentum, the win is much larger", "No-momentum regime · test accuracy at T = 7000, ε = 8");
  s.addTable([
    [{text:"Dataset",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Learned",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Constant",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Dynamic",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Median-Clip",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Δ vs Constant",options:{bold:true,fill:{color:C.accent},color:C.white}}],
    [{text:"MNIST",options:{bold:true}}, {text:"98.70",options:{bold:true,color:C.accent}}, "96.18", "98.05", "94.48",
     {text:"+2.52",options:{bold:true,color:C.accent2}}],
    [{text:"Fashion-MNIST",options:{bold:true}}, {text:"87.68",options:{bold:true,color:C.accent}}, "85.10", "86.66", "84.51",
     {text:"+2.58",options:{bold:true,color:C.accent2}}],
  ], { x:0.7,y:1.9,w:11.9, fontSize:15, rowH:0.7, valign:"middle",
       border:{type:"solid",color:C.line,pt:1}, color:C.ink, align:"center" });
  bullets(s, [
    {t:"Gains jump from ≈ +1 pt (with momentum) to ≈ +2.5 pt (without).", b:true, c:C.accent},
    "Learned also clears the strong Dynamic-DPSGD baseline at every cell.",
    "Without momentum's smoothing, how you spend the budget matters far more — exactly where the schedule pays off.",
  ], {x:0.7,y:4.0,w:11.9,h:2.2});
  s.addNotes("The bigger story. Strip momentum and the learned schedule's edge over constant roughly doubles to +2.5 points, and it still beats the strong dynamic baseline everywhere. Intuition: momentum was doing some of the smoothing for free; without it the schedule earns its keep.");

  // R: no-momentum delta grows with T
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "And the gain grows with training length", "No-momentum · (Learned − Constant) accuracy vs T");
  fig(s, "NM_DELTA", {x:3.3,y:1.6,w:9.4,h:4.4});
  caption(s, "Above zero everywhere — including ε = 1 — and rising with T on MNIST.", {x:3.3,y:6.35,w:9.4});
  s.addText([
    {text:"Read-off\n\n",options:{bold:true,fontSize:15,color:C.ink}},
    {text:"Positive at every ε, including ε = 1 (unlike the momentum case).\n\n",options:{fontSize:13,color:C.accent}},
    {text:"MNIST: gap widens steadily to ≈ +2.5 pt at T = 7000.\n\n",options:{fontSize:13,color:C.ink}},
    {text:"Longer DP-SGD runs benefit more from a learned schedule.",options:{fontSize:13,italic:true,color:C.accent2}},
  ], { x:0.5,y:1.7,w:2.7,h:4.4, valign:"top" });
  s.addNotes("Contrast with the momentum delta slide. Here the gain is positive at every epsilon including 1, and grows with T (clearest on MNIST). The longer you train under DP, the more a learned allocation helps.");

  // R: no-momentum sigma contrast
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Same qualitative schedule, lower noise floor", "No-momentum · learned σ(t)");
  fig(s, "NM_SIGMA", {x:3.2,y:1.7,w:9.5,h:4.0});
  caption(s, "Same early-spike-then-plateau shape; overall σ scale is lower than with momentum.", {x:3.2,y:6.05,w:9.5});
  s.addText([
    {text:"Read-off\n\n",options:{bold:true,fontSize:15,color:C.ink}},
    {text:"The front-loaded spike + decaying plateau is robust — it is not an artifact of momentum.\n\n",options:{fontSize:13,color:C.ink}},
    {text:"The learned shape is a stable, transferable motif.",options:{fontSize:13,italic:true,color:C.accent}},
  ], { x:0.5,y:1.7,w:2.6,h:4.3, valign:"top" });
  s.addNotes("The learned shape survives removing momentum — same early-spike, decaying-plateau motif at a lower overall scale. Strengthens the claim that this is a real property of the DP-SGD objective, not an optimizer artifact.");

  // R: architecture robustness
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "The benefit transfers across architectures", "ε = 8 · (Learned − Constant) vs parameter count");
  fig(s, "ARCH", {x:3.5,y:1.6,w:6.6,h:4.55});
  caption(s, "Every architecture family sits well above zero, for both datasets.", {x:3.5,y:6.3,w:6.6});
  s.addText([
    {text:"Read-off\n\n",options:{bold:true,fontSize:15,color:C.ink}},
    {text:"MLP width/depth and CNN width all show +1 to +2 pt gains.\n\n",options:{fontSize:13,color:C.accent}},
    {text:"No family dips to zero — the schedule is not tuned to one model.\n\n",options:{fontSize:13,color:C.ink}},
    {text:"Mild upward trend with size on MNIST.",options:{fontSize:13,italic:true,color:C.muted}},
  ], { x:0.5,y:1.7,w:3.0,h:4.4, valign:"top" });
  s.addNotes("Architecture ladders at eps=8. Across MLP width, MLP depth, and CNN width, the learned-minus-constant gap stays well above zero on both datasets. The method is not overfit to a single architecture.");

  // R: summary table
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Results at a glance", "Test accuracy; Δ = Learned − Constant (percentage points)");
  s.addTable([
    [{text:"Setting",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Learned",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Constant",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Δ",options:{bold:true,fill:{color:C.accent},color:C.white}}],
    [{text:"MNIST · ε8 · T7000 · momentum",options:{}}, {text:"98.68",options:{color:C.accent,bold:true}}, "97.74", {text:"+0.94",options:{color:C.accent2,bold:true}}],
    [{text:"MNIST · ε8 · T7000 · no momentum",options:{fill:{color:C.tint}}}, {text:"98.70",options:{color:C.accent,bold:true,fill:{color:C.tint}}}, {text:"96.18",options:{fill:{color:C.tint}}}, {text:"+2.52",options:{color:C.accent2,bold:true,fill:{color:C.tint}}}],
    [{text:"Fashion · ε8 · T7000 · momentum",options:{}}, {text:"87.57",options:{color:C.accent,bold:true}}, "86.42", {text:"+1.15",options:{color:C.accent2,bold:true}}],
    [{text:"Fashion · ε8 · T7000 · no momentum",options:{fill:{color:C.tint}}}, {text:"87.68",options:{color:C.accent,bold:true,fill:{color:C.tint}}}, {text:"85.10",options:{fill:{color:C.tint}}}, {text:"+2.58",options:{color:C.accent2,bold:true,fill:{color:C.tint}}}],
    [{text:"MNIST · ε1 · T1500 · no momentum",options:{}}, {text:"96.65",options:{color:C.accent,bold:true}}, "96.06", {text:"+0.59",options:{color:C.accent2,bold:true}}],
  ], { x:0.7,y:1.9,w:11.9, fontSize:14, rowH:0.62, valign:"middle",
       border:{type:"solid",color:C.line,pt:1}, color:C.ink, align:"center" });
  s.addText("Learned wins (or ties) in essentially every cell at ε ≥ 3; the no-momentum regime is where it shines.",
            { x:0.7,y:5.9,w:11.9,h:0.5, fontSize:14, italic:true, color:C.muted });
  s.addNotes("Headline numbers in one place. The no-momentum rows (shaded) carry the strongest gains; momentum rows are smaller but still positive.");

  // ================= SECTION 5: DISCUSSION =================
  sectionSlide(pptx, "PART 5", "Discussion & Conclusion").addNotes("Wrap up: when, caveats, next.");

  // D: when does it help
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "When does a learned schedule help?");
  s.addTable([
    [{text:"Factor",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Helps MORE when…",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Helps LESS when…",options:{bold:true,fill:{color:C.ink},color:C.white}}],
    [{text:"Budget ε",options:{bold:true}}, {text:"Looser (ε = 3–8)",options:{color:C.accent}}, {text:"Tight (ε = 1, noise-dominated)",options:{color:C.accent2}}],
    [{text:"Training length T",options:{bold:true}}, {text:"Longer runs (gain grows with T)",options:{color:C.accent}}, "Very short runs"],
    [{text:"Inner optimizer",options:{bold:true}}, {text:"Plain SGD (no momentum)",options:{color:C.accent}}, {text:"Momentum (already smooths)",options:{color:C.accent2}}],
    [{text:"Architecture",options:{bold:true}}, {text:"Robust across MLP/CNN sizes",options:{color:C.accent}}, "—"],
  ], { x:0.55,y:1.85,w:12.2, fontSize:14, rowH:0.7, valign:"middle",
       border:{type:"solid",color:C.line,pt:1}, color:C.ink });
  s.addText("Rule of thumb: the more the budget actually constrains training, the more a learned allocation buys you.",
            { x:0.55,y:6.0,w:12.2,h:0.5, fontSize:14, italic:true, color:C.accent });
  s.addNotes("The regime map — arguably the most useful slide for practitioners. Looser budget, longer training, plain SGD: all amplify the benefit. Robust across architectures.");

  // D: limitations
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Limitations & honest caveats");
  bullets(s, [
    {t:"Tight-budget regime:", b:true, c:C.accent2}, {t:"at ε = 1 with momentum the gain can vanish or go slightly negative.", sub:true},
    {t:"Scope of data:", b:true}, {t:"results are MNIST-family + small MLP/CNN — not yet ImageNet/transformers.", sub:true},
    {t:"Cost:", b:true}, {t:"the outer loop differentiates through full T-step runs — training the schedule is expensive.", sub:true},
    {t:"Coverage:", b:true}, {t:"a handful of sweep cells are missing runs (artifact/history gaps), excluded from aggregates.", sub:true},
    {t:"Transfer:", b:true}, {t:"schedules are learned per setting; cross-setting reuse is future work.", sub:true},
  ], {w:11.8});
  s.addNotes("Be upfront. Tight-budget-with-momentum is the weak spot. Scope is MNIST-family. The outer loop is compute-heavy. A few missing runs were excluded. Whether a learned shape transfers to new settings is open.");

  // D: conclusion
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Conclusion");
  bullets(s, [
    {t:"DP-SGD's noise/clip schedule is a tunable hyperparameter — and tuning it helps.", b:true, c:C.accent},
    "We learn σ(t), C(t) by a bilevel outer loop that differentiates through training and projects onto the exact (ε, δ) budget.",
    {t:"Learned schedules consistently match or beat constant, dynamic, and median-clip baselines (up to +2.6 pt).", b:true},
    {t:"They reveal a stable motif: front-loaded noise + a mid-training clip hump.", b:true, c:C.accent2},
    {t:"Next:", b:true}, {t:"larger models & datasets; cheaper outer-loop estimators; transferring learned shapes.", sub:true, c:C.muted},
  ], {w:11.8});
  s.addNotes("Land the three points: (1) the schedule matters, (2) our constrained bilevel method optimizes it safely, (3) consistent gains plus an interpretable learned shape. Then future work.");

  // REFERENCES
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "References");
  const REFS = [
    ["1","M. Abadi et al. “Deep Learning with Differential Privacy.” ACM CCS, 2016."],
    ["2","C. Dwork & A. Roth. “The Algorithmic Foundations of Differential Privacy.” Found. & Trends in TCS, 2014."],
    ["3","J. Dong, A. Roth & W. J. Su. “Gaussian Differential Privacy.” J. Royal Statistical Society: Series B, 2022."],
    ["4","D. Maclaurin, D. Duvenaud & R. P. Adams. “Gradient-based Hyperparameter Optimization through Reversible Learning.” ICML, 2015."],
    ["5","L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi & M. Pontil. “Bilevel Programming for Hyperparameter Optimization and Meta-Learning.” ICML, 2018."],
    ["6","J. Du, S. Li, X. Chen, S. Chen & M. Hong. “Dynamic Differential-Privacy Preserving SGD.” arXiv preprint, 2021."],
    ["7","G. Andrew, O. Thakkar, B. McMahan & S. Ramaswamy. “Differentially Private Learning with Adaptive Clipping.” NeurIPS, 2021."],
    ["8","R. Shokri, M. Stronati, C. Song & V. Shmatikov. “Membership Inference Attacks Against Machine Learning Models.” IEEE S&P, 2017."],
  ];
  s.addText([].concat(...REFS.map(r => ([
      {text:"["+r[0]+"]  ",options:{bold:true,color:C.accent}},
      {text:r[1]+"\n",options:{color:C.ink}},
    ]))),
    { x:0.6,y:1.75,w:12.1,h:4.9, fontSize:13.5, color:C.ink, valign:"top", lineSpacingMultiple:1.45, paraSpaceAfter:6 });
  s.addNotes("References. [1] DP-SGD; [2] DP foundations; [3] Gaussian-DP accounting; [4,5] differentiating through training / bilevel HPO; [6] Dynamic-DPSGD baseline; [7] adaptive/median clipping baseline; [8] membership inference motivation.");

  // CLOSING
  s = pptx.addSlide({ masterName:"TITLE" });
  s.addText("Thank you", { x:0.8,y:2.6,w:11.7,h:1.0, fontSize:44, bold:true, color:C.white });
  s.addText("Questions?", { x:0.8,y:3.8,w:11.7,h:0.7, fontSize:22, color:"CBD2D9" });
  s.addText("Learned noise & clip schedules for DP-SGD  ·  front-loaded noise + mid-training clip hump  ·  up to +2.6 pt over constant",
            { x:0.8,y:4.7,w:11.0,h:0.8, fontSize:14, italic:true, color:"9AA5B1" });
  s.addNotes("Close. Restate the one-sentence takeaway and invite questions. Have the deep-dive slides ready if asked about the privacy accounting or projection.");
}
"""

HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Generate deck — DP Percentages</title>
<style>body{font-family:system-ui,sans-serif;margin:3rem;color:#1F2933}
button{font-size:1rem;padding:.5rem 1rem;border:0;background:#2F6F4E;color:#fff;border-radius:6px;cursor:pointer}</style>
</head>
<body>
  <h2>Differentiable Privacy Percentages — talk</h2>
  <p>Your download should start automatically. If not, <button onclick="build()">download talk.pptx</button>.</p>
  <script src="https://cdn.jsdelivr.net/npm/pptxgenjs@4/dist/pptxgen.bundle.js"></script>
  <script>
  let pptx;
  __JS__
  function build(){
    pptx = new PptxGenJS();
    pptx.defineLayout({ name:"W16x9", width:13.333, height:7.5 });
    pptx.layout = "W16x9";
    pptx.author = "Paul Saunders";
    pptx.title = "Differentiable Privacy Percentages";
    defineMasters(pptx);
    buildSlides(pptx);
    pptx.writeFile({ fileName:"DP-Percentages-overview.pptx" });
  }
  window.addEventListener("load", build);
  </script>
</body>
</html>
"""


def main():
    js = JS.replace("__FIGMAP__", fig_map_js())
    out = HTML.replace("__JS__", js)
    dest = ROOT / "presentation.html"
    dest.write_text(out)
    mb = dest.stat().st_size / 1e6
    print(f"wrote {dest}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
