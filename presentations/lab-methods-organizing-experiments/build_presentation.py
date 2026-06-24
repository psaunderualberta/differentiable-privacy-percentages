#!/usr/bin/env python3
"""Generate presentation.html — a self-contained PptxGenJS deck for the
45-50 min lab-methods talk: "Treating a Research Repo Like a System".

The talk teaches *transferable* practices for organizing experiments and
results. Every slide leads with a project-agnostic principle; this repo is
shown only as one worked illustration (the "EXAMPLE · THIS REPO" boxes).

Run:  python build_presentation.py
Then: open presentation.html in a browser; it downloads lab-methods.pptx.
"""

import base64
import json
import pathlib
import struct

# project root (presentations/<slug>/build_presentation.py -> ../../)
ROOT = pathlib.Path(__file__).parent.parent.parent
LAD = ROOT / "src" / "cache" / "results" / "psaunder__TestLadderSweep" / "plots" / "sgd"

# JS key -> figure path (only two real figures, used as "payoff" proof-of-output)
FIGS = {
    "FOREST": LAD / "ladders" / "overall" / "arch_forest_delta.png",
    "SIGMA": LAD / "sigma_shape.png",
}


def png_dims(d: bytes):
    w, h = struct.unpack(">II", d[16:24])
    return w, h


def fig_map_js() -> str:
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
            tint:"F4F1EA", muted:"7B8794", line:"DDDDDD", white:"FFFFFF",
            code:"22272E", codefg:"E6EDF3" };
const TALK = "Treating a Research Repo Like a System — organizing experiments & results";
const FIG = __FIGMAP__;

// rt(): "_x ^x _{..} ^{..}" markup -> PptxGenJS runs with real sub/superscript
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

function defineMasters(pptx){
  pptx.defineSlideMaster({
    title:"CONTENT", background:{color:C.white},
    objects:[
      { rect:{ x:0,y:0,w:"100%",h:0.12, fill:{color:C.accent} } },
      { text:{ text:TALK, options:{ x:0.5,y:7.06,w:11.4,h:0.3, fontSize:9, color:C.muted } } },
    ],
    slideNumber:{ x:12.7,y:7.06, fontSize:9, color:C.muted },
  });
  pptx.defineSlideMaster({ title:"SECTION", background:{color:C.ink},
    objects:[ { rect:{ x:0,y:0,w:0.18,h:"100%", fill:{color:C.accent} } } ] });
  pptx.defineSlideMaster({ title:"TITLE", background:{color:C.ink},
    objects:[ { rect:{ x:0,y:6.9,w:"100%",h:0.6, fill:{color:C.accent} } } ] });
}

function head(s, headline, sub){
  s.addText(headline, { x:0.5,y:0.32,w:12.3,h:0.95, fontSize:25, bold:true, color:C.ink, valign:"top" });
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
  s.addText(title, { x:0.8,y:3.0,w:11.7,h:1.6, fontSize:38, bold:true, color:C.white });
  s.addShape(pptx.ShapeType.line, { x:0.85,y:4.5,w:2.5,h:0, line:{color:C.accent,width:3} });
  return s;
}

// dark monospace code/tree box
function mono(s, body, box, opt){
  opt = opt||{};
  s.addShape(pptx.ShapeType.roundRect, { x:box.x,y:box.y,w:box.w,h:box.h,
    fill:{color:opt.fill||C.code}, line:{color:opt.line||C.ink,width:1} });
  s.addText(body, { x:box.x+0.18,y:box.y+0.13,w:box.w-0.34,h:box.h-0.26,
    fontFace:"Consolas", fontSize:opt.fs||11.5, color:opt.color||C.codefg,
    valign:"top", lineSpacingMultiple:1.12, align:"left" });
}

// "EXAMPLE · THIS REPO" callout — separates the concrete instance from the principle
function exbox(s, runs, box){
  s.addShape(pptx.ShapeType.roundRect, { x:box.x,y:box.y,w:box.w,h:box.h,
    fill:{color:"EEF3F6"}, line:{color:C.accent3,width:1} });
  s.addText([{text:"EXAMPLE · THIS REPO\n",options:{bold:true,fontSize:10,color:C.accent3,charSpacing:2}}].concat(runs),
    { x:box.x+0.22,y:box.y+0.16,w:box.w-0.44,h:box.h-0.32, valign:"top",
      fontSize:12.5, color:C.ink, lineSpacingMultiple:1.22 });
}

// "Adopt on Monday" green-bordered takeaway box
function callout(s, title, lines, box, color){
  color = color||C.accent;
  s.addShape(pptx.ShapeType.roundRect, { x:box.x,y:box.y,w:box.w,h:box.h,
    fill:{color:C.tint}, line:{color:color,width:1.5} });
  const runs = [{text:title+"\n",options:{bold:true,fontSize:14,color:color}}];
  lines.forEach(l => runs.push({text:"✓  "+l+"\n",options:{fontSize:13,color:C.ink}}));
  s.addText(runs, { x:box.x+0.25,y:box.y+0.18,w:box.w-0.45,h:box.h-0.3,
    valign:"top", lineSpacingMultiple:1.3 });
}

// horizontal pipeline of labelled chips with ▸ separators
function pipeline(s, steps, box, color){
  color = color||C.accent;
  const n = steps.length, gap = 0.22;
  const w = (box.w - gap*(n-1))/n;
  steps.forEach((st,i)=>{
    const x = box.x + i*(w+gap);
    s.addShape(pptx.ShapeType.roundRect, { x:x,y:box.y,w:w,h:box.h,
      fill:{color: i%2 ? C.tint : "E7F0EA"}, line:{color:color,width:1.25} });
    s.addText([{text:(st.k?st.k+"\n":""),options:{bold:true,fontSize:12.5,color:color}},
               {text:st.t,options:{fontSize:10.5,color:C.ink}}],
      { x:x+0.06,y:box.y+0.08,w:w-0.12,h:box.h-0.16, align:"center", valign:"middle" });
    if(i<n-1) s.addText("▸", { x:x+w-0.04,y:box.y,w:gap+0.08,h:box.h,
      align:"center", valign:"middle", fontSize:15, color:color, bold:true });
  });
}

function buildSlides(pptx){
  let s;

  // ---------- TITLE ----------
  s = pptx.addSlide({ masterName:"TITLE" });
  s.addText("Treating a Research Repo Like a System", { x:0.8,y:1.95,w:11.7,h:1.0, fontSize:38, bold:true, color:C.white });
  s.addText("Transferable practices for organizing experiments & results", { x:0.8,y:3.05,w:11.7,h:0.7, fontSize:21, color:"CBD2D9" });
  s.addText([{text:"Paul Saunders",options:{bold:true}},{text:"   ·   University of Alberta   ·   Lab meeting",options:{color:"9AA5B1"}}],
            { x:0.8,y:4.35,w:11.7,h:0.5, fontSize:15, color:C.white });
  s.addText("One principle, six practices, and the smallest changes that pay off most — framed so you can lift them into your own project, whatever your tooling.",
            { x:0.8,y:5.05,w:11.0,h:0.9, fontSize:13, italic:true, color:"9AA5B1" });
  s.addNotes("Framing: my supervisors liked how I organize experiments/results and asked me to share it. This is a methods talk, not a results talk. Everything is meant to transfer — I'll show my DP-schedules repo only as a worked example, never as something you need to adopt verbatim. Promise the audience: by the end you'll have a short checklist you can start on Monday.");

  // ---------- AGENDA ----------
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Roadmap — six practices, one loop");
  bullets(s, [
    {t:"Why bother — research code rots, and the cost is deferred", b:true, c:C.accent},
    {t:"1 · Fix the vocabulary first", b:true}, {t:"a glossary before tooling", sub:true, c:C.muted},
    {t:"2 · Record decisions, not just code", b:true}, {t:"ADRs & lightweight RFCs", sub:true, c:C.muted},
    {t:"3 · Design the experiment grid deliberately", b:true}, {t:"isolate one variable; generate it", sub:true, c:C.muted},
    {t:"4 · Make metadata self-describing & discoverable", b:true}, {t:"tags & self-describing files", sub:true, c:C.muted},
    {t:"5 · One source of truth → results, no manual steps", b:true}, {t:"a pipeline, not a folder of scripts", sub:true, c:C.muted},
    {t:"6 · Treat interpretation as an artifact", b:true}, {t:"dated, provenance-tagged", sub:true, c:C.muted},
  ], {w:7.6, y:1.55, fontSize:14});
  s.addShape(pptx.ShapeType.roundRect, { x:8.6,y:1.75,w:4.2,h:4.4, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"Then:\n\n",options:{bold:true,fontSize:15,color:C.ink}},
    {text:"▸ Does it pay off?\n",options:{fontSize:14,color:C.accent}},
    {text:"   evidence from the repo\n\n",options:{fontSize:12,color:C.muted}},
    {text:"▸ Small changes, big payoff\n",options:{fontSize:14,color:C.accent2,bold:true}},
    {text:"   the 80/20 to start with\n\n",options:{fontSize:12,color:C.muted}},
    {text:"▸ A starter checklist\n",options:{fontSize:14,color:C.accent}},
  ], { x:8.85,y:2.05,w:3.7,h:3.9, valign:"top" });
  s.addNotes("Walk the six practices quickly. Stress they're ordered roughly by how foundational they are: language underpins everything; interpretation is the payoff end. Flag the two 'meta' sections near the end — evidence it works, and the small-changes-big-payoff distillation for anyone who tunes out the details.");

  // ---------- THESIS (big type) ----------
  s = pptx.addSlide({ masterName:"CONTENT" });
  s.addShape(pptx.ShapeType.line, { x:0.55,y:1.0,w:2.6,h:0, line:{color:C.accent,width:3} });
  s.addText("The whole talk in one line", { x:0.5,y:1.15,w:12.3,h:0.5, fontSize:16, italic:true, color:C.muted });
  s.addText([
    {text:"Nothing is hand-managed.\n",options:{color:C.ink}},
    {text:"Everything is ",options:{color:C.ink}},
    {text:"named",options:{color:C.accent}},
    {text:", ",options:{color:C.ink}},
    {text:"recorded",options:{color:C.accent}},
    {text:",\n",options:{color:C.ink}},
    {text:"generated",options:{color:C.accent}},
    {text:", and ",options:{color:C.ink}},
    {text:"discovered",options:{color:C.accent}},
    {text:".",options:{color:C.ink}},
  ], { x:0.7,y:2.2,w:12.0,h:2.8, fontSize:40, bold:true, lineSpacingMultiple:1.05, valign:"top" });
  s.addText([
    {text:"named ",options:{bold:true,color:C.accent}},{text:"a shared vocabulary    ",options:{color:C.muted}},
    {text:"recorded ",options:{bold:true,color:C.accent}},{text:"decisions & interpretations as files    ",options:{color:C.muted}},
    {text:"generated ",options:{bold:true,color:C.accent}},{text:"experiments from code    ",options:{color:C.muted}},
    {text:"discovered ",options:{bold:true,color:C.accent}},{text:"structure from conventions",options:{color:C.muted}},
  ], { x:0.6,y:5.4,w:12.2,h:1.2, fontSize:14, lineSpacingMultiple:1.4, valign:"top" });
  s.addNotes("The single takeaway. Each of the four verbs maps to practices: named=glossary; recorded=ADRs+interpretations; generated=experiment generation+pipeline; discovered=tags/conventions. If they remember only this sentence, the talk worked. Everything else is how to live it.");

  // ================= WHY =================
  sectionSlide(pptx, "MOTIVATION", "Why bother organizing at all?").addNotes("Earn the right to give advice: name the pain everyone in the room has felt.");

  // why 1 — research code rots
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Research code rots faster than product code");
  bullets(s, [
    {t:"It optimizes for the next result, not the next reader.", b:true},
    "Experiments multiply faster than anyone can remember them.",
    "The same word means three things across three scripts.",
    "Runs get re-launched because no one is sure the old ones are trustworthy.",
    "The reason a choice was made lives only in your head — until it doesn't.",
    {t:"None of this shows up in code review. It shows up six months later.", b:true, c:C.accent2},
  ], {w:6.4});
  s.addShape(pptx.ShapeType.roundRect, { x:7.1,y:1.95,w:5.6,h:4.2, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"Sound familiar?\n\n",options:{bold:true,fontSize:16,color:C.ink}},
    {text:"“Which run produced this plot?”\n\n",options:{fontSize:15,italic:true,color:C.ink}},
    {text:"“Wait, does ‘batch size’ mean the\n  inner or the outer one here?”\n\n",options:{fontSize:15,italic:true,color:C.ink}},
    {text:"“Why did we drop that metric?”\n\n",options:{fontSize:15,italic:true,color:C.ink}},
    {text:"“Just re-run it to be safe.”",options:{fontSize:15,italic:true,color:C.accent2}},
  ], { x:7.4,y:2.3,w:5.0,h:3.6, valign:"top" });
  s.addNotes("Get nods. These four quotes are the symptoms; the six practices are each a direct antidote to one or more. The point isn't discipline for its own sake — it's that the cost is real and deferred onto future-you and your readers/reviewers.");

  // why 2 — cost paid by future-you
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "The cost is paid later — by future-you and your reader");
  bullets(s, [
    {t:"Disorganization is a loan against your future time, at high interest.", b:true},
    "Writing up the thesis/paper: you reverse-engineer what you already knew.",
    "A reviewer asks “what about architecture X?” — and you can't tell if you ran it.",
    "A labmate inherits the project and starts from zero.",
    {t:"Good organization is just paying that cost down to near-zero, up front.", b:true, c:C.accent},
    {t:"Every practice here trades a few minutes now for hours — or a result — later.", b:true, c:C.accent},
  ], {w:11.8});
  s.addNotes("Reframe organization as an investment with a concrete return, not as tidiness or perfectionism. This sets up the 'small changes, big payoff' section: the best practices have a tiny up-front cost and a huge deferred return.");

  // why 3 — the loop
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "One loop you can adopt incrementally", "You don't need all of it at once — each stage stands alone and compounds");
  pipeline(s, [
    {k:"NAME", t:"shared vocabulary"},
    {k:"DECIDE", t:"record the why"},
    {k:"STRUCTURE", t:"design the grid"},
    {k:"RUN", t:"generate + fan out"},
    {k:"COMPILE", t:"discover + plot"},
    {k:"INTERPRET", t:"dated artifact"},
  ], { x:0.55,y:2.4,w:12.2,h:1.5 });
  s.addText("↺  interpretation feeds the next decision — the loop closes",
            { x:0.55,y:4.15,w:12.2,h:0.5, fontSize:14, italic:true, color:C.accent, align:"center" });
  bullets(s, [
    {t:"Practices 1–6 map onto these six stages.", b:true},
    "Start anywhere a pain is sharpest — a glossary file, or one decision record — and grow outward.",
    "The value is cumulative: naming makes discovery possible; generation makes interpretation cheap.",
  ], {w:12.2, y:4.8, fontSize:14});
  s.addNotes("This is the spine of the talk and we'll return to it at the end. Emphasize: it is NOT all-or-nothing. The most common failure is trying to adopt a whole 'system' and bouncing off. Pick the stage where you hurt most.");

  // ================= PRACTICE 1 =================
  sectionSlide(pptx, "PRACTICE 1", "Fix the vocabulary first").addNotes("Language is load-bearing and almost free. Lead here.");

  // p1 — overloaded words
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Overloaded words silently corrupt experiments");
  bullets(s, [
    {t:"Every project grows words that mean two things.", b:true},
    "Two people say “batch size” and mean different numbers — and nobody notices.",
    "A plot axis labelled “loss” could be train, val, inner, or outer.",
    "The bug isn't in the code; it's in the conversation about the code.",
    {t:"This is the cheapest mistake to prevent and the most expensive to debug.", b:true, c:C.accent2},
  ], {w:6.3});
  exbox(s, [
    {text:'In one repo, four words each meant two things:\n\n',options:{}},
    {text:'“run”',options:{bold:true,fontFace:"Consolas"}},{text:'  = a W&B training run  ',options:{}},
    {text:'OR',options:{bold:true,color:C.accent2}},{text:'  a regression search\n',options:{}},
    {text:'“weights”',options:{bold:true,fontFace:"Consolas"}},{text:'  = model params  ',options:{}},
    {text:'OR',options:{bold:true,color:C.accent2}},{text:'  the schedule’s params\n',options:{}},
    {text:'“checkpoint”',options:{bold:true,fontFace:"Consolas"}},{text:'  = two unrelated files\n',options:{}},
    {text:'“timesteps”',options:{bold:true,fontFace:"Consolas"}},{text:'  = inner steps  ',options:{}},
    {text:'OR',options:{bold:true,color:C.accent2}},{text:'  outer steps\n\n',options:{}},
    {text:'Each collision had already caused a real bug.',options:{italic:true,color:C.muted}},
  ], { x:7.1,y:1.85,w:5.6,h:4.4 });
  s.addNotes("The example is concrete and a little embarrassing on purpose — it makes the point that this happens to careful people. Note 'timesteps' actually caused a renamed CLI flag. The fix is not cleverness; it's writing the words down.");

  // p1 — glossary format
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "A glossary is three columns: term · definition · aliases-to-avoid");
  bullets(s, [
    {t:"The third column is the one most people skip — and it's the important one.", b:true},
    {t:"Term", b:true}, {t:"the one canonical name. Pick it once.", sub:true},
    {t:"Definition", b:true}, {t:"precise enough to settle an argument.", sub:true},
    {t:"Aliases to avoid", b:true}, {t:"the tempting wrong names, listed so they get corrected on sight.", sub:true},
    {t:"One markdown table. No tooling. Versioned with the code.", b:true, c:C.accent},
  ], {w:5.9});
  s.addTable([
    [{text:"Term",options:{bold:true,fill:{color:C.ink},color:C.white,fontSize:12}},
     {text:"Definition",options:{bold:true,fill:{color:C.ink},color:C.white,fontSize:12}},
     {text:"Avoid",options:{bold:true,fill:{color:C.ink},color:C.white,fontSize:12}}],
    [{text:"outer step",options:{bold:true,color:C.accent}}, "one iteration of the meta-loop", {text:"timestep, epoch",options:{color:C.accent2,italic:true}}],
    [{text:"schedule weights",options:{bold:true,color:C.accent}}, "the learnable T-vector", {text:"weights (bare)",options:{color:C.accent2,italic:true}}],
    [{text:"val loss",options:{bold:true,color:C.accent}}, "loss on held-out set", {text:"validation loss",options:{color:C.accent2,italic:true}}],
    [{text:"minibatch size",options:{bold:true,color:C.accent}}, "inner DP-SGD batch", {text:"batch size (bare)",options:{color:C.accent2,italic:true}}],
  ], { x:6.4,y:2.0,w:6.4, colW:[1.7,3.0,1.7], fontSize:11.5, rowH:0.6, valign:"middle",
       border:{type:"solid",color:C.line,pt:1}, color:C.ink });
  caption(s, "Excerpt, lightly adapted, from this repo's UBIQUITOUS_LANGUAGE.md", { x:6.4,y:6.35,w:6.4 });
  s.addNotes("Show how lightweight it is — a plain markdown table in the repo, no tool to install. The 'avoid' column turns the glossary from passive documentation into an active linter you run in your head during review. Mention the idea is borrowed from domain-driven design's 'ubiquitous language'.");

  // p1 — flagged ambiguities
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "The highest-value entries are the ambiguities you flag");
  bullets(s, [
    {t:"A glossary that only lists clean terms is half a glossary.", b:true},
    "Add a section that names each overloaded word and rules on it explicitly.",
    "“Word X means A here and B there; always qualify which.”",
    "This is where you encode the hard-won knowledge of past confusion.",
    {t:"It's also the first thing to hand a new collaborator.", b:true, c:C.accent},
  ], {w:6.1});
  exbox(s, [
    {text:'A “flagged ambiguities” section, verbatim idea:\n\n',options:{}},
    {text:'“run” ',options:{bold:true,color:C.accent}},
    {text:'is overloaded across subsystems: a\nW&B run (one training run) vs a regression\nsearch (keyed by its own id). The resume &\ncheckpoint paths differ and share no code.\n',options:{}},
    {text:'→ Say “W&B run” or “synthesis”, never bare “run”.\n\n',options:{bold:true,color:C.accent2}},
    {text:'Each ruling is one paragraph. There are six.',options:{italic:true,color:C.muted}},
  ], { x:7.0,y:1.85,w:5.7,h:4.3 });
  s.addNotes("This section is what makes the glossary live. It records not just 'what we call things' but 'the mistakes we keep almost making.' Reading it is the fastest way to onboard. Six paragraphs took maybe 30 minutes to write and have saved far more.");

  // p1 — takeaway
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Practice 1 — name things once, everywhere");
  bullets(s, [
    {t:"The principle:", b:true, c:C.accent},
    "Shared language is the substrate every other practice is built on — you can't generate, tag, or discover what you can't name consistently.",
    {t:"Why it's worth it:", b:true, c:C.accent},
    "Minutes to write; prevents a class of bug that code review can't catch and that surfaces at the worst time (write-up, review).",
  ], {w:6.4});
  callout(s, "Adopt on Monday", [
    "Create GLOSSARY.md — three columns.",
    "Add the 5 terms you've already seen confused.",
    "Add a “flagged ambiguities” paragraph for each two-meaning word.",
    "Version it next to the code; fix aliases on sight in review.",
  ], { x:7.0,y:1.7,w:5.8,h:3.6 });
  s.addNotes("Land the principle, then the cost/benefit, then the concrete first step. Keep repeating this three-part structure (principle / why / Monday) for every practice so the audience can pattern-match.");

  // ================= PRACTICE 2 =================
  sectionSlide(pptx, "PRACTICE 2", "Record decisions, not just code").addNotes("Code shows what; it never shows why-not.");

  // p2 — what's an ADR
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Code records what you did — not why, or what you rejected");
  bullets(s, [
    {t:"A diff answers “what changed.” It never answers “why this and not that.”", b:true},
    "An Architecture Decision Record (ADR) is one short markdown file per non-obvious choice.",
    "Numbered, append-only, versioned with the code. You never edit history — you supersede it.",
    "Not just for software architecture — use it for experiment-design decisions too.",
    {t:"The test: “will someone (maybe me) later wonder why this is this way?” If yes, write one.", b:true, c:C.accent},
  ], {w:6.5});
  mono(s, [
    {text:"docs/adr/\n",options:{color:"7EE787",bold:true}},
    {text:"  0001-fit-metrics.md\n",options:{}},
    {text:"  0002-ladder-experiments.md\n",options:{}},
    {text:"  0003-baseline-checkpointing.md\n",options:{}},
    {text:"  0004-self-describing-sweeps.md\n",options:{}},
    {text:"  0005-content-addressed-outputs.md\n",options:{}},
    {text:"  0006-template-regression.md\n",options:{}},
  ], { x:7.2,y:2.6,w:5.5,h:2.6 });
  caption(s, "Six decisions, six files — the project's reasoning, browsable", { x:7.2,y:5.35,w:5.5 });
  s.addNotes("Define ADR plainly. Stress the append-only/supersede discipline (mirrors how you'd cite a retracted decision). And stress the broadening: in a research repo most of the consequential decisions are about experiment design, not class hierarchies. The trigger question is the practical rule.");

  // p2 — anatomy
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "ADR anatomy: five short sections");
  s.addTable([
    [{text:"Section",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Answers",options:{bold:true,fill:{color:C.ink},color:C.white}}],
    [{text:"Context",options:{bold:true,color:C.accent}}, "What forced a decision? What's the situation?"],
    [{text:"Decision",options:{bold:true,color:C.accent}}, "What we're doing — stated as one assertive sentence."],
    [{text:"Why",options:{bold:true,color:C.accent}}, "The reasoning a future reader would otherwise have to reconstruct."],
    [{text:"Considered & rejected",options:{bold:true,color:C.accent2}}, "The alternatives — and the specific reason each lost."],
    [{text:"Consequence",options:{bold:true,color:C.accent}}, "What this now commits us to (good and bad)."],
  ], { x:0.55,y:1.85,w:8.0, colW:[2.6,5.4], fontSize:13.5, rowH:0.78, valign:"middle",
       border:{type:"solid",color:C.line,pt:1}, color:C.ink });
  s.addShape(pptx.ShapeType.roundRect, { x:8.85,y:1.85,w:3.9,h:4.3, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"The title is the\ndecision itself\n\n",options:{bold:true,fontSize:15,color:C.ink}},
    {text:"Write the filename\nas a claim:\n\n",options:{fontSize:13,color:C.muted}},
    {text:"“Fit with scale-aware\nmetrics, not\ncorrelation”\n\n",options:{fontSize:14,italic:true,color:C.accent}},
    {text:"so the index reads\nas a list of\npositions held.",options:{fontSize:13,color:C.muted}},
  ], { x:9.1,y:2.15,w:3.4,h:3.8, valign:"top" });
  s.addNotes("This skeleton is the whole format — reusable for any project. Two craft tips: (1) the Decision should be a single assertive sentence; (2) the title states the decision so the folder listing is a readable summary of every position the project has taken.");

  // p2 — considered & rejected
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "“Considered & rejected” is the section that saves you");
  bullets(s, [
    {t:"Future-you will re-propose the rejected idea — unless it's written down.", b:true},
    "It pre-empts the reviewer's “why didn't you just…” before they ask.",
    "It records the dead ends so nobody pays to rediscover them.",
    {t:"Each rejected option gets one line: the option, then the specific reason it lost.", b:true, c:C.accent},
  ], {w:6.0});
  exbox(s, [
    {text:'From a decision on how to fit schedule equations —\nwhat was rejected and why:\n\n',options:{}},
    {text:'✘ Pooled R² as headline ',options:{bold:true,color:C.accent2}},
    {text:'— hides the per-\ncondition variance that is the whole concern.\n\n',options:{}},
    {text:'✘ Category = raw timestep ',options:{bold:true,color:C.accent2}},
    {text:'— up to 3000\nfree constants; pure memorization.\n\n',options:{}},
    {text:'✘ Category = per seed ',options:{bold:true,color:C.accent2}},
    {text:'— fits seed noise;\npool seeds as repeats instead.\n',options:{}},
  ], { x:7.0,y:1.85,w:5.7,h:4.4 });
  s.addNotes("This is the single most valuable habit in the talk for research specifically. Reviewers and committee members live in the 'why not X' space. Each rejection here is one sentence but encodes a real experiment-design insight. Note the parallel-structure '✘ option — reason' makes it skimmable.");

  // p2 — RFCs
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Decisions you haven't made yet: lightweight RFCs", "Same idea, pointed forward instead of backward");
  bullets(s, [
    {t:"An ADR records a decision taken. An RFC proposes one not yet taken.", b:true},
    "Same numbered-markdown discipline, with a Status: Proposed / Accepted / Superseded.",
    "Lets you think a change through — and get feedback — before touching code.",
    "Great for the refactor you keep meaning to do but haven't justified yet.",
    {t:"Together: a paper trail of both where the project has been and where it's going.", b:true, c:C.accent},
  ], {w:6.4});
  mono(s, [
    {text:"refactors/\n",options:{color:"7EE787",bold:true}},
    {text:"  001-scoped-run-context.md\n",options:{}},
    {text:"  002-budget-projection.md\n",options:{}},
    {text:"  003-schedule-update-step.md\n",options:{}},
    {text:"  004-run-lifecycle.md\n\n",options:{}},
    {text:"# RFC 001\n",options:{color:"7EE787"}},
    {text:"Status: Proposed\n",options:{color:"F0883E"}},
    {text:"Problem: …\nProposed interface: …",options:{}},
  ], { x:7.2,y:2.0,w:5.5,h:3.9 });
  s.addNotes("RFCs are the forward-looking sibling. The only real difference from an ADR is the Status header and that it's written before the work. This is where you reason about a refactor or a new experiment family in the open, and it doubles as a to-do list with justification attached.");

  // p2 — takeaway
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Practice 2 — record the why and the roads not taken");
  bullets(s, [
    {t:"The principle:", b:true, c:C.accent},
    "A decision that isn't written down will be silently re-litigated. Capture the reasoning and the rejected alternatives while they're fresh.",
    {t:"Why it's worth it:", b:true, c:C.accent},
    "Ten minutes per decision; directly answers the reviewer/committee “why not X,” and stops you from re-deriving dead ends.",
  ], {w:6.4});
  callout(s, "Adopt on Monday", [
    "Make a docs/adr/ folder.",
    "Next non-obvious choice → write 0001 (5 sections).",
    "Always fill “Considered & rejected.”",
    "Title the file with the decision itself.",
  ], { x:7.0,y:1.7,w:5.8,h:3.4 });
  s.addNotes("Same three-part close. The reviewer-facing benefit is the one to emphasize for an academic audience — it converts organization directly into smoother paper revisions and defenses.");

  // ================= PRACTICE 3 =================
  sectionSlide(pptx, "PRACTICE 3", "Design the experiment grid deliberately").addNotes("How you lay out the sweep determines what you can conclude.");

  // p3 — confounded sweeps
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "A confounded sweep answers no question cleanly");
  bullets(s, [
    {t:"If two things vary at once, no result can be attributed to either.", b:true},
    "It's the experimental-design lesson, applied to your own runs.",
    "The trap: a single “vary the architecture” sweep that quietly changes two knobs.",
    {t:"Rule: each sweep holds everything fixed but the one variable under study.", b:true, c:C.accent},
    "Cheaper to design right than to re-run after a reviewer points it out.",
  ], {w:6.2});
  s.addShape(pptx.ShapeType.roundRect, { x:7.0,y:1.95,w:5.7,h:4.2, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"Confounded vs controlled\n\n",options:{bold:true,fontSize:16,color:C.ink,align:"center"}},
    {text:"✘  one “arch sweep”:\n",options:{fontSize:15,bold:true,color:C.accent2}},
    {text:"     (16)  →  (512, 256)\n",options:{fontFace:"Consolas",fontSize:13,color:C.ink}},
    {text:"     width AND depth move\n\n",options:{fontSize:13,italic:true,color:C.muted}},
    {text:"✓  two controlled families:\n",options:{fontSize:15,bold:true,color:C.accent}},
    {text:"     width:  (16)(64)(256)\n",options:{fontFace:"Consolas",fontSize:13,color:C.ink}},
    {text:"     depth:  (128)(128,128)…\n",options:{fontFace:"Consolas",fontSize:13,color:C.ink}},
    {text:"     one knob each",options:{fontSize:13,italic:true,color:C.muted}},
  ], { x:7.3,y:2.3,w:5.2,h:3.7, valign:"top" });
  s.addNotes("Concrete confound: a sweep from a tiny net to a wide-and-deep net changes width and depth together, so a difference in the learned result can't be pinned on either. Splitting into one-knob-at-a-time families is the fix. This is the 'change one variable' rule everyone knows but few enforce in their own sweeps.");

  // p3 — name the structure
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Give the structure of your sweep a small vocabulary");
  bullets(s, [
    {t:"Once experiments have structure, name the parts — so plots, tags, and talks agree.", b:true},
    "A handful of words for “the whole sweep direction,” “one controlled family,” “one point,” “the shared reference point.”",
    "The exact words don't matter; having agreed words does.",
    {t:"This is Practice 1 applied to your experiment design.", b:true, c:C.accent},
  ], {w:6.2});
  exbox(s, [
    {text:'The vocabulary this repo settled on:\n\n',options:{}},
    {text:'Axis  ',options:{bold:true,color:C.accent}},{text:'a top-level sweep direction\n',options:{}},
    {text:'Ladder  ',options:{bold:true,color:C.accent}},{text:'one family, one knob varied\n',options:{}},
    {text:'Rung  ',options:{bold:true,color:C.accent}},{text:'one architecture in a ladder\n',options:{}},
    {text:'Anchor  ',options:{bold:true,color:C.accent}},{text:'the rung shared by ladders,\n          the common reference point\n\n',options:{}},
    {text:'Now every plot, tag and conversation uses\nthe same four words. Pick your own — but\npick.',options:{italic:true,color:C.muted}},
  ], { x:7.0,y:1.85,w:5.7,h:4.4 });
  s.addNotes("Don't sell the specific words (Axis/Ladder/Rung/Anchor) — sell the move. The audience should leave thinking 'I should name the parts of my own grid,' not 'I should use the word ladder.' The payoff is that downstream tooling and plot folders can use these names as keys (next practice).");

  // p3 — generate from knobs
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Generate the grid from knobs — don't hand-enumerate it");
  bullets(s, [
    {t:"A hand-typed list of configs is a list of typos waiting to happen.", b:true},
    "Define the knobs; let code expand them into the full set of runs.",
    "Derived quantities (param-matched sizes, memory estimates) get computed, not transcribed.",
    "Adding a condition is editing one knob, not pasting twenty rows.",
    {t:"The generator becomes the single source of truth for “what experiments exist.”", b:true, c:C.accent},
  ], {w:6.3});
  mono(s, [
    {text:"# experiments/architectures.py\n",options:{color:"7EE787"}},
    {text:"LADDERS = {\n",options:{}},
    {text:'  "mlp-width":  widths([16,64,256]),\n',options:{}},
    {text:'  "mlp-depth":  depths(1..4, w=128),\n',options:{}},
    {text:'  "mlp-depth-pm": param_matched(\n',options:{}},
    {text:"        anchor),  # sizes computed\n",options:{color:"8B949E"}},
    {text:"  ...\n}\n\n",options:{}},
    {text:"create_experiments.py\n",options:{color:"79C0FF"}},
    {text:"  expands LADDERS → one run per rung",options:{color:"8B949E"}},
  ], { x:7.2,y:1.95,w:5.5,h:4.2 });
  s.addNotes("The registry-of-knobs pattern. Key insight: anything derived (param-matched widths, per-run memory) is computed from the knobs, never typed by hand, so it can't drift. The generator is now the authoritative answer to 'which experiments are there,' which feeds the self-describing sweep file in Practice 4.");

  // p3 — takeaway
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Practice 3 — isolate one variable, generate the grid");
  bullets(s, [
    {t:"The principle:", b:true, c:C.accent},
    "Design sweeps so each one varies a single thing, name the structure, and produce the run list from code rather than by hand.",
    {t:"Why it's worth it:", b:true, c:C.accent},
    "Clean attribution in your results; no transcription errors; adding conditions is a one-line edit.",
  ], {w:6.4});
  callout(s, "Adopt on Monday", [
    "For your next sweep, write down the one knob it varies.",
    "Name the parts of your grid (4–5 words).",
    "Move the config list into a small generator function.",
    "Compute derived values; never paste them.",
  ], { x:7.0,y:1.7,w:5.8,h:3.6 });
  s.addNotes("Three-part close. The 'one knob per sweep' point is the experimental-rigor win; 'generate' is the engineering win. Both feed Practice 4, where the generated structure becomes machine-discoverable.");

  // ================= PRACTICE 4 =================
  sectionSlide(pptx, "PRACTICE 4", "Make metadata self-describing & discoverable").addNotes("Let tooling read structure off conventions, not hardcoded lists.");

  // p4 — tag don't duplicate
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Tag membership — don't duplicate runs to label them");
  bullets(s, [
    {t:"When one run belongs to several groups, copying it per group is waste and drift.", b:true},
    "Instead, run it once and attach a tag per group it belongs to.",
    "One artifact, multiple memberships — no re-running just to re-label.",
    {t:"Tags are cheap metadata; reruns are expensive compute.", b:true, c:C.accent},
  ], {w:6.2});
  exbox(s, [
    {text:'The shared reference architecture belongs to\nthree controlled families.\n\n',options:{}},
    {text:'✘ Naive: ',options:{bold:true,color:C.accent2}},{text:'run it 3×, once per family\n     — 3× the compute, configs drift\n\n',options:{}},
    {text:'✓ Tagged: ',options:{bold:true,color:C.accent}},{text:'run it once, attach\n',options:{}},
    {text:'     ladder:mlp-width\n     ladder:mlp-depth\n     ladder:mlp-depth-pm\n',options:{fontFace:"Consolas",fontSize:11.5,color:C.ink}},
    {text:'  to the single run.',options:{}},
  ], { x:7.0,y:1.85,w:5.7,h:4.3 });
  s.addNotes("Dedup-by-tag. The anchor architecture is the worst case — it's in three families, so naive duplication triples that run. A prefixed tag per membership keeps it as one run. This sets up the real point: the prefix is a convention tooling can read.");

  // p4 — discover from conventions
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Let tooling discover structure from a convention, not a hardcoded list");
  bullets(s, [
    {t:"A hardcoded list of groups is a thing you must remember to update. You won't.", b:true},
    "Adopt a naming convention — e.g. a tag prefix — and have downstream code key off the prefix.",
    "New group appears? It's discovered automatically; no code edit, nothing to forget.",
    {t:"The convention is the API between “what I ran” and “what I plot.”", b:true, c:C.accent},
  ], {w:6.3});
  mono(s, [
    {text:"# compile step, schematically\n",options:{color:"7EE787"}},
    {text:"for tag in run.tags:\n",options:{}},
    {text:'  if tag.startswith("ladder:"):\n',options:{}},
    {text:"      name = tag.split(\":\")[1]\n",options:{}},
    {text:'      row[f"in_{name}"] = True\n\n',options:{}},
    {text:"# a new ladder:foo tag becomes an\n",options:{color:"8B949E"}},
    {text:"# in_foo column with zero code changes\n",options:{color:"8B949E"}},
  ], { x:7.2,y:2.1,w:5.5,h:3.0 });
  caption(s, "Structure is read off the prefix — the group list is never maintained by hand", { x:7.2,y:5.25,w:5.5 });
  s.addNotes("This is the heart of 'discovered' from the thesis sentence. The compiler doesn't know the ladder names; it learns them from the tags. Adding a family is purely additive. Contrast with the brittle alternative — an enum or list of group names that someone has to keep in sync.");

  // p4 — self-describing files
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Make hand-off files self-describing");
  bullets(s, [
    {t:"Files that flow between tools should carry their own meaning.", b:true},
    "When a per-item value differs across rows, put it in the file — named after the flag it feeds.",
    "The consumer maps column → --flag blindly; it never reconstructs state to recompute.",
    {t:"Adding a new per-item field costs one column, not a parser rewrite.", b:true, c:C.accent},
  ], {w:6.2});
  mono(s, [
    {text:"# sweep file: header = flag names\n",options:{color:"7EE787"}},
    {text:"run_id        mem_per_gpu\n",options:{color:"79C0FF",bold:true}},
    {text:"a1b2c3        16\n",options:{}},
    {text:"d4e5f6        32\n",options:{}},
    {text:"g7h8i9        16\n\n",options:{}},
    {text:"# consumer:\n",options:{color:"8B949E"}},
    {text:"#   for col in header[1:]:\n",options:{color:"8B949E"}},
    {text:"#     args += f\"--{col}={row[col]}\"",options:{color:"8B949E"}},
  ], { x:7.2,y:1.95,w:5.5,h:4.2 });
  s.addNotes("The self-describing-file pattern. The crucial trick: the column header is literally the flag name, so the submit script is a dumb forwarder that needs no knowledge of what the columns mean. Adding a per-run flag (here: memory, which varies by run) is one more column — the open-schema idea.");

  // p4 — takeaway
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Practice 4 — metadata your tooling can read generically");
  bullets(s, [
    {t:"The principle:", b:true, c:C.accent},
    "Encode structure in conventions (prefixes, named columns) so downstream code discovers it, instead of maintaining a parallel list of what exists.",
    {t:"Why it's worth it:", b:true, c:C.accent},
    "New conditions flow through automatically; no brittle lists to keep in sync; no reruns to re-label.",
  ], {w:6.4});
  callout(s, "Adopt on Monday", [
    "Tag runs by membership; never duplicate to label.",
    "Pick a tag/naming convention with a prefix.",
    "Make compile code key off the prefix, not a list.",
    "Name hand-off file columns after the flags they feed.",
  ], { x:7.0,y:1.7,w:5.8,h:3.6 });
  s.addNotes("Three-part close. The unifying idea across all four bullets: replace a thing-you-must-remember-to-update with a convention-that-updates-itself.");

  // ================= PRACTICE 5 =================
  sectionSlide(pptx, "PRACTICE 5", "One source of truth → results, no manual steps").addNotes("A pipeline, not a folder of scripts run by memory.");

  // p5 — pipeline diagram
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "From one config to every plot — automatically", "Each arrow is code; no step is “remember to run X by hand”");
  pipeline(s, [
    {k:"GENERATE", t:"configs → sweep file"},
    {k:"FAN OUT", t:"sweep → cluster jobs"},
    {k:"TRACK", t:"runs → experiment tracker"},
    {k:"COMPILE", t:"runs → tidy table"},
    {k:"PLOT", t:"table → figures"},
  ], { x:0.55,y:2.3,w:12.2,h:1.5 });
  bullets(s, [
    {t:"One definition of the experiments flows all the way to the figures.", b:true},
    "Every stage is re-runnable and leaves a durable artifact (file or tracked run).",
    "A new person can run the whole chain without a tour of “which script first.”",
    {t:"The folder-of-scripts-run-by-memory is exactly what this replaces.", b:true, c:C.accent2},
  ], {w:12.2, y:4.2, fontSize:14});
  s.addNotes("Show the spine. In this repo the stages are create_experiments → run-starter (SLURM via parallel) → W&B → compile_results_fetch → compile_results_plot, but name them generically. The test of a good pipeline: a newcomer reproduces every figure without asking you the order. No 'oh you also have to run this notebook' steps.");

  // p5 — dumb forwarders
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Each stage is a dumb forwarder — it never reconstructs state");
  bullets(s, [
    {t:"A stage should consume its input and pass along — not re-derive what an earlier stage knew.", b:true},
    "If the submit script has to rebuild a config to size a job, knowledge has leaked across a boundary.",
    "Put per-item facts in the artifact at the stage that knows them (Practice 4), and forward.",
    {t:"Thin stages are easy to test, easy to swap, and hard to get subtly wrong.", b:true, c:C.accent},
    "When a stage needs context it shouldn't have, that's a design smell — push the fact upstream.",
  ], {w:6.4});
  s.addShape(pptx.ShapeType.roundRect, { x:7.1,y:1.95,w:5.6,h:4.0, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"Where does this fact live?\n\n",options:{bold:true,fontSize:16,color:C.ink}},
    {text:"✘ submit script recomputes\n   per-run memory from a\n   rebuilt config\n",options:{fontSize:14,color:C.accent2}},
    {text:"   — leaks model details into\n     the launcher\n\n",options:{fontSize:12,italic:true,color:C.muted}},
    {text:"✓ generator writes memory\n   into the sweep file; submit\n   just forwards the column\n",options:{fontSize:14,color:C.accent}},
    {text:"   — launcher stays generic",options:{fontSize:12,italic:true,color:C.muted}},
  ], { x:7.4,y:2.25,w:5.0,h:3.5, valign:"top" });
  s.addNotes("This connects Practice 4 and 5: self-describing files exist precisely so stages can stay dumb. The diagnostic question 'where should this fact live?' is the reusable tool. A launcher that needs to know about model architectures is a sign a fact is in the wrong place.");

  // p5 — resumability
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Design resumability in — don't bolt it on");
  bullets(s, [
    {t:"Long jobs will be interrupted (timeouts, preemption, crashes). Plan for it.", b:true},
    "Checkpoint at a granularity finer than your failure unit — so a kill loses minutes, not days.",
    "Make reruns idempotent: re-submitting resumes, it doesn't restart or double-count.",
    "Decide what an ungraceful kill is allowed to lose — and write that down (an ADR).",
    {t:"Resumability is a design decision, with trade-offs — not an afterthought.", b:true, c:C.accent},
  ], {w:6.5});
  exbox(s, [
    {text:'A real checkpointing decision (its own ADR):\n\n',options:{}},
    {text:'Granularity: ',options:{bold:true,color:C.accent}},{text:'per-candidate, because a\nwhole sweep can exceed the wall clock —\nsweep-level checkpoints wouldn’t prevent\nthe crash they’re meant to.\n\n',options:{}},
    {text:'Accepted cost: ',options:{bold:true,color:C.accent2}},{text:'an ungraceful kill loses\nin-progress progress; that’s recomputable,\nso we don’t pay to insure it.\n',options:{}},
  ], { x:7.0,y:1.85,w:5.7,h:4.3 });
  s.addNotes("Two transferable rules: checkpoint finer than your failure unit; make resubmits idempotent. And crucially, the choice of WHAT you protect vs WHAT you let recompute is itself a decision worth an ADR — ties Practice 5 back to Practice 2.");

  // p5 — content-address outputs
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Content-address outputs so parallel runs can't collide");
  bullets(s, [
    {t:"If two variants write to the same path, one silently corrupts the other.", b:true},
    "Derive the output directory from a hash of everything that defines the job.",
    "Same inputs → same path (safe to resume); different inputs → different path (safe to run side-by-side).",
    {t:"Add a short human-readable prefix so the hash dir is still skimmable.", b:true, c:C.accent},
  ], {w:6.3});
  mono(s, [
    {text:"# directory = f(everything that\n#   defines this job), hashed\n",options:{color:"8B949E"}},
    {text:"identity = {filters, search-space,\n            data-selection, …}\n",options:{}},
    {text:"slug = human_prefix + sha1(identity)[:8]\n\n",options:{}},
    {text:"out/mnist-a1b2c3d4/…\n",options:{color:"7EE787"}},
    {text:"out/fashion-9f8e7d6c/…\n\n",options:{color:"7EE787"}},
    {text:"# same identity → resume\n",options:{color:"8B949E"}},
    {text:"# diff  identity → coexist",options:{color:"8B949E"}},
  ], { x:7.2,y:1.95,w:5.5,h:4.2 });
  s.addNotes("Content-addressing, the same idea git uses. Before this the repo keyed outputs only by target, so filtering one sweep to two datasets collided — one warm-started from and overwrote the other's state, silently corrupting both. Hashing the full job identity makes concurrent variants safe. The human prefix keeps it debuggable.");

  // p5 — takeaway
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Practice 5 — single source of truth, dumb forwarders, resumable");
  bullets(s, [
    {t:"The principle:", b:true, c:C.accent},
    "Wire experiments into a pipeline where one definition flows to figures, each stage is thin and re-runnable, and outputs are addressed so reruns resume and parallel runs don't collide.",
    {t:"Why it's worth it:", b:true, c:C.accent},
    "Reproducible by anyone; survives cluster interruptions; no silent corruption across variants.",
  ], {w:6.4});
  callout(s, "Adopt on Monday", [
    "Write down your stages; make each one re-runnable.",
    "Move per-item facts into the artifact (no re-deriving).",
    "Checkpoint finer than your failure unit.",
    "Derive output paths from a hash of the job's inputs.",
  ], { x:7.0,y:1.7,w:5.8,h:3.6 });
  s.addNotes("Three-part close. This is the most engineering-heavy practice; reassure the audience they can adopt it incrementally — even just 'each stage leaves a durable artifact' is a big step up from notebooks run by memory.");

  // ================= PRACTICE 6 =================
  sectionSlide(pptx, "PRACTICE 6", "Treat interpretation as an artifact").addNotes("The result isn't the plot — it's the dated reading of the plot.");

  // p6 — layout mirrors questions
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Lay out results to mirror the questions you're asking");
  bullets(s, [
    {t:"A flat dump of 200 PNGs answers nothing; a folder tree can answer by its shape.", b:true},
    "Group figures by the question they serve — per-family detail vs the cross-cutting summary.",
    "The path becomes navigation: you find a plot by the question, not by guessing a filename.",
    {t:"Organize outputs the way you'd organize the results section of the paper.", b:true, c:C.accent},
  ], {w:6.1});
  mono(s, [
    {text:"plots/<optimizer>/\n",options:{color:"7EE787"}},
    {text:"  ladders/\n",options:{}},
    {text:"    mlp-width/    ",options:{}},{text:"# one family\n",options:{color:"8B949E"}},
    {text:"    mlp-depth/\n",options:{}},
    {text:"    cnn-width/\n",options:{}},
    {text:"    overall/      ",options:{}},{text:"# cross-family\n",options:{color:"8B949E"}},
    {text:"  shape_variants/\n",options:{}},
    {text:"  curves/\n",options:{}},
  ], { x:7.2,y:2.0,w:5.5,h:3.6 });
  caption(s, "Per-family detail and the cross-family summary live in different places by design", { x:7.2,y:5.7,w:5.5 });
  s.addNotes("The folder hierarchy encodes the question structure: each controlled family gets its own directory of detail plots; 'overall' holds the one figure that compares across families. Finding a result is navigating the questions, not grepping filenames.");

  // p6 — honest chart (real figure)
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Pick the chart that answers the question honestly");
  bullets(s, [
    {t:"The default chart is often the dishonest one.", b:true},
    "A box plot over 3 seeds implies a distribution you don't have.",
    "When the question is “does A beat B everywhere?”, show every point and the comparison directly.",
    {t:"A forest plot: each condition's seeds as dots, a mean marker, a min–max bar — honest about n.", b:true, c:C.accent},
    "Let the chart type follow the claim, not habit.",
  ], {w:5.7});
  fig(s, "FOREST", { x:6.5,y:1.7,w:6.4,h:4.6 });
  caption(s, "Forest plot: per-condition Δ vs baseline, dots = seeds, bar = min–max (no fake box)", { x:6.5,y:6.35,w:6.4 });
  s.addNotes("Real figure from the repo. The decision to use a forest plot instead of a box plot (n=3 is too few for a five-number summary) is itself an ADR. The transferable point: the chart should be chosen to match the claim and be honest about sample size, not picked because it's the library default.");

  // p6 — dated interpretation files
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Write interpretations as dated files, not chat messages");
  bullets(s, [
    {t:"A finding explained in Slack is lost the moment the thread scrolls.", b:true},
    "Write a short markdown file per result set: what it is, what it shows, what's suspicious.",
    "Date it. Reference the exact artifacts it reads. Version it with the code that made them.",
    {t:"Now “what did we conclude from that sweep?” has a durable, citable answer.", b:true, c:C.accent},
  ], {w:6.3});
  mono(s, [
    {text:"results/interpretations/\n",options:{color:"7EE787"}},
    {text:"  dc-sweep-sgd.md\n\n",options:{}},
    {text:"# Interpretation: DC sweep (SGD)\n",options:{color:"79C0FF"}},
    {text:"Artifacts: arch_table.tex,\n",options:{}},
    {text:"  t_sweep_main.png, …\n",options:{}},
    {text:"Date: 2026-05-29\n\n",options:{}},
    {text:"## What it shows\n",options:{color:"79C0FF"}},
    {text:"## Rigor concerns",options:{color:"79C0FF"}},
  ], { x:7.2,y:1.95,w:5.5,h:4.2 });
  s.addNotes("The interpretation is itself an artifact with provenance. It names the exact files it reads and is dated, so when you revisit you know what was true when you wrote it. This is the input to the next decision — closing the loop back to Practice 2.");

  // p6 — shown/inferred/not-shown
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Separate what the plot shows from what you claim");
  bullets(s, [
    {t:"The most common analysis error is sliding from “shown” to “believed” without noticing.", b:true},
    "Tag every statement: is it read directly off the figure, inferred, or not actually shown?",
    "It keeps your own reasoning honest and makes the gaps in evidence visible.",
    {t:"Reviewers do this to you anyway — do it to yourself first.", b:true, c:C.accent},
  ], {w:6.2});
  s.addShape(pptx.ShapeType.roundRect, { x:7.1,y:1.95,w:5.6,h:4.1, fill:{color:C.tint}, line:{color:C.line,width:1} });
  s.addText([
    {text:"Three tags, used inline\n\n",options:{bold:true,fontSize:16,color:C.ink}},
    {text:"[shown]  ",options:{bold:true,fontFace:"Consolas",fontSize:14,color:C.accent}},
    {text:"Learned wins every\n  row at ε ≥ 3\n\n",options:{fontSize:13,color:C.ink}},
    {text:"[inferred]  ",options:{bold:true,fontFace:"Consolas",fontSize:14,color:C.accent3}},
    {text:"gains grow as the\n  budget loosens\n\n",options:{fontSize:13,color:C.ink}},
    {text:"[not shown]  ",options:{bold:true,fontFace:"Consolas",fontSize:14,color:C.accent2}},
    {text:"number of seeds;\n  δ; identical grids",options:{fontSize:13,color:C.ink}},
  ], { x:7.4,y:2.25,w:5.0,h:3.6, valign:"top" });
  s.addNotes("This three-way tagging is a discipline that prevents over-claiming. '[not shown]' is the most valuable — it's the list of things a reviewer will ask about, written by you before they do. It also tells future-you exactly what extra evidence to go collect.");

  // p6 — rigor concerns
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Always record the rigor concerns — especially your own results'");
  bullets(s, [
    {t:"A finding without its caveats is a trap you set for yourself.", b:true},
    "Write down what would make the result wrong: tiny n, missing error bars, absent baseline, confounds.",
    "Catch the implausible before a reviewer does — e.g. “±0.000 std means n=1, not zero variance.”",
    {t:"This section is where organization turns into scientific rigor.", b:true, c:C.accent},
  ], {w:6.3});
  exbox(s, [
    {text:'Self-critique recorded next to the result:\n\n',options:{}},
    {text:'⚠ Variance ±0.000 on baselines is\n   implausibly small — likely n=1 or\n   eval noise, not seed spread. Confirm\n   what ± measures and over how many seeds.\n\n',options:{}},
    {text:'⚠ Some bolded margins are smaller than\n   the reported spread — borderline, not\n   significant. Don’t over-claim.',options:{color:C.ink}},
  ], { x:7.0,y:1.85,w:5.7,h:4.3 });
  s.addNotes("This is the payoff of treating interpretation as an artifact: the same file that records the finding records its weaknesses. The repo's interpretation files literally flag '±0.000 is suspicious' — that's the author reviewing themselves. It both protects against embarrassment and tells you what to fix next.");

  // p6 — takeaway
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Practice 6 — interpretation has provenance");
  bullets(s, [
    {t:"The principle:", b:true, c:C.accent},
    "A result is not the figure — it's the dated, artifact-referencing reading of the figure, with claims tagged by evidence and caveats written down.",
    {t:"Why it's worth it:", b:true, c:C.accent},
    "Findings survive past the conversation; over-claiming is caught early; the write-up is half-drafted already.",
  ], {w:6.4});
  callout(s, "Adopt on Monday", [
    "Lay out the results folder by question.",
    "Pick chart types honest about your sample size.",
    "Write one dated interpretation.md per result set.",
    "Tag claims shown / inferred / not-shown + list rigor concerns.",
  ], { x:7.0,y:1.7,w:5.8,h:3.6 });
  s.addNotes("Three-part close, and the loop is now complete: interpretation feeds the next decision (Practice 2). Note the write-up bonus — dated interpretation files are essentially first drafts of your results section.");

  // ================= PAYOFF =================
  sectionSlide(pptx, "DOES IT PAY OFF?", "What this buys you").addNotes("Show the dividend, briefly.");

  // payoff 1 — figures fall out
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "The payoff: publishable figures fall out of the pipeline");
  bullets(s, [
    {t:"Because the chain is automated and discoverable, the figures regenerate themselves.", b:true},
    "No figure is hand-assembled; re-run the compile/plot stages and they refresh.",
    "Every plot traces back through a tidy table, to tracked runs, to a generated config.",
    {t:"Organization isn't overhead — it's what makes the results trustworthy and cheap to produce.", b:true, c:C.accent},
  ], {w:5.7});
  fig(s, "SIGMA", { x:6.5,y:1.7,w:6.4,h:4.6 });
  caption(s, "Learned noise-scale shape across conditions — emitted by the plot stage, not drawn by hand", { x:6.5,y:6.35,w:6.4 });
  s.addNotes("Real figure. The point isn't the science (front-loaded noise etc.) — it's that this came out of the pipeline with no manual steps, fully traceable to its inputs. That traceability is what lets you trust it and answer 'where did this come from' instantly.");

  // payoff 2 — backfill not rebuild
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Extending a study becomes a backfill, not a rebuild");
  bullets(s, [
    {t:"Reviewer: “what about more privacy budgets / another dataset / a deeper net?”", b:true},
    "Because experiments are generated and structure is discovered, you add a knob value and re-run the chain.",
    "New runs get tagged, discovered, compiled, and plotted with zero downstream edits.",
    {t:"The difference between a one-line backfill and a two-week rebuild is the whole talk.", b:true, c:C.accent2},
    "An exploratory sweep at one budget can grow to the full grid later — by design.",
  ], {w:11.8});
  s.addNotes("This is the consequence the audience cares about most: revisions. The generate+discover machinery means scope changes are additive. Contrast the alternative — a reviewer request triggering a frantic re-wiring of bespoke scripts. Mention the repo deliberately ran ladders at one budget first, with full breadth as a planned backfill.");

  // payoff 3 — newcomer
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "A newcomer can navigate by the artifacts alone");
  bullets(s, [
    {t:"The onboarding path is the repo itself — not a meeting with you.", b:true},
    {t:"GLOSSARY → ", b:true, c:C.accent}, {t:"learn the language", sub:true},
    {t:"ADRs → ", b:true, c:C.accent}, {t:"learn why it's built this way", sub:true},
    {t:"the pipeline → ", b:true, c:C.accent}, {t:"reproduce every figure", sub:true},
    {t:"interpretations → ", b:true, c:C.accent}, {t:"learn what we concluded and what's shaky", sub:true},
    {t:"This is also future-you, six months from now, with no memory of today.", b:true, c:C.accent2},
  ], {w:11.8});
  s.addNotes("Tie it together: the four documentation artifacts form a complete self-serve onboarding. The most important newcomer is future-you. This is the concrete answer to the motivation section's 'the cost is paid later' — here's what you bought.");

  // ================= SMALL CHANGES, BIG PAYOFF =================
  sectionSlide(pptx, "THE 80/20", "Small changes, outsized payoff").addNotes("If they forget everything else, this is the section to remember.");

  // sc 1 — effort x impact map
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Not all of these cost the same — start where effort is low and payoff is high");
  s.addTable([
    [{text:"Practice",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Effort",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"Payoff",options:{bold:true,fill:{color:C.ink},color:C.white}},
     {text:"",options:{bold:true,fill:{color:C.ink},color:C.white}}],
    [{text:"Glossary + flagged ambiguities",options:{bold:true}}, {text:"Tiny",options:{color:C.accent}}, {text:"High",options:{color:C.accent,bold:true}}, {text:"★ start here",options:{bold:true,color:C.accent2}}],
    [{text:"ADRs (esp. considered-&-rejected)",options:{bold:true}}, {text:"Tiny",options:{color:C.accent}}, {text:"High",options:{color:C.accent,bold:true}}, {text:"★ start here",options:{bold:true,color:C.accent2}}],
    [{text:"One naming convention tooling reads",options:{bold:true}}, {text:"Low",options:{color:C.accent}}, {text:"High",options:{color:C.accent,bold:true}}, {text:"★ start here",options:{bold:true,color:C.accent2}}],
    [{text:"Dated interpretation files",options:{bold:true}}, {text:"Low",options:{color:C.accent}}, {text:"High",options:{color:C.accent,bold:true}}, ""],
    [{text:"One-knob sweeps + a generator",options:{bold:true}}, {text:"Medium",options:{color:C.accent3}}, {text:"High",options:{color:C.accent,bold:true}}, ""],
    [{text:"Full automated, resumable pipeline",options:{bold:true}}, {text:"High",options:{color:C.accent2}}, {text:"High",options:{color:C.accent,bold:true}}, {text:"grow into it",options:{italic:true,color:C.muted}}],
  ], { x:0.55,y:1.8,w:12.2, colW:[5.1,2.0,2.0,3.1], fontSize:13, rowH:0.62, valign:"middle",
       border:{type:"solid",color:C.line,pt:1}, color:C.ink });
  s.addText("All six are worth it eventually — but the top three are almost free and pay off immediately.",
            { x:0.55,y:6.35,w:12.2,h:0.4, fontSize:14, italic:true, color:C.accent });
  s.addNotes("This is the anti-overwhelm slide. The detailed practices can blur together; this table re-sorts them by leverage. The three starred rows are pure markdown, no infrastructure, and return value the same week. The pipeline is high-effort — explicitly say 'grow into it,' don't lead with it.");

  // sc 2 — the three to do first
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "If you do three things this week, do these");
  s.addShape(pptx.ShapeType.roundRect, { x:0.55,y:1.8,w:3.95,h:4.2, fill:{color:C.tint}, line:{color:C.accent,width:1.5} });
  s.addText([
    {text:"1\n",options:{bold:true,fontSize:30,color:C.accent}},
    {text:"GLOSSARY.md\n\n",options:{bold:true,fontSize:17,color:C.ink}},
    {text:"≈ 5 minutes.\n\n",options:{fontSize:14,italic:true,color:C.muted}},
    {text:"Three columns + a flagged-ambiguities paragraph.\n\n",options:{fontSize:13.5,color:C.ink}},
    {text:"Saves: a class of bug review can't catch.",options:{fontSize:13.5,bold:true,color:C.accent}},
  ], { x:0.8,y:2.0,w:3.5,h:3.9, valign:"top" });
  s.addShape(pptx.ShapeType.roundRect, { x:4.7,y:1.8,w:3.95,h:4.2, fill:{color:C.tint}, line:{color:C.accent,width:1.5} });
  s.addText([
    {text:"2\n",options:{bold:true,fontSize:30,color:C.accent}},
    {text:"docs/adr/\n\n",options:{bold:true,fontSize:17,color:C.ink}},
    {text:"≈ 10 min per decision.\n\n",options:{fontSize:14,italic:true,color:C.muted}},
    {text:"Plain markdown. Always fill “considered & rejected.”\n\n",options:{fontSize:13.5,color:C.ink}},
    {text:"Saves: the reviewer's “why not X,” answered in advance.",options:{fontSize:13.5,bold:true,color:C.accent}},
  ], { x:4.95,y:2.0,w:3.5,h:3.9, valign:"top" });
  s.addShape(pptx.ShapeType.roundRect, { x:8.85,y:1.8,w:3.95,h:4.2, fill:{color:C.tint}, line:{color:C.accent,width:1.5} });
  s.addText([
    {text:"3\n",options:{bold:true,fontSize:30,color:C.accent}},
    {text:"one convention\n\n",options:{bold:true,fontSize:17,color:C.ink}},
    {text:"≈ an afternoon.\n\n",options:{fontSize:14,italic:true,color:C.muted}},
    {text:"A tag/name prefix your compile code keys off — not a hardcoded list.\n\n",options:{fontSize:13.5,color:C.ink}},
    {text:"Saves: every future condition wires itself in.",options:{fontSize:13.5,bold:true,color:C.accent}},
  ], { x:9.1,y:2.0,w:3.5,h:3.9, valign:"top" });
  s.addText("Two are just markdown files. None needs new infrastructure. All three return value the same week.",
            { x:0.55,y:6.2,w:12.2,h:0.4, fontSize:14, italic:true, color:C.accent2, align:"center" });
  s.addNotes("The single most actionable slide. Keep it concrete: a file, a folder, a convention. Emphasize two of the three are literally just writing markdown — no tooling, no buy-in needed, can start during this meeting. This is the slide to leave on screen during Q&A.");

  // ================= RECAP =================
  sectionSlide(pptx, "RECAP", "The loop, and where to start").addNotes("Close the frame opened at the start.");

  // recap — the loop revisited
  s = pptx.addSlide({ masterName:"CONTENT" });
  head(s, "Six practices, one loop — named, recorded, generated, discovered");
  pipeline(s, [
    {k:"NAME", t:"1 · glossary"},
    {k:"DECIDE", t:"2 · ADRs / RFCs"},
    {k:"STRUCTURE", t:"3 · controlled grid"},
    {k:"RUN", t:"4–5 · tag, generate, pipeline"},
    {k:"COMPILE", t:"4–5 · discover, plot"},
    {k:"INTERPRET", t:"6 · dated artifact"},
  ], { x:0.55,y:2.3,w:12.2,h:1.5 });
  s.addText("↺  each interpretation becomes the next decision",
            { x:0.55,y:4.1,w:12.2,h:0.5, fontSize:14, italic:true, color:C.accent, align:"center" });
  bullets(s, [
    {t:"None of it is exotic — it's markdown files, naming conventions, and thin scripts.", b:true},
    "The discipline is the product: name it, record it, generate it, discover it.",
    {t:"Adopt one stage where you hurt most; let it pull the others in.", b:true, c:C.accent},
  ], {w:12.2, y:4.7, fontSize:14});
  s.addNotes("Return to the spine from slide 6, now annotated with the practice numbers. Reassure: nothing here required special tools — it's mostly markdown and conventions. The meta-advice: don't boil the ocean; start at your sharpest pain.");

  // closing
  s = pptx.addSlide({ masterName:"TITLE" });
  s.addText("Name it. Record it. Generate it. Discover it.", { x:0.8,y:2.45,w:11.7,h:1.0, fontSize:34, bold:true, color:C.white });
  s.addText("Start with a GLOSSARY.md and an ADR folder — today.", { x:0.8,y:3.65,w:11.7,h:0.7, fontSize:20, color:"CBD2D9" });
  s.addText([{text:"Thanks — happy to walk anyone through the repo: glossary, ADRs, the pipeline, the interpretation files.",options:{italic:true}}],
            { x:0.8,y:4.7,w:11.0,h:0.8, fontSize:14, color:"9AA5B1" });
  s.addText("Questions?", { x:0.8,y:5.5,w:11.7,h:0.6, fontSize:20, bold:true, color:C.accent });
  s.addNotes("Close on the one-liner. Concrete call to action: two markdown files, startable today. Offer to give anyone a tour of the actual artifacts in the repo — the live examples are more convincing than the slides. Leave the 'three things' slide up during Q&A if you can.");
}
"""


HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Generate deck — Lab methods</title>
<style>body{font-family:system-ui,sans-serif;margin:3rem;color:#1F2933}
button{font-size:1rem;padding:.5rem 1rem;border:0;background:#2F6F4E;color:#fff;border-radius:6px;cursor:pointer}</style>
</head>
<body>
  <h2>Treating a Research Repo Like a System — lab methods talk</h2>
  <p>Your download should start automatically. If not, <button onclick="build()">download lab-methods.pptx</button>.</p>
  <script src="https://cdn.jsdelivr.net/npm/pptxgenjs@4/dist/pptxgen.bundle.js"></script>
  <script>
  let pptx;
  __JS__
  function build(){
    pptx = new PptxGenJS();
    pptx.defineLayout({ name:"W16x9", width:13.333, height:7.5 });
    pptx.layout = "W16x9";
    pptx.author = "Paul Saunders";
    pptx.title = "Treating a Research Repo Like a System";
    defineMasters(pptx);
    buildSlides(pptx);
    pptx.writeFile({ fileName:"lab-methods-organizing-experiments.pptx" });
  }
  window.addEventListener("load", build);
  </script>
</body>
</html>
"""


def main():
    js = JS.replace("__FIGMAP__", fig_map_js())
    out = HTML.replace("__JS__", js)
    dest = pathlib.Path(__file__).parent / "presentation.html"
    dest.write_text(out)
    mb = dest.stat().st_size / 1e6
    print(f"wrote {dest}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
