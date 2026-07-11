# EasyOPD Demo — Cell-by-Cell Recording Script (~60s live-notebook segment)

Follows the 70s slide deck → total video ~130s. **Read the bold "SAY" lines verbatim.**
Timings are targets; English at a normal pace is ~2.7 words/sec.

---

## ⚙️ BEFORE RECORDING (do this off-camera)

1. Open `examples/demo/demo.ipynb`; pick kernel **"Python (OpenAgentRL-sj)"** (top-right).
2. **Run All once** to warm the cache — Cell 4 (`list_methods`) takes ~2 min the first time.
3. When it finishes, **do NOT restart**. Instead just **clear the outputs** of Cells 6, 8, 12, 14
   (Edit ▸ Clear Outputs, or click each cell's output ✗). Keep the kernel alive.
   → Now every code cell re-runs in **under 1 second** on camera.
4. Scroll back to the top. Start recording.

> If you must restart the kernel, re-run Cells 2 and 4 silently first (Cell 4 = the slow 2-min one), then clear outputs again.

---

## 🎬 RECORDING — read the SAY lines, do the DO lines

### CELL 0 — Title (markdown)  · ~7s
**DO:** Show the title cell; the 3-row table (SimCT / SDPO / SOD + baselines) is visible.
**SAY:**
> "This is EasyOPD — a unified on-policy distillation framework built on verl. One API drives three settings: cross-tokenizer, self-distillation, and step-wise OPD, each with its baselines."

*(Skip Cells 1–2, the path setup — scroll past silently, or keep them collapsed.)*

---

### CELL 4 — Warm-up (code: `list_methods()`)  · ~3s
**DO:** Scroll past. It's already run (cached). Don't dwell on it.
**SAY:** *(nothing, or)* "First we import EasyOPD."

---

### CELL 6 — Discover methods (code: `EasyOPD.list_methods()`)  · ~10s
**DO:** Click Run. The list of 17 method names prints instantly.
**SAY:**
> "`list_methods` shows everything that's registered — seventeen methods in total: our SimCT, SDPO, and SOD, alongside baselines like ULD, ALM, DSKD, and GRPO."

---

### CELL 8 — One call, any method (code: three `from_hparams`)  · ~16s
**DO:** Click Run. Three lines print: `simct / sdpo / sod -> loss_mode=...`.
Hover or point at the three `from_hparams` lines while it runs.
**SAY:**
> "Loading any method is a single call. The *same* `from_hparams` selects cross-tokenizer SimCT, self-distillation SDPO, or step-wise SOD — only the method name and its YAML config change. Every baseline loads exactly the same way."

---

### CELL 9 + CELL 10 — Launch training  · ~12s
**DO (Cell 9, markdown):** Show the three `bash run_*.sh` commands.
**DO (Cell 10, code):** Uncomment `!bash examples/simct/run_simct.sh`, run it, let it scroll
until the line `Cross-tokenizer teacher sidecar enabled` or the first `simct/xtok_kd_loss`
appears, then **interrupt the kernel (■)**.
**➡ In editing: speed this clip up 4–8× — model loading takes a few minutes.**
**SAY:**
> "Training is one launch script per method. Here we start SimCT live — EasyOPD brings up the teacher sidecar and immediately begins logging cross-tokenizer distillation metrics."

---

### CELL 12 — Supervision-specific diagnostics (code)  · ~13s
**DO:** Run Cell 12. Three blocks print (SimCT / SDPO / SOD diagnostics).
**SAY:**
> "Each method activates only the hooks it needs, and logs diagnostics you can't get from accuracy alone: SimCT reports its aligned-span KD loss, SDPO reports the EMA self-teacher and reprompt fractions, and SOD reports the step-level KL it re-weights by."

*(CELL 14 is optional — run it only if you want to show the raw log lines. If time is tight, skip it.)*

---

### CELL 15 — Closing (markdown)  · ~6s
**DO:** Scroll to the final cell; the GitHub link is visible.
**SAY:**
> "Same entry point, three supervision regimes — you switch methods by changing the YAML only. EasyOPD is open-source on GitHub."

---

## ⏱️ TIMELINE (target ~60s)

| Time | Cell | What's on screen |
|------|------|------------------|
| 0:00–0:07 | 0 | Title + 3-setting table |
| 0:07–0:10 | 4 | (scroll past warmed-up import) |
| 0:10–0:20 | 6 | 17 methods printed |
| 0:20–0:36 | 8 | 3× from_hparams → loss modes |
| 0:36–0:48 | 9,10 | launch SimCT (sped up) |
| 0:48–1:01 | 12 | diagnostics for 3 methods |
| 1:01–1:07 | 15 | closing + GitHub link |

## 🔑 ONE-LINE TAKEAWAY (if you improvise)
> "One entry point, one config change — switch across cross-tokenizer, self-distillation, and step-wise OPD."

## ⚠️ REMINDERS
- The only slow cell is #4 (first-time discovery, ~2 min) — **always pre-run it**; everything else is sub-second once cached.
- For the live launch (Cell 10): don't wait for convergence — show the launch + first diagnostic line, then cut and speed-ramp.
- If short on time, drop Cell 14 (raw logs) and keep Cell 12 (summary).
