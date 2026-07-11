# EasyOPD Demo — Video Recording Script (~60s Jupyter segment)

This is the **~60-second live-notebook segment** that follows the 70s slide deck,
giving a total video of ~120-150s. It walks through `demo.ipynb` cell by cell.

- **Narration** lines are what you say (English, to match the paper video).
- **Action** lines are what you do on screen.
- Timings are approximate; adjust to your pace.

---

## BEFORE RECORDING (do NOT film this)

1. Open `examples/demo/demo.ipynb`.
2. Select the kernel **"Python (OpenAgentRL-sj)"** (top-right kernel picker).
3. **Restart Kernel & Run All once** to warm the cache. The first
   `list_methods()` (cell 4) takes ~2 minutes because it imports all 17 method
   modules; after this, every call is instant.
4. After the warm-up run finishes, **Restart Kernel** and run only the
   **Setup cell (cell 2)** + **Warm-up cell (cell 4)** silently, then clear the
   outputs of cells 6/8/12/14. Now during recording every remaining cell
   returns in well under a second.
5. Have `demo_logs/` present next to the notebook (already there).

> Net effect: while recording, nothing hangs — the slow discovery is already cached.

---

## RECORDING (the ~60s you film)

### [0:00–0:06] Title cell (cell 0, markdown)
- **Action:** Show the title cell with the 3-setting table already visible.
- **Narration:**
  > "EasyOPD is a unified on-policy distillation framework built on verl.
  > The same API drives three settings — cross-tokenizer, self-distillation,
  > and step-wise OPD — and their baselines."

### [0:06–0:16] Discover methods (cell 6)
- **Action:** Run `EasyOPD.list_methods()`. The list of 17 methods appears instantly.
- **Narration:**
  > "Every released method is registered in one place. `list_methods` returns
  > all of them — SimCT, SDPO, SOD, and the baselines ULD, ALM, DSKD, GRPO."

### [0:16–0:32] One call, any method (cell 8)
- **Action:** Run the three `from_hparams` lines. Point at the three lines while it prints.
- **Narration:**
  > "Loading a method is a single call. The *same* `from_hparams` selects
  > cross-tokenizer SimCT, self-distillation SDPO, or step-wise SOD —
  > only the method name and its YAML config change. Baselines load the same way."

### [0:32–0:44] Launch training (cell 10 + markdown 9)
- **Action:** Show the markdown cell 9 with the three `bash run_*.sh` commands.
  Then in cell 10, uncomment `!bash examples/simct/run_simct.sh` and run it.
  Let it scroll until the first `Cross-tokenizer teacher sidecar enabled` /
  `simct/xtok_kd_loss` line appears, then **interrupt the kernel**.
  *(Speed this part up 4–8x in editing — model loading takes a few minutes.)*
- **Narration:**
  > "Training is one launch script per method. Here we launch SimCT live;
  > EasyOPD spins up the teacher sidecar and starts logging cross-tokenizer
  > distillation metrics."

### [0:44–0:56] Supervision-specific diagnostics (cell 12, and/or 14)
- **Action:** Run cell 12 (the summary) — or cell 14 to show the *raw* lines
  pulled straight from the finished-run logs in `demo_logs/`.
- **Narration:**
  > "Each method activates only the hooks it needs and logs diagnostics you
  > can't read off accuracy: SimCT reports aligned-span KD loss, SDPO reports
  > its EMA self-teacher and reprompt fractions, and SOD reports the step-level
  > KL it reweights by."

### [0:56–1:00] Closing (cell 15, markdown)
- **Action:** Show the final markdown cell with the GitHub link.
- **Narration:**
  > "Same entry point, three supervision regimes — switch methods by changing
  > the YAML only. EasyOPD is open-source on GitHub."

---

## CELL MAP (quick reference)

| Cell | Type | Role | Film? |
|------|------|------|-------|
| 0  | md   | Title + 3-setting table | show |
| 1  | md   | Setup heading | skip/scroll |
| 2  | code | Path setup (run before recording) | pre-run |
| 3  | md   | Warm-up heading | skip |
| 4  | code | `list_methods()` warm-up (~2min first time) | **pre-run, do not film** |
| 5  | md   | "1. Discover" heading | show |
| 6  | code | `list_methods()` → 17 methods | **film** |
| 7  | md   | "2. Same call" heading | show |
| 8  | code | 3× `from_hparams` (simct/sdpo/sod) | **film** |
| 9  | md   | "3. Launch training" + bash commands | show |
| 10 | code | live `!bash run_simct.sh` (uncomment) | **film (speed up)** |
| 11 | md   | "4. Diagnostics" heading | show |
| 12 | code | diagnostics summary | **film** |
| 13 | md   | "raw log lines" note | optional |
| 14 | code | raw lines from `demo_logs/*.txt` | optional |
| 15 | md   | Closing + GitHub link | show |

---

## PACING TIPS

- Total live segment target: **~60s**. If short on time, drop cell 14 (raw logs)
  and keep cell 12 (summary).
- The only genuinely slow step is cell 4 (first-time discovery, ~2min) — always
  pre-run it. Everything else is sub-second once cached.
- For cell 10, don't wait for real convergence; the goal is to *show the launch
  and the first diagnostic line*, then cut. Speed-ramp in the editor.
