# Addition accuracy notebook lab log

## 2026-09-10 — Separate plots for the later five-shot runs

Updated `notebooks/addition_accuracy.ipynb` to pin the completed five-shot
summaries from `runs/deepseek-v4-flash/one-fact-addition-5shot-full/` and
`runs/deepseek-v4-flash/two-fact-addition-5shot-cyclic/`. Their configurations
were created on September 3 at 22:00 and 22:05 UTC, respectively, at repository
revision `92e3e69ff4e9c5006ba28553fae9d4ab365da37f`. Both used DeepSeek V4 Flash
0731 MXFP4/BF16, five fixed demonstrations, seed 42, greedy decoding, and
filler lengths 0, 10, 20, 50, and 100 before `Answer:` in every user turn.

The notebook now has exactly two figures: one-fact exact-answer accuracy
(262 facts per condition) and two-fact exact-answer accuracy (260 mixed-fact
cyclic pairs per condition). The cyclic run retains the full mixed-fact scope;
the later atomic-only 200-pair run is a separate experiment. Removed the old
combined accuracy and paired-change displays. Reused the existing plotting
helper with explicit summary paths, preserving its defaults for other callers.
The notebook documents the 95% Wilson formula and asymmetric error-bar lengths.
Its pointwise intervals do not account for shared facts across cyclic pairs,
paired differences across conditions, or variation between model runs.

| Dots | One-fact correct / 262 | Two-fact correct / 260 |
|---:|---:|---:|
| 0 | 137 | 20 |
| 10 | 194 | 13 |
| 20 | 182 | 17 |
| 50 | 206 | 15 |
| 100 | 214 | 17 |

Validation: executed all five code cells using `.venv-sglang/bin/python` with
IPython's terminal shell and one CPU thread; checked five demonstrations,
condition coverage, and counts against run configurations, then independently
reconciled every plotted correct/total count with saved `results.json` records.
Confirmed exactly two embedded PNG outputs. Initial execution through the base
IPython shell failed because it does not implement the GUI hook used by the
inline magic; terminal-shell execution succeeded. PNGs were serialized explicitly
because terminal display defaults omit rich image output. No inference or Slurm
job action was performed.

Exports: `runs/deepseek-v4-flash/plots/{one-fact,two-fact}-addition-5shot-accuracy-vs-filler.{pdf,png}`.

## 2026-09-10 — One-fact baseline band and separate markers

Updated the one-fact notebook cell to pass `connect_points=False` and
`baseline_band=True` to `filler/addition/accuracy_plot.py`. The helper now
supports these optional presentation settings: nonzero filler conditions retain
their markers and Wilson error bars, and the zero-filler point and tick are
replaced by a full-width gray baseline interval with a centered white
“Baseline” label. Its bounds are 0.462528–0.582612 (137/262 correct).
Returned data retain all five conditions for the notebook's numerical table.
Preserved the user's removal of the legend. The two-fact cell retains its
existing plotting options.

Reexecuted all five notebook code cells with `.venv-sglang/bin/python`,
IPython's terminal shell, and one CPU thread, refreshing embedded images and
the PNG/PDF exports above. Checked the four nonzero x coordinates, absent
connecting lines and legend, baseline band bounds and white label, and retained
five-condition data; visually inspected the exported one-fact PNG. No model
inference or scheduler action was needed. The process exited successfully;
environment cleanup emitted MUNGE socket warnings after validation completed.

## 2026-09-12 — Added the five-shot k=5 supplement

The one-fact plot now includes **k=0,5,10,20,50,100**, retaining all 262 fact/addend pairs. The new k5 run scored **186/262 (70.9924%)**, with 95% Wilson interval **[65.2255%,76.1525%]**. The other five lengths retain their original 1,310 records. The derived [summary](runs/deepseek-v4-flash/one-fact-addition-5shot-with-k5/summary.json) and [report](runs/deepseek-v4-flash/one-fact-addition-5shot-with-k5/REPORT.md) identify the different runtimes; intervals do not estimate runtime variance.

Updated notebook condition validation and explanatory text, retained the baseline band, separate markers and no legend, and refreshed the existing 600-DPI notebooks/one_fact_eval.png plus standard PNG/PDF exports. Preserved the two-fact notebook output and figure files. Full preflight had 126 passing tests; the independent audit verified all merged counts, original records and Wilson endpoints. [Publication checks](runs/deepseek-v4-flash/one-fact-patching-filler-repeat-1to4/publication.json).

## 2026-09-12 — Random versus copy comparison on discovery prompts

Added a dedicated comparison plot to `notebooks/addition_accuracy.ipynb`, below
its five-shot repeat plot. The random/copy experiments share 96 discovery
prompts and canonical single-token greedy scoring; the existing five-shot
plot uses 262 examples and full-response integer parsing. The new plot therefore
has its own axes and clean reference, with explicit dataset and scoring labels.
The x-axis is j+1 untouched fillers (1–6), matching the preceding repeat plot's
axis convention; small horizontal offsets separate the two policies visually.

Loaded the completed random campaign's `comparison/summary.json` and reconciled
all 12 counts against its 1,152-row `comparison/per_example.csv`. Random correct
counts are 70,74,61,60,63,76 and historical copy counts are 62,31,40,49,64,70,
each out of 96. All clean correctness labels agree, at 83/96. The plot uses the
previously independently audited 95% whole-panel bootstrap intervals (2,000
shared resamples, seed 42), including the clean reference band. The notebook
explains the single-noise-realization limitation, differing norm policies,
runtime baselines, and changing replacement count.

Executed the added cell in the notebook's selected `mech-int-sglang` Jupyter
kernel, using `.venv-sglang/bin/python`, Matplotlib 3.10.6, and the user's actual
`~/.config/matplotlib/matplotlibrc`: serif Computer Modern, size 11, TeX enabled.
Verified `cmr10.tfm` lookup and actual inline PNG output, preserved the existing
notebook cells/outputs, and visually inspected the new figure. Labels, legend,
and all confidence intervals fit without clipping or overlap. The sandbox
blocked Jupyter local sockets; the rendering check succeeded through host
execution. Kernel cleanup emitted a benign destructor warning after successful
validation. No model execution or scheduler action was performed.

Exports: [PNG](notebooks/plots/filler-random-vs-copy-accuracy.png) and
[PDF](notebooks/plots/filler-random-vs-copy-accuracy.pdf).
[Validation and input hashes](runs/deepseek-v4-flash/one-fact-patching-filler-random/notebook-validation/validation.json),
[reproduction script](runs/deepseek-v4-flash/one-fact-patching-filler-random/notebook-validation/validate.py),
and [inline image](runs/deepseek-v4-flash/one-fact-patching-filler-random/notebook-validation/inline.png).
