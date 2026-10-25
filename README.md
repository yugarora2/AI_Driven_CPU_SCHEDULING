
# Adaptive CPU Scheduler: ML shortlist project

**Research question:** can a machine-learning model look at a workload and shortlist
the 2-3 scheduling algorithms most likely to be best, so we only simulate those
instead of all of them, without losing much quality?

```
workload  ->  features (10 numbers)  ->  ML ranks the schedulers  ->  keep top-k
          ->  simulate ONLY those k   ->  pick the best of them
```

## Files

| File | What it does |
|---|---|
| `config.py` | Shared settings: families, score weights, reserved test seeds, `SIM_VERSION` |
| `schedulers.py` | Your algorithms (FCFS, SJF, SRTF, Priority NP/P, RR, MLFQ), now returning metrics |
| `candidates.py` | The list of schedulers the model chooses between (RR with 5 quanta, 2 MLFQ setups, ...) |
| `workloads.py` | Synthetic workload generator (6 families, seeded) |
| `features.py` | Turns a workload into 10 numbers |
| `scoring.py` | Turns raw metrics into a score and the "best scheduler" label |
| `generate_data.py` | **Teammates run this** to make data |
| `merge_data.py` | Combines everyone's files into `data/dataset.csv` |
| `inspect_data.py` | Checks the labels are interesting before training |
| `train.py` | Trains models and prints the results tables |
| `test_schedulers.py` | Correctness tests for the simulator |
| `demo_cli.py` | Your original interactive menu (Gantt chart output) |

## Step 0: setup (everyone)

```bash
python -m venv venv
venv\Scripts\activate          # Windows      (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python test_schedulers.py      # must print: All 6 tests passed.
```
If a test fails, STOP and fix it before generating data. A bug in the simulator
silently corrupts every row of the dataset.

## Step 1: generate data (teammates, in parallel)

Same seed = same workload, so every person gets their own **seed range** and nobody
duplicates work. Each run makes `6 families x per-family` workloads.

| Person | Command |
|---|---|
| 1 | `python generate_data.py --seed-start 0     --per-family 500 --out data/part_p1.csv` |
| 2 | `python generate_data.py --seed-start 10000 --per-family 500 --out data/part_p2.csv` |
| 3 | `python generate_data.py --seed-start 20000 --per-family 500 --out data/part_p3.csv` |
| 4 | `python generate_data.py --seed-start 30000 --per-family 500 --out data/part_p4.csv` |
| 5 | `python generate_data.py --seed-start 40000 --per-family 500 --out data/part_p5.csv` |
| **one person only** | `python generate_data.py --seed-start 900000 --per-family 100 --out data/part_final.csv` |

* The `900000` file is the **final test set**. Nobody trains, tunes or peeks at it
  until the very end (`train.py` enforces this).
* Rules: use the same `config.py` / code version as everyone; send your CSV to whoever merges.
* Speed: about 30 seconds per 3000 workloads on one CPU core. That is cheap, so more
  data costs almost nothing.

## Step 2: merge and inspect (one person)

Put all `part_*.csv` files in `data/`, then:
```bash
python merge_data.py       # writes data/dataset.csv (refuses mixed SIM_VERSIONs, drops duplicates)
python inspect_data.py     # label distribution, per-family winners, verdict
```
You want a **spread** of winners, with no single scheduler above ~60%. If the verdict
says "too easy", change `WEIGHTS` in `config.py`. Labels are recomputed from raw
metrics every time, so no regeneration is needed.

## Step 3: train and evaluate

```bash
python train.py            # development results (cross-validation + leave-one-family-out)
python train.py --final    # ONE time, at the end, on the reserved seeds
```
Outputs `ranker.joblib` (saved models) and `tree_rules.txt` (the decision tree as readable
if/else rules, great for the report).

How to read the table:
* **hit rate**: how often the shortlist contained a truly best scheduler.
* **regret**: score gap to the best scheduler after verifying the shortlist (0 = perfect).
* **avg WT / avg RT**: real average waiting / response time achieved. These columns
  need not move together, because the label optimises the combined score, not one metric.
* **static**: always the same shortlist (best average candidates). The ML must beat this.
* **random**, **oracle** (runs everything), **SRTF reference** (cheats by knowing burst times).

## Key design decisions (write these in the report)

1. **Best = lowest weighted score** of waiting time, response time, worst-case waiting and
   context switches, each normalised across the candidates **per workload**. Turnaround and
   throughput are left out on purpose: turnaround = waiting + burst, and every scheduler here
   is work-conserving, so they carry no extra information.
2. **SJF and SRTF are excluded from the label** (`CLAIRVOYANT` in `config.py`). They need
   the true burst times, which a real OS never has, and SRTF is provably optimal for waiting
   time so it would win ~70-90% of workloads. They are still simulated and reported as
   reference lines. Set `LABEL_INCLUDE_CLAIRVOYANT = True` to include them.
3. **Splits:** random cross-validation (did it learn anything?), leave-one-family-out (does it
   generalise to unseen kinds of workload?), reserved final test seeds (touched once).
4. **Offline tool.** The model needs the whole workload to compute features, so this is a
   "replay a known workload and find a good scheduler fast" tool, not a live adaptive kernel
   scheduler. Say so in the report.

## Common mistakes

* Evaluating on training data, or training on the final-test seeds.
* Changing schedulers/workloads/features without bumping `SIM_VERSION`.
* Reporting numbers you have not measured. Write results in the CV/LinkedIn bullet only after
  you have run `train.py` on the full merged dataset.

## What to do next (extension ideas, one per teammate)

* New workload families, or real traces (Parallel Workloads Archive, SWF format) as an extra test set.
* More candidates (RR with other quanta, MLFQ with a priority boost, SJF using predicted bursts).
* Sensitivity analysis: re-label with 3 different `WEIGHTS` and check the conclusions hold.
* Plots: hit rate vs k, regret vs k, winner share per family (matplotlib).
* The live version: windowed features from past jobs only, switching policy every N time units.
