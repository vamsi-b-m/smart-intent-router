# Smart Router — Phase 0 (local model + app, no MLOps tooling yet)

An intent-routing model for a chat application: given an incoming message,
predict which downstream agent should handle it (`billing_agent`,
`tech_support_agent`, `chitchat_agent`, `general_agent`, or
`fallback_human` for out-of-scope queries).

This phase has **no DVC, no MLflow, no Kubernetes** on purpose — the goal
is to prove the model + serving idea works before wrapping any tooling
around it. That comes in later phases.

## What's here

```
src/
  data/load_data.py    # downloads CLINC150, maps 151 intents -> 5 routes, writes CSVs
  train/baseline.py    # TF-IDF + Logistic Regression, saves a joblib pipeline
  eval/evaluate.py     # scores the trained pipeline on the held-out test set
  serve/app.py          # FastAPI app exposing POST /route
tests/test_app.py       # pytest suite against the live app (via TestClient)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run it end to end

```bash
# 1. Download CLINC150 and build train/val/test CSVs
python src/data/load_data.py

# 2. Train the baseline pipeline
python src/train/baseline.py

# 3. Evaluate on the test set
python src/eval/evaluate.py

# 4. Run the tests
pytest tests/ -v

# 5. Serve it
uvicorn src.serve.app:app --reload --port 8000
```

Then, in another terminal:

```bash
curl -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"text": "why was my payment declined"}'
```

## Results (this run)

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 0.944 | 0.819 |
| Macro-F1 | 0.843 | 0.771 |

**Known limitation:** `fallback_human` recall on the test set is only
0.17 — the model rarely recognizes truly out-of-scope messages, because
CLINC's OOS test split (1,000 examples) is 10x larger than what the model
saw of that class in training (100 examples). It also has no real semantic
understanding — e.g. "what's the weather in Paris" gets misrouted to
`tech_support_agent` with 96% confidence, because TF-IDF only matches
surface vocabulary, not meaning.

These aren't bugs to silently fix — they're exactly the kind of gap that
later phases are designed to catch and respond to:
- **Phase 1 (MLflow)** gives you a place to track this metric run over run
  as you try fixes (e.g. oversampling OOS, or swapping in DistilBERT).
- **Phase 6 (Evidently + monitoring)** is what would catch this kind of
  silent failure in production traffic, not just in a one-off eval script.

## Next: Phase 1

Introduce DVC (data versioning against S3) and MLflow (experiment
tracking + model registry) — still fully local/no-cluster, but now every
training run is versioned and comparable instead of being one throwaway
`joblib` file.






# Smart Router — Phase 1 (DVC + MLflow, still fully local)

An intent-routing model for a chat application: given an incoming message,
predict which downstream agent should handle it (`billing_agent`,
`tech_support_agent`, `chitchat_agent`, `general_agent`, or
`fallback_human` for out-of-scope queries).

Phase 1 adds **DVC** (data + pipeline versioning) and **MLflow**
(experiment tracking + model registry) on top of the Phase 0 model/app —
still no Kubernetes, no cloud, everything runs on your laptop. The point
of this phase is that every training run is now reproducible, comparable,
and the serving app pulls whichever model version was actually promoted,
instead of a hardcoded file.

## What's new in this phase

```
params.yaml               # hyperparameters DVC tracks/diffs across experiments
dvc.yaml                  # pipeline: prepare_data -> train -> evaluate
dvc.lock                  # exact data/code hashes DVC reproduced last time
.dvc/config               # remote storage config (S3, see below)

src/train/baseline.py     # now reads params.yaml, logs everything to MLflow,
                            # registers each run as a new Model Registry version
src/eval/evaluate.py      # scores on test set, logs test_* metrics onto the
                            # SAME mlflow run, and runs promotion logic:
                            # compares against the current @champion alias,
                            # promotes only if strictly better
src/serve/app.py          # loads models:/smart-router@champion straight from
                            # the MLflow registry (falls back to local joblib
                            # if MLflow is unavailable)
```

## Setup

```bash
pip install -r requirements.txt
```

Note: this pins `pathspec==0.11.2` — newer `pathspec>=1.0` breaks DVC 3.55
with a `cannot import name '_DIR_MARK'` error. If you hit that, that's why.

## Run the whole pipeline through DVC

```bash
dvc repro
```

This runs `prepare_data → train → evaluate` in order, skipping any stage
whose inputs haven't changed since the last run. Each `train` run:
- logs params + `val_accuracy`/`val_macro_f1` to MLflow
- registers the fitted pipeline as a new version of the `smart-router`
  registered model
Each `evaluate` run:
- logs `test_accuracy`/`test_macro_f1` onto that same MLflow run
- compares the new version against whichever version currently holds the
  `champion` alias, and promotes it (via `set_registered_model_alias`) only
  if it's strictly better

## Try a few experiment variants

```bash
dvc exp run --name exp-low-C --set-param train.C=1.0
dvc exp run --name exp-unigram-noweight --set-param train.ngram_max=1 --set-param train.class_weight=none

dvc exp show --md   # compare params + all logged metrics across every run
```

In one real run of this: dropping `class_weight: balanced` caused
`fallback_human` F1 to collapse to **0.00** (the model stopped predicting
that class at all) — and the promotion logic correctly refused to
promote it, leaving the better-performing version as champion. That's
the actual value of this phase: bad experiments get caught automatically
instead of silently becoming "the model in production."

Apply an experiment's changes back to your main workspace/git history:

```bash
dvc exp apply exp-low-C
git add -A && git commit -m "adopt exp-low-C as new baseline"
```

## Look at what MLflow tracked

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` — you'll see every run's params/metrics, the
registered `smart-router` model with however many versions you've created,
and which one currently holds the `champion` alias.

## Data versioning / remote storage

```bash
dvc remote add -d storage s3://<your-bucket>/data   # replace with your own bucket
dvc push                                              # uploads data/ + models/ to S3
```

This repo's `.dvc/config` points at a placeholder bucket
(`s3://smart-router-dvc-store/data`) — swap in a real bucket you control
and configure AWS credentials (`aws configure` or env vars) before
`dvc push` will work. Without that, everything above still works locally;
you just won't have an off-machine backup of the DVC cache yet.

## Serve it (now backed by the registry, not a hardcoded file)

```bash
uvicorn src.serve.app:app --reload --port 8000
curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true,"model_version":"smart-router@champion (v2)"}
```

Re-run `dvc exp run` with better params, let it get promoted, restart the
app — it now serves the new version automatically, with zero code changes.

## Results so far (champion as of this run: v2)

| Version | C | ngram_max | class_weight | test_accuracy | test_macro_f1 | Promoted? |
|---|---|---|---|---|---|---|
| v1 (baseline) | 5.0 | 2 | balanced | 0.819 | 0.7714 | ✅ (first version) |
| v2 (exp-low-C) | 1.0 | 2 | balanced | 0.813 | **0.7717** | ✅ (marginally better) |
| v3 (exp-unigram-noweight) | 1.0 | 1 | none | 0.777 | 0.6996 | ❌ (fallback_human F1 → 0.00) |

## Next: Phase 2

Containerize training + serving (Dockerfiles), run MLflow + FastAPI +
Postgres together via Docker Compose, and push images somewhere real —
the last local step before any cloud infra shows up in Phase 3.
 