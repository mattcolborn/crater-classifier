# Crater Classifier

A binary image classifier that distinguishes cratered from non-cratered terrain using transfer learning with ResNet18 and PyTorch, deployed as a live Streamlit dashboard with an automated daily inference pipeline.

**Live app:** <https://crater-classifier-cqc7brbhahwawnxds3uhzh.streamlit.app>

---

## What it does

- Classifies 128px terrain patches as **crater** or **no crater**
- Serves an interactive dashboard with three tabs: single-image prediction, training curves, and a log of automated daily runs
- Runs unattended: a scheduled GitHub Actions job performs inference every morning and logs results to a hosted database, which the dashboard reads back

## How it runs

```
GitHub Actions (cron, 06:00 UTC daily)
        │
        ├── loads models/crater_classifier.pth
        ├── runs inference over demo_images/
        │
        ▼
   Supabase  ──  daily_runs table
        │
        ▼
Streamlit Community Cloud  ──  "Daily runs" dashboard tab
```

| Component | Host | Notes                                                                         |
|---|---|-------------------------------------------------------------------------------|
| Dashboard | Streamlit Community Cloud | Deployed from `main`, builds from `requirements.txt`                          |
| Scheduled inference | GitHub Actions | `.github/workflows/daily-run.yml`, cron 06:00 UTC, also manually dispatchable |
| Prediction log | Supabase (Postgres) | `daily_runs` table, London region                                             |
| Model weights | Committed to repo | `models/crater_classifier.pth` (~43 MB)                                       |

Note that the schedule only fires from the default branch, so workflow changes have to reach `main` before they take effect.

## Tech stack

**Machine learning**
- PyTorch, torchvision — ResNet18 with two-stage transfer learning
- Pillow, matplotlib — image handling and training curve plots

**Application and infrastructure**
- Streamlit — dashboard UI, hosted on Streamlit Community Cloud
- Supabase — Postgres backend for prediction logging
- GitHub Actions — scheduled daily inference
- pandas — tabular display of run history

**Tooling and code quality**
- Python 3.12
- Poetry — dependency management and virtual environments
- ruff — linting and formatting
- mypy — static type checking; all modules are fully type-annotated
- Jupyter — exploratory notebook in `notebooks/`

## Project structure

```
crater-classifier/
├── src/
│   └── crater_classifier/
│       ├── __init__.py
│       ├── config.py       # all settings and hyperparameters
│       ├── data.py         # data loading, augmentation, splitting
│       ├── model.py        # ResNet18 model builder
│       ├── train.py        # training loop and fine-tuning
│       ├── evaluate.py     # test evaluation and plotting
│       ├── predict.py      # single image inference
│       └── utils.py        # save and load model
├── notebooks/
│   └── crater_classifier.ipynb
├── .github/
│   └── workflows/          # scheduled daily inference workflow
├── assets/
│   └── training_curves.png # bundled for the dashboard
├── demo_images/            # 19 labelled 128px patches (label encoded in filename)
├── models/
│   └── crater_classifier.pth
├── app.py                  # Streamlit dashboard
├── daily_run.py            # scheduled inference + Supabase logging
├── main.py                 # training entry point
├── pyproject.toml          # Poetry project definition
├── requirements.txt        # dependency list for Streamlit Cloud
├── .gitignore
└── README.md
```

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/mattcolborn/crater-classifier.git
cd crater-classifier
```

**2. Install dependencies with Poetry** (requires Python 3.12)

```bash
poetry install
```

**3. Register the environment as a Jupyter kernel** (optional, for the notebook)

```bash
poetry run python -m ipykernel install --user \
  --name crater-classifier --display-name "crater-classifier"
```

## Configuration

### Local settings

Edit `src/crater_classifier/config.py` to set paths and hyperparameters:

```python
DATA_DIR   = "/path/to/your/data"
OUTPUT_DIR = "/path/to/your/output"
IMAGE_SIZE = 128
NUM_EPOCHS = 20
```

### Secrets

Nothing sensitive is committed. Credentials are supplied in three places depending on context:

| Context | Where | Keys |
|---|---|---|
| GitHub Actions | Repository Secrets | `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` |
| Local dashboard | `.streamlit/secrets.toml` (gitignored) | Supabase URL + publishable key |
| Streamlit Cloud | App settings → Secrets | Supabase URL + publishable key |

The service key is used only by the Actions job, which writes; the dashboard reads with the publishable key.

## Data

Prepare image data in the following structure:

```
data/
├── craters/
│   ├── image_001.jpg
│   └── ...
└── no_craters/
    ├── image_001.jpg
    └── ...
```

Images are automatically split 80% train / 10% validation / 10% test.

## Training

```bash
poetry run python main.py
```

The best model (by validation accuracy) is saved to `OUTPUT_DIR/crater_classifier.pth`. The checkpoint stores both `model_state_dict` and `class_to_idx`, so inference does not depend on directory ordering at load time.

### Training approach

Two-stage transfer learning with ResNet18 pretrained on ImageNet:

- **Stage 1** (epochs 1–10): backbone frozen, only the classifier head is trained
- **Stage 2** (epochs 11–20): last ResNet block unfrozen for fine-tuning at a lower learning rate

## Inference

**Via the dashboard** — upload an image on the Predict tab of the live app, or run it locally:

```bash
poetry run streamlit run app.py
```

**Via the daily job** — run the scheduled pipeline manually:

```bash
poetry run python daily_run.py
```

This loads the committed checkpoint, predicts across `demo_images/`, and inserts one row per prediction into the Supabase `daily_runs` table.

## Results and limitations

On the 19-image demo set the model predicts 16/19 correctly; the misses are craters classified as terrain, several of which are visually ambiguous at 128px.

**Known limitation:** the model fails on oblique-angle photographs. The training set consisted mostly of overhead views, in which craters appear as approximate circles, so the learned features do not transfer to the elliptical profiles seen from low angles. This is a dataset composition problem rather than an architecture one, and would be addressed by augmenting the training set with perspective-transformed and obliquely photographed examples.

## Development

```bash
poetry run ruff check .      # lint
poetry run ruff format .     # format
poetry run mypy src          # type check
```

Changes are proposed via pull request and reviewed before merge to `main`.

