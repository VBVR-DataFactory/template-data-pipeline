# Template Data Pipeline

A minimal template for converting existing datasets into the standardized VBVR format. Fork this repo and customize it for your dataset.

---

## Ground Truth Video Philosophy

> There may be many videos that could score 100% on EVAL — but our ground truth video **must** score 100%.

The ground truth video is the canonical reference answer. It is not merely *a* correct solution; it is *the* definitive solution that the evaluation system is measured against. If the ground truth itself does not achieve a perfect score on EVAL, then either the ground truth or the evaluation is broken — and that must be fixed before anything else.

---

## Design Philosophy

This template is built around two simple ideas:

1. **Download** — Every dataset needs to be fetched from somewhere (HuggingFace, S3, local files, APIs, etc.). The canonical download orchestration lives in `core/download.py`, which delegates to your custom logic in `src/download/`. You write the downloader; the core handles the plumbing.

2. **Pipeline** — Every dataset needs to be transformed into the standardized VBVR format. The base pipeline machinery lives in `core/pipeline.py`, which delegates to your custom logic in `src/pipeline/`. You write the transforms and field mappings; the core handles writing, validation, and orchestration.

That's it. Download the data, then transform it.

There is also an **Eval** module (`eval/`). It is standalone and optional, but should contain everything needed to evaluate the task — whether that's instructions for human evaluation, rule-based scoring, VLM-as-judge prompts, or anything else. It doesn't depend on `core/` or `src/`.

**Each repo is one task.** Fork this template once per dataset/task you want to convert.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/your-dataset-pipeline.git
cd your-dataset-pipeline

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Generate dataset
python examples/generate.py --num-samples 50
```

---

## Structure

```
template-data-pipeline/
├── core/                        # KEEP: Standard utilities (don't modify)
│   ├── download.py             # Download orchestration — delegates to src/download
│   └── pipeline.py             # Pipeline base class, output writer, schemas — delegates to src/pipeline
├── src/                         # CUSTOMIZE: Your dataset logic
│   ├── download/               # Custom download module
│   │   ├── __init__.py
│   │   └── downloader.py      #    Your download logic (called by core/download.py)
│   └── pipeline/               # Custom pipeline module
│       ├── __init__.py
│       ├── pipeline.py        #    Your pipeline (subclasses BasePipeline)
│       ├── transforms.py      #    Your field mappings (source → standard format)
│       └── config.py          #    Your configuration
├── examples/
│   └── generate.py             # Entry point
├── eval/                        # STANDALONE: Evaluation (optional)
│   ├── verify.py              #    Automated evaluation script
│   └── EVAL.md                #    Evaluation guide & instructions
├── raw/                         # Downloaded raw data (gitignored)
└── data/questions/              # Processed output (gitignored)
```

---

## Output Format

Every pipeline produces:

```
data/questions/{domain}_task/{task_id}/
├── first_frame.png          # Initial state (REQUIRED)
├── final_frame.png          # Goal state (optional)
├── prompt.txt               # Instructions (REQUIRED)
├── first_video.mp4          # Opening segment video (optional)
├── last_video.mp4           # Closing segment video (optional)
├── ground_truth.mp4         # Full video, beginning to end (optional)
└── metadata.json            # Task metadata (optional)
```

---

## Customization (Two Modules to Modify)

`core/download.py` always calls `src/download`, and `core/pipeline.py` always calls `src/pipeline`. Customize the `src/` modules for your dataset.

### 1. Update `src/download/downloader.py`

Define how your dataset is downloaded:

```python
from core.download import HuggingFaceDownloader

class TaskDownloader:
    def __init__(self, config):
        self.hf_downloader = HuggingFaceDownloader(
            repo_id=config.hf_repo,
            split=config.split,
        )

    def download(self, limit=None):
        yield from self.hf_downloader.download(limit=limit)

def create_downloader(config):
    return TaskDownloader(config)
```

### 2. Update `src/pipeline/pipeline.py`

Define how raw samples are processed:

```python
from core.pipeline import BasePipeline, SampleProcessor
from core.download import run_download
from . import transforms

class TaskPipeline(BasePipeline):
    def download(self):
        yield from run_download(self.task_config)

    def process_sample(self, raw_sample, idx):
        return SampleProcessor.build_sample(
            task_id=f"my_dataset_{idx:05d}",
            domain=self.task_config.domain,
            first_image=transforms.extract_first_image(raw_sample),
            prompt=transforms.extract_prompt(raw_sample),
        )
```

### 3. Update `src/pipeline/transforms.py`

Map your source dataset fields to the standard format:

```python
def extract_first_image(raw_sample: dict):
    return raw_sample.get("image")

def extract_prompt(raw_sample: dict) -> str:
    return raw_sample.get("question") or "Solve this task."
```

### 4. Update `src/pipeline/config.py`

Set your dataset-specific parameters:

```python
from core.pipeline import PipelineConfig
from pydantic import Field

class TaskConfig(PipelineConfig):
    domain: str = Field(default="my_dataset")
    hf_repo: str = Field(default="org/dataset-name")
```

**Single entry point:** `python examples/generate.py --num-samples 50`

---

## Eval Module

The `eval/` directory is standalone — it does not depend on `core/` or `src/`. It should contain everything needed to evaluate the task outputs. This could be:

- **Rule-based evaluation** — automated scoring scripts (see `eval/verify.py`)
- **Human evaluation** — rubrics, guidelines, comparison templates
- **VLM-as-judge** — prompts and scripts for using vision-language models as evaluators
- **Any combination** — whatever fits your task

See `eval/EVAL.md` for the full evaluation guide.

---

## S3 Upload / Download

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID="key"
export AWS_SECRET_ACCESS_KEY="secret"
export AWS_DEFAULT_REGION="us-east-1"

# Upload
python -c "from core import upload_directory_to_s3; upload_directory_to_s3('data/questions', 'BUCKET', 'datasets/')"

# Download
python -c "from core import download_from_s3; download_from_s3('BUCKET', 'datasets/', 'data/questions')"
```
