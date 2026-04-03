# Template Data Pipeline 📦

A minimal template for converting existing datasets into the standardized VBVR format. Fork this and customize it for your dataset (HuggingFace, local files, APIs, etc.).

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

# 4. Process dataset
python examples/process.py --num-samples 50
```

---

## Structure

```
template-data-pipeline/
├── core/                        # ✅ KEEP: Standard utilities
│   ├── download.py             # Download raw data (HuggingFace, S3) — delegates to src.download
│   └── pipeline.py             # Process data (BasePipeline, OutputWriter, schemas) — delegates to src.pipeline
├── src/                         # ⚠️ CUSTOMIZE: Your dataset logic
│   ├── download/               # Custom download module
│   │   ├── __init__.py
│   │   └── downloader.py      #    Your download logic (called by core/download.py)
│   └── pipeline/               # Custom pipeline module
│       ├── __init__.py
│       ├── pipeline.py        #    Your pipeline (subclasses BasePipeline)
│       ├── transforms.py      #    Your field mappings (source → standard format)
│       └── config.py          #    Your configuration
├── examples/
│   └── process.py              # Entry point
├── raw/                         # Downloaded raw data (gitignored)
├── eval/                        # (optional) Evaluation tools
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

**Single entry point:** `python examples/process.py --num-samples 50`

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
