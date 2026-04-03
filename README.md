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
├── core/                    # ✅ KEEP: Standard utilities
│   ├── base_pipeline.py    # Abstract base class (download + process)
│   ├── schemas.py          # Pydantic models (TaskSample, PipelineConfig)
│   ├── downloader.py       # Download module (HuggingFace, etc.)
│   ├── processor.py        # Process module (build standardized samples)
│   ├── output_writer.py    # Write to standardized folder structure
│   ├── image_utils.py      # Image conversion helpers
│   ├── validator.py        # Format validation
│   └── s3.py               # S3 upload / download
├── src/                     # ⚠️ CUSTOMIZE: Your dataset logic
│   ├── pipeline.py         # Your pipeline (subclasses BasePipeline)
│   ├── transforms.py       # Your field mappings (source → standard format)
│   └── config.py           # Your configuration
├── examples/
│   └── process.py          # Entry point
├── eval/                    # (optional) Evaluation tools
└── data/questions/          # Output
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

## Customization (3 Files to Modify)

### 1. Update `src/pipeline.py`

Replace the example VideoThinkBench pipeline with your dataset:

```python
from core import BasePipeline, HuggingFaceDownloader, SampleProcessor, TaskSample

class TaskPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)
        self.downloader = HuggingFaceDownloader(
            repo_id=config.hf_repo,
            split=config.split,
        )

    def download(self):
        yield from self.downloader.download(limit=self.config.num_samples)

    def process_sample(self, raw_sample, idx):
        return SampleProcessor.build_sample(
            task_id=f"my_dataset_{idx:05d}",
            domain=self.task_config.domain,
            first_image=raw_sample["image"],
            prompt=raw_sample["question"],
            final_image=raw_sample.get("target_image"),
        )
```

### 2. Update `src/transforms.py`

Map your source dataset fields to the standard format:

```python
def extract_first_image(raw_sample: dict):
    return raw_sample.get("image")

def extract_prompt(raw_sample: dict) -> str:
    return raw_sample.get("question") or "Solve this task."

def extract_final_image(raw_sample: dict):
    return raw_sample.get("target_image")
```

### 3. Update `src/config.py`

Set your dataset-specific parameters:

```python
from core import PipelineConfig
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
