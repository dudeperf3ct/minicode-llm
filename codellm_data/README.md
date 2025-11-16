# Data

Write-up: https://dudeperf3ct.github.io/projects/train_llm_part0/

## Pre-requisites

- `uv`

## Getting Started

1. Install the dependencies:

  ```bash
  uv sync
  ```

2. Configure the settings in `config.yaml` as needed.

3. Run the dataset parsing script. It downloads the dataset on the first run if not already present at `data/raw_dataset/<name-of-dataset>` path.

  ```bash
  python codellm_data/main.py --parse
  ```
4. Run the dataset downloading script.

  ```bash
  python codellm_data/main.py --download
  ```
