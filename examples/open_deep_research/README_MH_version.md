# Benchmarking smolagents framework with GAIA

## Create conda env
```bash
conda create -n smolagents-env python=3.11
conda activate smolagents-env
cd smolagents/examples/open_deep_research/
pip install -r requirements.txt
```
## Set up env vars
```bash
# agent model
export DEEPSEEK_API_KEY
# for VLM used in visualizer tool
export TOGETHER_API_KEY
# for web_search tool
export GOOGLE_CSE_ID
export GOOGLE_API_KEY
```

## Run the GAIA benchmark
1. Download the dataset and set env var
```bash
export DATAPATH=<path/to/the/downloaded/data/subset>
```
2. (Optional) Sample the dataset
3. Run the benchmark
```bash
# by default sampled dataset with text only questions will be run
bash run_gaia_benchmark.sh
```