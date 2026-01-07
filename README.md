# CuCu

> This is an official repository of *"From National Curricula to Cultural Awareness: Constructing Open-Ended Culture-Specific Question Answering Dataset."*

## Requirements

```bash
pip install openai huggingface_hub pandas tqdm langchain langchain-openai langchain-upstage
export UPSTAGE_API_KEY="..."
export OPENAI_API_KEY="..."
```

## Runs

```bash

python generate_queries.py \
  --input_csv ./data/learning_outcomes.csv \
  --detail_csv ./outputs/detail.csv \
  --simple_csv ./outputs/simple.csv \
  --hq_csv ./outputs/queries.csv \
  --resume
```

```bash
python generate_responses_hf.py \
  --input_csv ./data/queries.csv \
  --output_csv ./outputs/responses.csv \
  --models  openai/gpt-oss-120b,Qwen/Qwen3-235B-A22B,Qwen/Qwen3-Next-80B-A3B-Thinking,deepseek-ai/DeepSeek-R1 \
  --judge_model gpt-4o \
  --resume

```