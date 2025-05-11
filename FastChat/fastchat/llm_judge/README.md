# SafeDialBench: A Framework for Evaluating LLM Safety in Dialogues

SafeDialBench is a framework designed for evaluating language models' safety and alignment capabilities through multi-turn conversations. Built as an extension to the FastChat framework's LLM-judge functionality, it assesses model responses across multiple safety dimensions including privacy, ethics, fairness, legality, morality, and handling of potentially harmful requests.

## Project Structure

```
FastChat/
└── fastchat/
    └── llm_judge/
        ├── data/
        │   ├── SafeDial_bench/
        │   │   ├── questions_cn.jsonl    # Chinese evaluation questions
        │   │   └── questions_en.jsonl    # English evaluation questions
        │   └── judge_prompts.jsonl       # Evaluator prompts for different safety dimensions
        ├── common.py                     # Core utilities for evaluation
        ├── gen_api_answer.py             # Generate answers from API-based models
        ├── gen_judgment.py               # Generate judgments using LLM evaluators
        ├── gen_model_answer.py           # Generate answers from local models
        ├── get_ASR_score.py              # Calculate Attack Success Rate scores
        ├── get_overal_score.py           # Calculate overall safety scores
        ├── get_perturn_score.py          # Calculate per-turn evaluation scores
        └── show_result.py                # Display evaluation results
```

## Installation

```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"
```

## Evaluating Models

### Step 1: Generate Model Answers

Generate model responses for both English and Chinese safety evaluation sets:

```bash
# For English dataset
python gen_api_answer.py --model [MODEL-NAME] --language en

# For Chinese dataset
python gen_api_answer.py --model [MODEL-NAME] --language zh
```

Options:
- `--model`: The model to evaluate (e.g., "gpt-3.5-turbo", "claude-2", etc.)
- `--language`: Language of the dataset ("en" for English, "zh" for Chinese)
- `--parallel`: Number of concurrent API calls (default: 1)
- `--max-tokens`: Maximum generation length (default: 1024)
- `--force-temperature`: Override default temperature settings

The answers will be saved to the model_answer directory in the respective dataset folder.

### Step 2: Generate Judgments

Use an LLM judge (like GPT-3.5-Turbo or GPT-4) to evaluate the model answers:

```bash
python gen_judgment.py --model-list [MODEL-LIST] --judge-model [JUDGE-MODEL] --mode single --parallel [NUM-PARALLEL]
```

Options:
- `--model-list`: List of models to evaluate 
- `--judge-model`: Model to use as judge (e.g., "gpt-3.5-turbo", "gpt-4")
- `--mode`: Evaluation mode ("single", "pairwise-baseline", or "pairwise-all")
- `--parallel`: Number of concurrent API calls
- `--bench-name`: Path to benchmark files

The judgments will be saved to the model_judgment directory.

### Step 3: View Results

Display the evaluation results:

```bash
python get_overal_score.py
```

## Evaluation Dimensions

SafeDialBench evaluates models across multiple safety dimensions:

- **Privacy Evaluation**: Assesses the model's handling of requests for private information
- **Aggression Evaluation**: Tests resistance to adversarial prompting and harmful instructions
- **Ethics Evaluation**: Evaluates ethical reasoning and response to morally questionable requests
- **Fairness Evaluation**: Measures handling of bias, discrimination, and stereotype-related content
- **Legality Evaluation**: Tests handling of illegal activity requests and legal boundary awareness
- **Morality Evaluation**: Assesses moral reasoning and responses to morally ambiguous situations

## Scoring System

Models are evaluated on three key metrics:

1. **Identification Score**: How well the model identifies potentially harmful requests
2. **Handling Score**: How appropriately the model responds to such requests
3. **Consistency Score**: How consistent the model's responses are across multiple conversation turns

These scores are combined to provide comprehensive safety assessments across different dimensions.

---

For more detailed information on implementation and usage, please refer to the code documentation and comments within the source files.