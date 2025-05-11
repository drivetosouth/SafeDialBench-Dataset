#!/usr/bin/env python3
"""
Generate judgments using GPT for SafeDialBench dataset.

Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import dataclasses
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append('D:\FastChat')
from fastchat.llm_judge.common import (
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
)

import os


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = False
    multi_turn: bool = True


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = True


@dataclasses.dataclass
class MatchPair:
    question: dict
    model_1: str
    model_2: str
    answer_1: dict
    answer_2: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = True
def load_questions_mt_bench101(question_file, first_n=None):
    questions = []
    with open(question_file, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if first_n is not None and idx >= first_n:
                break
            question = json.loads(line.strip())
            questions.append(question)
    return questions


def load_model_answers_mt_bench101(answer_path):
    model_answers = {}
    if os.path.isdir(answer_path):
        for model_file in os.listdir(answer_path):
            model_name = args.model_list[0] if args.model_list else os.path.splitext(model_file)[0]
            model_answers[model_name] = {}
            with open(os.path.join(answer_path, model_file), "r", encoding="utf-8") as fin:
                for line in fin:
                    answer = json.loads(line.strip())
                    q_id = answer["id"]
                    model_answers[model_name][q_id] = answer
    else:
        model_name = args.model_list[0] if args.model_list else os.path.splitext(os.path.basename(answer_path))[0]
        model_answers[model_name] = {}
        with open(answer_path, "r", encoding="utf-8") as fin:
            for line in fin:
                answer = json.loads(line.strip())
                q_id = answer["id"]
                model_answers[model_name][q_id] = answer

    return model_answers


def check_data_mt_bench101(questions, model_answers, ref_answers, models, judges):
    missing_questions = {}
    for model in models:
        assert model in model_answers, f"Model {model} not found in model_answers."
        missing_questions[model] = []
        for q in questions:
            q_id = q["id"]
            if q_id not in model_answers[model]:
                missing_questions[model].append(q_id)
                print(f"Warning: Question {q_id} missing in model {model} answers. Marking as missing.")
                model_answers[model][q_id] = {"id": q_id, "choices": [{"turns": [{"message": "NO_ANSWER_AVAILABLE"}]}]}

    print(f"Proceeding with all {len(questions)} questions, with placeholders for missing answers.")

    if ref_answers:
        for judge in judges.values():
            if judge.ref_based:
                assert judge.model_name in ref_answers, f"Reference answers for judge {judge.model_name} not found."
                for q in questions:
                    q_id = q["id"]
                    if q_id not in ref_answers[judge.model_name]:
                        print(f"Warning: Reference answer for question {q_id} missing. Creating placeholder.")
                        ref_answers[judge.model_name][q_id] = {"id": q_id, "reference_answer": "NO_REFERENCE_AVAILABLE"}


def make_match_pair_mt_bench101(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=True,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["history"]) <= 1:
            continue
        q_id = q["id"]
        for i in range(len(models)):
            m_1 = models[i]
            m_2 = baseline_model if baseline_model else models[(i + 1) % len(models)]
            if m_1 == m_2:
                continue
            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[m_2][q_id]
            if ref_answers is not None and judge.ref_based:
                ref = ref_answers[judge.model_name][q_id]
                match = MatchPair(
                    dict(q),
                    m_1,
                    m_2,
                    a_1,
                    a_2,
                    judge,
                    ref_answer=ref,
                    multi_turn=multi_turn,
                )
            else:
                match = MatchPair(
                    dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                )
            matches.append(match)
    return matches


def make_match_single_mt_bench101(
    questions,
    models,
    model_answers,
    judge,
    ref_answers=None,
    multi_turn=True,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["history"]) <= 1:
            print(q["id"])
            continue
        q_id = q["id"]
        for m in models:
            a = model_answers[m][q_id]
            if ref_answers is not None and judge.ref_based:
                ref = ref_answers[judge.model_name][q_id]
                match = MatchSingle(
                    dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                )
            else:
                match = MatchSingle(
                    dict(q), m, a, judge, multi_turn=multi_turn
                )
            matches.append(match)
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def determine_question_type(question):
    """根据问题的任务类型确定使用哪种评估器"""

    task = question.get("task", "")

    # 映射表
    type_mapping = {
        "Privacy": "privacy-evaluation",
        "Aggression": "attack-evaluation",
        "Ethics": "ethics-evaluation",
        "Fairness": "fair-evaluation",
        "Legality": "legal-evaluation",
        "Morality": "moral-evaluation",
        "攻击性": "attack-evaluation",
        "伦理性": "ethics-evaluation",
        "隐私性": "privacy-evaluation",
        "公平性": "fair-evaluation",
        "合法性": "legal-evaluation",
        "道德性": "moral-evaluation"
    }
    if task in type_mapping:
        return type_mapping[task]
    return "ethics-evaluation"  # 默认值

def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["privacy-evaluation"] = Judge(
        judge_model,
        judge_prompts["privacy-evaluation"],
        ref_based = False,
        multi_turn = True,
    )
    judges["ethics-evaluation"] = Judge(
        judge_model,
        judge_prompts["ethics-evaluation"],
        ref_based = False,
        multi_turn = True,
    )
    judges["attack-evaluation"] = Judge(
        judge_model,
        judge_prompts["attack-evaluation"],
        ref_based = False,
        multi_turn = True,
    )
    judges["fair-evaluation"] = Judge(
        judge_model,
        judge_prompts["fair-evaluation"],
        ref_based = False,
        multi_turn = True,
    )
    judges["legal-evaluation"] = Judge(
        judge_model,
        judge_prompts["legal-evaluation"],
        ref_based = False,
        multi_turn = True,
    )
    judges["moral-evaluation"] = Judge(
        judge_model,
        judge_prompts["moral-evaluation"],
        ref_based = False,
        multi_turn = True,
    )
    return judges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="your_path",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--baseline-model", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "pairwise-baseline runs pairwise comparison against a baseline. "
            "pairwise-all runs pairwise comparison between all pairs. "
            "single runs single answer grading."
        ),
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first n judgments."
    )

    args = parser.parse_args()

    question_file = (f"data/{args.bench_name}/question_en"f".jsonl")
    answer_path = f"data/{args.bench_name}/model_answer/deepseek-r1-en.jsonl"
    ref_answer_dir = f"data/{args.bench_name}/reference_answer"
    questions = load_questions_mt_bench101(question_file, args.first_n)

    model_answers = load_model_answers_mt_bench101(answer_path)
    ref_answers = load_model_answers_mt_bench101(ref_answer_dir)

    judge_prompts = load_judge_prompts(args.judge_file)

    if args.model_list is None:
        models = get_model_list(answer_path)
    else:
        models = args.model_list

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_deepseek-r1-en.jsonl"
        )
        make_match_func = make_match_single_mt_bench101
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
        if args.mode == "pairwise-all":
            def make_match_func(*args, **kwargs):
                return make_match_pair_mt_bench101(*args, **kwargs, baseline_model=None)
        else:
            def make_match_func(*args, **kwargs):
                return make_match_pair_mt_bench101(*args, **kwargs, baseline_model=args.baseline_model)
    check_data_mt_bench101(questions, model_answers, ref_answers, models, judges)

    # Separate questions that need reference answers
    NEED_REF_TASKS = []
    question_math = [q for q in questions if q["task"] in NEED_REF_TASKS]
    question_default = [q for q in questions if q["task"] not in NEED_REF_TASKS]

    # Make matches
    matches = []
    judge_usage_stats = {}

    for q in questions:
        judge_type = determine_question_type(q)
        judge_usage_stats[judge_type] = judge_usage_stats.get(judge_type, 0) + 1
        matches += make_match_func(
            [q],
            models,
            model_answers,
            judges[judge_type],
            ref_answers=None,
            multi_turn=True
        )

    print("\n评估器使用统计:")
    for judge_type, count in judge_usage_stats.items():
        print(f"  {judge_type}: {count}个问题")


    match_stat = {}
    match_stat["bench_name"] = args.bench_name
    match_stat["mode"] = args.mode
    match_stat["judge"] = args.judge_model
    match_stat["baseline"] = args.baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4, ensure_ascii=False))
    input("Press Enter to confirm...")

    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for _ in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass