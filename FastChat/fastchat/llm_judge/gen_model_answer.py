#!/usr/bin/env python3
"""
Generate answers with local models for SafeDialBench dataset.

Usage:
python3 gen_model_answer.py --model-path path/to/your/model --model-id your_model_id
"""

import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype



def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
):
    questions = load_questions(question_file, question_begin, question_end)
    random.shuffle(questions)

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray

        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    num_chunks = num_gpus_total // num_gpus_per_model
    chunk_size = (len(questions) + num_chunks - 1) // num_chunks
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
            )
        )

    if use_ray:
        import ray

        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
):
    model, tokenizer = load_model(
        model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    for question in tqdm(questions):
        temperature = 0.7  

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            history = question["history"]
            method = question["method"]

            for idx, turn in enumerate(history):
                user_msg = turn["user"]
                bot_msg = turn.get("bot", None)

                conv.append_message(conv.roles[0], user_msg)

                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids
                attention_mask = torch.as_tensor(tokenizer([prompt]).attention_mask).cuda()

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                try:
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    output = tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                    )

                    if conv.stop_str and output.find(conv.stop_str) > -1:
                        output = output[: output.find(conv.stop_str)]

                    output = output.strip()

                except RuntimeError as e:
                    print(f"Error: Question ID {question['id']}, Error Message: {e}")
                    output = "ERROR"

                # conv.update_last_message(output)
                turns.append({"role": "assistant", "message": output})


                conv.update_last_message(bot_msg)
                

            choices.append({"index": i, "turns": turns})

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
            ans_json = {
                "id": question["id"],
                "task": question["task"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "method": method,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


def reorg_answer_file(answer_file):
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as fin:
        for l in fin:
            ans = json.loads(l)
            qid = ans["id"]
            answers[qid] = l

    qids = sorted(answers.keys())
    with open(answer_file, "w", encoding="utf-8") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model weights, either a local folder or a Hugging Face repository ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="Custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="SafeDial",
        help="Name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="Start index of the questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="End index of the questions."
    )
    parser.add_argument("--answer-file", type=str, help="Path to the output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="Number of answer choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="Number of GPUs to use per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="Total number of GPUs to use."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maximum GPU memory for model weights.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. Uses float16 on GPU and float32 on CPU if not set.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Version of the model to load.",
    )
    parser.add_argument(
        "--language", 
        type=str, 
        default="zh", 
        choices=["zh", "en"],
        help="Language of the dataset (zh for Chinese, en for English)"
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    # Set dataset file based on language
    if args.lang == "zh":
        question_file = f"data/{args.bench_name}/datasets_zh.jsonl"
    else:
        question_file = f"data/{args.bench_name}/datasets_en.jsonl"
    
    # Set output file based on language
    if args.answer_file:
        answer_file = args.answer_file
    else:
        if args.lang == "zh":
            answer_file = f"data/{args.bench_name}/model_answer/{args.model}_zh.jsonl"
        else:
            answer_file = f"data/{args.bench_name}/model_answer/{args.model}_en.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
    )

    reorg_answer_file(answer_file)