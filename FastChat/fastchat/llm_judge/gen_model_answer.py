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

from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype


def load_questions_mt_bench101(question_file, question_begin=None, question_end=None):
    questions = []
    with open(question_file, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if question_begin is not None and idx < question_begin:
                continue
            if question_end is not None and idx >= question_end:
                break
            question = json.loads(line.strip())
            questions.append(question)
    return questions


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
    questions = load_questions_mt_bench101(question_file, question_begin, question_end)
    # 随机打乱问题列表，平衡负载
    random.shuffle(questions)

    # 将问题列表分块，分配给多个 GPU
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
        # 根据任务类型设置温度（可选）
        temperature = 0.7  # 默认温度

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
                # print(idx,": ", prompt)
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

                    # 处理生成的输出
                    output = tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                    )

                    # 去除停止词（如果有）
                    if conv.stop_str and output.find(conv.stop_str) > -1:
                        output = output[: output.find(conv.stop_str)]

                    output = output.strip()

                except RuntimeError as e:
                    print(f"错误：问题 ID {question['id']}，错误信息：{e}")
                    output = "ERROR"

                # conv.update_last_message(output)
                turns.append({"role": "assistant", "message": output})


                conv.update_last_message(bot_msg)
                
                # if bot_msg is not None and idx < len(history) - 1:
                #     # 已有机器人的回复，添加到对话中
                #     conv.append_message(conv.roles[1], bot_msg)
                #     turns.append({"role": "assistant", "message": bot_msg})
                # else:
                #     # 需要模型生成回复
                #     conv.append_message(conv.roles[1], None)
                #     prompt = conv.get_prompt()
                #     print(prompt)
                #     input_ids = tokenizer([prompt]).input_ids
                #     attention_mask = torch.as_tensor(tokenizer([prompt]).attention_mask).cuda()
                #     # print(tokenizer([prompt]))

                #     if temperature < 1e-4:
                #         do_sample = False
                #     else:
                #         do_sample = True

                #     try:
                #         output_ids = model.generate(
                #             torch.as_tensor(input_ids).cuda(),
                #             do_sample=do_sample,
                #             temperature=temperature,
                #             max_new_tokens=max_new_token,
                #             attention_mask=attention_mask,
                #             pad_token_id=tokenizer.eos_token_id,
                #         )
                #         if model.config.is_encoder_decoder:
                #             output_ids = output_ids[0]
                #         else:
                #             output_ids = output_ids[0][len(input_ids[0]) :]

                #         # 处理生成的输出
                #         output = tokenizer.decode(
                #             output_ids,
                #             skip_special_tokens=True,
                #             spaces_between_special_tokens=False,
                #         )

                #         # 去除停止词（如果有）
                #         if conv.stop_str and output.find(conv.stop_str) > -1:
                #             output = output[: output.find(conv.stop_str)]

                #         output = output.strip()

                #     except RuntimeError as e:
                #         print(f"错误：问题 ID {question['id']}，错误信息：{e}")
                #         output = "ERROR"

                #     conv.update_last_message(output)
                #     turns.append({"role": "assistant", "message": output})

            choices.append({"index": i, "turns": turns})

        # 保存答案
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
    """按照问题 ID 排序并去重答案文件"""
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
        help="模型权重的路径，可以是本地文件夹或 Hugging Face 的仓库 ID。",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="模型的自定义名称。"
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt-bench-101",
        help="基准问题集的名称。",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="调试选项。问题的起始索引。",
    )
    parser.add_argument(
        "--question-end", type=int, help="调试选项。问题的结束索引。"
    )
    parser.add_argument("--answer-file", type=str, help="输出答案文件的路径。")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="生成的最大新标记数。",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="生成的答案选项数量。",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="每个模型使用的 GPU 数量。",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="总共使用的 GPU 数量。"
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="每个 GPU 用于模型权重的最大显存。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="覆盖默认的 dtype。如果未设置，在 GPU 上使用 float16，在 CPU 上使用 float32。",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="要加载的模型版本。",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/datasets_all.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer_ours/{args.model_id}.jsonl"

    print(f"输出答案到 {answer_file}")

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