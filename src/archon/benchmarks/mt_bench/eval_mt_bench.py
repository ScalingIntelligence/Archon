import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import numpy as np
from tqdm import tqdm
from loguru import logger

import sys
import os

# Add the parent directory to the path for FastChat imports
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "FastChat"))

from .FastChat.fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)


def make_match(
    questions,
    models,
    model_answers,
    judge,
    baseline_model,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m_1 = models[i]
            m_2 = baseline_model
            if m_1 == m_2:
                continue
            if q_id in model_answers[m_1] and q_id in model_answers[m_2]:
                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[baseline_model][q_id]
                if ref_answers is not None:
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


def make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                q_id = q["question_id"]
                m_1 = models[i]
                m_2 = models[j]
                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[m_2][q_id]
                if ref_answers is not None:
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


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
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


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def main(args):
    question_file = (
        f"archon/benchmarks/mt_bench/FastChat/fastchat/llm_judge/data/{args.bench_name}/question.jsonl"
    )
    if args.question_file:
        if os.path.exists(args.question_file):
            question_file = args.question_file
            logger.info(f"Questions File: {question_file}")
        else:
            logger.warning(
                f"question_file {args.question_file} does not exist. Defaulting to {question_file}"
            )

    output_dir = "outputs"
    if args.output_dir:
        output_dir = (
            args.output_dir[0] if type(args.output_dir) == list else args.output_dir
        )
        if not os.path.exists(output_dir):
            output_dir = "outputs"
            logger.warning(
                f"output_dir {args.output_dir} does not exist. Defaulting to {output_dir}"
            )

    answer_dir = f"{output_dir}/mt_bench/model_answer"
    if args.answer_dir:
        if os.path.exists(args.answer_dir):
            answer_dir = args.answer_dir
        else:
            logger.warning(
                f"answer_dir {args.answer_dir} does not exist. Defaulting to {answer_dir}"
            )
    print(f"{answer_dir=}")
    ref_answer_dir = (
        f"mt_bench/FastChat/fastchat/llm_judge/data/{args.bench_name}/reference_answer"
    )
    if args.ref_answer_dir:
        if os.path.exists(args.ref_answer_dir):
            ref_answer_dir = args.ref_answer_dir
        else:
            logger.warning(
                f"ref_answer_dir {args.ref_answer_dir} does not exist. Defaulting to {ref_answer_dir}"
            )

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = f"{output_dir}/{args.bench_name}/model_judgment/{args.judge_model}-judge/single.jsonl"
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair

        if args.mode == "pairwise-all":
            output_file = f"{output_dir}/{args.bench_name}/model_judgment/{args.judge_model}-judge/pair.jsonl"
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

            output_file = f"{output_dir}/{args.bench_name}/model_judgment/{args.judge_model}-judge/{args.baseline_model}-baseline/pair.jsonl"
    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["bench_name"] = args.bench_name
    match_stat["mode"] = args.mode
    match_stat["judge"] = args.judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    # Skip confirmation
    # input("Press Enter to confirm...")
    # Play matches
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="mt_bench/FastChat/fastchat/llm_judge/data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-4")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--answer-dir",
        type=str,
        default="outputs/mt_bench/model_answer",
        help="Path to where generated answers are located",
    )

    parser.add_argument(
        "--question-file",
        type=str,
        default=None,
        help="Path to where questions are located (in FastChat)",
    )

    parser.add_argument(
        "--ref-answer-dir",
        type=str,
        default=None,
        help="Path to where refereance answers are located (usually in FastChat)",
    )

    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="+",
        default=None,
        help="output directory for judge scores",
    )

    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )

    parser.add_argument(
        "--first_n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    args = parser.parse_args()

    main(args)
