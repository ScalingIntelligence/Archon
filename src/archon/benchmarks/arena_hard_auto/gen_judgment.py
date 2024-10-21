"""
Usage: python3 -m arena_hard_auto/gen_judgment.py --model-list 'model1, model2']
"""

import json
import argparse
import os
import re
import concurrent.futures
from loguru import logger
from tqdm import tqdm
from ...completions import Archon
from .arena_hard_auto_utils import (
    load_questions,
    chat_completion_archon as get_judgment_answer,
    load_questions,
    load_model_answers,
    make_config,
)


def get_score(judgment, pattern, pairwise=True):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        return None, False


def judgment(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    configs = args["configs"]
    output_file = args["output_file"]
    judge_model = configs["judge_model"]
    temperature = configs["temperature"]
    judge_name = judge_model.config["name"]

    num_games = 2 if configs["pairwise"] else 1

    output = {
        "question_id": question["question_id"],
        "model": answer["model_id"],
        "judge": judge_name,
        "games": []
    }

    for game in range(num_games):
        conv = [{"role": "system", "content": configs["system_prompt"]}]

        for template in configs["prompt_template"]:
            prompt_args = {}

            for i, turn in enumerate(question["turns"]):
                prompt_args[f"question_{i+1}"] = turn["content"]
            base = 1

            if baseline:
                if game % 2 == 1: # swap position
                    answer, baseline = baseline, answer

                for i, turn in enumerate(baseline["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+1}"] = turn["content"]
                    base += 1
            if answer:
                for i, turn in enumerate(answer["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+base}"] = turn["content"]

            if reference:
                for j, ref_answer in enumerate(reference):
                    for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                        prompt_args[f"ref_answer_{i+j+1}"] = turn["content"]
            
            user_prompt = template.format(**prompt_args)
            conv.append({"role": "user", "content": user_prompt})

        judgment = ""
        for _ in range(configs['number_of_judgment_attempts']):
            new_judgment = get_judgment_answer(
                judge_model,
                conv,
                temperature
            )

            judgment += ("\n" + new_judgment)

            score, try_again = get_score(judgment, args["regex_pattern"])

            conv.append({"role": "assistant", "content": new_judgment})

            if not try_again:
                break

            conv.append({"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})

        result = {
            "user_prompt": conv[1]["content"],
            "judgment": judgment,
            "score": score
        }
        output["games"].append(result)

    with open(output_file, "a") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")

def generate_judgments(configs, save_directory:str = "outputs/", question_file: str = None, parallel: int=1):
    print(f'judge model path: {configs["judge_model"]}, baseline: {configs["baseline"]}, baseline model: {configs["baseline_model"]}, reference: {configs["reference"]}, '
          + f'reference models: {configs["ref_model"]}, pairwise: {configs["pairwise"]}')
    
    # Load the path to the archon config for the judge
    archon_config = None
    try:
        with open(configs["judge_model"], "r") as model_file:
            archon_config = json.load(model_file)
    except FileNotFoundError:
        logger.error(f'Judge file not found: {configs["judge_model"]}')
    except json.JSONDecodeError:
        logger.error(f'Invalid JSON in Judge file: {configs["judge_model"]}')
    except Exception as e:
        logger.error(f'Could not load valid (judge) Archon config at {configs["judge_model"]}')
        

    judge_model = Archon(archon_config)
    judge_name = judge_model.config['name']
    
    # Replace path with instantiated model
    configs["judge_model"] = judge_model
    if configs["regex_pattern"]:
        pattern = re.compile(configs["regex_pattern"])

    question_file_path = os.path.join(save_directory, configs["bench_name"], "question.jsonl")
    if question_file:
        question_file_path = question_file
    answer_dir = os.path.join(save_directory, configs["bench_name"], "model_answer")
    ref_answer_dir = os.path.join(save_directory, configs["bench_name"], "reference_answer")

    questions = load_questions(question_file_path)
    model_answers = load_model_answers(answer_dir)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]

    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = [ref_answers[model] for model in configs["ref_model"]]
    
    output_files = {}
    output_dir = f"{save_directory}/{configs['bench_name']}/model_judgment/{judge_name}-judge/{configs['baseline_model']}-baseline/"
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        if os.path.exists(output_file):
            logger.warning(f"Output file {output_file} already exists. Delete or rename this file if you want to generate new judgments.")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_answers(output_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        for model in models:
            count = 0
            for question in questions:
                question_id = question["question_id"]

                kwargs = {}
                kwargs["question"] = question
                if model in model_answers and question_id not in model_answers[model]:
                    print(f"Warning: {model} answer to {question['question_id']} cannot be found.")
                    continue

                if model in existing_judgments and question_id in existing_judgments[model]:
                    logger.info(f"Existing judgment found for {model}")
                    count += 1
                    continue
                
                if model not in model_answers:
                    logger.error(f"{model} is not in model_answer directory. Try running gen_answers for the model")
                    continue

                kwargs["answer"] = model_answers[model][question_id]
                if ref_answers:
                    kwargs["reference"] = [ref_answer[question_id] for ref_answer in ref_answers]
                    assert len(kwargs["reference"]) == len(configs["ref_model"])
                else:
                    kwargs["reference"] = None
                if configs["baseline"]:       
                    try:
                        kwargs["baseline_answer"] = model_answers[configs["baseline_model"]][question_id]
                    except KeyError:
                        logger.error(f"Baseline model {configs['baseline_model']} answer to {question['question_id']} cannot be found. Try generating answers for the baseline model.")
                        kwargs["baseline_answer"] = None
                else:
                    kwargs["baseline_answer"] = None
                kwargs["configs"] = configs
                kwargs["endpoint_dict"] = parallel
                kwargs["output_file"] = output_files[model]
                kwargs["regex_pattern"] = pattern
                future = executor.submit(judgment, **kwargs)
                futures.append(future)

            if count > 0:
                print(f"{count} number of existing judgments")

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

def generate_pairwise_config(baseline: str, judge: str, model_list: list[str] = None, temperature: float = 0.0):
    return {
        'name': 'judgment config file for Arena Hard',
        'bench_name': 'arena_hard_auto',
        'judge_model': judge,
        'reference': False,
        'ref_model': None,
        'baseline': True,
        'baseline_model': baseline,
        'pairwise': True,
        'temperature': temperature,
        'regex_pattern': '\\[\\[([AB<>=]+)\\]\\]',
        'number_of_judgment_attempts': 2,
        "system_prompt": (
            "Please act as an impartial judge and evaluate the quality of the responses "
            "provided by two AI assistants to the user prompt displayed below. You will "
            "be given assistant A's answer and assistant B's answer. Your job is to "
            "evaluate which assistant's answer is better.\n\n"
            "Begin your evaluation by generating your own answer to the prompt. You must "
            "provide your answers before judging any answers.\n\n"
            "When evaluating the assistants' answers, compare both assistants' answers "
            "with your answer. You must identify and correct any mistakes or inaccurate "
            "information.\n\n"
            "Then consider if the assistant's answers are helpful, relevant, and concise. "
            "Helpful means the answer correctly responds to the prompt or follows the "
            "instructions. Note when user prompt has any ambiguity or more than one "
            "interpretation, it is more helpful and appropriate to ask for clarifications "
            "or more information from the user than providing an answer based on assumptions. "
            "Relevant means all parts of the response closely connect or are appropriate "
            "to what is being asked. Concise means the response is clear and not verbose "
            "or excessive.\n\n"
            "Then consider the creativity and novelty of the assistant's answers when needed. "
            "Finally, identify any missing important information in the assistants' answers "
            "that would be beneficial to include when responding to the user prompt.\n\n"
            "After providing your explanation, you must output only one of the following "
            "choices as your final verdict with a label:\n\n"
            "1. Assistant A is significantly better: [[A>>B]]\n"
            "2. Assistant A is slightly better: [[A>B]]\n"
            "3. Tie, relatively the same: [[A=B]]\n"
            "4. Assistant B is slightly better: [[B>A]]\n"
            "5. Assistant B is significantly better: [[B>>A]]\n\n"
            "Example output: \"My final verdict is tie: [[A=B]]\"."
        ),
        "prompt_template": ["<|User Prompt|>\n{question_1}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>"],
        'model_list': model_list,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-list", type=str, help="Names of models in outputs/model_answer directory. input as 'model1,model2,model3...'")
    parser.add_argument("--baseline", type=str, default="gpt-4-0314", help="Name of baseline model in outputs/model_answer directory")
    parser.add_argument("--judge-config", type=str, default="configs/individual_models/gpt-4-turbo-20240620.json", help="Archon model used to compare baseline to model list members")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature used for LM generated judgments")
    parser.add_argument("--save-directory", type=str, default="outputs")
    parser.add_argument("--question-file", type=str, default="arena_hard_auto/arena_questions.jsonl")
    # Note: Using a setting file will override all other parser args
    parser.add_argument("--setting-file", type=str, default=None)
    args = parser.parse_args()
    args.model_list = args.model_list.replace(" ", "").split(",")
    print(args)

    configs = None
    if args.setting_file is not None:
        # Run arena hard auto using a yaml file setup
        configs = make_config(args.setting_file)
    else:
        # Run using command line arguments
        configs = generate_pairwise_config(args.baseline, args.judge_config, args.model_list, args.temperature)

    generate_judgments(configs, args.save_directory, args.question_file, parallel=1)
