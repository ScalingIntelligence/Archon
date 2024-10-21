import os
import json
import time
import yaml
import random
import requests
from ...completions import Archon
from typing import List
from glob import glob

import copy

# API setting constants
API_MAX_RETRY = 16  # 3 #16
API_RETRY_SLEEP = 10  # 5 #10
API_ERROR_OUTPUT = "$ERROR$"


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        if 'arena_hard_battles' in filename or 'bootstrapping_results' in filename:
            continue
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(endpoint_list)[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs

def chat_completion_archon(model, messages, temperature=None):
    if not isinstance(model, Archon):
        raise ValueError("Model must be an instance of Archon")
    output = model.generate(messages, temperature=temperature)
    return output

##################################################


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


##################################################


def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = "https://api.together.xyz/v1/chat/completions"

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )
            if "error" in res.json():

                print("------------------------------------------")
                print(f"Model with Error: {model}")
                print(res.json())
                print("------------------------------------------")

                if res.json()["error"]["type"] == "invalid_request_error":
                    return None

            output = res.json()["choices"][0]["message"]["content"]

            break

        except Exception as e:
            print(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return output

    output = output.strip()

    return output


##################################################


def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)

    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):

        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_together,
):

    if len(references) > 0:

        messages = inject_references_to_messages(messages, references)

    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_MoA_response(
    aggregator_model: str,
    reference_models: List[str],
    rounds: int,
    messages: dict,
    temperature: float,
    max_tokens: int,
    generate_fn=generate_together,
):

    references = []

    if len(reference_models) > 0:

        prev_references = []

        for i_round in range(rounds):

            print(f"Round {i_round+1}/{rounds} to collecting reference responses.")

            references = []

            for reference_model in reference_models:

                reference = generate_with_references(
                    model=reference_model,
                    messages=messages,
                    references=prev_references,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    generate_fn=generate_fn,
                )

                if reference is not None:

                    references.append(reference)

            if i_round < rounds - 1:

                prev_references = references

                references = []

    output = generate_with_references(
        model=aggregator_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        generate_fn=generate_fn,
        references=references,
    ).strip()

    return output


##################################################


def generate_candidates_with_archon_using_together_api(
    messages: dict,
    temperature: float,
    max_new_tokens: int,
    archon_sample_models: list,
    archon_sample_count: int,
    archon_fusion_model: str,
):

    print("Beginning Archon Sample Generation")
    total_initial_outputs = []
    for sample_count in range(archon_sample_count):
        print(f"Archon Sample: {sample_count}")
        output = generate_MoA_response(
            aggregator_model=archon_fusion_model,
            reference_models=archon_sample_models,
            rounds=1,
            messages=messages,
            temperature=0.7,
            max_tokens=max_new_tokens,
            generate_fn=generate_together,
        )
        # breakpoint()
        total_initial_outputs.append(output)

    ###################################

    if len(total_initial_outputs) == 1:
        return total_initial_outputs[0]
    else:
        final_aggregated_output = generate_with_references(
            model=archon_fusion_model,
            messages=messages,
            references=total_initial_outputs,
            temperature=0.0,
            max_tokens=max_new_tokens,
            generate_fn=generate_together,
        ).strip()

        assert final_aggregated_output is not None and len(final_aggregated_output) > 0
        return final_aggregated_output
