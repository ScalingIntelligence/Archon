import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse


def evaluate_pass(answers: list):
    corrects = []
    for i, response_dict in tqdm(enumerate(answers)):
        correct = False

        ground_truth_answer = response_dict["answer"].split("####")[1].strip()
        response_answer_section = response_dict["output"].split("The answer is:")[1].strip()
        if ground_truth_answer in response_answer_section:
            correct = True
        corrects.append(correct)

    corrects = np.array(corrects)
    return corrects


def evaluate_gsm8k_answers(answers_json_path: str):
    """
    Evaluate the GSM8k answers against the evaluation split.

    Parameters:
    answers_json_path (str): The path to the JSON file containing the GSM8k answers.

    Returns:
    dict: The evaluation results.
    """
    with open(answers_json_path, "r") as f:
        answers = json.load(f)

    print("Evaluating MATH answers...")

    overall_corrects = None
    if len(answers) != 0 and isinstance(answers[0], list):
        for answers_pass in answers:
            corrects = evaluate_pass(answers_pass)

            if overall_corrects is None:
                overall_corrects = np.zeros_like(corrects, dtype=bool)

            overall_corrects = np.logical_or(overall_corrects, corrects)

    else:
        overall_corrects = evaluate_pass(answers)

    evaluation_results = {}
    evaluation_results["accuracy"] = np.mean(overall_corrects)

    return evaluation_results


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-json-path", type=str, required=True, help="The path to the JSON file containing the GSM8k answers.")
    args = parser.parse_args()

    evaluation_results = evaluate_gsm8k_answers(args.answers_json_path)
    print(evaluation_results)

if __name__ == "__main__":
    main()

# Evaluate the GSM8k answers against the evaluation split
# python gsm8k_evaluation.py --answers-json-path gsm8k_answers.json
