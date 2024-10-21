
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import argparse
import re

def evaluate_pass(answers: list):
    corrects = []
    problematic_queries = []

    for i, response_dict in tqdm(enumerate(answers)):
        ground_truth_answer = re.search(r'\\boxed\{([^}]*)\}', response_dict["solution"]).group(1)
        assert len(ground_truth_answer) > 0
        
        try:
            response_answer_section = response_dict["output"].split("answer is:")[1].strip()
            
            if ground_truth_answer in response_answer_section:
                corrects.append(True)
            else:
                corrects.append(False)

            problematic_queries.append(False)
        except:
            print(f"Error evaluating response {i}: {response_dict['output']}")
            corrects.append(False)
            problematic_queries.append(True)

    corrects = np.array(corrects)
    problematic_queries = np.array(problematic_queries)

    return corrects, problematic_queries

def evaluate_math_answers(answers_json_path: str):
    """
    Evaluate the MATH answers against the evaluation split.

    Parameters:
    answers_json_path (str): The path to the JSON file containing the MATH answers.

    Returns:
    dict: The evaluation results.
    """
    with open(answers_json_path, "r") as f:
        answers = json.load(f)

    print("Evaluating MATH answers...")

    overall_corrects = None
    overall_problematic = None

    if len(answers) != 0 and isinstance(answers[0], list):
        for answers_pass in answers:
            corrects, problematic_queries = evaluate_pass(answers_pass)

            if overall_corrects is None:
                overall_corrects = np.zeros_like(corrects, dtype=bool)

            if overall_problematic is None:
                overall_problematic = np.ones_like(problematic_queries, dtype=bool)

            overall_corrects = np.logical_or(overall_corrects, corrects)
            overall_problematic = np.logical_and(overall_problematic, problematic_queries)
    else:
        overall_corrects, overall_problematic = evaluate_pass(answers)

    evaluation_results = {}
    evaluation_results["accuracy"] = np.mean(overall_corrects)
    num_problematic = np.sum(overall_problematic)

    return evaluation_results, num_problematic

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-json-path", type=str, required=True, help="The path to the JSON file containing the MATH answers.")
    args = parser.parse_args()

    evaluation_results, problematic_queries = evaluate_math_answers(args.answers_json_path)
    print(evaluation_results)
    print("Problematic Queries Count: ", problematic_queries)

if __name__ == "__main__":
    main()

# Evaluate the MATH answers against the evaluation split
# python math_evaluation.py --answers-json-path math_answers.json
