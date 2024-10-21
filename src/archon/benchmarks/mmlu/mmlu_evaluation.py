import json
import numpy as np
from tqdm import tqdm
import argparse

choices = ["A", "B", "C", "D"]


def evaluate_mmlu_answers(answers_json_path: str):
  """
  Evaluate the answers in MMLU against the provided correct answers.

  Parameters:
  answers_json_path (str): The path to the JSON file containing the MMLU answers.

  Returns:
  dict: The evaluation results.
  """
  with open(answers_json_path, "r") as f:
    answers = json.load(f)

  corrects = []
  problematic_queries = 0
  print("Evaluating MMLU answers...")

  for i, response_dict in tqdm(enumerate(answers)):
    correct_answer_index = response_dict["answer"]
    correct_answer = choices[correct_answer_index]
    try:
      # Extract the model's final answer from its output
      model_output = response_dict["output"].rsplit("answer is:",1)[-1].strip(" ")

      # Compare the model's final answer with the correct one
      if model_output.lower() == correct_answer.lower():
        corrects.append(True)
      else:
        corrects.append(False)
    except Exception as e:
      print(f"Error evaluating response {i}: {response_dict['output']} - Error: {str(e)}")
      corrects.append(False)
      problematic_queries += 1

  corrects = np.array(corrects)

  evaluation_results = {}
  evaluation_results["accuracy"] = np.mean(corrects)

  return evaluation_results, problematic_queries


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--answers-json-path", type=str, required=True,
                      help="The path to the JSON file containing the MMLU answers.")
  args = parser.parse_args()

  evaluation_results, problematic_queries = evaluate_mmlu_answers(args.answers_json_path)
  print(evaluation_results)
  print("Problematic Queries Count: ", problematic_queries)


if __name__ == "__main__":
  main()

# Example usage:
# python mmlu_evaluation.py --answers-json-path mmlu_answers.json
