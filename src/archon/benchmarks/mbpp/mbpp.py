from .human_eval.evaluation import evaluate_functional_correctness
import argparse
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-json-path", type=str, required=True, help="The path to the JSON file containing the GSM8k answers.")
    parser.add_argument("--language", type=str, default="python")
    args = parser.parse_args()
    problem_file = os.path.join('mbpp/data', f"mbpp.jsonl")
    
    result = evaluate_functional_correctness(
        input_file=args.answers_json_path,
        tmp_dir='tmp',
        problem_file=os.path.join('mbpp/data', f"mbpp_test.jsonl"),
        language='python',
        is_mbpp=True
    )
    print(result)

if __name__ == "__main__":
    main()

# Evaluate the GS