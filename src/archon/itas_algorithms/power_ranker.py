import os
import json
from ..completions import Archon
from ..completions.gen_answers import main as gen_answers_main
from loguru import logger
import argparse

from ..benchmarks.mt_bench.eval_mt_bench import main as eval_mt_main
from ..benchmarks.mt_bench.show_mt_bench_result import display_result_pairwise
from ..benchmarks.arena_hard_auto.gen_judgment import generate_judgments, generate_pairwise_config
from ..benchmarks.arena_hard_auto.show_arena_hard_auto_result import rank_model_performance

# Needed for FastChat
from dotenv import load_dotenv
load_dotenv()

def parse_model_name(model_path: str) -> str:
    parsed_name = os.path.basename(model_path)
    if parsed_name.endswith("json"):
        return parsed_name[:-5]
    return parsed_name


QUESTION_MAP = {
    "arena_hard_auto": "archon/benchmarks/arena_hard_auto/arena_questions.jsonl",
    "mt_bench": "archon/benchmarksmt_bench/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl",
}


class PowerRanker:
    def __init__(
        self,
        benchmark_name: str,
        judge_path: str,
        baseline_path: str,
        model_list_paths: list[str],
        output_dir: str = "outputs/",
        dataset_sample: float = 1.0,
    ):
        if benchmark_name not in QUESTION_MAP:
            raise ValueError(f"{benchmark_name} is not a valid benchmark name")
        self.benchmark = {
            "name": benchmark_name,
            "question_file": QUESTION_MAP[benchmark_name],
        }
        self.output_dir = output_dir
        self.judge = self.base_initializer(judge_path)
        self.baseline = self.base_initializer(baseline_path)
        self.eval_models = [
            self.eval_initializer(model_path) for model_path in model_list_paths
        ]
        self.results = None
        self.model_to_score_dict = None
        self.dataset_sample = dataset_sample

    def base_initializer(self, model_path: str):
        # Model keys: config_path, archon_model, name, answer_path
        new_model = {}
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist.")
        new_model["config_path"] = model_path

        # Create an archon model from model_path
        with open(model_path, "r") as file:
            try:
                new_model["archon_model"] = Archon(json.load(file))
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in config file: {model_path}")

        # Get the name from the archon config
        new_model["name"] = new_model["archon_model"].config["name"]

        # Set predicted answer path
        new_model["answer_path"] = (
            f"{self.output_dir}/{self.benchmark['name']}/model_answer/{new_model['name']}.json"
        )

        return new_model

    def eval_initializer(self, model_path: str):
        # Model keys: config_path, archon_model, name, answer_path, judgment_path
        new_model = self.base_initializer(model_path)

        # Set predicted judgments path
        if self.benchmark["name"] == "arena_hard_auto":
            new_model["judgment_path"] = (
                f"{self.output_dir}/{self.benchmark['name']}/model_judgment/{self.judge['name']}-judge/{self.baseline['name']}-baseline/{new_model['name']}.json"
            )

        # TODO: Update mt_bench save file
        if self.benchmark["name"] == "mt_bench":
            new_model["judgment_path"] = (
                f"{self.output_dir}/{self.benchmark['name']}/model_judgment/{self.judge['name']}_pair.jsonl"
            )
        return new_model

    def gen_model_answers(self):
        # Need to generate answers for baselines, and all eval_models
        config_paths = [model["config_path"] for model in self.eval_models]
        config_paths.append(self.baseline["config_path"])
        for config in config_paths:
            args_dict = {
                "benchmark": self.benchmark["name"],
                "dataset_sample": self.dataset_sample,
                "config": config,
                "output_dir": self.output_dir,
                "parallel": 16,
                "debug_data": False,
                "debug": False,
                "debug_verifier": False,
                "debug_archon": False,
                "append": False,
                "debug_unit_test_generator": False,
                "samples": 1
            }
            args = argparse.Namespace(**args_dict)
            gen_answers_main(args)

    def compare_against_arena(self):
        model_name_list = [model["name"] for model in self.eval_models]
        judge_config = generate_pairwise_config(
            baseline=self.baseline["name"],
            judge=self.judge["config_path"],
            model_list=model_name_list,
            temperature=0.0,
        )
        generate_judgments(
            configs=judge_config,
            save_directory=self.output_dir,
            question_file=self.benchmark["question_file"],
        )
        self.results, self.model_to_score_dict = rank_model_performance(
            save_directory=self.output_dir,
            judge_name=self.judge["name"],
            baseline=self.baseline["name"],
        )
        return self.results, self.model_to_score_dict

    def compare_against_mt(self):
        # Generate Judgments
        model_name_list = [model["name"] for model in self.eval_models]
        args_dict = {
            "bench_name": "mt_bench",
            "judge_file": "archon/benchmarks/mt_bench/FastChat/fastchat/llm_judge/data/judge_prompts.jsonl",
            "judge_model": self.judge["name"],
            "baseline_model": self.baseline["name"],
            "mode": "pairwise-baseline",
            "answer_dir": f"{self.output_dir}/mt_bench/model_answer",
            "model_list": model_name_list,  # Replace with configs once functionality is built
            "parallel": 1,
            "first_n": None,
            "question_file": self.benchmark["question_file"],
            "ref_answer_dir": "archon/benchmarks/mt_bench/FastChat/fastchat/llm_judge/data/mt_bench/reference_answer",
            "output_dir": self.output_dir,
        }
        args = argparse.Namespace(**args_dict)
        eval_mt_main(args)

        # Power Rank Results
        args_dict = {
            "bench_name": "mt_bench",
            "model_list": model_name_list,  # ["archon-MoA-lite"]
            "mode": "pairwise-baseline",
            "input_file": f"{self.output_dir}/mt_bench/model_judgment/{self.judge['name']}-judge/{self.baseline['name']}-baseline/pair.jsonl",
            "judge_model": self.judge["name"],
            "baseline_model": self.baseline["name"],
            "return_ranking": True,
        }
        args = argparse.Namespace(**args_dict)

        self.results, self.model_to_score_dict = display_result_pairwise(args)
        print(self.results)
        return self.results, self.model_to_score_dict

    def rank_models(self):
        if self.benchmark["name"] == "mt_bench":
            return self.compare_against_mt()
        if self.benchmark["name"] == "arena_hard_auto":
            return self.compare_against_arena()
        logger.error(f"{self.benchmark['name']} is not supported")


if __name__ == "__main__":
    """
    Example command:
    python -m power_ranker --benchmark_name mt_bench --judge_path configs/individual_models/gpt-4-turbo-2024-04-09.json --output_dir outputs --baseline_path configs/individual_models/gpt-4-turbo-2024-04-09.json --model_list_paths configs/individual_models/gpt-4-turbo-2024-04-09.json configs/individual_models/claude-3-haiku-20240307.json --dataset_sample 1.0
    """
    parser = argparse.ArgumentParser(description="Power Ranker for model evaluation.")
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default="arena_hard_auto",
        help="Name of the benchmark",
    )
    parser.add_argument(
        "--judge_path",
        type=str,
        required=True,
        help="Path to the judge archon configuration",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for output files (defaults to outputs)",
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        required=True,
        help="Path to the baseline model configuration",
    )
    parser.add_argument(
        "--model_list_paths",
        type=str,
        nargs="+",
        required=True,
        help="List of model configuration paths",
    )
    parser.add_argument(
        "--dataset_sample",
        type=float,
        default=1.0,
        help="Proportion of the the dataset to use",
    )
    args = parser.parse_args()

    test = PowerRanker(
        args.benchmark_name,
        args.judge_path,
        args.baseline_path,
        args.model_list_paths,
        args.output_dir,
        args.dataset_sample
    )

    test.gen_model_answers()
    results, model_to_score_dict = test.rank_models()
    print(test)
