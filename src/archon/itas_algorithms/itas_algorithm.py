
import os
from datetime import datetime
from bayes_opt import BayesianOptimization
import numpy as np

from functools import partial
from tabulate import tabulate

import argparse
import json
from datetime import datetime
from pathlib import Path


######################################################
def append_json(file_path: str) -> str:
    if file_path.endswith(".json"):
        return file_path
    return file_path + ".json"


from .power_ranker import PowerRanker, parse_model_name


class ITAS:
    # TODO: Get this from the archon config
    MODEL_TO_API_DICT = {
        "gpt-3.5-turbo-0125": "OpenAI_API",
        "gpt-4-0314": "OpenAI_API",
        "gpt-4-1106-preview": "OpenAI_API",
        "gpt-4o": "OpenAI_API",
        "gpt-4-turbo-2024-04-09": "OpenAI_API",
        "gpt-4o-2024-05-13": "OpenAI_API",
        "gpt-4-turbo-20240620": "OPENAI_API",
        "gpt-4o-mini": "OpenAI_API",
        "claude-3-sonnet-20240229": "Anthropic_API",
        "claude-3-opus-20240229": "Anthropic_API",
        "claude-3-5-sonnet-20240620": "Anthropic_API",
        "claude-3-haiku-20240307": "Anthropic_API",
        "Qwen/Qwen1.5-72B-Chat": "Together_API",
        "Qwen/Qwen1.5-110B-Chat": "Together_API",
        "Qwen/Qwen2-72B-Instruct": "Together_API",
        "microsoft/WizardLM-2-8x22B": "Together_API",
        "mistralai/Mixtral-8x22B-Instruct-v0.1": "Together_API",
        "meta-llama/Llama-3-70b-chat-hf": "Together_API",
        "databricks/dbrx-instruct": "Together_API",
        "Qwen/Qwen1.5-7B-Chat": "HuggingFace",
        "Nexusflow/Starling-LM-7B-beta": "HuggingFace",
        "meta-llama/Meta-Llama-3-8B-Instruct": "HuggingFace",
        "berkeley-nest/Starling-LM-7B-alpha": "HuggingFace",
        "teknium/OpenHermes-2.5-Mistral-7B": "HuggingFace",
        "mistralai/Mistral-7B-Instruct-v0.2": "HuggingFace",
        "cognitivecomputations/dolphin-2.2.1-mistral-7b": "HuggingFace",
        "microsoft/Phi-3-mini-4k-instruct": "HuggingFace",
        "HuggingFaceH4/zephyr-7b-beta": "HuggingFace",
        "microsoft/Phi-3-small-8k-instruct": "HuggingFace",
        "Qwen/Qwen2-7B-Instruct": "HuggingFace",
        "princeton-nlp/Llama-3-Instruct-8B-SimPO": "HuggingFace",
        "princeton-nlp/Llama-3-Instruct-8B-IPO": "HuggingFace",
        "princeton-nlp/Llama-3-Instruct-8B-RDPO": "HuggingFace",
        "princeton-nlp/Llama-3-Instruct-8B-DPO": "HuggingFace",
    }

    def __init__(self, search_config, general_model_config):
        self.search_config = search_config
        self.general_model_config = general_model_config
        self.model_power_ranking = self.get_model_power_ranking()

    def get_model_power_ranking(self):
        if os.path.exists(self.search_config["power_ranking_save_path"]):
            print(
                f"Loading power ranking from file: {self.search_config['power_ranking_save_path']}"
            )
            with open(self.search_config["power_ranking_save_path"], "r") as file:
                model_power_ranking = json.load(file)

            # Make sure all the models available are in the power ranking
            for model in self.search_config["models_available"]:
                assert model in model_power_ranking
        else:

            os.makedirs(
                os.path.dirname(self.search_config["power_ranking_save_path"]),
                exist_ok=True,
            )
            os.makedirs(
                os.path.dirname(self.search_config["answers_save_path"]), exist_ok=True
            )

            # Create configs for every model in power ranking and baseline model
            for model in self.search_config["models_available"] + [
                self.search_config["baseline_model"]
            ]:
                model_config = {
                    "name": model,
                    "model_power_ranking": [model],
                    "model_top_k": 1,
                    "sample_top_k": 1,
                    "fuser_layer_1": 0,
                    "fuser_layer_2": 0,
                    "fuser_layer_3": 0,
                    "final_fuser_model": model,
                    "critic_model": model,
                    "ranker_model": model,
                    "add_final_fuser": False,
                }

                save_path = append_json(
                    os.path.join(self.search_config["answers_save_path"], model)
                )  # Save the config to correct folder
                self.create_archon_config(model_config, save_path=save_path)
                print(
                    f"Created Archon config for {model}, saved to: {self.get_save_path(model_config)}"
                )

            if self.search_config["benchmark"] in ["mt_bench", "arena_hard_auto"]:
                ranker = PowerRanker(
                    benchmark_name=self.search_config["benchmark"],
                    judge_path=self.search_config["judge_model"],
                    baseline_path=self.search_config["baseline_model"],
                    model_list_paths=self.search_config["models_available"],
                    output_dir=self.search_config["save_directory"],
                )
                ranker.gen_model_answers()
                model_power_ranking, _ = ranker.rank_models()
            else:
                raise ValueError("Invalid benchmark type!")

            # Remove baseline model from ranking list if it is not in the models available
            if (
                self.search_config["baseline_model"]
                not in self.search_config["models_available"]
            ):
                model_power_ranking.remove(
                    parse_model_name(self.search_config["baseline_model"])
                )
                assert self.search_config["baseline_model"] not in model_power_ranking

            with open(self.search_config["power_ranking_save_path"], "w") as file:
                json.dump(model_power_ranking, file)
                print(
                    "Saved power ranking to file:",
                    self.search_config["power_ranking_save_path"],
                )

        return model_power_ranking

    def run_benchmark(self, archon_json, dataset_sample):
        if self.search_config["benchmark"] in ["mt_bench", "arena_hard_auto"]:
            # return get_benchmark_results(
            #     baseline_model=self.search_config["baseline_model"],
            #     model_list=[archon_json["name"]],
            #     answers_save_path=self.search_config["answers_save_path"],
            #     save_directory=self.search_config["save_directory"],
            #     judge_model=self.search_config["judge_model"],
            #     benchmark=self.search_config["benchmark"],
            #     dataset_sample=dataset_sample
            # )
            ranker = PowerRanker(
                benchmark_name=self.search_config["benchmark"],
                judge_path=self.search_config["judge_model"],
                baseline_path=self.search_config["baseline_model"],
                model_list_paths=[archon_json["save_path"]],
                output_dir=self.search_config["save_directory"],
            )
            ranker.gen_model_answers(dataset_sample=dataset_sample)
            return ranker.rank_models()
        else:
            raise ValueError("Invalid benchmark type!")

    def calculate_total_inference_calls(self, archon_json):
        return sum(
            model["samples"] for layer in archon_json["layers"] for model in layer
        )

    def create_archon_config(self, archon_config_dict, save_path=None):
        archon_config = {"name": archon_config_dict["name"], "layers": []}

        def add_critic_layer(critic_model):
            return [
                {
                    "type": "critic",
                    "model": critic_model,
                    "model_type": self.MODEL_TO_API_DICT[critic_model],
                    "temperature": self.general_model_config["temperature"],
                    "max_context_length": self.general_model_config["max_tokens"],
                    "samples": 1,
                }
            ]

        def add_ranker_layer(ranker_model, ranker_top_k=5):
            return [
                {
                    "type": "ranker",
                    "model": ranker_model,
                    "model_type": self.MODEL_TO_API_DICT[ranker_model],
                    "temperature": self.general_model_config["temperature"],
                    "max_context_length": self.general_model_config["max_tokens"],
                    "top_k": ranker_top_k,
                    "use_critiques": True
                }
            ]

        # Proposer Layer
        archon_config["layers"].append(
            [
                {
                    "type": "generator",
                    "model": archon_config_dict["model_power_ranking"][i],
                    "model_type": self.MODEL_TO_API_DICT[
                        parse_model_name(archon_config_dict["model_power_ranking"][i])
                    ],
                    "temperature": self.general_model_config["temperature"],
                    "max_tokens": self.general_model_config["max_tokens"],
                    "samples": 1,
                }
                for i in range(archon_config_dict["model_top_k"])
            ]
        )

        # Fuser and Critic Layers
        for fuser_layer_key in ["fuser_layer_1", "fuser_layer_2", "fuser_layer_3"]:
            if archon_config_dict[fuser_layer_key] > 0:
                archon_config["layers"].append(
                    add_critic_layer(archon_config_dict["critic_model"]),
                    add_ranker_layer(archon_config_dict["ranker_model"], archon_config_dict["ranker_top_k"]),
                )
                archon_config["layers"].append(
                    [
                        {
                            "type": "fuser",
                            "model": archon_config_dict["model_power_ranking"][j],
                            "model_type": self.MODEL_TO_API_DICT[
                                archon_config_dict["model_power_ranking"][j]
                            ],
                            "temperature": self.general_model_config["temperature"],
                            "max_tokens": self.general_model_config["max_tokens"],
                            "samples": 1,
                        }
                        for j in range(archon_config_dict[fuser_layer_key])
                    ]
                )

        # Final Fuser Layer
        if archon_config_dict["add_final_fuser"]:
            archon_config["layers"].append(
                [
                    {
                        "type": "fuser",
                        "model": archon_config_dict["final_fuser_model"],
                        "model_type": self.MODEL_TO_API_DICT[
                            archon_config_dict["final_fuser_model"]
                        ],
                        "temperature": self.general_model_config["temperature"],
                        "max_tokens": self.general_model_config["max_tokens"],
                        "samples": 1,
                    }
                ]
            )

        # Save the config
        if save_path is None:
            save_path = self.get_save_path(archon_config_dict)

        archon_config["save_path"] = save_path
        if os.path.exists(save_path):
            os.remove(save_path)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as file:
            json.dump(archon_config, file)

        return archon_config

    def get_save_path(self, archon_config_dict):
        return os.path.join(
            self.search_config["answers_save_path"],
            append_json(self.create_name(archon_config_dict)),
        )

    def create_name(self, archon_config_dict):
        return (
            f"search_run_{self.search_config['search_config_name']}"
            f"_model_top_k_{archon_config_dict['model_top_k']}"
            f"_sample_top_k_{archon_config_dict['sample_top_k']}"
            f"_fuser_layer_1_{archon_config_dict['fuser_layer_1']}"
            f"_fuser_layer_2_{archon_config_dict['fuser_layer_2']}"
            f"_fuser_layer_3_{archon_config_dict['fuser_layer_3']}"
        )

    def find_nearest_value(self, value, choices):
        return min(choices, key=lambda x: abs(x - int(np.round(value))))

    def run_archon_config(
        self,
        model_top_k_choice,
        sample_top_k_choice,
        fuser_layer_1_choice,
        fuser_layer_2_choice,
        fuser_layer_3_choice,
    ):
        # Map the choices to the closest valid value
        model_top_k_choice = self.find_nearest_value(
            model_top_k_choice, self.search_config["model_top_k_choices"]
        )
        sample_top_k_choice = self.find_nearest_value(
            sample_top_k_choice, self.search_config["sample_top_k_choices"]
        )
        fuser_layer_1_choice = self.find_nearest_value(
            fuser_layer_1_choice, self.search_config["fuser_layer_1_choices"]
        )
        fuser_layer_2_choice = self.find_nearest_value(
            fuser_layer_2_choice, self.search_config["fuser_layer_2_choices"]
        )
        fuser_layer_3_choice = self.find_nearest_value(
            fuser_layer_3_choice, self.search_config["fuser_layer_3_choices"]
        )

        print(
            f"Running with Model Top-K: {model_top_k_choice}, Sample Top-K: {sample_top_k_choice}, "
            f"Fuser Layer 1: {fuser_layer_1_choice}, Fuser Layer 2: {fuser_layer_2_choice}, Fuser Layer 3: {fuser_layer_3_choice}"
        )

        archon_config_dict = {
            "name": self.create_name(
                {
                    "model_top_k": model_top_k_choice,
                    "sample_top_k": sample_top_k_choice,
                    "fuser_layer_1": fuser_layer_1_choice,
                    "fuser_layer_2": fuser_layer_2_choice,
                    "fuser_layer_3": fuser_layer_3_choice,
                    "search_name": self.search_config["search_config_name"],
                }
            ),
            "model_power_ranking": self.model_power_ranking,
            "model_top_k": model_top_k_choice,
            "sample_top_k": sample_top_k_choice,
            "fuser_layer_1": fuser_layer_1_choice,
            "fuser_layer_2": fuser_layer_2_choice,
            "fuser_layer_3": fuser_layer_3_choice,
            "final_fuser_model": self.model_power_ranking[0],
            "critic_model": self.model_power_ranking[0],
            "ranker_model": self.model_power_ranking[0],
            "ranker_top_k": 5,
            "add_final_fuser": True,
        }

        archon_json = self.create_archon_config(archon_config_dict)
        total_inference_calls = self.calculate_total_inference_calls(archon_json)
        if total_inference_calls > self.search_config["maximum_inference_calls"]:
            return -1
        else:
            _, model_to_score_dict = self.run_benchmark(
                archon_json, self.search_config["dataset_sample_for_search"]
            )

            return list(model_to_score_dict.values())[0]

    def perform_search(self):
        if self.search_config["search_algorithm_choice"] == "random_search":
            print("Starting Random Search!")
            return self.perform_random_search()
        elif self.search_config["search_algorithm_choice"] == "grid_search":
            print("Starting Grid Search!")
            return self.perform_grid_search()
        elif self.search_config["search_algorithm_choice"] == "bayesian_optimization":
            print("Starting Bayesian Optimization!")
            return self.perform_bayesian_optimization()
        else:
            raise ValueError("Invalid search algorithm choice!")

    def perform_random_search(self):
        archon_save_path_to_score = {}
        while (
            len(archon_save_path_to_score)
            < self.search_config["random_search_iterations"]
        ):
            random_choices = {
                "model_top_k": np.random.choice(
                    self.search_config["model_top_k_choices"]
                ),
                "sample_top_k": np.random.choice(
                    self.search_config["sample_top_k_choices"]
                ),
                "fuser_layer_1": np.random.choice(
                    self.search_config["fuser_layer_1_choices"]
                ),
                "fuser_layer_2": np.random.choice(
                    self.search_config["fuser_layer_2_choices"]
                ),
                "fuser_layer_3": np.random.choice(
                    self.search_config["fuser_layer_3_choices"]
                ),
            }

            save_path = self.get_save_path(random_choices)
            if save_path not in archon_save_path_to_score:
                archon_score = self.run_archon_config(**random_choices)
                archon_save_path_to_score[save_path] = archon_score

        return archon_save_path_to_score

    def perform_grid_search(self):
        archon_save_path_to_score = {}
        for model_top_k_choice in self.search_config["model_top_k_choices"]:
            for sample_top_k_choice in self.search_config["sample_top_k_choices"]:
                for fuser_layer_1_choice in self.search_config["fuser_layer_1_choices"]:
                    for fuser_layer_2_choice in self.search_config[
                        "fuser_layer_2_choices"
                    ]:
                        for fuser_layer_3_choice in self.search_config[
                            "fuser_layer_3_choices"
                        ]:
                            archon_score = self.run_archon_config(
                                model_top_k_choice,
                                sample_top_k_choice,
                                fuser_layer_1_choice,
                                fuser_layer_2_choice,
                                fuser_layer_3_choice,
                            )
                            save_path = self.get_save_path(
                                {
                                    "model_top_k": model_top_k_choice,
                                    "sample_top_k": sample_top_k_choice,
                                    "fuser_layer_1": fuser_layer_1_choice,
                                    "fuser_layer_2": fuser_layer_2_choice,
                                    "fuser_layer_3": fuser_layer_3_choice,
                                }
                            )
                            archon_save_path_to_score[save_path] = archon_score

        return archon_save_path_to_score

    def generate_grid_configs(self, best_config):
        grid_configs = []
        for model_top_k in self.search_config["model_top_k_choices"]:
            for sample_top_k in self.search_config["sample_top_k_choices"]:
                for fuser_layer_1 in self.search_config["fuser_layer_1_choices"]:
                    for fuser_layer_2 in self.search_config["fuser_layer_2_choices"]:
                        for fuser_layer_3 in self.search_config["fuser_layer_3_choices"]:
                            config = {
                                "model_top_k": model_top_k,
                                "sample_top_k": sample_top_k,
                                "fuser_layer_1": fuser_layer_1,
                                "fuser_layer_2": fuser_layer_2,
                                "fuser_layer_3": fuser_layer_3,
                            }
                            if config != best_config:
                                grid_configs.append(config)
        
        return grid_configs

    def perform_greedy_search(self):
        archon_save_path_to_score = {}
        best_config = {
            "model_top_k": self.search_config["model_top_k_choices"][0],
            "sample_top_k": self.search_config["sample_top_k_choices"][0],
            "fuser_layer_1": self.search_config["fuser_layer_1_choices"][0],
            "fuser_layer_2": self.search_config["fuser_layer_2_choices"][0],
            "fuser_layer_3": self.search_config["fuser_layer_3_choices"][0],
        }
        best_score = float('-inf')
        
        # Optimize parameters individually
        for param in best_config.keys():
            for value in self.search_config[f"{param}_choices"]:
                current_config = best_config.copy()
                current_config[param] = value
                
                archon_score = self.run_archon_config(**current_config)
                save_path = self.get_save_path(current_config)
                archon_save_path_to_score[save_path] = archon_score
                
                if archon_score > best_score:
                    best_score = archon_score
                    best_config[param] = value
        
        # Use remaining iterations for grid search
        remaining_iterations = self.search_config["search_iterations"] - len(archon_save_path_to_score)
        if remaining_iterations > 0:
            grid_configs = self.generate_grid_configs(best_config)
            for config in grid_configs[:remaining_iterations]:
                archon_score = self.run_archon_config(**config)
                save_path = self.get_save_path(config)
                archon_save_path_to_score[save_path] = archon_score
        
        return archon_save_path_to_score

    def perform_bayesian_optimization(self):
        archon_save_path_to_score = {}
        pbounds = {
            "model_top_k_choice": (
                min(self.search_config["model_top_k_choices"]),
                max(self.search_config["model_top_k_choices"]),
            ),
            "sample_top_k_choice": (
                min(self.search_config["sample_top_k_choices"]),
                max(self.search_config["sample_top_k_choices"]),
            ),
            "fuser_layer_1_choice": (
                min(self.search_config["fuser_layer_1_choices"]),
                max(self.search_config["fuser_layer_1_choices"]),
            ),
            "fuser_layer_2_choice": (
                min(self.search_config["fuser_layer_2_choices"]),
                max(self.search_config["fuser_layer_2_choices"]),
            ),
            "fuser_layer_3_choice": (
                min(self.search_config["fuser_layer_3_choices"]),
                max(self.search_config["fuser_layer_3_choices"]),
            ),
        }

        optimizer = BayesianOptimization(
            f=partial(self.run_archon_config),
            pbounds=pbounds,
            random_state=42,
            verbose=2,
        )

        optimizer.maximize(
            init_points=self.search_config["init_points"],
            n_iter=self.search_config["n_iter"],
        )

        print("Best result from Bayesian Optimization:", optimizer.max)

        sorted_results = sorted(optimizer.res, key=lambda x: x["target"], reverse=True)
        for result in sorted_results:
            choices = {
                "model_top_k": self.find_nearest_value(
                    result["params"]["model_top_k_choice"],
                    self.search_config["model_top_k_choices"],
                ),
                "sample_top_k": self.find_nearest_value(
                    result["params"]["sample_top_k_choice"],
                    self.search_config["sample_top_k_choices"],
                ),
                "fuser_layer_1": self.find_nearest_value(
                    result["params"]["fuser_layer_1_choice"],
                    self.search_config["fuser_layer_1_choices"],
                ),
                "fuser_layer_2": self.find_nearest_value(
                    result["params"]["fuser_layer_2_choice"],
                    self.search_config["fuser_layer_2_choices"],
                ),
                "fuser_layer_3": self.find_nearest_value(
                    result["params"]["fuser_layer_3_choice"],
                    self.search_config["fuser_layer_3_choices"],
                ),
            }
            save_path = self.get_save_path(choices)
            archon_save_path_to_score[save_path] = result["target"]

        return archon_save_path_to_score

    def display_itas_results(self, itas_results):
        table_data = []
        for i, (archon_config, score) in enumerate(
            sorted(itas_results.items(), key=lambda x: x[1], reverse=True)
        ):
            with open(archon_config, "r") as file:
                archon_json = json.load(file)
            table_data.append(
                [
                    i + 1,
                    archon_config.split("search_run_search_config_")[1][16:].replace(
                        ".json", ""
                    ),
                    score,
                    self.calculate_total_inference_calls(archon_json),
                ]
            )
        print(
            tabulate(
                table_data,
                headers=["Rank", "Archon Config", "Score", "Inference Calls"],
                tablefmt="grid",
            )
        )

    def itas_algorithm(self):
        archon_save_path_to_score = self.perform_search()
        top_10_archon_save_paths = sorted(
            archon_save_path_to_score.items(), key=lambda x: x[1]
        )[: self.search_config["number_of_archon_configs_for_final_ranking"]]

        archon_save_path_to_score_complete_dataset = {}
        for archon_config_tuple in top_10_archon_save_paths:

            # Create new unique configs for the top 10
            with open(archon_config_tuple[0], "r") as file:
                archon_json = json.load(file)

            archon_json["name"] = archon_json["name"] + "_final_ranking"
            archon_json["save_path"] = archon_json["save_path"].replace(
                ".json", "_final_ranking.json"
            )
            # archon_json["save_path"] = self.get_save_path(archon_config_dict=

            with open(archon_json["save_path"], "w") as file:
                json.dump(archon_json, file)

            #####################

            _, model_to_score_dict = self.run_benchmark(
                archon_json, self.search_config["dataset_sample_for_final_ranking"]
            )
            archon_save_path_to_score_complete_dataset[archon_json["save_path"]] = list(
                model_to_score_dict.values()
            )[0]

        self.display_itas_results(archon_save_path_to_score_complete_dataset)
        return archon_save_path_to_score_complete_dataset


################################################


def load_search_config(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run the itas algorithm with a given search configuration."
    )
    parser.add_argument(
        "--search-config",
        type=Path,
        required=True,
        help="Path to the JSON file containing the search configuration.",
    )

    args = parser.parse_args()

    # Load the search configuration from the provided file
    search_config = load_search_config(args.search_config)

    # Update search_config_name to include the current timestamp
    search_config_name = "search_config_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    search_config["search_config_name"] = search_config_name

    # Update paths in the search_config to include the search_config_name
    base_output_path = os.path.join(
        search_config["base_output_directory"], search_config_name
    )
    search_config["power_ranking_save_path"] = os.path.join(
        base_output_path, "power_ranking_config.json"
    )
    search_config["answers_save_path"] = os.path.join(
        base_output_path, "configs", "individual_model_configs"
    )
    search_config["save_directory"] = os.path.join(
        base_output_path, "model_generations"
    )

    # Define the general model configuration
    general_model_config = {"temperature": 0.7, "max_tokens": 2048}

    # Initialize and run the itas algorithm
    itas = itas(search_config=search_config, general_model_config=general_model_config)
    itas.itas_algorithm()


if __name__ == "__main__":
    main()
