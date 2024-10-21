import os
import datasets
import json
import shortuuid
import time
import tiktoken
import random
from loguru import logger
import yaml
from .mix_eval.utils import (
    construct_mix_eval_prompt_freeform,
    construct_mix_eval_prompt_multichoice,
)
from .code_contests.utils import get_python_solutions
import os
import re



class Benchmark:
    def __init__(self, dataset_sample: float = 1.0, debug_data=False):
        self.dataset_sample = dataset_sample
        self.debug_data = debug_data
        self.dataset = None
        self.save_type = "json"

    def load_dataset(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_answer(self):
        raise NotImplementedError("Subclasses should implement this method")

    def process_results(self):
        raise NotImplementedError("Subclasses should implement this method")

    def save_answers(self):
        raise NotImplementedError("Subclasses should implement this method")


class AlpacaEvalBenchmark(Benchmark):

    def __init__(self, dataset_sample=1.0, debug_data=False):
        super().__init__(debug_data)

    def load_dataset(self):
        self.dataset = datasets.load_dataset(
            "tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", trust_remote_code=True
        )["eval"]
        self.dataset = self.dataset.remove_columns(["output", "generator"])

        if self.debug_data:
            self.dataset = self.dataset.select(range(5))

        return self.dataset

    def get_answer(self, item, model, config, **kwargs):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item["instruction"]},
        ]
        output = model.generate(messages)
        return {"output": output, "generator": config["name"]}

    def process_results(self, results):
        self.dataset = self.dataset.add_column("output", [r["output"] for r in results])
        self.dataset = self.dataset.add_column(
            "generator", [r["generator"] for r in results]
        )
        return self.dataset

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset

        with open(output_path, "w") as f:
            json.dump(list(answers), f, indent=2)


class MtBenchBenchmark(Benchmark):

    def __init__(self, dataset_sample: float = 1.0, debug_data=False):
        super().__init__(dataset_sample=dataset_sample, debug_data=debug_data)
        self.save_type = "jsonl"

    def load_dataset(self):
        question_file = (
            "archon/benchmarks/mt_bench/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
        )
        self.dataset = datasets.load_dataset("json", data_files=question_file)["train"]
        if self.dataset_sample < 1.0:
            self.dataset = self.dataset.select(
                range(int(len(self.dataset) * self.dataset_sample))
            )
        elif self.debug_data:
            self.dataset = self.dataset.select(range(5))

        print("MT Bench Dataset Length: ", len(self.dataset))

        return self.dataset

    def get_answer(self, item, model, config, num_choices=1, **kwargs):

        temperature_config = {
            "writing": 0.7,
            "roleplay": 0.7,
            "extraction": 0.0,
            "math": 0.0,
            "coding": 0.0,
            "reasoning": 0.0,
            "stem": 0.1,
            "humanities": 0.1,
            "arena-hard-200": 0.0,
        }

        temperature = None
        if "required_temperature" in item.keys():
            temperature = item["required_temperature"]
        elif item["category"] in temperature_config:
            temperature = temperature_config[item["category"]]

        choices = []
        for i in range(num_choices):
            turns = []
            conv = [
                {"role": "system", "content": "You are a helpful assistant."},
            ]

            for j in range(len(item["turns"])):
                conv.append({"role": "user", "content": item["turns"][j]})
                output = model.generate(conv, temperature=temperature)

                conv.append({"role": "assistant", "content": output})
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        ans = {
            "answer_id": shortuuid.uuid(),
            "model_id": config["name"],
            "choices": choices,
            "tstamp": time.time(),
        }
        return ans

    def process_results(self, results):

        # TODO: Better way to do this lol

        self.dataset = self.dataset.add_column(
            "answer_id", [r["answer_id"] for r in results]
        )
        self.dataset = self.dataset.add_column(
            "model_id", [r["model_id"] for r in results]
        )
        self.dataset = self.dataset.add_column(
            "choices", [r["choices"] for r in results]
        )
        self.dataset = self.dataset.add_column("tstamp", [r["tstamp"] for r in results])

        return self.dataset

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset

        # mt_bench expects jsonl format
        with open(output_path, "w") as f:
            for result in answers:
                f.write(json.dumps(result) + "\n")


class ArenaHardAutoBenchmark(Benchmark):

    def __init__(self, dataset_sample: float = 1.0, debug_data=False):
        super().__init__(dataset_sample=dataset_sample, debug_data=debug_data)
        self.save_type = "jsonl"

    def load_dataset(self):
        question_file = "archon/benchmarks/arena_hard_auto/arena_questions.jsonl"
        self.dataset = datasets.load_dataset("json", data_files=question_file)["train"]
        if self.dataset_sample < 1.0:
            self.dataset = self.dataset.select(
                range(int(len(self.dataset) * self.dataset_sample))
            )
        elif self.debug_data:
            self.dataset = self.dataset.select(range(5))
        return self.dataset

    def get_answer(self, item, model, config, num_choices=1, **kwargs):
        temperature = 0.7
        encoding = tiktoken.encoding_for_model(
            "gpt-3.5-turbo"
        )  # arena benchmarks on gpt 3.5 encoding
        choices = []
        for i in range(num_choices):
            turns = []
            conv = [
                {"role": "system", "content": "You are a helpful assistant."},
            ]

            for j in range(len(item["turns"])):
                conv.append({"role": "user", "content": item["turns"][j]["content"]})
                output = model.generate(conv, temperature=temperature)

                conv.append({"role": "assistant", "content": output})
                turns.append(
                    {
                        "content": output,
                        "token_len": len(
                            encoding.encode(output, disallowed_special=())
                        ),
                    }
                )

            choices.append({"index": i, "turns": turns})

        ans = {
            "question_id": item["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": config["name"],
            "choices": choices,
            "tstamp": time.time(),
        }
        return ans

    def process_results(self, results):

        # TODO: Better way to do this lol
        self.dataset = self.dataset.add_column(
            "answer_id", [r["answer_id"] for r in results]
        )
        self.dataset = self.dataset.add_column(
            "model_id", [r["model_id"] for r in results]
        )
        self.dataset = self.dataset.add_column(
            "choices", [r["choices"] for r in results]
        )
        self.dataset = self.dataset.add_column("tstamp", [r["tstamp"] for r in results])

        return self.dataset

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset

        with open(output_path, "w") as f:
            for result in answers:
                f.write(json.dumps(result) + "\n")


class MixEvalBenchmark(Benchmark):
    def __init__(self, dataset_sample=1.0, debug_data=False):
        super().__init__(debug_data)
        self.save_type = "jsonl"

    def load_dataset(self):
        question_file = "MixEval/MixEval"
        dataset = datasets.load_dataset(question_file, "MixEval")

        # Concatenate the two sections together
        self.dataset = datasets.concatenate_datasets(
            [dataset["free_form"], dataset["multiple_choice"]]
        )

        if self.debug_data:
            self.dataset = self.dataset.select(range(5))

        return self.dataset

    def get_answer(self, item, model, config, **kwargs):
        # Determine the problem type and construct the appropriate prompt
        if item["problem_type"] == "free-form":
            # Free-form question
            formatted_input = construct_mix_eval_prompt_freeform(item)
            problem_type = "free-form"
        else:
            # Multiple-choice question
            formatted_input = construct_mix_eval_prompt_multichoice(item)
            problem_type = "multiple-choice"

        # Prepare the input messages for the model
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_input},
        ]

        # Generate the output from the model
        output = model.generate(messages)

        # Construct the answer object in the required format for logging
        answer = {
            "problem_type": problem_type,
            "context": item.get("context", None),
            "prompt": item["prompt"],
            "target": item.get("target", []),
            "benchmark_name": config.get("benchmark_name", "Unknown"),
            "formatted_input": formatted_input,
            "id": item.get("id", "unknown"),
            "generator": config["name"],
            "output": output,
        }

        return answer

    def process_results(self, results):
        self.dataset = self.dataset.add_column("output", [r["output"] for r in results])
        self.dataset = self.dataset.add_column(
            "generator", [r["generator"] for r in results]
        )
        self.dataset = self.dataset.add_column(
            "formatted_input", [r["formatted_input"] for r in results]
        )
        return self.dataset

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset

        # Create directory based on the filename without extension
        dir_name = os.path.splitext(output_path)[0]
        os.makedirs(dir_name, exist_ok=True)

        # Update the file paths to be inside the created directory
        base_filename = os.path.basename(output_path)
        free_form_path = os.path.join(
            dir_name, base_filename.replace(".jsonl", "_free_form.jsonl")
        )
        mult_choice_path = os.path.join(
            dir_name, base_filename.replace(".jsonl", "_mult_choice.jsonl")
        )

        free_form_answers = [a for a in answers if a["problem_type"] == "free-form"]
        mult_choice_answers = [
            a for a in answers if a["problem_type"] == "multiple-choice"
        ]

        with open(free_form_path, "w") as f:
            for result in free_form_answers:
                f.write(json.dumps(result) + "\n")

        with open(mult_choice_path, "w") as f:
            for result in mult_choice_answers:
                f.write(json.dumps(result) + "\n")


class MixEvalHardBenchmark(Benchmark):
    def __init__(self, dataset_sample=1.0, debug_data=False):
        super().__init__(debug_data)
        self.save_type = "jsonl"

    def load_dataset(self):
        question_file = "MixEval/MixEval"
        dataset = datasets.load_dataset(question_file, "MixEval_Hard")

        # Concatenate the two sections together
        self.dataset = datasets.concatenate_datasets(
            [dataset["free_form"], dataset["multiple_choice"]]
        )

        if self.debug_data:
            self.dataset = self.dataset.select(range(5))

        return self.dataset

    def get_answer(self, item, model, config, **kwargs):
        # Determine the problem type and construct the appropriate prompt
        if item["problem_type"] == "free-form":
            # Free-form question
            formatted_input = construct_mix_eval_prompt_freeform(item)
            problem_type = "free-form"
        else:
            # Multiple-choice question
            formatted_input = construct_mix_eval_prompt_multichoice(item)
            problem_type = "multiple-choice"

        # Prepare the input messages for the model
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_input},
        ]

        # Generate the output from the model
        output = model.generate(messages)

        # Construct the answer object in the required format for logging
        answer = {
            "problem_type": problem_type,
            "context": item.get("context", None),
            "prompt": item["prompt"],
            "target": item.get("target", []),
            "benchmark_name": config.get("benchmark_name", "Unknown"),
            "formatted_input": formatted_input,
            "id": item.get("id", "unknown"),
            "generator": config["name"],
            "output": output,
        }

        return answer

    def process_results(self, results):
        self.dataset = self.dataset.add_column("output", [r["output"] for r in results])
        self.dataset = self.dataset.add_column(
            "generator", [r["generator"] for r in results]
        )
        self.dataset = self.dataset.add_column(
            "formatted_input", [r["formatted_input"] for r in results]
        )
        return self.dataset

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset

        # Create directory based on the filename without extension
        dir_name = os.path.splitext(output_path)[0]
        os.makedirs(dir_name, exist_ok=True)

        # Update the file paths to be inside the created directory
        base_filename = os.path.basename(output_path)
        free_form_path = os.path.join(
            dir_name, base_filename.replace(".jsonl", "_free_form_hard.jsonl")
        )
        mult_choice_path = os.path.join(
            dir_name, base_filename.replace(".jsonl", "_mult_choice_hard.jsonl")
        )

        free_form_answers = [a for a in answers if a["problem_type"] == "free-form"]
        mult_choice_answers = [
            a for a in answers if a["problem_type"] == "multiple-choice"
        ]

        with open(free_form_path, "w") as f:
            for result in free_form_answers:
                f.write(json.dumps(result) + "\n")

        with open(mult_choice_path, "w") as f:
            for result in mult_choice_answers:
                f.write(json.dumps(result) + "\n")

CC_IMAGE_TAGS = ["<image>", "[Image]"]
CC_PROMPT = "Q: Write python code to solve the following coding problem that obeys the constraints and passes the example test cases. The output code needs to read from and write to standard IO. Please wrap your code answer using ```:"

class CodeContestsBenchmark(Benchmark):

    def __init__(self, dataset_sample=1.0, debug_data=False):
        super().__init__(dataset_sample=dataset_sample, debug_data=debug_data)
        self.save_type = "yaml"
        self.num_few_shot = 1
        self.limit = None
        self.offset = None
        self.stride = None

    def has_image_tags(self, description):
        for tag in CC_IMAGE_TAGS:
            if tag in description:
                return True
        return False

    def load_dataset(self):

        dataset = datasets.load_dataset("deepmind/code_contests")
        few_shot_dataset = [p for p in dataset["train"]]
        test_dataset = [p for p in dataset["test"]]

        random.seed(0)

        if self.debug_data:
            logger.info("Getting few_shot_items_with_solutions")

        few_shot_items_with_solutions = []
        for i, data in enumerate(few_shot_dataset):
            python_solutions = get_python_solutions(data)
            data["python_solutions"] = python_solutions
            if len(python_solutions) > 0 and not self.has_image_tags(
                data["description"]
            ):
                few_shot_items_with_solutions.append(data)

        if self.debug_data:
            logger.info("Getting no_image_test_dataset")

        no_image_test_dataset = []
        for i, data in enumerate(test_dataset):
            if self.has_image_tags(data["description"]):
                continue
            few_shot_items = random.sample(
                few_shot_items_with_solutions, self.num_few_shot
            )
            data["few_shot_items"] = few_shot_items
            no_image_test_dataset.append(data)

        random.shuffle(no_image_test_dataset)

        limit = self.limit if self.limit else len(no_image_test_dataset)
        stride = self.stride if self.stride else 1
        offset = self.offset if self.offset else 0

        self.dataset = no_image_test_dataset[offset:limit:stride]

        if self.debug_data:
            self.dataset = self.dataset[:5]
        if self.debug_data:
            logger.debug(f"peak: {self.dataset[0].keys()}")

        print(f"Total number of items to process: {len(self.dataset)}")
        return self.dataset

    def problem_to_prompt(self, problem, add_solution=True):
        prompt = f"{CC_PROMPT}\n{problem['description']}\nA:"
        if add_solution:
            prompt += f" ```{problem['python_solutions'][0].strip()}```"

        return prompt

    def get_prompt(self, item):
        prompt = "\n".join(
            [
                self.problem_to_prompt(few_shot_item)
                for few_shot_item in item["few_shot_items"]
            ]
        )
        prompt += "\n" + self.problem_to_prompt(item, add_solution=False)
        return prompt

    def get_test_cases(self, item):
        return {
            "input": item["public_tests"]["input"]
            + item["private_tests"]["input"]
            + item["generated_tests"]["input"],
            "output": item["public_tests"]["output"]
            + item["private_tests"]["output"]
            + item["generated_tests"]["output"],
        }

    def get_timeout(self, item):
        timeout_seconds = 0
        if item["time_limit"] is not None:
            timeout_seconds += item["time_limit"]["seconds"]
            timeout_seconds += item["time_limit"]["nanos"] / 1_000_000_000

        if timeout_seconds == 0:
            timeout_seconds = None
        return timeout_seconds

    def get_answer(self, item, model, config, samples=1, **kwargs):
        prompt = self.get_prompt(item)

        if self.debug_data:
            logger.info(prompt)

        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": prompt},
        ]
        output = []
        for sample in range(samples):
            output.append(model.generate(messages))

        ans = {
            "prompt": prompt,
            "question": item["description"],
            "name": item["name"],
            "samples": output,
            "test_cases": self.get_test_cases(item),
            "timeout": self.get_timeout(item),
        }

        return ans

    def process_results(self, results):
        self.dataset = results
        return results

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset

        with open(output_path, "w") as f:
            yaml.dump(answers, f)


class GSM8KBenchmark(Benchmark):
    def __init__(self, dataset_sample=1.0, debug_data=False):
        super().__init__(dataset_sample=dataset_sample, debug_data=debug_data)
        self.dataset_sample = dataset_sample
        self.debug_data = debug_data
        self.dataset = None
        self.save_type = "json"

        self.prompt = "Answer the following mathematics question. Provide your reasoning by showing your work before your answer. "
        self.prompt += "At the end of your response, output your final answer in the format: 'The answer is: [answer]'. "
        self.prompt += "You must provide the separator 'The answer is: ' before your final answer. "
        self.prompt += "Question: "

    def load_dataset(self):

        self.dataset = datasets.load_dataset("openai/gsm8k", "main")["test"]
        self.dataset = self.dataset.select(
            range(int(len(self.dataset) * self.dataset_sample))
        )

        random.seed(0)

        if self.debug_data:
            self.dataset = self.dataset.select(range(5))
        if self.debug_data:
            logger.debug(f"peak: {self.dataset[0].keys()}")

        print(f"Total number of items to process: {len(self.dataset)}")
        return self.dataset

    def get_answer(self, item, model, config, **kwargs):

        prompt = self.prompt + item["question"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        output = model.generate(messages)
        return {"prompt": prompt, "output": output, "generator": config["name"]}

    def process_results(self, results):
        self.dataset = self.dataset.add_column("prompt", [r["prompt"] for r in results])
        self.dataset = self.dataset.add_column("output", [r["output"] for r in results])
        self.dataset = self.dataset.add_column(
            "generator", [r["generator"] for r in results]
        )
        return self.dataset

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset

        # Append if file not deleted
        answers = list(answers)
        prev_answers = None

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                prev_answers = json.load(f)

            if isinstance(prev_answers, list):
                if len(prev_answers) != 0 and not isinstance(prev_answers[0], list):
                    prev_answers = [prev_answers]

                prev_answers.append(answers)
                answers = prev_answers
            else:
                answers = [answers]

        else:
            answers = [answers]

        with open(output_path, "w") as f:
            json.dump(answers, f, indent=2)


class HumanEvalBenchmark(Benchmark):
    def __init__(self, dataset_sample=1.0, debug_data=False):
        language = "python"
        temp_dir = "tmp"
        self.lang = language
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        self.save_type = "json"
        problem_file = os.path.join("human_eval/data", f"humaneval-{self.lang}.jsonl")

        self.examples = [json.loads(x) for x in open(problem_file) if x.strip()]
        self.examples = self.examples
        print("Read {} examples for evaluation over.".format(len(self.examples)))

    def load_dataset(self):
        self.dataset = datasets.Dataset.from_list(self.examples)
        return self.dataset

    def build_deepseekcoder_instruction(self, languge: str, question: str):
        return """
        Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
        ```{}
        {}
        ```
        """.strip().format(
            languge.lower(), question.strip()
        )

    def get_answer(self, item, model, config, **kwargs):
        from human_eval.utils.utils import extract_generation_code, language_settings

        prompt = self.build_deepseekcoder_instruction(
            language_settings[self.lang]["full_name"], item["prompt"]
        )

        messages = [{"role": "user", "content": prompt}]
        output = model.generate(messages)
        item["output"] = output
        return extract_generation_code(item, lang_code=self.lang)

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset

        with open(output_path, "w", encoding="utf-8") as fw:
            for ex in answers:
                fw.write(json.dumps(ex) + "\n")

    def process_results(self, results):
        if "prompt" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                "prompt", [r["prompt"] for r in results]
            )
        if "output" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                "output", [r["output"] for r in results]
            )
        if "generation" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                "generation", [r["generation"] for r in results]
            )
        return self.dataset


class MBPPBenchmark(Benchmark):
    def __init__(self, dataset_sample=1.0, debug_data=False):
        language = "python"
        temp_dir = "tmp"
        self.lang = language
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        self.save_type = "json"
        problem_file = os.path.join("mbpp/data", f"mbpp.jsonl")

        self.examples = list(self.read_test_examples(problem_file))
        print("Read {} examples for evaluation over.".format(len(self.examples)))

    def read_test_examples(self, data_path):
        def format_test_example(q, tests, code: str = None):
            prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(
                q.strip(), "\n".join(tests)
            )
            if code:
                code = code.replace("\r", "").replace("\t", "    ")
                prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
            return prompt

        examples = [json.loads(x) for x in open(data_path)]
        print("Read all {} examples from {} over!".format(len(examples), data_path))

        # test_cases
        examples_str = []
        for i in range(1, 4):
            ex = examples[i]
            q, test, code = ex["text"], ex["test_list"], ex["code"]
            ex_prompt = format_test_example(q, test, code)
            example_prompt = "- Example {}:\n{}".format(i, ex_prompt)
            examples_str += [example_prompt]

        for i in range(10, 510):
            ex = examples[i]
            q, test, code = ex["text"], ex["test_list"], ex["code"]

            prompt = format_test_example(q, test, code=None)

            prompt_with_shots = """
            Please refer the given examples and generate a python function for my problem.
            Examples are listed as follows:
            {}

            Here is my problem:
            {}
            """.strip().format(
                "\n\n".join(examples_str), prompt
            )
            yield {"task_id": ex["task_id"], "prompt": prompt_with_shots}

    def load_dataset(self):
        self.dataset = datasets.Dataset.from_list(self.examples)
        return self.dataset

    def get_answer(self, item, model, config, **kwargs):
        messages = [{"role": "user", "content": item["prompt"]}]
        output = model.generate(messages)
        item["output"] = output
        return self.convert_for_evaluation(item)

    def convert_for_evaluation(self, example):
        gpt_completion = example["output"]
        generation = gpt_completion
        try:
            code_block: str = re.findall(
                f"```python\n(.*?)```", gpt_completion, re.DOTALL | re.IGNORECASE
            )[0]
            generation = code_block
        except Exception as ex:
            print("Failed to extract codeblock:\n{}".format(gpt_completion))

        example["generation"] = generation
        return example

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset

        with open(output_path, "w", encoding="utf-8") as fw:
            for ex in answers:
                fw.write(json.dumps(ex) + "\n")

    def process_results(self, results):
        if "prompt" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                "prompt", [r["prompt"] for r in results]
            )
        if "output" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                "output", [r["output"] for r in results]
            )
        if "generation" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                "generation", [r["generation"] for r in results]
            )
        if "task_id" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column(
                "task_id", [r["task_id"] for r in results]
            )
        return self.dataset


class MATHBenchmark(Benchmark):
    def __init__(self, dataset_sample=1.0, debug_data=False):
        super().__init__(dataset_sample=dataset_sample, debug_data=debug_data)
        self.dataset_sample = dataset_sample
        self.debug_data = debug_data
        self.dataset = None
        self.save_type = "json"

        self.prompt = "Answer the following mathematics question. Provide your reasoning by showing your work before your answer. "
        self.prompt += "At the end of your response, output your final answer in the format: 'The answer is: [answer]'. "
        self.prompt += "You must provide the separator 'The answer is: ' before your final answer. "
        self.prompt += "Question: "

    def load_dataset(self):

        self.dataset = datasets.load_dataset("hendrycks/competition_math")["test"]
        self.dataset = self.dataset.select(
            range(int(len(self.dataset) * self.dataset_sample))
        )

        random.seed(0)

        if self.debug_data:
            self.dataset = self.dataset.select(range(5))
        if self.debug_data:
            logger.debug(f"peak: {self.dataset[0].keys()}")

        print(f"Total number of items to process: {len(self.dataset)}")
        return self.dataset

    def get_answer(self, item, model, config, **kwargs):

        prompt = self.prompt + item["problem"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        output = model.generate(messages)
        return {"prompt": prompt, "output": output, "generator": config["name"]}

    def process_results(self, results):
        self.dataset = self.dataset.add_column("prompt", [r["prompt"] for r in results])
        self.dataset = self.dataset.add_column("output", [r["output"] for r in results])
        self.dataset = self.dataset.add_column(
            "generator", [r["generator"] for r in results]
        )
        return self.dataset

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset

        # Append if file not deleted
        answers = list(answers)
        prev_answers = None

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                prev_answers = json.load(f)

            if isinstance(prev_answers, list):
                if len(prev_answers) != 0 and not isinstance(prev_answers[0], list):
                    prev_answers = [prev_answers]

                prev_answers.append(answers)
                answers = prev_answers
            else:
                answers = [answers]

        else:
            answers = [answers]

        with open(output_path, "w") as f:
            json.dump(answers, f, indent=2)


class MiniF2FBenchmark(Benchmark):
    def __init__(self, dataset_sample=1.0, debug_data=False):
        super().__init__(dataset_sample=dataset_sample, debug_data=debug_data)
        self.dataset_sample = dataset_sample
        self.debug_data = debug_data
        self.dataset = None
        self.save_type = "json"

        self.prompt = "Answer the following mathematics question. Provide your reasoning by showing your work before your answer. "
        self.prompt += "At the end of your response, output your final answer in the format: 'The answer is: [answer]'. "
        self.prompt += "You must provide the separator 'The answer is: ' before your final answer. "
        self.prompt += "Question: "

    def load_dataset(self):

        self.dataset = datasets.load_dataset("cat-searcher/minif2f-lean4")["test"]
        self.dataset = self.dataset.select(
            range(int(len(self.dataset) * self.dataset_sample))
        )

        random.seed(0)

        if self.debug_data:
            self.dataset = self.dataset.select(range(5))
        if self.debug_data:
            logger.debug(f"peak: {self.dataset[0].keys()}")

        print(f"Total number of items to process: {len(self.dataset)}")
        return self.dataset

    def get_answer(self, item, model, config):

        prompt = self.prompt + item["informal_stmt"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        output = model.generate(messages)
        return {"prompt": prompt, "output": output, "generator": config["name"]}

    def process_results(self, results):
        self.dataset = self.dataset.add_column("prompt", [r["prompt"] for r in results])
        self.dataset = self.dataset.add_column("output", [r["output"] for r in results])
        self.dataset = self.dataset.add_column(
            "generator", [r["generator"] for r in results]
        )
        return self.dataset

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset

        with open(output_path, "w") as f:
            json.dump(list(answers), f, indent=2)


class MMLUBenchmark(Benchmark):
    def __init__(self, dataset_sample=1.0, debug_data=False):
        super().__init__(dataset_sample=dataset_sample, debug_data=debug_data)
        self.save_type = "jsonl"
        # Set task from the MMLU dataset:
        self._local_task = "elementary_mathematics"

    def load_dataset(self):
        # Load the MMLU dataset (local_task):
        self.dataset = datasets.load_dataset("cais/mmlu", self._local_task)["test"]

        if self.dataset_sample < 1.0:
            self.dataset = self.dataset.select(
                range(int(len(self.dataset) * self.dataset_sample))
            )
        elif self.debug_data:
            self.dataset = self.dataset.select(range(5))

        print(f"Total number of items to process: {len(self.dataset)}")
        return self.dataset

    def get_answer(self, item, model, config, **kwargs):

        # Format the prompt for the model
        prompt = f"Question: {item['question']}\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nC. {item['choices'][2]}\nD. {item['choices'][3]}\n\nAnswer:"
        prompt += "At the end of your response, output your final answer in the format: 'The answer is: [answer]'. "
        # Prepare the input messages for the model
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # Generate the output from the model
        output = model.generate(messages)

        # Construct the answer object in the required format for logging
        answer = {
            "prompt": prompt,
            "output": output,
            "generator": config["name"],
            "target": item["answer"],
            "task": item["subject"].replace("_", " "),
        }

        return answer

    def process_results(self, results):
        # Add the generated answers to the dataset
        self.dataset = self.dataset.add_column("prompt", [r["prompt"] for r in results])
        self.dataset = self.dataset.add_column("output", [r["output"] for r in results])
        self.dataset = self.dataset.add_column(
            "generator", [r["generator"] for r in results]
        )
        return self.dataset

    def save_answers(self, output_path, answers=None):
        if answers is None:
            answers = self.dataset
        with open(output_path, "w") as f:
            json.dump(list(answers), f, indent=2)


BENCHMARK_CLASSES = {
    "alpaca_eval": AlpacaEvalBenchmark,
    "mt_bench": MtBenchBenchmark,
    "arena_hard_auto": ArenaHardAutoBenchmark,
    "mix_eval": MixEvalBenchmark,
    "mix_eval_hard": MixEvalHardBenchmark,
    "code_contests": CodeContestsBenchmark,
    "gsm8k": GSM8KBenchmark,
    "human_eval": HumanEvalBenchmark,
    "mbpp": MBPPBenchmark,
    "math": MATHBenchmark,
    "minif2f": MiniF2FBenchmark,
    "mmlu": MMLUBenchmark,
}


def load_benchmark(benchmark_name, dataset_sample: float = 1.0, debug_data=False):

    if benchmark_name in BENCHMARK_CLASSES:
        return BENCHMARK_CLASSES[benchmark_name](
            dataset_sample=dataset_sample, debug_data=debug_data
        )
    else:
        supported_benchmarks = ", ".join(BENCHMARK_CLASSES.keys())
        raise ValueError(
            f"Unsupported benchmark: {benchmark_name}. Only {supported_benchmarks} are supported."
        )
