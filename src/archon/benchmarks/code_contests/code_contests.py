import unittest
from datasets import load_dataset
from tqdm import tqdm
from .code_contests_utils import execution_server_client
from .utils import get_python_solutions


class TestCheckCompletions(unittest.TestCase):
    def setUp(self):
        dataset = load_dataset("deepmind/code_contests")
        train_problems = [p for p in dataset["train"]]

        self.solutions, self.input_expected_output_pairs, self.expected_corrects = (
            [],
            [],
            [],
        )
        for problem in train_problems:
            correct_python_solutions = get_python_solutions(problem)
            if len(correct_python_solutions) == 0:
                continue
            # can't add generated tests because the dataset has false negatives, see https://github.com/ScalingIntelligence/code-contests-analysis
            input_expected_output_pairs = list(
                zip(
                    problem["private_tests"]["input"],
                    problem["private_tests"]["output"],
                )
            )
            self.solutions.append(correct_python_solutions)
            self.input_expected_output_pairs.append(input_expected_output_pairs)
            self.expected_corrects.append([True] * len(correct_python_solutions))

    def test_check_completions(self):
        with execution_server_client.ExecutionServerClient() as client:
            for solutions, input_expected_output_pairs, expected_corrects in tqdm(
                zip(
                    self.solutions,
                    self.input_expected_output_pairs,
                    self.expected_corrects,
                )
            ):
                corrects = []
                for code in solutions:
                    is_correct = client.execute_code(
                        code,
                        input_expected_output_pairs,
                        timeout=10,
                        memory_limit_bytes=2_000_000_000_000,
                    )
                    corrects.append(is_correct)
                self.assertEqual(corrects, expected_corrects)


if __name__ == "__main__":
    unittest.main()
