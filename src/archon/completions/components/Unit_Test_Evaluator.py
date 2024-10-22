import pdb
import re
import ast
from .Generator import Generator
from .Component import Component
from loguru import logger
from .prompts import make_unit_test_evaluator_prompt


class Unit_Test_Evaluator(Component):
    def __init__(self, config):
        """
        Initialize the Unit_Test_Evaluator with configuration settings.

        Parameters:
        config (dict): Configuration dictionary containing model settings and other parameters.
        """
        self.config = config
        self.initialize_unit_test_evaluator()

    def initialize_unit_test_evaluator(self):
        """
        Initialize the unit test evaluator model with the specified settings.
        """
        self.model_name = self.config["model"]
        self.model_type = self.config["model_type"]
        self.temperature = self.config["temperature"]
        self.samples = self.config["samples"]
        self.unit_test_cap = self.config.get("unit_test_cap", None)
        self.remove_unit_tests_from_prompt = self.config[
            "remove_unit_tests_from_prompt"
        ]
        self.top_k = self.config.get("top_k", None)

        self.first_k = self.config.get("first_k", None)

        # backwards compatibility
        self.first_pass = self.config.get("first_pass", None)
        if self.first_k is None and self.first_pass:
            self.first_k = 1

        self.unit_test_evaluator = Generator(config=self.config)

        print(f"Unit_Test_Evaluator model initialized: {self.model_name}")

    def run(self, conversation, prev_state, state):
        """
        Run a component and updates the state accordingly.

        Args:
            conversation (list[dict]): A list of dictionaries representing the conversation with Archon. 
                Each dictionary contains role and content
            prev_state (dict): A dictionary representing the state from the previous layer.
            state (dict): A dictionary holding the values that will be updated from the previous layer to be sent to the next layer
        """

        candidates = prev_state["candidates"]
        unit_tests = prev_state["unit_tests"]

        evaluated_candidates = self.evaluate_unit_tests(conversation, candidates, unit_tests)
        
        state["candidates"].extend(evaluated_candidates)

        return
    
    def evaluate_unit_tests(
        self,
        conversation: list,
        candidate_responses: list,
        unit_tests: list,
    ):
        """
        Generate unit tests for a given conversation.

        Args:
            conversation (list): A list of the conversation so far
            candidates (list):  The list of candidates to generate responses from.
            unit_tests (list, optional): A list of unit tests

        Returns:
            list: The top_k ranked candidates passed off passed unit_tests.
        """

        # If it is a multi-stage conversation, extract all the user queries from the conversation
        query = conversation[-1]["content"]
        query = query.strip()

        if self.remove_unit_tests_from_prompt:
            query = query.split("Your response should pass the following unit tests:")[
                0
            ].strip()

        assert isinstance(query, str) and len(query) > 10
        assert (
            isinstance(candidate_responses, list)
            and len(candidate_responses) > 0
            and all([isinstance(response, str) for response in candidate_responses])
        )
        assert (
            isinstance(unit_tests, list)
            and len(unit_tests) > 0
            and all([isinstance(unit_test, str) for unit_test in unit_tests])
        )

        ########################################

        candidate_responses = [response.strip() for response in candidate_responses]
        verdict_scores = []  # List of scores for each candidate response

        so_far_pass_all = []
        for response in candidate_responses:

            assert isinstance(response, str) and len(response) > 0
            evaluator_prompt = make_unit_test_evaluator_prompt(
                query, response, unit_tests
            )

            messages = (
                [
                    {
                        "role": "system",
                        "content": "You are a unit test evaluator",
                    }
                ]  # system
                + [
                    message for message in conversation[:-1] if message["role"] != "system"
                ]  # rest of conversation without query
                + [{"role": "user", "content": evaluator_prompt}]  # prompt
            )

            for retry in range(10):
                try:
                    output = self.unit_test_evaluator.generate_from_messages(
                        messages, self.temperature
                    )
                    unit_test_verdicts = self.parse_unit_tests_evaluations(
                        output[0], len(unit_tests)
                    )
                    unit_test_pass_count = sum(
                        [1 for verdict in unit_test_verdicts if "Passed" in verdict]
                    )  # Count the number of passed unit tests

                    if self.first_k and unit_test_pass_count == len(unit_tests):
                        so_far_pass_all.append(response)
                        if len(so_far_pass_all) == self.first_k:
                            return so_far_pass_all

                    verdict_scores.append(unit_test_pass_count)
                    break
                except Exception as e:
                    print(f"Error generating unit tests: {e}")
                    print(f"Problematic messages: " + messages[-1]["content"])
                    print(
                        f"Problematic unit tests: {output[0] if len(output) > 0 else 'NA'}"
                    )
                    print(f"Retry #{retry + 1}...")

            # raise ValueError("Failed to generate unit tests with unit test evaluator!")

        ##############################

        # Rank the candidate responses based on the number of passed unit tests
        ranked_candidate_responses = [
            response
            for _, response in sorted(
                zip(verdict_scores, candidate_responses), reverse=True
            )
        ]
        ranked_candidate_response_verdict_counts = [
            score for score in sorted(verdict_scores, reverse=True)
        ]

        return ranked_candidate_responses[: self.top_k]

    def parse_unit_tests_evaluations(self, output: str, num_tests: int):
        """
        Parse the output from the unit test evaluator to extract unit test evaluations.

        Parameters:
        output (str): The raw output from the unit test evaluator.
        num_tests (int): The number of unit tests to evaluate.

        Returns:
        list: A list of generated unit test evaluations.
        """

        verdict_text = output

        verdicts = []
        for i in range(num_tests):

            start = verdict_text.rfind(
                f"Unit Test #{i+1}:"
            )  # Get last occurrence of unit test
            end = (
                verdict_text.rfind(f"Unit Test #{i+2}:")
                if i < num_tests - 1
                else len(verdict_text)
            )
            test_verdict = verdict_text[start:end]

            if "[Passed]" in test_verdict and "[Failed]" in test_verdict:
                print("Both Passed and Failed found in test verdict.")
                verdicts.append("[Unknown]")
            elif (
                "[Passed]" in test_verdict
                or "passed" in test_verdict.lower()
                or "pas" in test_verdict.lower()
            ):
                verdicts.append("[Passed]")
            elif (
                "[Failed]" in test_verdict
                or "failed" in test_verdict.lower()
                or "fai" in test_verdict.lower()
            ):
                verdicts.append("[Failed]")
            else:
                verdicts.append("[Unknown]")

        assert [
            "[Passed]" in verdict or "[Failed]" in verdict for verdict in verdicts
        ], f"Verdicts do not contain Passed or Failed. Verdicts: {verdicts}"
        assert (
            len(verdicts) == num_tests
        ), f"Number of verdicts does not match number of unit tests. Verdicts: {verdicts}, Num Tests: {num_tests}"

        return verdicts

    def parse_unit_tests_output(self, output):
        """
        Parse the output from the unit test generator to extract unit tests.

        Parameters:
        output (str): The raw output from the unit test generator.

        Returns:
        list: A list of generated unit tests.
        """
        # pdb.set_trace()
        if (
            isinstance(output, list)
            and len(output) > 0
            and [isinstance(test, str) for test in output]
        ):
            return output
        else:
            # Remove newlines and extra spaces
            assert isinstance(output, str) and len(output) > 0
            output = " ".join(output.split())

            # Remove the outer square brackets
            if output.startswith("[") and output.endswith("]"):
                output = output[1:-1]

            # Use regex to split the string into individual test cases
            pattern = r"""(?:[^,'"]|"(?:\\.|[^"])*"|'(?:\\.|[^'])*')+"""
            test_cases = re.findall(pattern, output)

            # Process each test case
            unit_tests = []
            for test in test_cases:
                # Remove leading/trailing whitespace and quotes
                test = test.strip().strip("'\"")
                # Unescape quotes
                test = test.replace("\\'", "'").replace('\\"', '"')
                unit_tests.append(test)

            # pdb.set_trace()
            assert (
                isinstance(unit_tests, list)
                and len(unit_tests) > 0
                and [isinstance(test, str) for test in unit_tests]
            )

            return unit_tests
