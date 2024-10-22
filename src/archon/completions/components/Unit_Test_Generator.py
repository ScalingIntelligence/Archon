import re
from .Generator import Generator
from .Component import Component
from .. import utils
from loguru import logger
from .prompts import make_unit_test_generator_prompt


class Unit_Test_Generator(Component):
    def __init__(self, config):
        """
        Initialize the Unit_Test_Generator with configuration settings.

        Parameters:
        config (dict): Configuration dictionary containing model settings and other parameters.
        """
        self.config = config
        self.initialize_unit_test_generator()

    def initialize_unit_test_generator(self):
        """
        Initialize the unit test generator model with the specified settings.
        """
        self.model_name = self.config["model"]
        self.model_type = self.config["model_type"]
        self.temperature = self.config["temperature"]
        self.samples = self.config["samples"]
        self.unit_test_cap = (
            self.config["unit_test_cap"] if "unit_test_cap" in self.config else None
        )

        self.unit_test_generator = Generator(config=self.config)

        print(f"Unit_Test_Generator model initialized: {self.model_name}")

    def run(self, conversation, prev_state, state):
        """
        Run a component and updates the state accordingly.

        Args:
            conversation (list[dict]): A list of dictionaries representing the conversation with Archon. 
                Each dictionary contains role and content
            prev_state (dict): A dictionary representing the state from the previous layer.
            state (dict): A dictionary holding the values that will be updated from the previous layer to be sent to the next layer
        """

        unit_tests = self.generate_unit_tests(conversation)
        
        state["unit_tests"].extend(unit_tests)

        return

    def generate_unit_tests(self, conversation: list) -> list:
        """
        Generate unit tests for a given conversation.

        Parameters:
        conversation (list): The conversation to generate unit testst for.

        Returns:
        list: A list of generated unit tests.
        """

        # If it is a multi-stage conversation, extract all the user queries from the messages
        query = conversation[-1]["content"]
        query = query.strip()

        assert isinstance(query, str) and len(query) > 0

        ########################################

        unit_test_generator_prompt = make_unit_test_generator_prompt(
            query, self.unit_test_cap
        )

        ########################################

        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a unit test generator.",
                }
            ]  # system
            + [
                message for message in conversation[:-1] if message["role"] != "system"
            ]  # rest of conversation without query
            + [{"role": "user", "content": unit_test_generator_prompt}]  # prompt
        )

        for retry in range(10):
            try:
                output = self.unit_test_generator.generate_from_messages(
                    messages, self.temperature
                )
                unit_tests = self.parse_unit_tests_output(output[0])

                if utils.DEBUG_UNIT_TEST_GENERATOR:
                    logger.debug(f"{len(unit_tests)=}")

                if (
                    self.unit_test_cap is not None
                    and len(unit_tests) != self.unit_test_cap
                ):
                    print(unit_tests)
                    raise f"Unit tests doesn't match unit_test cap"

                unit_tests = (
                    unit_tests[: self.unit_test_cap]
                    if self.unit_test_cap is not None
                    else unit_tests
                )

                # add unit tests to prompt (by refernce)
                assert conversation[-1]["role"] == "user"
                current_query = conversation[-1]["content"]
                current_query += (
                    " Your response should pass the following unit tests: \n"
                )
                current_query += "- " + "\n- ".join(unit_tests)
                conversation[-1]["content"] = current_query

                return unit_tests
            except Exception as e:
                print(f"Error generating unit tests: {e}")
                print(f"Problematic messages: " + messages[-1]["content"][:50])
                print(
                    f"Problematic unit tests: {output[0] if len(output) > 0 else 'NA'}"
                )
                print(f"Retry #{retry + 1}...")

        raise ValueError("Failed to generate unit tests with unit test generator!")

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
