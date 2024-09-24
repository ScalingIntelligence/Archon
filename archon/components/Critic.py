import re

from .Generator import Generator
from .prompts import make_critic_prompt


class Critic:
    def __init__(self, config):
        """
        Initialize the Critic with configuration settings.

        Parameters:
        config (dict): Configuration dictionary containing model settings and other parameters.
        """
        self.config = config
        self.initialize_critic()

    def initialize_critic(self):
        """
        Initialize the critic model and tokenizer with the specified settings.
        """
        self.model_name = self.config["model"]
        self.model_type = self.config["model_type"]
        self.temperature = self.config["temperature"]
        self.samples = self.config["samples"]

        self.critic = Generator(config=self.config)

        print(f"Critic model initialized: {self.model_name}")

    def evaluate_candidates(self, messages, candidates, temperature=None):
        """
        Evaluate the strengths and weaknesses of each candidate.

        Parameters:
        query (str): The input query.
        candidates (list of str): The list of candidate generations to evaluate.
        temperature (float, optional): Sampling temperature.

        Returns:
        dict: A dictionary with strengths and weaknesses for each candidate.
        """
        assert isinstance(messages, list) and len(messages) > 0
        assert isinstance(candidates, list) and len(candidates) > 0

        if temperature is None:
            temperature = self.temperature

        query = messages[-1]["content"]
        critic_prompt = make_critic_prompt(query, candidates)

        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a critical evaluator.",
                }
            ]  # system
            + [
                message for message in messages[:-1] if message["role"] != "system"
            ]  # rest of conversation without query
            + [{"role": "user", "content": critic_prompt}]  # prompt
        )

        for retry in range(10):
            try:
                output = self.critic.generate_from_messages(messages)
                # breakpoint()
                evaluations = self.parse_evaluation_output(output[0], candidates)
                return evaluations
            except Exception as e:
                print(f"Error for critic: {e}")
                print(f"Retry #{retry + 1}...")
                continue

        raise ValueError("Failed to evaluate candidates with critic!")

    def parse_evaluation_output(self, output, candidates):
        """
        Parse the output from the evaluation model to extract strengths and weaknesses.

        Parameters:
        output (str): The raw output from the evaluation model.

        Returns:
        list: A list of strings with strengths and weaknesses for each candidate.
        """

        assert isinstance(output, str) and len(output) > 0
        output = (
            output.replace("\n\n\n\n", "\n\n").replace("\n\n", "\n").replace("---", "")
        )
        segments = re.split(
            r"\[\d+\]", output
        )  # what happens here is that it someties references previous answers. Leading to segmenting where not assumed.

        # join segments that come in between "strengths" in case critique was split unexpectedly from above
        left = None
        new_segments = []
        for right, segment in enumerate(segments):
            if left is None and "strengths:" in segment.lower():
                left = right

            if left and "strengths:" in segment.lower():
                new_segments.append("".join(segments[left:right]))
                left = right

        new_segments.append("".join(segments[left : len(segments)]))

        evaluations = [segment.strip() for segment in new_segments if len(segment) > 10]

        ####################################

        if len(evaluations) != len(candidates):
            print(
                f"Problematic Evaluations Length: {len(evaluations)} != {len(candidates)} candidates. Evals were derived from {len(segments)} segments"
            )
            print(f"")
            raise ValueError("Number of evaluations should match number of candidates")

        for i, eval in enumerate(evaluations):
            if (
                len(eval) < 10
                or "strength" not in eval.lower()
                or "weakness" not in eval.lower()
            ):
                raise ValueError(f"Invalid evaluation for candidate {i+1}")

        return evaluations
