import re
from .Component import Component
from .Generator import Generator
from .prompts import make_critic_prompt


class Critic(Component):
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
        critiques = self.evaluate_candidates(conversation, candidates)
        
        state["critiques"].extend(critiques)

        return

    def evaluate_candidates(self, conversation: list, candidates: list) -> list:
        """Evaluate the strengths and weaknesses of each candidate.

        Args:
            conversation (list): The conversation so far
            candidates (list): The list of candidate generations to evaluate.

        Returns:
            list: A list of the critiques, where each index corresponds to the candidate
        """        
    
        assert isinstance(conversation, list) and len(conversation) > 0
        assert isinstance(candidates, list) and len(candidates) > 0

        query = conversation[-1]["content"]
        critic_prompt = make_critic_prompt(query, candidates)

        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a critical evaluator.",
                }
            ]  # system
            + [
                message for message in conversation[:-1] if message["role"] != "system"
            ]  # rest of conversation without query
            + [{"role": "user", "content": critic_prompt}]  # prompt
        )

        for retry in range(10):
            try:
                output = self.critic.generate_from_messages(messages, self.temperature)
                # breakpoint()
                evaluations = self.parse_evaluation_output(output[0], candidates)
                return evaluations
            except Exception as e:
                print(f"Error for critic: {e}")
                print(f"Retry #{retry + 1}...")
                continue

        raise ValueError("Failed to evaluate candidates with critic!")

    def parse_evaluation_output(self, output: str, candidates: list) -> list:
        """Parse the output from the evaluation model to extract strengths and weaknesses.

        Args:
            output (str): The raw output from the evaluation model.
            candidates (list): A list of the candidates

        Returns:
            list: a list of the critiques per candidate
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
