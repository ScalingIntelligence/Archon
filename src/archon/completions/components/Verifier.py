import re
from .Generator import Generator
from .Component import Component
from .. import utils
from loguru import logger
from .prompts import make_verifier_reasoning_prompt, make_verifier_verdict_prompt


class Verifier(Component):
    def __init__(self, config):
        """
        Initialize the Verifier with configuration settings.

        Parameters:
        config (dict): Configuration dictionary containing model settings and other parameters.
        """
        self.config = config
        self.initialize_verifier()

    def initialize_verifier(self):
        """
        Initialize the verifier model and tokenizer with the specified settings.
        """
        self.model_name = self.config["model"]
        self.model_type = self.config["model_type"]
        self.temperature = self.config["temperature"]
        self.samples = self.config["samples"]
        self.first_k = self.config.get("first_k", None)

        self.verifier = Generator(config=self.config)

        print(f"Verifier model initialized: {self.model_name}")

    def parse_list(self, output):
        reasons = " ".join(output.split())
        # left = output.find("[")
        # right = output.rfind("]")
        # reasons = output[left + 1 : right]
        reasons_list = reasons.split("!!!")
        if utils.DEBUG_VERIFIER:
            logger.debug(f"Parsed list: {reasons_list}")

        return reasons_list

    def generate_reasoning(self, conversation, candidate):
        """
        Generate reasoning for a candidate.

        Parameters:
        conversation (list): The conversation so far.
        candidates (str): The candidate for reasoning

        Returns:
        list[str]: The reasonings for the candidate.
        """
        query = conversation[-1]["content"]
        assert isinstance(query, str) and len(query) > 0
        assert isinstance(candidate, str)

        reasoning_prompt = make_verifier_reasoning_prompt(query, candidate)

        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a reasoning generator for instructions and their responses.",
                }
            ]  # system
            + [
                message for message in conversation[:-1] if message["role"] != "system"
            ]  # rest of conversation without query
            + [{"role": "user", "content": reasoning_prompt}]  # prompt
        )

        # if utils.DEBUG_VERIFIER:
        #      logger.debug(f"Message being sent to generate reasonings: {messages[-1]["content"][:10]}")

        for retry in range(10):
            try:
                reasoning = self.verifier.generate_from_messages(
                    messages, self.temperature
                )[0]

                if utils.DEBUG_VERIFIER:
                    logger.debug(f"Output from generated reasoning: {reasoning[:10]}")

                return reasoning
            except Exception as e:
                print(f"Error generating reasoning: {e}")
                print(f"Retry #{retry + 1}...")
                continue

        raise ValueError("Failed to generate reasoning with verifier!")

    def extract_verdict(generated_response: str):
        """
        Extract the verdict from the generated response.
        """
        assert (
            "[Correct]" in generated_response or "[Incorrect]" in generated_response
        ), f"Verdict not found in generated response. Found: {generated_response}"
        assert not (
            "[Correct]" in generated_response and "[Incorrect]" in generated_response
        ), f"Both '[Correct]' and '[Incorrect]' found in generated response. Found: {generated_response}"
        # return "[Correct]" if "[Correct]" in generated_response else "[Incorrect]"
        return 1 if "[Correct]" in generated_response else 0

    def verify_query_reasoning_pairs(self, conversation: list, candidate: str, reasoning: str):
        """
        Verify the query-reasoning pair.

        Parameters:
        conversation (list): The conversation so far.
        candidate (str): The candidate generation.
        reasoning (str): The reasoning for the candidate.

        Returns:
        int: 1 if the reasoning is correct, 0 otherwise.
        """
        query = conversation[-1]["content"]
        assert isinstance(query, str) and len(query) > 0
        assert isinstance(candidate, str) and len(candidate) > 0
        assert isinstance(reasoning, str) and len(reasoning) > 0

        verdict_prompt = make_verifier_verdict_prompt(query, candidate, reasoning)

        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a verification system for judging responses and their reasoning.",
                }
            ]  # system
            + [
                message for message in conversation[:-1] if message["role"] != "system"
            ]  # rest of conversation without query
            + [{"role": "user", "content": verdict_prompt}]  # prompt
        )

        # if utils.DEBUG_VERIFIER:
        #     logger.debug(f"Messages being sent to verifier: {messages}")

        for retry in range(10):
            try:
                verdict = self.verifier.generate_from_messages(
                    messages, self.temperature
                )[0]

                if utils.DEBUG_VERIFIER:
                    logger.debug(f"Output from verifier: {verdict[-10:]}")
                # breakpoint()

                verification_result = self.parse_verification_output(verdict)

                return verification_result
            except Exception as e:
                print(f"Error verifying query-reasoning pair: {e}")
                print(f"Retry #{retry + 1}...")
                continue

        raise ValueError("Failed to verify query-reasoning pair with verifier!")

    def parse_verification_output(self, output):
        """
        Parse the output from theƒ verification model to extract the verdict.

        Parameters:
        output (str): The raw output from the verification model.

        Returns:
        int: 1 if the reasoning is correct, 0 otherwise.
        """
        assert isinstance(output, str) and len(output) > 0

        if "[Correct]" in output and "[Incorrect]" in output:
            raise ValueError(
                "Both '[Correct]' and '[Incorrect]' found in verification output."
            )
        elif "[Correct]" in output:
            return 1
        elif "[Incorrect]" in output:
            return 0
        else:
            if utils.DEBUG_VERIFIER:
                logger.error(f"Verdict not found in verification output: {output}")
            raise ValueError("Verdict not found in verification output.")
   
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
        critiques = prev_state.get("critiques", None)
        verified_candidates, verified_critiques = self.verify(conversation, candidates, critiques)

        state["candidates"].extend(verified_candidates)
        if verified_critiques:
            state["critiques"].extend(verified_critiques)

        return

    def verify(self, conversation: list, candidates: list, critiques:list=None):
        """
        Filter responses based on verification results.

        Args:
            conversation (list): A list of the conversation so far
            candidates (list):  The list of candidates to generate responses from.
            critiques (list, optional): A list of critiques, one per candidates. Defaults to None.

        Returns:
            list: A list of verified correct candidate responses.
        """

        assert isinstance(conversation, list) and isinstance(conversation[-1], dict)
        assert conversation[-1]["role"] == "user"
        assert isinstance(candidates, list) and len(candidates) > 0
        query = conversation[-1]["content"]

        ####################################

        verified_responses = []
        verified_critiques = []
        incorrect_responses = []

        if critiques:
            assert isinstance(critiques, list) and all(
                isinstance(critique, str) for critique in critiques
            )
            assert len(critiques) == len(candidates)

        for i in range(0, len(candidates)):
            cand = candidates[i]
            try:

                reasoning = self.generate_reasoning(conversation, cand)

                verification_result = self.verify_query_reasoning_pairs(
                    conversation, cand, reasoning
                )
                if utils.DEBUG_VERIFIER:
                    logger.debug(f"{verification_result}")

                if verification_result == 1:
                    verified_responses.append(cand)
                    if critiques is not None:
                        verified_critiques.append(critiques[i])
                    if self.first_k and len(verified_responses) == self.first_k:
                        return verified_responses, verified_critiques
                else:
                    incorrect_responses.append(cand)
            except Exception as e:
                print(f"Error processing candidate for verification: {e}")

        ####################################

        if utils.DEBUG_VERIFIER:
            print(f"Verified Responses Length: {len(verified_responses)}")
            print(f"Incorrect Responses Length: {len(incorrect_responses)}")

        verified_critiques = verified_critiques if len(verified_critiques) > 0 else None

        if len(verified_responses) == 0:
            verified_responses = incorrect_responses
            verified_critiques = critiques
            logger.warning(
                "No responses passed verification. Passing all responses to next layer"
            )

        return verified_responses, verified_critiques