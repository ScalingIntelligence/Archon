from .Generator import Generator
from .Component import Component
from loguru import logger
from .. import utils
from .prompts import make_fuser_prompt


class Fuser(Component):
    def __init__(self, config):
        """
        Initialize the Fuser with configuration settings.

        Parameters:
        config (dict): Configuration dictionary containing model settings and other parameters.
        """
        self.config = config
        self.generator = None
        self.initialize_fuser()

    def initialize_fuser(self):
        """
        Initialize the generator with the specified model and generation function.
        """
        self.model = self.config["model"]
        # self.top_k_generations = self.config["top_k_generations"]
        self.temperature = self.config["temperature"]
        self.samples = self.config.get("samples", 1)

        self.fuser = Generator(config=self.config)

        self.length_control = self.config.get("length_control", False)
        print(f"Fuser initialized with model: {self.model}")

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
        fused_candidates = self.fuse(conversation, candidates, critiques)

        state["candidates"].extend(fused_candidates)
        
        return
    
    def fuse(self, conversation: list, candidates: list, critiques: list=None) -> list:
        """
        Fuse the generations from multiple models

        Args:
            conversation (list): A list of the conversation so far
            candidates (list):  The list of candidates to generate responses from.
            critiques (list, optional): A list of critiques, one per candidates. Defaults to None.

        Returns:
            list: fused responses
        """        

        assert isinstance(conversation, list) and isinstance(conversation[0], dict)
        for item in conversation:
            assert isinstance(item, dict) and "role" in item and "content" in item
        # assert init_conv[0]["role"] == "user" and len(init_conv[0]["content"]) > 0
        assert (
            isinstance(candidates, list)
            and all(isinstance(context, str) for context in candidates)
            and len(candidates) > 0
        )

        if utils.DEBUG:
            logger.debug(f"Length of candidates: {len(candidates)}")
            logger.debug(
                f"Length of critiques: {len(critiques) if critiques else 'NA'}"
            )
            logger.debug(f"{candidates=}")
            logger.debug(f"{conversation=}")

        # breakpoint()

        fuser_prompt = make_fuser_prompt(
            conversation, candidates, critiques, length_control=self.length_control
        )

        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who fuses multiple answers",
                }
            ]  # system
            + [
                message for message in conversation[:-1] if message["role"] != "system"
            ]  # rest of conversation without query
            + [{"role": "user", "content": fuser_prompt}]  # fuser prompt
        )

        fuser_generations = []

        output = self.fuser.generate_from_messages(messages, self.temperature)
        if output is not None:
            fuser_generations.extend(output)

        return fuser_generations