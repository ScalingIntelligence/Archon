from .Generator import Generator
from loguru import logger
from .. import utils
from .prompts import make_fuser_prompt


class Fuser:
    def __init__(self, config):
        """
        Initialize the Fuser with configuration settings.

        Fuser class is responsible for handling the inputs
        Adding appropriate system messages

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

    def fuse(self, messages, contexts, critiques=None):
        """
        Fuse the generations from multiple models based on the provided query and contexts.

        Parameters:
        init_conv (list of dict): The conversation of the user.
        contexts (list of str): The list of contexts to generate responses from.

        Returns:
        list of str: The top_k_fused_generations fused results.
        """

        assert isinstance(messages, list) and isinstance(messages[0], dict)
        for item in messages:
            assert isinstance(item, dict) and "role" in item and "content" in item
        # assert init_conv[0]["role"] == "user" and len(init_conv[0]["content"]) > 0
        assert (
            isinstance(contexts, list)
            and all(isinstance(context, str) for context in contexts)
            and len(contexts) > 0
        )

        if utils.DEBUG:
            logger.debug(f"Length of contexts: {len(contexts)}")
            logger.debug(
                f"Length of critiques: {len(critiques) if critiques else 'NA'}"
            )
            logger.debug(f"{contexts=}")
            logger.debug(f"{messages=}")

        # breakpoint()

        fuser_prompt = make_fuser_prompt(
            messages, contexts, critiques, length_control=self.length_control
        )

        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who fuses multiple answers",
                }
            ]  # system
            + [
                message for message in messages[:-1] if message["role"] != "system"
            ]  # rest of conversation without query
            + [{"role": "user", "content": fuser_prompt}]  # fuser prompt
        )

        fuser_generations = []

        output = self.fuser.generate_from_messages(messages, self.temperature)
        if output is not None:
            fuser_generations.extend(output)

        return fuser_generations
