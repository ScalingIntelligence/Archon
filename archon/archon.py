from components import (
    Generator,
    Ranker,
    Fuser,
    Critic,
    Verifier,
    Unit_Test_Generator,
    Unit_Test_Evaluator,
    Component,
)
import utils as utils
from loguru import logger
import random

MODEL_TYPE_MAP = {
    "generator": Generator,
    "ranker": Ranker,
    "fuser": Fuser,
    "critic": Critic,
    "verifier": Verifier,
    "unit_test_generator": Unit_Test_Generator,
    "unit_test_evaluator": Unit_Test_Evaluator,
}


class Layer:
    def __init__(self, layer_config, custom_components, custom_generators):
        self.config = layer_config
        self.models = []
        self.custom_components = custom_components
        self.custom_generators = custom_generators
        self.initialize_layer()

    def initialize_layer(self):

        model_list = self.config
        if isinstance(self.config, dict):
            model_list = self.config["models"]

        for model_config in model_list:
            model_type = model_config["type"]

            if model_type in MODEL_TYPE_MAP:
                if model_type == "generator":
                    self.models.append(
                        MODEL_TYPE_MAP[model_type](model_config, self.custom_generators)
                    )
                else:
                    self.models.append(MODEL_TYPE_MAP[model_type](model_config))
            else:
                try:
                    # trying for custom
                    component = self.custom_components[model_config["type"]]
                    self.models.append(component(model_config))
                except Exception as e:
                    logger.error(e)
                    raise ValueError(
                        f"Unsupported object type: {model_config['type']}. Check config (set Custom to true), add custom component before initiliaziation, and make sure custom component has been correctly made"
                    )
        logger.info(f"Initialized layer with {len(self.models)} models")

    def process(
        self,
        conv,
        prev_outputs=[],
        prev_critiques=None,
        temperature=None,
        unit_tests=None,
    ):

        query = conv[-1]["content"]
        output = []
        output_critiques = None

        # if self.prev_layer is not None:
        #     prev_outputs, prev_critques = self.prev_layer.process()
        # else:
        #     prev_outputs = []
        #     prev_critques = []

        if len(prev_outputs) > 32:
            logger.info(
                f"WARNING: Previous inputs of length ({len(prev_outputs)}) are too long! Will likely exceed context window of generator LMs"
            )

        for model in self.models:

            # add unit tests to query if present
            if unit_tests is not None:
                assert conv[-1]["role"] == "user"
                current_query = conv[-1]["content"]
                current_query += (
                    " Your response should pass the following unit tests: \n"
                )
                current_query += "- " + "\n- ".join(unit_tests)
                conv[-1]["content"] = current_query
                # unit_tests = None

            if isinstance(model, Generator):  # Generating responses from ensemble
                assert (
                    len(prev_outputs) == 0
                ), "Likely that model type not in first layer. Check config"

                output.extend(model.generate_from_messages(conv, temperature))

            elif isinstance(
                model, Fuser
            ):  # Generating fused responses from ensemble candidates

                fused_output = model.fuse(conv, prev_outputs, critiques=prev_critiques)
                output.extend(fused_output)

            elif isinstance(
                model, Ranker
            ):  # Ranking the responses from ensemble candidates
                assert len(self.models) == 1 and isinstance(model, Ranker)
                ranked_outputs, ranked_critiques = model.rank(
                    conv, prev_outputs, critiques=prev_critiques
                )
                output.extend(ranked_outputs)
                output_critiques = ranked_critiques

            elif isinstance(
                model, Critic
            ):  # Evaluating the responses from ensemble candidates
                assert len(self.models) == 1 and isinstance(model, Critic)
                evaluations = model.evaluate_candidates(conv, prev_outputs, temperature)

                output = prev_outputs
                output_critiques = evaluations

            elif isinstance(
                model, Verifier
            ):  # Verifying the responses from ensemble candidates

                assert len(self.models) == 1 and isinstance(model, Verifier)
                
                verified_outputs, verified_critiques = model.verify(
                    conv, prev_outputs, prev_critiques
                )
                output.extend(verified_outputs)
                output_critiques = verified_critiques

            elif isinstance(
                model, Unit_Test_Generator
            ):  # Generating unit tests for the responses from ensemble candidates
                assert len(self.models) == 1 and isinstance(model, Unit_Test_Generator)
                unit_tests = model.generate_unit_tests(conv, temperature)

            elif isinstance(
                model, Unit_Test_Evaluator
            ):  # Evaluating the responses from ensemble candidates using unit tests
                assert len(self.models) == 1 and isinstance(model, Unit_Test_Evaluator)
                ranked_outputs = model.evaluate_unit_tests(
                    messages=conv,
                    candidate_responses=prev_outputs,
                    unit_tests=unit_tests,
                    temperature=temperature,
                )
                output.extend(ranked_outputs)
                output_critiques = None
            elif isinstance(model, Component):
                custom_output = model.generate(
                    messages=conv,
                    prev_outputs=prev_outputs,
                    prev_critiques=prev_critiques,
                    unit_tests=unit_tests,
                    temperature=temperature,
                )
                output.extend(custom_output)

            else:
                raise ValueError(f"Unsupported object type: {type(model).__name__}")

        return output, output_critiques, unit_tests


class Archon:
    """
    Archon class to generate responses given an input by applying multiple layers
    of inference times techniques sequentially.
    """

    def __init__(self, config, api_key_file=None):
        """
        Initialize the Archon with configuration settings.

        Parameters:
        config (dict): Configuration dictionary containing layers and other settings.
        api_key_file (str, optional): path to JSON of api_keys to use on generation. Defaults to None and use environment variables otherwise.
        """
        self.config = config
        self.initialized = False

        # attributes for custom 
        self.custom = config.get("custom", False)
        self.custom_components = {}
        self.custom_generators = {}
        

        if api_key_file:
            # initialize an object that is used in utils.py to handle key swapping
            utils.KEYS = utils.keyHandler(api_key_file)

        # if custom, user has to manually initialize
        if not self.custom:
            self.initialize()
        else:
            logger.warning(
                "Custom model, make sure to add custom components before initializing."
            )

    def add_component(self, name: str, component: Component):
        """ add a custom component for use in archon configuration

        Args:
            name (str): Name of component, must match name in archon config
            component (Component): Component to be called during inference time
        """        
        self.custom_components[name] = component

    def add_generator(self, name: str, generator):
        """add a custom generator for use in archon configuration

        Args:
            name (str): Name of generator, must match name in archon config
            generator (): _description_
        """        
        self.custom_generators[name] = generator

    def initialize(self):
        
        """
        Initialize the archon model, layer by layer.
        """

        self.layers = []
        for layer_config in self.config["layers"]:
            layer = Layer(layer_config, self.custom_components, self.custom_generators)
            self.layers.append(layer)

        print(f"Archon initialized with {len(self.layers)} layers.")
        self.initialized = True

    def generate(self, conv, temperature=None):
        """_summary_

        Args:
            conv (list): A list of the conversation so far
            temperature (float, optional): temperature to use for generation. Defaults to None.

        Returns:
            str: generated answer to given conversation
        """        
        if not self.initialized:
            raise Exception(
                f"Initialize your archon model before generating. This most likely happens because you have a custom component"
            )

        # if only query was given
        if isinstance(conv, str):
            conv = {
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": conv},
            }
        elif conv[0]["role"] != "system":  # add a system message if missing
            conv = [{"role": "system", "content": "You are a helpful assistant"}] + [
                message for message in conv
            ]

        response, critique = self._generate(conv, temperature)

        if utils.DEBUG_ARCHON:
            if not len(response) > 0:
                logger.error(f"Response is empty: {response}")
            if not isinstance(response, list):
                logger.error(f"Response is not a list: {response}")
            if not isinstance(response[0], str):
                logger.error(
                    f"First element of response is not a string: {response[0]}"
                )

        assert (
            len(response) > 0
            and isinstance(response[0], str)
            and isinstance(response, list)
        ), f"response not valid: {response}"

        if len(response) > 1:
            # pick one random. Not want first element cause that can
            # produce biased towards first one added to list because of speed or order in config
            response = [random.choice(response)]

        return response[0]

    def _generate(self, conv, temperature=None):
        """
        Generate responses by applying each layer sequentially to the inputs.

        Parameters:
        conv (str): list of dicts

        Returns:
        list of str: The final generated responses.
        """

        # messages should just be a single list of optional system and user messages
        # query = conv[-1]["content"]

        prev_output = []
        prev_critiques = []
        unit_tests = None

        for i in range(len(self.layers)):

            layer = self.layers[i]
            layer_output = []
            layer_critique = []

            if utils.DEBUG:
                logger.debug(
                    f"Running layer {i}, with {len(prev_output)} previous outputs and {len(prev_critiques) if prev_critiques else 0} previous critiques"
                )

            if utils.DEBUG:
                logger.debug(f"Running layer {i}")

            layer_output, layer_critique, unit_tests = layer.process(
                conv,
                prev_output,
                prev_critiques,
                temperature=temperature,
                unit_tests=unit_tests,
            )

            prev_output = layer_output

            # None if empty
            prev_critiques = layer_critique if layer_critique else None

        if len(prev_output) == 0:
            logger.warning("No output generated by Archon!")
        elif len(prev_output) > 1:
            logger.warning(
                f"Multiple outputs generated by Archon! Returning a random candidate from the set of {len(prev_output)} choices."
            )
            prev_output = [random.choice(prev_output)]

        return prev_output, prev_critiques
