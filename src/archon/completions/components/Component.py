from abc import ABC, abstractmethod


class Component(ABC):

    @abstractmethod
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def generate(self, **kwargs):
        """
        Currently, generate will take in these arguements,
        although you are not required to use them.
            messages=conv,
            prev_outputs=prev_outputs,
            prev_critiques=prev_critiques,
            unit_tests=unit_tests,
            custom_state=custom_state,

        The custom state, is a dict that is passed between all custom components.
        This is provided in case you want multiple components sending information to eachother
        (and not being limited by our keyword arguments)

        Output: your output is a list of str outputs to be passed to the next layer
        """
        pass
