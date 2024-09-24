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
            messages=messages,
            prev_outputs=prev_outputs,
            prev_critiques=prev_critiques,
            unit_tests=unit_tests,
            temperature=temperature,

        Output: your output is a list of str outputs to be passed to the next layer
        """
        pass
