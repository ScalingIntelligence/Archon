from abc import ABC, abstractmethod


class Component(ABC):

    @abstractmethod
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self, conversation: list, prev_state: dict, state: dict):
        """
        Run a component and updates the state accordingly.

        Args:
            conversation (list[dict]): A list of dictionaries representing the conversation with Archon. 
                Each dictionary contains role and content
            prev_state (dict): A dictionary representing the state from the previous layer.
            state (dict): A dictionary holding the values that will be updated from the previous layer to be sent to the next layer
        """
        
