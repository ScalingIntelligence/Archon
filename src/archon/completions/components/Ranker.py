from .Generator import Generator
from .Component import Component
import threading
from .. import utils
from loguru import logger
import re
from .prompts import make_ranker_prompt


class Ranker(Component):
    def __init__(self, config):
        """
        Initialize the Ranking class with configuration settings.

        Parameters:
        config (dict): Configuration dictionary containing model settings and other parameters.
        """
        self.config = config
        self.ranker = None
        self.initialize_ranker()

    def initialize_ranker(self):
        """
        Initialize the ranker with the specified model and checkpoint.
        """
        self.model_name = self.config["model"]
        self.model_type = self.config["model_type"]
        self.top_k = self.config["top_k"]
        self.temperature = self.config["temperature"]
        self.use_critiques = self.config.get("use_critiques", False)

        if self.model_name == "llm-blender/PairRM":
            # multiple threads will use the same tokenizer. So it has to be locked
            self.ranker_lock = threading.Lock()

            # TODO: loading a blender will take a long time. I wonder if having
            # multiple rankers will be too long of an initializaion
            self.ranker_batch_size = self.config["ranker_batch_size"]
            import llm_blender

            self.ranker = llm_blender.Blender()
            self.ranker.loadranker(self.model_name)
        else:
            self.config["samples"] = 1
            # expecting one sample
            self.ranker = Generator(config=self.config)

        print(f"Ranker initialized with model: {self.model_name}")

    def extract_ranking(self, output: str, candidates: list):
        answer_str = output[0].partition("\n")[0]
        ranks_str = re.findall(r"\[(\d+)\]", answer_str)

        # Check that length of ranks_str matches length of candidates
        # and that all the ranks are from 1 to len(candidates), inclusive
        if len(ranks_str) == len(candidates) and all(
            1 <= int(rank) <= len(candidates) for rank in ranks_str
        ):
            return ranks_str
        else:
            # Check for occurences of multi-bracked items e.g. [3-5]
            # and expand them to individual ranks
            for rank in ranks_str:
                if "-" in rank:
                    start, end = map(int, rank.strip("[]").split("-"))
                    ranks_str.remove(rank)
                    ranks_str += [str(i) for i in range(start, end + 1)]

            # Add missing generation indices to the end of the list
            ranks_str += [
                str(i)
                for i in range(1, len(candidates) + 1)
                if str(i) not in ranks_str
            ]

            # remove duplicates that come after
            final = []
            [
                final.append(x)
                for x in ranks_str
                if x not in final and 1 <= int(x) <= len(candidates)
            ]

            assert len(final) == len(candidates) and all(
                1 <= int(rank) <= len(candidates) for rank in final
            )

            return final

    def llm_rank(self, conversation, candidates, critiques=None):
        """
        Rank the candidates based on the provided query and critiques.

        Parameters:
        conversation (list): The conversation so far
        candidates (list of str): The list of candidates to rank.
        critiques (list of str, optional): The list of critiques corresponding to each generation.

        Returns:
        list of str: The top_k ranked candidates.
        """

        if critiques and self.use_critiques:
            assert len(candidates) == len(
                critiques
            ), "Number of critiques must match number of candidates."

        query = conversation[-1]["content"]
        ranking_prompt = make_ranker_prompt(candidates, query, critiques)

        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who ranks multiple answers",
                }
            ]  # system
            + [
                message for message in conversation[:-1] if message["role"] != "system"
            ]  # rest of conversation without query
            + [{"role": "user", "content": ranking_prompt}]  # prompt
        )

        output = self.ranker.generate_from_messages(messages)
        ranks_str = self.extract_ranking(output, candidates)

        ranks = [int(i) for i in ranks_str]
        ranking = [candidates[i - 1] for i in ranks]
        top_k_contexts = ranking[: self.top_k]

        top_k_critiques = None
        if critiques:
            critique_ranking = [critiques[i - 1] for i in ranks]
            top_k_critiques = critique_ranking[: self.top_k]
            assert len(top_k_critiques) == len(
                top_k_contexts
            ), "Number of TOP critiques must match number of TOP candidates."

        if utils.DEBUG:
            logger.debug(f"{output=}")
            logger.debug(f"{ranks_str=}")
            logger.debug(f"{ranks=}")
            logger.debug(f"{ranking=}")
            logger.debug(f"{len(top_k_contexts)=}")

        return top_k_contexts, top_k_critiques

    def pairrm_rank(self, query, generations):

        with self.ranker_lock:
            scores = self.ranker.rank(
                [query],  # 1 query (1D)
                [candidates],  # 1 set of candidates for query (2D)
                return_scores=True,
                batch_size=self.ranker_batch_size,
                disable_tqdm=True,
            )

            # TODO: Some unneeded weird list stuff happening here.
            # Originally designed for multi query at the same time,
            # but we are just doing 1 query at a time
            ranks = [
                sorted(range(len(score)), key=lambda i: score[i], reverse=True)
                for score in scores
            ]

            if utils.DEBUG:
                logger.debug(f"{scores=}")
                logger.debug(f"{ranks=}")

            ranking = [candidates[i] for i in ranks[0]]

            top_k_contexts = ranking[: self.top_k]

            if utils.DEBUG:
                logger.debug(f"{len(top_k_contexts)=}")
                logger.debug(f"{top_k_contexts=}")

        return (top_k_contexts, None)

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
        top_k_contexts, top_k_critiques = self.rank(conversation, candidates, critiques)

        state["candidates"].extend(top_k_contexts)
        if top_k_critiques:
            state["critiques"].extend(top_k_critiques)
        
        return

    def rank(self, conversation: list, candidates, critiques: list=None):
        """
        Rank the candidates based on the provided conversation.

        Args:
            conversation (list): A list of the conversation so far
            candidates (list):  The list of candidates to generate responses from.
            critiques (list, optional): A list of critiques, one per candidates. Defaults to None.

        Returns:
            list: The top_k ranked candidates.
        """

        assert isinstance(conversation, list) and isinstance(conversation[-1], dict)
        assert conversation[-1]["role"] == "user"
        assert isinstance(candidates, list) and len(candidates) > 0
        query = conversation[-1]["content"]

        if utils.DEBUG:
            logger.debug(
                f"Ranking {len(candidates)} candidates with {self.model_name}"
            )

        if self.model_name == "llm-blender/PairRM":
            top_k_contexts, top_k_critiques = self.pairrm_rank(query, candidates)
        else:
            top_k_contexts, top_k_critiques = self.llm_rank(
                conversation, candidates, critiques
            )

        return top_k_contexts, top_k_critiques
