from .Generator import Generator
import threading
from .. import utils
from loguru import logger
import re
from .prompts import make_ranker_prompt


class Ranker:
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

        # TODO: Add multi-sampling

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

    def extract_ranking(self, output: str, generations: list):
        answer_str = output[0].partition("\n")[0]
        ranks_str = re.findall(r"\[(\d+)\]", answer_str)

        # Check that length of ranks_str matches length of generations
        # and that all the ranks are from 1 to len(generations), inclusive
        if len(ranks_str) == len(generations) and all(
            1 <= int(rank) <= len(generations) for rank in ranks_str
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
                for i in range(1, len(generations) + 1)
                if str(i) not in ranks_str
            ]

            # remove duplicates that come after
            final = []
            [
                final.append(x)
                for x in ranks_str
                if x not in final and 1 <= int(x) <= len(generations)
            ]

            assert len(final) == len(generations) and all(
                1 <= int(rank) <= len(generations) for rank in final
            )

            return final

    def llm_rank(self, messages, generations, critiques=None):
        """
        Rank the generations based on the provided query and critiques.

        Parameters:
        query (str): The input query.
        generations (list of str): The list of generations to rank.
        critiques (list of str, optional): The list of critiques corresponding to each generation.

        Returns:
        list of str: The top_k ranked generations.
        """

        if critiques and self.use_critiques:
            assert len(generations) == len(
                critiques
            ), "Number of critiques must match number of generations."

        query = messages[-1]["content"]
        ranking_prompt = make_ranker_prompt(generations, query, critiques)

        messages = (
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who ranks multiple answers",
                }
            ]  # system
            + [
                message for message in messages[:-1] if message["role"] != "system"
            ]  # rest of conversation without query
            + [{"role": "user", "content": ranking_prompt}]  # prompt
        )

        output = self.ranker.generate_from_messages(messages)
        ranks_str = self.extract_ranking(output, generations)

        ranks = [int(i) for i in ranks_str]
        ranking = [generations[i - 1] for i in ranks]
        top_k_contexts = ranking[: self.top_k]

        top_k_critiques = None
        if critiques:
            critique_ranking = [critiques[i - 1] for i in ranks]
            top_k_critiques = critique_ranking[: self.top_k]
            assert len(top_k_critiques) == len(
                top_k_contexts
            ), "Number of TOP critiques must match number of TOP generations."

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
                [generations],  # 1 set of generations for query (2D)
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

            ranking = [generations[i] for i in ranks[0]]

            top_k_contexts = ranking[: self.top_k]

            if utils.DEBUG:
                logger.debug(f"{len(top_k_contexts)=}")
                logger.debug(f"{top_k_contexts=}")

        return (top_k_contexts, None)

    def rank(self, messages, generations, critiques=None):
        """
        Rank the generations based on the provided query.

        Parameters:
        query (str): The input query.
        generations (list of str): The list of generations to rank.

        Returns:
        list of str: The top_k ranked generations.
        """

        assert isinstance(messages, list) and isinstance(messages[-1], dict)
        assert messages[-1]["role"] == "user"
        assert isinstance(generations, list) and len(generations) > 0
        query = messages[-1]["content"]

        if utils.DEBUG:
            logger.debug(
                f"Ranking {len(generations)} generations with {self.model_name}"
            )

        if self.model_name == "llm-blender/PairRM":
            top_k_contexts, top_k_critiques = self.pairrm_rank(query, generations)
        else:
            top_k_contexts, top_k_critiques = self.llm_rank(
                messages, generations, critiques
            )

        return top_k_contexts, top_k_critiques
