from .archon import Archon
from loguru import logger
from functools import partial
import argparse
import concurrent.futures
from tqdm import tqdm
from .utils import load_config
from . import utils
import os
from ..benchmarks.benchmarks import BENCHMARK_CLASSES, load_benchmark
import time


def initialize_debug(args):
    if args.debug:
        utils.DEBUG = 1
        logger.debug("In DEBUG mode")
    if args.debug_verifier:
        utils.DEBUG_VERIFIER = 1
        logger.debug("In DEBUG VERIFIER mode")
    if args.debug_archon:
        utils.DEBUG_ARCHON = 1
        logger.debug("In DEBUG ARCHON mode")
    if args.debug_unit_test_generator:
        utils.DEBUG_UNIT_TEST_GENERATOR = 1
        logger.debug("In DEBUG UNIT TEST GENERATOR mode")


def main(args):
    logger.info(f"Start.")

    # initialize debug modes
    initialize_debug(args)

    # Load Archon configuration
    logger.info("loading: " + args.config)
    archon_config = load_config(args.config)

    if "name" not in archon_config:
        name = "archon-" + time.strftime("%m%d%Y-%H:%M:%S")
        logger.warning(
            "Make sure to add 'name' to your config. Will continue run with {name}"
        )
        archon_config["name"] = name

    if utils.DEBUG:
        logger.debug(f"{archon_config=}")

    api_key_data = None
    if hasattr(args, 'api_keys'):
        api_key_data = args.api_keys
    # initialize Archon
    archon = Archon(config=archon_config, api_key_data=api_key_data)
    logger.info("Finished initializing archon")

    benchmark = load_benchmark(
        benchmark_name=args.benchmark,
        dataset_sample=args.dataset_sample,
        debug_data=args.debug_data,
    )
    eval_set = benchmark.load_dataset()

    results = []
    # run Archon on eval questions in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        results = list(
            tqdm(
                executor.map(
                    partial(
                        benchmark.get_answer,
                        model=archon,
                        config=archon_config,
                        samples=args.samples,
                    ),
                    eval_set,
                ),
                total=len(eval_set),
            )
        )

    # update benchmark data
    eval_set = benchmark.process_results(results)

    ########### Save Output #########
    output_dir = f"{args.output_dir.rstrip('/')}/{args.benchmark}/model_answer"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test = "" if not args.debug_data else "TEST"
    samples = "" if not args.samples or args.samples == 1 else f"@{str(args.samples)}"

    timestamp = time.strftime("%I%M%S%p%m%d%Y")
    output_path = (
        f"{output_dir}/{archon_config['name']}{test}{samples}.{benchmark.save_type}"
    )

    # If output path exists and we do not want to append, delete it to keep newest answers
    if os.path.isfile(output_path) and not args.append:
        print(f"File already exists, deleting it: {output_path}")
        os.remove(output_path)
        assert not os.path.isfile(output_path)

    # add timestamp if it already exits and we do not want to append
    if os.path.isfile(output_path) and not args.append:
        output_path = f"{output_dir}/{archon_config['name']}{test}-{timestamp}.{benchmark.save_type}"

    logger.info(f"Saving outputs to {output_path}.")

    benchmark.save_answers(output_path, eval_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # arguments for gen_answers.py

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=BENCHMARK_CLASSES.keys(),
        required=True,
        help="The benchmark to use for evaluation",
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Archon config to gen answers from"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="output directory to save answers",
    )

    parser.add_argument(
        "--parallel", type=int, default=16, help="The number of concurrent API calls."
    )

    parser.add_argument(
        "--samples", type=int, default=1, help="Number of samples from archon."
    )

    parser.add_argument("--append", action="store_true", help="Append answers to file")

    parser.add_argument("--debug-data", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-verifier", action="store_true")
    parser.add_argument("--debug-archon", action="store_true")
    parser.add_argument("--debug-unit-test-generator", action="store_true")

    parser.add_argument(
        "--dataset-sample",
        type=float,
        help="Sample dataset to use for evaluation",
        default=1.0,
    )

    parser.add_argument("--api-keys", type=str, default=None)

    args = parser.parse_args()

    main(args)
