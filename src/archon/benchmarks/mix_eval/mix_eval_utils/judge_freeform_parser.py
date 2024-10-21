from tqdm import tqdm
import time
import random
import os
import json
from dotenv import load_dotenv
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, AzureOpenAI
from openai._exceptions import RateLimitError, BadRequestError
from httpx import Timeout
from ....completions import Archon
import pdb
from ..prompts.judge_prompts import archon_judge_for_closeended_freeform
from .common_utils import extract_basemodel_response_3e

########################-Archon-########################
class ArchonJudgeCloseendFreeform: 
    def __init__(self, args):
        self.args = args
        self.JUDGE = args.freeform_judge
        self.FIX_INTERVAL_SECOND = 0
        self.MAX_RETRY_NUM = 99
        self.MAX_NEW_TOKENS = 999
        self.judge_model = self._load_judge_model(self.JUDGE)

    def format_prompts(self, inputs):
        prompt, gold_ans, response = inputs
        gold_ans = '; '.join([f"<answer {i+1}> {ans}" for i, ans in enumerate(gold_ans)])
        formated = archon_judge_for_closeended_freeform(prompt, gold_ans, response)
        return formated

    def _load_judge_model(self, model):
        # Load the path to the archon config for the judge
        archon_config = None
        try:
            with open(f'configs/{model}.json', "r") as model_file:
                archon_config = json.load(model_file)
        except FileNotFoundError:
            logger.error(f'Judge file not found: {model}')
        except json.JSONDecodeError:
            logger.error(f'Invalid JSON in Judge file: {model}')
        except Exception as e:
            logger.error(f'Could not load valid (judge) Archon config at {model}')
        
        judge_model = Archon(archon_config)
        return judge_model

            
    def chat_completion_archon(self, messages, temperature=None):
        output = self.judge_model.generate(messages, temperature=temperature)
        return output
        
    def _archon_decode(self, inputs):
        completion = self.chat_completion_archon(messages=self.format_prompts(inputs))
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion


    def archon_decode(self, inputs):
        delay = 1
        blocked = 0
        for i in range(self.MAX_RETRY_NUM):
            try:
                completion = self._archon_decode(inputs)
                return completion
            except RateLimitError as e:
                exponential_base = 2
                delay *= exponential_base * (1 + random.random())
                print(f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                print(e)
                time.sleep(delay)
                continue
            except BadRequestError as e:
                blocked += 1
                if blocked >= 10:
                    print("Blocked too many times, skipping...")
                    return 'Blocked'
                print(f"Input is blocked, retrying...")
                print(e)
                time.sleep(1)
                continue
            except Exception as e:
                print(f"Error in archon_decode, retrying...")
                print(e)
                time.sleep(1)
                continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return 'Error'


    def annotate_p(self, task):    
        prompt = task['prompt']
        gold_ans = task['target']
        response = task['output']
        
        if hasattr(self.args, 'model_type'):
            if self.args.model_type == 'BaseModel':
                response = extract_basemodel_response_3e(response)
                task['response_extracted'] = response
            elif self.args.model_type == 'ChatModel':
                pass
            elif self.args.model_type == 'APIModelBase':
                pass
            else:
                raise ValueError(f"Model type {self.args.model_type} not supported.")
        
        if not isinstance(gold_ans, list):
            print(f"Invalid target: {gold_ans}")
            return None
        
        inputs = (prompt, gold_ans, response)
        
        completion = self.archon_decode(inputs)
        if completion == 'Error':
            print(f"Error in archon_decode, the entry {task} will be retried later...")
            task['judge_response'] = None
            return task
        elif completion == 'Blocked':
            print(f"{task}: \n\nBlocked, the entry treated as bad entry.")
            task['judge_response'] = '[[0.0]]'
            return task
        annotation = completion
        task['judge_response'] = annotation
        return task


    def annotate_parallel(self, tasks):
        print(f"Parsing in parallel, in total {self.args.api_parallel_num} threads.")
        results = []
        with ThreadPoolExecutor(self.args.api_parallel_num) as executor:
            for entry in tqdm(
                executor.map(self.annotate_p, tasks), total=len(tasks)
            ):
                results.append(entry)
        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")
        return results