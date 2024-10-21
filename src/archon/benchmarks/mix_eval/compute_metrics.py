'''
Usage:
python -m mix_eval.compute_metrics \
    --benchmark {mix_eval, mix_eval_hard} \
    --models-to-eval MODELS_TO_EVAL [MODELS-TO-EVAL ...] \
    [--model-response-dir RESULTS-DIR] \
    [--multichoice-judge MC-JUDGE-MODEL] \
    [--freeform-judge FF-JUDGE-MODEL] \
    [--api-parallel-num API-PARALLEL-NUM] \ 

Example:
python -m mix_eval.compute_metrics  \
    --benchmark mix_eval \
    --model-response_dir outputs \
    --api-parallel-num 20 \
    --multichoice-judge archon-gpt-3.5-turbo \
    --freeform-judge archon-gpt-3.5-turbo \
    --models-to-eval Qwen1.5-72B-Chat 

'''
import json
import argparse
import os
import time
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
from prettytable import PrettyTable
from .mix_eval_utils.common_utils import set_seed
from .mix_eval_utils.metric_utils import (
    parse_multi_choice_response_model,
    eval_multi_choice,
    eval_freeform_model
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", 
        type=str, 
        choices=["mix_eval", "mix_eval_hard"], 
        required=True,
        help="Benchmark to evaluate."
        )
    parser.add_argument(
        "--model-response-dir", 
        type=str, 
        default="mix_eval/data/model_responses/", 
        help="Path to model responses."
        )
    parser.add_argument(
        "--multichoice-judge",
        type=str, 
        default="archon-gpt-3.5-turbo", 
        help="Judge Archon model for multiple-choice score computation."
        )
    parser.add_argument(
        "--freeform-judge",
        type=str, 
        default="archon-gpt-3.5-turbo", 
        help="Judge Archon model for freeform score computation."
        )
    parser.add_argument(
        "--models-to-eval", 
        nargs='+',
        required=True,
        help="Please enter models you want to evaluate."
        )
    parser.add_argument(
        "--api-parallel-num", 
        type=int, 
        default=100, 
        help="Number of parallel threads for calling the model parser api if use model parsing." 
        "If you hit rate limit error frequently, try to reduce this number."
        )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Print verbose information."
        )
    
    args = parser.parse_args()

    args.model_answer_dir = os.path.join(args.model_response_dir, args.benchmark, "model_answer")
    args.model_judgement_dir = os.path.join(args.model_response_dir, args.benchmark, "model_judgement")
    return args

def compute_metric_closeended_freeform_modelparse(args):
    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        
    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
             
    for model in models:
        print(f"\n\n\nParsing model: {model} for {args.benchmark} \n\n\n")
        if args.benchmark == "mix_eval":
            split = "free_form"
        elif args.benchmark == "mix_eval_hard":
            split = "free_form_hard"
        else:
            raise ValueError(f"Invalid benchmark: {args.benchmark}.")
        
        ans_file = os.path.join(
            args.model_answer_dir, 
            model,
            f"{model}_{split}.jsonl"
            )
        tasks = []
        with open(ans_file, "r") as f:
            for line in f:
                ans_dict = json.loads(line)
                tasks.append(ans_dict)
        results = eval_freeform_model(args, tasks)
        
        score_dict_model = {}
        for judge_dict in results:
            judge_score = judge_dict["judge_score"]
            if 'overall' not in score_dict_model:
                score_dict_model['overall'] = []
            score_dict_model['overall'].append(judge_score)
            if judge_dict['benchmark_name'] not in score_dict_model:
                score_dict_model[judge_dict['benchmark_name']] = []
            score_dict_model[judge_dict['benchmark_name']].append(judge_score)
            
        for key, value in score_dict_model.items():
            score_dict_model[key] = round(sum(value)/len(value), 3)
        score_dict[model] = score_dict_model

        # make model_judgment dir if it does not exist yet
        os.makedirs(os.path.join(args.model_judgement_dir, model), exist_ok=True)        
        with open(os.path.join(
                args.model_judgement_dir, 
                model, 
                f"{args.freeform_judge}-judge-free-form.jsonl",
                ), "w") as f:
            for case in results:
                f.write(json.dumps(case) + "\n")
        
        print("Sleep 60 seconds to avoid ratelimit error ... ")
        time.sleep(60)
    
    if args.verbose:
        print(f"[Close-ended Free-form Model Parser]")
        for model, score in score_dict.items():
            print(f"{model}: {json.dumps(score, indent=4)}")
        
    return score_dict
        
def compute_metric_closeended_freeform(args):
    return compute_metric_closeended_freeform_modelparse(args)

def compute_metric_closeended_multichoice_modelparse(args):
    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
               
    for model in models:
        print(f"\n\n\nParsing model: {model} for {args.benchmark} \n\n\n")
        if args.benchmark == "mix_eval":
            split = "mult_choice"
        elif args.benchmark == "mix_eval_hard":
            split = "mult_choice_hard"
        else:
            raise ValueError(f"Invalid benchmark: {args.benchmark}.")
        ans_file = os.path.join(args.model_answer_dir, model, f"{model}_{split}.jsonl")

        with open(ans_file, "r") as f:
            ans_dicts = []
            for line in f:
                ans_dict = json.loads(line)
                ans_dicts.append(ans_dict)
                
            ans_dicts_withscore = parse_multi_choice_response_model(args, ans_dicts)
            
            total = 0
            correct = 0
            results = []
            error_cases = []
            for ans_dict_ws in ans_dicts_withscore:
                options = ans_dict_ws["options"]
                target = ans_dict_ws["target"]
                assert isinstance(target, list) and len(target) == 1, \
                    f"Invalid target: {target}"
                all_choices = [chr(ord("A") + i) for i in range(len(options))]
                model_choice = ans_dict_ws['judge_option']
                target_id = all_choices[int(target[0])]
                if eval_multi_choice(target_id, model_choice):
                    correct += 1
                else:
                    error_cases.append(ans_dict_ws)
                results.append(ans_dict_ws)
                total += 1
        
        score_dict_model = {}
        for judge_dict in results:
            options = judge_dict["options"]
            target = judge_dict["target"]
            assert isinstance(target, list) and len(target) == 1, \
                f"Invalid target: {target}"
            all_choices = [chr(ord("A") + i) for i in range(len(options))]
            model_choice = judge_dict['judge_option']
            target_id = all_choices[int(target[0])]
            judge_score = 1 if eval_multi_choice(target_id, model_choice) else 0
            
            # add score
            if 'overall' not in score_dict_model:
                score_dict_model['overall'] = []
            score_dict_model['overall'].append(judge_score)
            if judge_dict['benchmark_name'] not in score_dict_model:
                score_dict_model[judge_dict['benchmark_name']] = []
            score_dict_model[judge_dict['benchmark_name']].append(judge_score)
            
        for key, value in score_dict_model.items():
            score_dict_model[key] = round(sum(value)/len(value), 3)
        score_dict[model] = score_dict_model

        # make model_judgment dir if it does not exist yet
        os.makedirs(os.path.join(args.model_judgement_dir, model), exist_ok=True)    
        with open(os.path.join(
                args.model_judgement_dir, 
                model, 
                f"{args.multichoice_judge}-judge-multiple-choice.jsonl"
                ), "w") as f:
            for case in results:
                f.write(json.dumps(case) + "\n")
                
        print("Sleep 60 seconds to avoid ratelimit error ... ")
        time.sleep(60)
    
    if args.verbose:
        print(f"[Close-ended Multiple-choice Model Parser]")
        for model, score in score_dict.items():
            print(f"{model}: {json.dumps(score, indent=4)}")
        
    return score_dict

def compute_metric_closeended_multichoice(args):
    return compute_metric_closeended_multichoice_modelparse(args)

def print_table(data_dict):
    # Create a table
    table = PrettyTable()
    
    # Set the column names
    table.field_names = ["Split", "Score"]
    
    # Add rows from the dictionary
    for key, value in data_dict.items():
        table.add_row([key, value])
    
    # Print the table
    print(table) 
                
def compute_metric(args):
    score_dict_ff = compute_metric_closeended_freeform(args)
    score_dict_mp = compute_metric_closeended_multichoice(args)
    
    models_ff = set(score_dict_ff.keys())
    models_mp = set(score_dict_mp.keys())
    common_models = models_ff.intersection(models_mp)
    missing_models = models_ff.union(models_mp) - common_models
    if missing_models:
        print(f"Something went wrong when computing the free-form or multiple-choice "
              f"split of these models: \n{missing_models}\n\nA possible reason may be that they lack a model answer file. "
              "Skipping them...")
    
    score_dict = {}
    for model in common_models:
        score_dir = os.path.join(
            args.model_judgement_dir, 
            model
            )
        score_dict_model = {
            "overall score (final score)": (score_dict_ff[model]['overall'] + score_dict_mp[model]['overall']) / 2,
            **{k:v for k, v in score_dict_ff[model].items() if k != "overall"},
            **{k:v for k, v in score_dict_mp[model].items() if k != "overall"},
            }
        score_dict[model] = score_dict_model
        with open(os.path.join(score_dir, "score.json"), "w") as f:
            f.write(json.dumps(score_dict_model, indent=4) + "\n")
        print_table(score_dict_model)
    
    print(f"Saving the model scores to {os.path.join(args.model_judgement_dir, 'score.json')} ...")
    with open(os.path.join(args.model_judgement_dir, "score.json"), "w") as f:
        f.write(json.dumps(score_dict, indent=4) + "\n")
    
if __name__ == '__main__':
    set_seed()
    args = parse_args()
    compute_metric(args)