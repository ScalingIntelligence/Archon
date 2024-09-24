# Prompts for MixEval Benchmark
MULTI_CHOICE_PROMPT = "Answer with the option letter from the given choices directly."
FREE_FORM_PROMPT = "Answer the question shortly."
FREE_FORM_PROMPT_BBH = "Answer the question. \nLet's think step by step."
FREE_FORM_PROMPT_GSM8k = "Answer the question. \nLet's think step by step."
FREE_FORM_PROMPT_MATH = "Answer the question. \nLet's think step by step."

# Functions for generating prompts for MixEval Benchmark 
def parse_mix_eval_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_mix_eval_prompt_multichoice(entry):
    prompt = entry["prompt"]
    parsed_options = parse_mix_eval_options(entry["options"])
    if 'context' in entry and str(entry['context']).lower() != "none" and str(entry['context']).lower() != "null" and str(entry['context']).replace(" ", "") != "":
        context = entry['context']
        prompt = f"{context}\n{prompt}\n{parsed_options}\n{MULTI_CHOICE_PROMPT}"
    else:
        prompt = f"{prompt}\n{parsed_options}\n{MULTI_CHOICE_PROMPT}"
    return prompt

def construct_mix_eval_prompt_freeform(entry):
    prompt = entry["prompt"]
    if entry["benchmark_name"] == "BBH":
        prompt = f"{prompt}\n{FREE_FORM_PROMPT_BBH}"
    elif entry["benchmark_name"] == "GSM8k":
        prompt = f"{prompt}\n{FREE_FORM_PROMPT_GSM8k}"
    elif entry["benchmark_name"] == "MATH":
        prompt = f"{prompt}\n{FREE_FORM_PROMPT_MATH}"
    else:
        prompt = f"{prompt}\n{FREE_FORM_PROMPT}"
    
    if 'context' in entry and str(entry['context']).lower() != "none" and str(entry['context']).lower() != "null" and str(entry['context']).replace(" ", "") != "":
        context = entry['context']
        prompt = f"Question: {context}\n{prompt}"
    else: 
        prompt = f"Question: {prompt}"
    return prompt