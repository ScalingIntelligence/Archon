"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, GenerationConfig

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

from typing import List
from together import Together

##################################################

def search_string_in_jsonl(file_path, search_string):
    if not os.path.exists(file_path):
        return False
    
    found = False
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if search_string in line:
                found = True
                break
                #print(f"Found the string in line: {line.strip()}")
    #if not found:
     #   print(f"The string '{search_string}' was not found in the file.")
    return found

def generate_candidates_with_together_api(instruction:str, 
                                          model: str, 
                                          temperature: float,
                                          previous_turns: dict = None):
    
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    system_prompt = "You are an expert chatbot, capable of instruction-following and question-answering. You are tasked with following the given instruction for the provided input."
    user_prompt = instruction

    ###################################

    if previous_turns is None:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
    else:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": previous_turns["first_instruction"]},
                    {"role": "assistant", "content": previous_turns["system_response"]},
                    {"role": "user", "content": user_prompt}]
        
    #print("Messages: ", messages)

    response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                #top_p=generation_dict['top_p'],
                #top_k=generation_dict['top_k'],
            )

    output = response.choices[0].message.content

    return output

##################################################

def generate_candidates_with_huggingface_locally(instruction:str, 
                                                 pipeline: transformers.pipeline,
                                                 generation_config: GenerationConfig,
                                                 model_path: str,
                                                 previous_turns: dict = None):
    
    system_prompt = "You are an expert chatbot, capable of instruction-following and question-answering. You are tasked with following the given instruction for the provided input."
    user_prompt = instruction
            
    if model_path in ["mistralai/Mistral-7B-Instruct-v0.2", "cognitivecomputations/dolphin-2.2.1-mistral-7b", "microsoft/Phi-3-mini-4k-instruct", 
                      "upstage/SOLAR-10.7B-Instruct-v1.0", "microsoft/Phi-3-small-8k-instruct", "mistralai/Mistral-7B-Instruct-v0.3"]:
        if previous_turns is None:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            messages = [{"role": "user", "content": previous_turns["first_instruction"]},
                        {"role": "assistant", "content": previous_turns["system_response"]},
                        {"role": "user", "content": user_prompt}]
    else:
        if previous_turns is None:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}]
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": previous_turns["first_instruction"]},
                        {"role": "assistant", "content": previous_turns["system_response"]},
                        {"role": "user", "content": user_prompt}]
            
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    prompt_length = len(prompt)

    outputs = pipeline(
        prompt,
        batch_size=generation_config.batch_size,
        generation_config=generation_config
    )

    answer = outputs[0]["generated_text"][prompt_length:]
    return answer
                                                 

##################################################

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    model_type
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                model_type=model_type
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    model_type
):
    if model_type == "local":
        model, tokenizer = load_model(
            model_path,
            revision=revision,
            device="cuda",
            num_gpus=num_gpus_per_model,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
    elif model_type == "HuggingFace":
        
        model_id = model_path
        model = model_path
    
        if model == "microsoft/Phi-3-small-8k-instruct":
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                tokenizer=AutoTokenizer.from_pretrained(model_id, trust_remote_code=True),
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                trust_remote_code=True
            )
        else:
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                #model_kwargs={"torch_dtype": torch.bfloat16} if model == "meta-llama/Meta-Llama-3-8B-Instruct" else {"torch_dtype": "auto"},
                #model_kwargs={"torch_dtype": "auto"},
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                trust_remote_code=True
            )

        pipeline.model.config.pad_token_id = pipeline.tokenizer.eos_token_id
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
        if model in ["meta-llama/Meta-Llama-3-8B-Instruct", "princeton-nlp/Llama-3-Instruct-8B-SimPO", "princeton-nlp/Llama-3-Instruct-8B-IPO", 
                     "princeton-nlp/Llama-3-Instruct-8B-RDPO", "princeton-nlp/Llama-3-Instruct-8B-DPO"]:
            pipeline.tokenizer.padding_side = 'left'

        pipeline.model.config.is_encoder_decoder = False

        ########################################

        terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        ########################################

        generation_config, unused_kwargs = GenerationConfig.from_pretrained(
            model_id, 
            return_unused_kwargs=True
        )

        generation_config.batch_size = 1
        
        generation_config.max_new_tokens = max_new_token
        generation_config.do_sample = True
        #generation_config.temperature = temperature
        generation_config.top_p = 0.9
        generation_config.num_return_sequences = 1
        generation_config.is_encoder_decoder = False
        generation_config.eos_token_id = terminators if model in ["meta-llama/Meta-Llama-3-8B-Instruct"] else pipeline.tokenizer.eos_token_id
        if model in ["meta-llama/Meta-Llama-3-8B-Instruct", "princeton-nlp/Llama-3-Instruct-8B-SimPO", "princeton-nlp/Llama-3-Instruct-8B-IPO", 
                     "princeton-nlp/Llama-3-Instruct-8B-RDPO", "princeton-nlp/Llama-3-Instruct-8B-DPO"]:
            generation_config.pretraining_tp = 1
        
        pipeline.model.config = generation_config

        ########################################

    ################################################################################

    print("Total Questions: ", len(questions))

    for question in tqdm(questions):

        question_string = f'"question_id": {question["question_id"]}'
        if not search_string_in_jsonl(answer_file, question_string):

            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.7

            choices = []
            for i in range(num_choices):
                torch.manual_seed(i)
                conv = get_conversation_template(model_id)
                turns = []
                turn_instruction = []
                for j in range(len(question["turns"])):
                    
                    if model_type == "local":

                        qs = question["turns"][j]
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                        input_ids = tokenizer([prompt]).input_ids

                        if temperature < 1e-4:
                            do_sample = False
                        else:
                            do_sample = True

                        # some models may error out when generating long outputs
                        try:
                            output_ids = model.generate(
                                torch.as_tensor(input_ids).cuda(),
                                do_sample=do_sample,
                                temperature=temperature,
                                max_new_tokens=max_new_token,
                            )
                            if model.config.is_encoder_decoder:
                                output_ids = output_ids[0]
                            else:
                                output_ids = output_ids[0][len(input_ids[0]) :]

                            # be consistent with the template's stop_token_ids
                            if conv.stop_token_ids:
                                stop_token_ids_index = [
                                    i
                                    for i, id in enumerate(output_ids)
                                    if id in conv.stop_token_ids
                                ]
                                if len(stop_token_ids_index) > 0:
                                    output_ids = output_ids[: stop_token_ids_index[0]]

                            output = tokenizer.decode(
                                output_ids,
                                spaces_between_special_tokens=False,
                            )
                            if conv.stop_str and isinstance(conv.stop_str, list):
                                stop_str_indices = sorted(
                                    [
                                        output.find(stop_str)
                                        for stop_str in conv.stop_str
                                        if output.find(stop_str) > 0
                                    ]
                                )
                                if len(stop_str_indices) > 0:
                                    output = output[: stop_str_indices[0]]
                            elif conv.stop_str and output.find(conv.stop_str) > 0:
                                output = output[: output.find(conv.stop_str)]

                            for special_token in tokenizer.special_tokens_map.values():
                                if isinstance(special_token, list):
                                    for special_tok in special_token:
                                        output = output.replace(special_tok, "")
                                else:
                                    output = output.replace(special_token, "")
                            output = output.strip()

                            if conv.name == "xgen" and output.startswith("Assistant:"):
                                output = output.replace("Assistant:", "", 1).strip()
                        except RuntimeError as e:
                            print("ERROR question ID: ", question["question_id"])
                            output = "ERROR"

                        conv.update_last_message(output)
                        turns.append(output)
                        turn_instruction.append(qs)

                    ##########################################

                    elif model_type == "TogetherAI":

                        qs = question["turns"][j]
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                        #input_ids = tokenizer([prompt]).input_ids

                        previous_turns = None
                        if j == 1:
                            #breakpoint()
                            previous_turns = {"first_instruction": conv.messages[0][1], 
                                            "system_response": conv.messages[1][1]}

                        output = generate_candidates_with_together_api(instruction=qs,
                                                                    model=model_path,
                                                                    temperature=temperature,
                                                                    previous_turns=previous_turns)
                        
                        conv.update_last_message(output)
                        turns.append(output)
                        turn_instruction.append(qs)

                    elif model_type == "HuggingFace":

                        qs = question["turns"][j]
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                        #input_ids = tokenizer([prompt]).input_ids

                        previous_turns = None
                        if j == 1:
                            #breakpoint()
                            previous_turns = {"first_instruction": conv.messages[0][1], 
                                              "system_response": conv.messages[1][1]}
                            
                        #breakpoint()

                        generation_config.temperature = temperature if temperature != 0.0 else 0.7
                        output = generate_candidates_with_huggingface_locally(instruction=qs,
                                                                              pipeline=pipeline,
                                                                              generation_config=generation_config,
                                                                              model_path=model_path,
                                                                              previous_turns=previous_turns,)
                        
                        conv.update_last_message(output)
                        turns.append(output)
                        turn_instruction.append(qs)

                    else:

                        raise ValueError(f"Unknown model type {model_type}")

                    ##########################################

                choices.append({"index": i, "turns": turns})

            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": choices,
                    "tstamp": time.time(),
                    "turn_instruction": turn_instruction,
                }
                fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--model-type", type=str, required=True, help="Local model or Together AI endpoint."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        model_type=args.model_type,
    )

    reorg_answer_file(answer_file)
