# Archon Evaluation on MMLU

## Generate answers
First, you will have to generate your answers for the [Massive Multitask Language Understanding (MMLU)](https://huggingface.co/datasets/cais/mmlu) dataset using `gen_answers.py` from the archon subdirectory (the parent folder of this folder).

An example of how to do that will be to run this script.
```
python3 gen_answers.py --config configs/<your-config-file>.json --benchmark mmlu --parallel 8
```

If the `name` of the model in `configs/<your-config-file>.json` is `<model_name>`, this will save your output as `outputs/mmlu/model_answer/<model_name>.json` folder in the archon subdirectory. 

## Evaluate answers

After you have your answers, you can run this script to evaluate their correctness by running this script from the archon subdirectory:
```
python3 mmlu/mmlu_evaluation.py --answers-json-path <output_file_name>

```
This will directly print the accuracy, and the number of problematic queries to the terminal.


## Complete Workflow. 
Within `archon/`, you can run 

```
python3 gen_answers.py --config configs/archon-70Bx10_1_samples_then_fuser_with_Qwen2_72B.json --benchmark mmlu --parallel 8

python3 mmlu/mmlu_evaluation.py --answers-json-path outputs/gsm8k/model_answer/archon-70Bx10_1_samples_then_fuser_with_Qwen2_72B.json

```