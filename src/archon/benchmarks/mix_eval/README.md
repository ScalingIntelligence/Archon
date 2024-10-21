# Archon Evaluation on MixEval and MixEval-Hard

The [MixEval](https://github.com/Psycoy/MixEval) and MixEval-Hard benchmark are updated periodically every few months. This code uses the version populated in the HF dataset [here](https://huggingface.co/datasets/MixEval/MixEval). 

The [MixEval](https://github.com/Psycoy/MixEval) and [MixEval-Hard](https://github.com/Psycoy/MixEval) benchmark are updated periodically every few months. This code uses the version populated in the HF dataset [here](https://huggingface.co/datasets/MixEval/MixEval). 

## Generate answers

You will have to generate your answers for MixEval or MixEval-Hard using `gen_answers.py` from the archon subdirectory (the parent folder of this folder). An Example of how to do that will be to run this script.

```
python3 gen_answers.py --benchmark mix_eval --config configs/<your-config-file>.json --parallel 32
```

This will save your output to either the `outputs/mix_eval/model_answer/<your-config-file>` or `outputs/mix_eval_hard/model_answer/<your-config-file>` folder in the archon subdirectory. Within this directory, there will be two `.jsonl` files. 

For the free-form questions, the model generations can be found in `<your-config-file>_free_form.jsonl` (or `<your-config-file>_free_form.jsonl` for MixEval-Hard).

For the multiple-choice questions, the model generations can be found in `<your-config-file>_mult_choice_hard.jsonl` (or `<your-config-file>_mult_choice_hard.jsonl` for MixEval-Hard).

## Evaluate answers
There is **no** additional set up needed for this evaluation apart from the initial archon set up.

After you have your answers, you can run the following command.

`python -m mix_eval.compute_metrics  --benchmark mix_eval --model-response_dir outputs --multichoice-judge archon-gpt-3.5-turbo --freeform-judge archon-gpt-3.5-turbo --api-parallel-num 20 --models-to-eval Qwen1.5-72B-Chat`

or (for MixEval-Hard)

`python -m mix_eval.compute_metrics  --benchmark mix_eval_hard --model-response-dir outputs --multichoice-judge archon-gpt-3.5-turbo --freeform-judge archon-gpt-3.5-turbo --api-parallel-num 20 --models-to-eval Qwen1.5-72B-Chat`

You can pass in as arguments:
- benchmark: What benchmark (MixEval or MixEval-Hard) you want to run
- multichoice-judge: What Archon model you want use as a judge for free-form evaluation
- freeform-judge: What Archon model you want use as a judge for free-form evaluation
- api-parallel-num: Number of parallel API calls 
- models-to-eval: Archon model that you want to evaluate. This is the same as `<your-config-file>`

The results for this model can be found here: `outputs/mix_eval/model_judgement/<your-config-file>/score.json`

Here's an example of these results. You can also see a breakdown of how the Archon model performs on each sub-dataset:

```
{
    "overall score (final score)": 0.8414999999999999,
    "TriviaQA": 0.844,
    "DROP": 0.834,
    "GSM8k": 0.961,
    "MATH": 0.65,
    "BBH": 0.901,
    "AGIEval": 0.683,
    "MMLU": 0.801,
    "PIQA": 0.938,
    "BoolQ": 0.851,
    "HellaSwag": 0.886,
    "ARC": 0.967,
    "CommonsenseQA": 0.868,
    "SIQA": 0.806,
    "OpenBookQA": 0.909,
    "GPQA": 0.25,
    "WinoGrande": 0.667,
    "MBPP": 0.0
}
```


## Complete Workflow. 
Within `archon/`, if you wanted to evaluate Qwen1.5-72B-Chat the Archon config stored in `configs/Qwen1.5-72B-Chat.json`, you would run the following:


```
python3 gen_answers.py --benchmark mix_eval --parallel 32 --config configs/Qwen1.5-72B-Chat.json

python -m mix_eval.compute_metrics  --benchmark mix_eval --model-response-dir outputs --multichoice-judge archon-gpt-3.5-turbo --freeform-judge archon-gpt-3.5-turbo --api-parallel-num 20 --models-to-eval Qwen1.5-72B-Chat

```

The results for this model can be found here: `outputs/mix_eval/model_judgement/Qwen1.5-72B-Chat/score.json`

