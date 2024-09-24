# Archon Evaluation on MixEval and MixEval-Hard

The [MixEval](https://github.com/Psycoy/MixEval) and MixEval-Hard benchmark are updated periodically every few months. This code uses the version populated in the HF dataset [here](https://huggingface.co/datasets/MixEval/MixEval). 

The [MixEval](https://github.com/Psycoy/MixEval) and [MixEval-Hard](https://github.com/Psycoy/MixEval) benchmark are updated periodically every few months. This code uses the version populated in the HF dataset [here](https://huggingface.co/datasets/MixEval/MixEval). 

## An Overview of MixEval and MixEval-Hard

**MixEval** is an approach that bridges the gap between real-world user queries and efficient, reproducible evaluation by leveraging user queries mined from the web and matching them with similar queries from existing benchmarks. MixEval is also the proposed benchmark built with this approach.

**MixEval-Hard** is the hard version of MixEval, designed to enhance the benchmark's ability to distinguish strong models. It is sampled from MixEval based on model evaluation results, with a higher probability of selecting harder queries. To address distribution deviation, MixEval-Hard introduces a rejective sampling process to ensure that the distribution of MixEval-Hard aligns with that of wild queries.

Dynamic evaluation is introduced to mitigate the contamination issue. The data points in MixEval and MixEval-Hard are periodically updated using a fast, stable pipeline, which performs benchmark mixture with a different batch of wild queries from the same distribution, showing low variance (0.36 Std. on a 0-100 scale) and significant version difference (85% unique query ratio).

**MixEval offers five significant advantages for practitioners:**

- Accurate model ranking, demonstrated by a 0.96 correlation with Chatbot Arena1.
- Fast, cheap and reproducible execution, requiring only 6% the time and cost of MMLU and with no dependence on human input.
- Dynamic benchmarking enabled by low-effort and stable updating mechanism.
- A comprehensive and less biased query distribution, as it bases queries on a large-scale web corpus.
- A fair grading process, ensured by the ground-truth-based grading mechanism.

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

