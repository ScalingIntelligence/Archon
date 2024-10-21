# Archon Evaluation on MT_BENCH

## Generate answers
First, you will have to generate your answers for mt_bench using `gen_answers.py` from the archon subdirectory (the parent folder of this folder). An Example of how to do that will be to run this script.
```
python3 gen_answers.py --benchmark mt_bench --config configs/<your-config-file>.json --parallel 16
```

This will save your output to an `outputs/mt_bench/` folder in the archon subdirectory. 

## Evaluate answers

### Set up mt_bench evaluation 
Since mt_bench eval (and other evaluations) use different versions of packages, I highly recommend you have a seperate environment for each evaluation. You can set mt_bench up with my conda environment.

```
conda env create -f mt_env.yml
conda activate mt_env
```

You can then install the needed mt_bench with
```
cd FastChat
pip install -e ".[model_worker,llm_judge]"
``` 
or follow `/FastChat/fastchat/llm_judge/README.md`

### Run evaluations

After you have your answers, you can run this script to evaluate against Sonnet 3.5. It automatically looks at the `outputs/mt_bench` directory.
```
python3 eval_mt_bench.py --model <output_file_name> --mode pairwise-baseline --parallel 32 --bench-name mt_bench --baseline-model archon-claude-3-5-sonnet-20240620
```

or this script to evaluate directly with a judge (no comparison). For example on Qwen1.5-72B-Chat output
```
python3 eval_mt_bench.py --model Qwen1.5-72B-Chat --parallel 32
```

## Show answers

Then you will want to show your results
For sonnet 3.5 comparisons:
```
python3 show_mt_bench_result.py --mode pairwise-baseline --baseline-model archon-claude-3-5-sonnet-20240620
```

For single judge comparisons
```
python3 show_mt_bench_result.py
```

## Complete Workflow. 
Within `archon/`, you can run 

```
python3 gen_answers.py --config configs/archon-70Bx10_1_samples_then_fuser_with_Qwen2_72B.json --benchmark mt_bench  --parallel 24

python3 mt_bench/eval_mt_bench.py --model-list archon-70Bx10_1_samples_then_fuser_with_Qwen2_72B --mode pairwise-baseline --parallel 32 --bench-name mt_bench --baseline-model archon-claude-3-5-sonnet-20240620

python3 mt_bench/show_mt_bench_result.py --mode pairwise-baseline --baseline-model archon-claude-3-5-sonnet-20240620

```