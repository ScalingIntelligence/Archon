# Archon Evaluation on ARENA_HARD_AUTO

## Generate answers
First, you will have to generate your answers for arena_hard_auto using `gen_answers.py` from the archon subdirectory (the parent folder of this folder). An Example of how to do that will be to run this script.
```
python3 gen_answers.py --benchmark arena_hard_auto --config configs/<your-config-file>.json --parallel 16
```

This will save your output to an `outputs/arena_hard_auto/` folder in the archon subdirectory. 

## Evaluate answers

After you have your answers, you can generate judgments using the `gen_judgement.py` file. It automatically looks at the `outputs/arena_hard_auto` directory for your model's generated answers so make sure to run gen answers before.
```
python -m arena_hard_auto.gen_judgment --baseline <path to config baseline> --model-list "<path to config>, <path to config>"
```
## Show answers

Then you will want to show your results
For sonnet 3.5 comparisons:
```
python3 show_arena_hard_auto_result.py --baseline archon-claude-3-5-sonnet-20240620 --judge-name gpt-4-turbo-2024-04-09

```
## Complete Workflow. 
Within `archon/`, you can run 

```
python3 gen_answers.py --config configs/MoA_with_added_critic_layer_before_each_fuser.json --benchmark arena_hard_auto  --parallel 24

# Added MoA_with_added_critic_layer_before_each_fuser to the config
python3 arena_hard_auto/gen_judgement.py

python3 arena_hard_auto/show_arena_hard_auto_result.py --baseline archon-claude-3-5-sonnet-20240620 --judge-name gpt-4-turbo-2024-04-09

```

TODO: Write 2 examples, one with passing in archon configs directly and one with the setting file. Current examples need to be updated to be archon configs