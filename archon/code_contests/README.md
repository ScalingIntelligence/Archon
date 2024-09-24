# Archon Evaluation on CODE_CONTESTS

## Generate answers
First, you will have to generate your answers for code_contests using `gen_answers.py` from the archon subdirectory (the parent folder of this folder). An Example of how to do that will be to run this script.
```
python3 gen_answers.py --benchmark code_contests --config configs/<your-config-file>.json --parallel 32
```

This will save your output to an `outputs/code_contests/model_answer` folder in the archon subdirectory. 

## Evaluate answers

For evaluation, you will run this command. This command will make a subdirectory and a file for each problem in `outputs/code_contests/model_judgement`

```
python3 code_contests/eval_code_contests.py output_file=outputs/code_contests/model_answer/WizardLM-2-8x22B.yaml
```

## Show answers

You can then see how many code_contests problems it got right by passing in the directory of the evaluations

```
python3 code_contests/show_code_results.py --eval_path outputs/code_contests/model_judgement/WizardLM-2-8x22B/
```

## Complete Workflow. 
Within `archon/`, you can run 
```
python3 gen_answers.py --config configs/WizardLM-2-8x22B.json --benchmark code_contests  --parallel 64

python3 code_contests/eval_code_contests.py output_file=outputs/code_contests/model_answer/WizardLM-2-8x22B.yaml

python3 code_contests/show_code_results.py --eval_path outputs/code_contests/model_judgement/WizardLM-2-8x22B/
```