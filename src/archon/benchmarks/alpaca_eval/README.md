# Archon Evaluation on AlpacaEval
[Original Github](https://github.com/tatsu-lab/alpaca_eval)

AlpacaEval is an evaluation framework for language models that focuses on cheaply and quickly evaluating model responses. The original implementation compares your model's responses against ```gpt text-davinci-003``` or ```gpt-4-turbo``` for AlpacaEval 2.0 and outputs a Win Rate.

## Generate answers
First, you will have to generate your answers for alapca_eval using `gen_answers.py` from the archon subdirectory (the parent folder of this folder).
```
python3 gen_answers.py --benchmark alpaca_eval --config configs/<your-config-file>.json --parallel 16
```

This will save your output to an `outputs/model_aswer/alapca_eval/` folder in the archon subdirectory. 

## Evaluate answers

After you have your answers, you can run this command to evaluate, input your model output as the argument. FOr example

```
alpaca_eval --model_outputs outputs/model_answer/alpaca_eval/archon-claude-3-5-sonnet-sample_10_then_critic_then_fuse.json
```

## Show answers

The results will either be printed out by the alpaca_eval command ran above (recommended) or can be found in the leaderboard that will be generated after you run the above command.

## Complete Workflow. 
Within `archon/`, you can run 

```
python3 gen_answers.py --config configs/WizardLM-2-8x22B.json --benchmark alpaca_eval  --parallel 64

alpaca_eval --model_outputs outputs/alpaca_eval/WizardLM-2-8x22B.json
```
