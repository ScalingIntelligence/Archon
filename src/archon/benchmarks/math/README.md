# Archon Evaluation on MATH
MATH is a  dataset of 12,500 math problems. Each problem in MATH has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations. The dataset is broken up into a few categories: Algebra', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Number Theory', 'Prealgebra', and 'Precalculus.' The model is prompted to include the final number answer and is checked for correctness simply by a substring search for the correct answer.

## Generate answers
First, you will have to generate your answers for the [MATH](https://huggingface.co/datasets/lighteval/MATH) dataset using `gen_answers.py` from the archon subdirectory (the parent folder of this folder).

An example of how to do that will be to run this script.
```
python3 gen_answers.py --config configs/<your-config-file>.json --benchmark math --parallel 8
```

If the `name` of the model in `configs/<your-config-file>.json` is `<model_name>`, this will save your output as `outputs/math/model_answer/<model_name>.json` folder in the archon subdirectory. 

## Evaluate answers

After you have your answers, you can run this script to evaluate their correctness by running this script from the archon subdirectory:
```
python3 math/math_evaluation.py --answers-json-path <output_file_name>

```
This will directly print the accuracy, and the number of problematic queries to the terminal.


## Complete Workflow. 
Within `archon/`, you can run 

```
python3 gen_answers.py --config configs/archon-70Bx10_1_samples_then_fuser_with_Qwen2_72B.json --benchmark math  --parallel 8

python3 math/math_evaluation.py --answers-json-path outputs/math/model_answer/archon-70Bx10_1_samples_then_fuser_with_Qwen2_72B.json
```
