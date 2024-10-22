# generator has no custom prompt. Just the conversation


def make_critic_prompt(query, candidates):
    num = len(candidates)
    prompt = f"I will provide you with {num} responses, each indicated by a numerical identifier []. Evaluate the strengths and weaknesses of each response based on the instruction: {query}.\n"

    for j in range(len(candidates)):
        prompt += f"\n[{j+1}] {candidates[j]}"

    prompt += f"\n\nInstruction: {query}.\n\nEvaluate the {num} responses above based on their relevance to the instruction. "
    prompt += f"All the responses should be included and evaluated using identifiers. "
    # user_prompt += f"The output format should be in the form of strengths and weaknesses for each response. "
    prompt += f"For each response, start the critique with the numerical identifier (e.g. [1]) followed by the strengths and weaknesses. "
    prompt += f"You must include both strengths and weaknesses, even if there are more of one than the other. "
    # user_prompt += f"Only separate the strengths and weaknesses with a single new line. "
    prompt += f"At the end of each response's analysis, include two new lines to separate the critiques. "
    prompt += f"Do not include any preface or text after the critiques. Do not include any references to previous critiques within a critique. Start with the analysis for the first response and end with the analysis for the last response. "
    prompt += f"All of the {num} responses should be included and evaluated using identifiers. "
    prompt += f"Structure each response's analysis as follows: [1]\nStrengths:\n- <strength #1>\n- <strength #2>\n- <strength #n> \nWeaknesses:\n- <weakness #1>\n- <weakness #2>\n- <weakness #n>\n\n"
    return prompt


def make_fuser_prompt(conv, references, critiques=None, length_control=False):

    query = conv[-1]["content"]

    if critiques:

        prompt = f"You have been provided with a set of responses with their individual critiques of strengths/weaknesses from various open-source models to the latest user query, which is {query}. Your task is to \
            synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses and their provided critiques of \
            strengths/weaknesses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, \
            and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n"
        prompt += f"Once again, the query is: {query}\n"
        if length_control:
            prompt += "The fused response can only be as long as the longest response in the current candidate pool.\n"

        prompt += "Responses from models:\n\n"

        count = 0
        assert len(references) == len(critiques)
        for reference, critique in zip(references, critiques):
            prompt += f"{count+1}. {reference} \n\nCritique:\n{critique}"
            count += 1
            if count != len(references):
                prompt += "\n\n"

        return prompt

    else:

        prompt = f"You have been provided with a set of responses from various open-source models to the latest user query, which is {query}.\
            Your task is to synthesize these responses into a single, high-quality response. \
            It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. \
            Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. \
            Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n"
        prompt += f"Once again, the query is: {query}\n"

        prompt += "Responses from models:"

        for i, reference in enumerate(references):

            prompt += f"\n{i+1}. {reference}"

        return prompt


def make_ranker_prompt(generations, query, critiques=None):
    num = len(generations)

    # No longer uses self.use_critiques
    if critiques:
        prompt = f"I will provide you with {num} responses, each indicated by a numerical identifier []. Rank the responses based on their relevance to the instruction and their provided critique of strengths/weaknesses: {query}.\n"
    else:
        prompt = f"I will provide you with {num} responses, each indicated by a numerical identifier []. Rank the responses based on their relevance to the instruction: {query}.\n"

    for j in range(len(generations)):
        prompt += f"\n[{j+1}] {generations[j]}"
        if critiques:
            prompt += f"\n\nCritique:\n{critiques[j]}"

    if critiques:
        prompt += f"\n\nInstruction: {query}.\n\nRank the {num} responses above based on their relevance to the instruction and their provided critique of strengths/weaknesses. "
        prompt += f"All the responses should be included and listed using identifiers, in descending order of relevance to the instruction, using the provided critiques of strengths/weaknesses to assist in the ranking. "
    else:
        prompt += f"\n\nInstruction: {query}.\n\nRank the {num} responses above based on their relevance to the instruction. "
        prompt += f"All the responses should be included and listed using identifiers, in descending order of relevance to the instruction. "

    prompt += f"The output format should be [] > [], e.g., [4] > [2]."
    prompt += f"Please explain how you got to your final response."
    prompt += f"Your ranking should start with Answer: and be on the first line "

    return prompt


def make_verifier_reasoning_prompt(query, candidate):
    prompt = f"I will provide you with a response indicated by the identifier 'Response'. Provide reasoning for why the response accurately and completely addresses the instruction: {query}.\n"
    prompt += f"\nResponse: {candidate}"
    prompt += f"\n\nInstruction: {query}.\n\nProvide the reasoning for the response above based on its relevance, completeness, and accuracy when compared to the instruction. "
    prompt += f"Do not include any preface or text after the reasoning."

    return prompt


def make_verifier_verdict_prompt(query, candidate, reasoning):
    prompt = [
        f"Given the following query, response, and reasoning, evaluate whether or not the response is correct.\n"
        f"- In your evaluation, you should consider how the response aligns with the reasoning and query.\n"
        f"- You should also consider whether or not the logic in the reasoning is correct and complete.\n"
        f"- Provide an explanation for your verdict before you return your evaluation. At the end of your explanation, you should finish with your verdict of either '[Correct]' or '[Incorrect]'.\n"
        f"- You must include a verdict with one of these formatted options: '[Correct]' or '[Incorrect]'.\n\n"
        f"Query: {query}\n"
        f"Response: {candidate}\n"
        f"Reasoning: {reasoning}\n"
    ]

    prompt = "".join(prompt)
    return prompt


def make_unit_test_generator_prompt(query, unit_test_cap=None):
    if unit_test_cap is not None and unit_test_cap >= 1:
        prompt = [
            f"Given the following query, generate a set of {unit_test_cap} unit tests that would evaluate the correctness of responses to this query.\n"
        ]
    else:
        prompt = [
            f"Given the following query, generate a set of unit tests that would evaluate the correctness of responses to this query.\n"
        ]

    prompt.extend(
        [
            # f"Given the following query, generate a set of unit tests that would evaluate the correctness of responses to this query.\n",
            f"- The unit tests should cover various aspects of the query and ensure comprehensive evaluation.\n",
            f"- Each unit test should be clearly stated and should include the expected outcome.\n",
            f"- The unit tests should be in the form of assertions that can be used to validate the correctness of responses to the query.\n",
            f"- The unit test should be formatted like 'The answer mentions...', 'The answer states...', 'The answer uses...', etc. followed by the expected outcome.\n",
            f"- Solely provide the unit tests for the question below. Do not provide any text before or after the list. Only output the unit tests as a list of strings (e.g. ['unit test #1', 'unit test #2', 'unit test #3']).\n\n",
            f"Query: {query}\n",
        ]
    )
    prompt = "".join(prompt)
    return prompt


def make_unit_test_evaluator_prompt(query, response, unit_tests):
    prompt = "Given the following query, candidate response, and unit tests, evaluate whether or not the response passes each unit test.\n"
    prompt += "- In your evaluation, you should consider how the response aligns with the unit tests, retrieved documents, and query.\n"
    prompt += "- Provide reasoning before you return your evaluation.\n"
    prompt += "- At the end of your evaluation, you must finish with a list of verdicts corresponding to each unit test.\n"
    prompt += "- You must include a verdict with one of these formatted options: '[Passed]' or '[Failed]'.\n"
    prompt += "- Here is an example of the output format:\n"
    prompt += "Unit Test #1: [Passed]\n"
    prompt += "Unit Test #2: [Failed]\n"
    prompt += "Unit Test #3: [Passed]\n"
    prompt += "- Each verdict should be on a new line and correspond to the unit test in the same position.\n"

    prompt += "- Here is the query, response, and unit tests for your evaluation:\n\n"

    ##############################

    prompt += f"Query: {query}\n\n"
    prompt += f"Candidate Response: {response}\n\n"
    prompt += "Unit Tests:\n"
    for i, unit_test in enumerate(unit_tests):
        assert isinstance(unit_test, str) and len(unit_test) > 0
        prompt += f"Unit Test #{i+1}: {unit_test}\n"

    return prompt
