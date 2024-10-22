from archon.completions import Archon

# make sure to set your OPENAI_API_KEY environment variable
# Initialize Archon
single_gpt_config = {
    "name": "gpt-4o-single",
    "layers": [
        [
            {
                "type": "generator",
                "model": "gpt-4o",
                "model_type": "OpenAI_API",
                "top_k": 1,
                "temperature": 0.7,
                "max_tokens": 2048,
                "samples": 1,
            }
        ]
    ],
}


archon_gpt_config = {
    "name": "archon-gpt-multi-model",
    "layers": [
        [
            {
                "type": "generator",
                "model": "gpt-4o",
                "model_type": "OpenAI_API",
                "top_k": 1,
                "temperature": 0.7,
                "max_tokens": 2048,
                "samples": 10,
            }
        ],
        [
            {
                "type": "ranker",
                "model": "gpt-4o",
                "model_type": "OpenAI_API",
                "top_k": 5,
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        ],
        [
            {
                "type": "fuser",
                "model": "gpt-4o",
                "model_type": "OpenAI_API",
                "temperature": 0.7,
                "max_tokens": 2048,
                "samples": 1,
            }
        ],
    ],
}

#################################################

single_gpt = Archon(single_gpt_config)
archon_gpt = Archon(archon_gpt_config)

testing_instruction = [{"role": "user", "content": "How do I make a cake?"}]

single_gpt_response = single_gpt.generate(testing_instruction)
archon_gpt_response = archon_gpt.generate(testing_instruction)

print(f"Single GPT Query: {single_gpt_response}")
print("---------------------")
print(f"Archon GPT: {archon_gpt_response}")
