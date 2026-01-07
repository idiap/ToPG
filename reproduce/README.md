<!--
#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#
-->


# TopG Indexing

## 1. 📦 Fetch the JSON data from the Hugging Face repository

All dumps of the knowledge base needed to reproduce the paper’s results are available [here](https://huggingface.co/datasets/mdelmas/Topg-kb).

The complete list of prompts and demonstrations used during indexing is provided in:
- `reproduce/data/indexing/prompts` (for GraphRAG-Benchmark and UltraDomain)
- `reproduce/data/demonstrations-indexing`

For HotPotQA, MusiQue, PopQA, and agriculture we did not use custom prompts for NER or proposition extraction, only the default prompt.

## 2. 🔧 Load a knowledge base (example: HotPotQA)

First, create and configure the knowledge base:

```python
from topg import Topg

config_path = Topg.initialize(
    base_path="/path/to/kb",
    collection_name="Test-HotPotQA",
)
```

You can edit the generated `config.yaml` (e.g. set logging to `DEBUG`). To reproduce paper results, do not change `model_name_documents` or `model_name_entities` (both default to `BAAI/bge-large-en-v1.5`). The config stores paths, model names, and LLM settings.

Mount the system:

```python
import yaml
from topg import Topg

with open(config_path, "r") as f:
    system_config = yaml.safe_load(f)

system = Topg(config=system_config)
```

Provide an OpenAI-compatible key via environment variable `LLM_API_KEY` if using the default `gpt-4o-mini` test setting. (See documentation for other providers.)

The knowledge base is empty initially:

```python
answer, memory = system.query(
    mode="local",
    question="In which county is the town in which Raymond Robertsen was born ?"
)
```
This should lead to no outputs.


Load passages:

```python
system.load_passages_from_json(
    json_passages="/path/to/hotpotqa_passages.json"
)
```

Load propositions:

```python
system.load_hyperpropositions_from_json(
    json_hyperpropositions="/path/to/hotpotqa_hyperpropositions.json"
)
```

Reload graphs after updating:

```python
system.store.load_graphs()
```

This is a multi-hop query, so let's try again with `local` mode.

```python
answer, memory = system.query(
    mode="local",
    question="In which county is the town in which Raymond Robertsen was born?",
    top_k=20
)
```

## 3. 🔍 Naive / Local Search

### 3.1 System setup

```python
import yaml
from topg import Topg

config_path = "path/to/your/config.yaml"
with open(config_path, "r") as f:
    system_config = yaml.safe_load(f)

system = Topg(config=system_config)
```

### 3.2 🤖 LLM settings

Paper setup: Quantized Gemma-3-27B (`ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g`) served via [vLLM](https://docs.vllm.ai) on a single H100. So it should look like this:

```python
from topg.utils.llm import configure_llm

system.lm = configure_llm(
    llm_config={
        "llm_name": "hosted_vllm/ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        "api_base": "http://localhost:$VLLMPORT/v1",
        "max_tokens": 2048,
    }
)
system.lm("Hello!")
```

Alternatively use an OpenAI-compatible model if it is just for testing.

### 3.3 Sample questions

```python
some_questions = [
    {
        "_id": "5abe953b5542993f32c2a170",
        "question": "What is one of the stars of The Newcomers known for?",
        "answer": "superhero roles as the Marvel Comics"
    },
    {
        "_id": "5ab69f9a554299710c8d1ef8",
        "question": "The fictional private detective that appears in 'The Adventure of the Seven Clocks' was written by whom?",
        "answer": "Sir Arthur Conan Doyle"
    },
    {
        "_id": "5abc030e554299642a094bdc",
        "question": "The Distribution of Industry Act was passed by a man who was prime minister when?",
        "answer": "1945 to 1951"
    },
    {
        "_id": "5ae6c2285542995703ce8b9a",
        "question": "Sparking the Marian civil war, who helped the recently abdicated queen to escape her imprisonment?",
        "answer": "the Queen's gaoler"
    },
    {
        "_id": "5ab57fc4554299488d4d99c0",
        "question": "In which county is the town in which Raymond Robertsen was born?",
        "answer": "Finnmark county"
    }
]
```

Full evaluation sets: [HippoRAG_2](https://huggingface.co/datasets/osunlp/HippoRAG_2) for HotPotQA, MusiQue, PopQA.


### 3.4 📝 Custom prompt (HotPotQA-like QA)

```python
prompt_instruction = """
# TASK
You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents.
You will be given a question and a set of collected facts (potentially empty).

# FACTS
All the relevant facts that have been collected so far.
# GUIDELINES

By combining the information from the collected facts, determine if you can answer the question.

- If YES, return is_sufficient = True and answer the question.
- If NO, then it means you need more information from the fact bank to answer the question. Return is_sufficient = False and plan the `next questions` for collecting more facts.

When planning `next_questions`
1) Identify what is missing — what do you still need to know - considering the information from the already collected facts ?
2) What are the most relevant direction to explore ?

Reason strategically step by step.

When proposing a question:

- `entity` refer to the named entity or object onto which the question will apply.
- `question` formulates the request.

# ANSWER FORMAT
# Your answer should be a short answer and it can have different types depending on the question: Group / Org (eg. Cartoonito, Apalacheev; Location Fort (eg. Richardson, California); Date (eg. 10th or even 13th century); Number (eg. "79.92 million, 17); Artwork (eg. Die schweigsame Frau); Yes/No; Adjective (eg. conservative); Event (eg. Prix Benois de la Danse); Proper noun (eg. Cold War, Laban Movement); Common noun (Analysis, comedy, both men and women)
"""

# Specify the prompt in the evaluator component
system.query_manager.set_custom_instruction_prompt(
    mode="local",
    predictor_name="evaluator",
    instruction_prompt=prompt_instruction
)
```

### 3.5 📝 Demonstrations

```python
demonstrations = [
    {
        "question": "When did the people who captured Malakoff come to the region where Philipsburg is located?",
        "facts": "Philipsburg is the capital of Sint Maarten.",
        "reasoning": "The question asks for the date when the people who captured Malakoff came to the region where Philipsburg is located. I know Philipsburg is the capital of Sint Maarten, but I do not yet know who captured Malakoff or what larger geographic region Sint Maarten belongs to. Without knowing who captured Malakoff, I can't identify which people the question refers to, and without knowing the region, I can't determine when they arrived there. These are the next missing facts.",
        "is_sufficient": False,
        "answer": "",
        "next_questions": ["Who captured Malakoff?", "What geographic region is Sint Maarten located in?"]
    },
    {
        "question": "When did the people who captured Malakoff come to the region where Philipsburg is located?",
        "facts": "Philipsburg is the capital of Sint Maarten.\nMalakoff was captured by French forces during the Crimean War.\nSint Maarten is located in the Caribbean.",
        "reasoning": "I know that Malakoff was captured by the French, and that Philipsburg (the target city) is in Sint Maarten, which is located in the Caribbean. The question now reduces to: when did the French come to the Caribbean? This key fact is still missing and is needed to answer the question.",
        "is_sufficient": False,
        "answer": "",
        "next_questions": ["When did the French come to the Caribbean?"]
    },
    {
        "question": "When did the people who captured Malakoff come to the region where Philipsburg is located?",
        "facts": "Malakoff was captured by French forces during the Crimean War.\nPhilipsburg is the capital of Sint Maarten.\nSint Maarten is located in the Caribbean.\nThe French began settling the Caribbean around the year 1625.",
        "reasoning": "I know that the people who captured Malakoff were the French, that Philipsburg is in Sint Maarten (part of the Caribbean), and that the French began settling the Caribbean around 1625. Therefore, the people who captured Malakoff (the French) came to the region where Philipsburg is located around 1625. All needed information is now available.",
        "is_sufficient": True,
        "answer": "1625",
        "next_questions": []
    }
]

system.query_manager.set_evaluator_demonstrations(mode="local", demonstrations=demonstrations)
```

### 3.6 Run queries

We are going to use the default retrieval parameters.

```python
QA_MODE = "local"
MAX_ITER = 3
TOP_K = 20

q_and_a = []
for idx, item in enumerate(some_questions):
    print(f"Processing {idx+1}/{len(some_questions)}")
    try:
        predicted_answer, memory = system.query(
            item["question"],
            mode=QA_MODE,
            max_iter=MAX_ITER,
            top_k=TOP_K,
        )
        q_and_a.append({
            "id": item["_id"],
            "question": item["question"],
            "answer": item["answer"],
            "predicted_answer": predicted_answer,
            "memory": [blk.model_dump() for blk in memory],
        })
    except Exception as e:
        print(f"Error {item['_id']}: {e}")
        q_and_a.append({
            "id": item["_id"],
            "question": item["question"],
            "answer": item["answer"],
            "predicted_answer": "no answer",
            "memory": [],
        })
```

### 3.7 📊 Evaluation

```python
from topg.evaluation.metrics import calculate_exact_match, calculate_f1_score

gold_answers = [[item["answer"]] if isinstance(item["answer"], str) else item["answer"]
                for item in some_questions]
predicted_answers = [r["predicted_answer"] for r in q_and_a]

exact_match_eval = calculate_exact_match(gold_answers, predicted_answers)
f1_score = calculate_f1_score(gold_answers, predicted_answers)
print({"exact_match": exact_match_eval, "f1_score": f1_score})
```

To evaluate with the `naive` mode, simply change **`QA_MODE`** to "naive".


### GraphRAG-Benchmark

Given the questions of the GraphRAG benchmark, you can use the same process to reproduce the results. The benchmark uses its own evaluation framework (see [GraphRAG-Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)). No demonstrations were used for those experiments. Indexing prompts: `reproduce/data/indexing/demonstrations-indexing`. Inference used the default prompt.

## 4. 🌐 Global Search

Assuming you loaded an agriculture / cs / legal KB dump.

### 4.1 Sample questions (agriculture)

Example of some queries:
```python
some_abstract_questions = [
    "How does beekeeping activity vary by region and season?",
    "How can soil health practices mitigate climate change effects?"
]
```

### 4.2 ⚙️ Parameters

```python

QA_MODE = "global"

seeding_top_k = 25
top_k = 15
min_number_of_facts = 200
alpha = 1.0
beta = 0.0  # Change this if you want to test with feedback on query embeddings
gamma = 0.0 # Change this if you want to test with feedback on query embeddings
max_iter = 50
m = 10
budget = 12000
passage_token_limit = 500
max_tokens_report = 12000
max_tokens_community_chunks = 8000
min_community_size = 10
max_community_size = 150
refine_answer = True
response_format = "Comprehensive multiple paragraphs expert long report"
```

### 4.3 Run queries

```python
q_and_a = []
for idx, question in enumerate(some_abstract_questions):
    print(f"Processing {idx+1}/{len(some_abstract_questions)}")
    try:
        final_response, memory = system.query(
            mode=QA_MODE,
            question=question,
            m=m,
            budget=budget,
            seeding_top_k=seeding_top_k,
            top_k=top_k,
            max_iter=max_iter,
            min_number_of_facts=min_number_of_facts,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            passage_token_limit=passage_token_limit,
            max_tokens_report=max_tokens_report,
            max_tokens_community_chunks=max_tokens_community_chunks,
            min_community_size=min_community_size,
            max_community_size=max_community_size,
            response_format=response_format,
            refine_answer=refine_answer,
        )
        q_and_a.append({
            "question": question,
            "answer": final_response,
            "memory": [blk.model_dump() for blk in memory],
        })
    except Exception as e:
        print(f"Error {idx} ({question}): {e}")
        q_and_a.append({
            "question": question,
            "answer": "no answer",
            "memory": [],
        })
```

### 4.4 Comparative evaluation prompt (LightRAG-style)

You can find the complete list of generated questions we used here: `reproduce/data/questions-UltraDomain`.

```python
sys_prompt = """
---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
"""

prompt = f"""
You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.
- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question? A comprehensive answer should be thorough and complete, without being redundant or irrelevant. For example, if the question is ’What are the benefits and drawbacks of nuclear energy?’, a comprehensive answer would provide both the positive and negative aspects of nuclear energy, such as its efficiency, environmental impact, safety, cost, etc.
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question? A diverse answer should be multi-faceted and multi-dimensional, offering different viewpoints and angles on the question. For example, if the question is ’What are the causes and effects of climate change?’, a diverse answer would provide different causes and effects of climate change, such as greenhouse gas emissions, deforestation, natural disasters, biodiversity loss, etc.
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic? A good answer shows empowerment by clearly explaining the topic in a way that helps the reader understand it and make informed, thoughtful decisions.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall_Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }}
}}
"""
```
