<!--
#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#
-->

[![arXiv](https://img.shields.io/badge/arXiv-2601.04859-b31b1b.svg)](https://www.arxiv.org/abs/2601.04859)

TopG (Traversal Over Proposition Graphs) is a hybrid RAG (Retrieval-Augmented Generation) framework that builds a graph from passages, entities, and propositions (facts). It supports three query modes: **naive** (single-shot retrieval), **local** (iterative multi-hop reasoning), and **global** (community-based decomposition and synthesis).

To reproduce the results from the paper, please go to the directory `reproduce`.

## Table of Contents

- [Installation](#installation)
- [Initialization](#initialization-and-configuration)
- [Indexing Documents](#indexing-documents)
- [Querying](#querying-naive-local-and-Global-modes)
- [Export and Import](#export-and-import)
- [Advanced settings and usage](#advanced-settings-and-usage)
---

## Installation

**Requirements:**
- Python 3.10–3.14

**Using `pip`**

```bash
pip install git+https://github.com/idiap/ToPG
```
**Environment:**
Set your LLM API key:
```bash
export LLM_API_KEY=sk-...
```

We use [dspy.LM](https://dspy.ai/learn/programming/language_models) under the hood to configure LLM calls. It supports a large range of providers (OpenAI, Ollama, Anthropic, etc.)
By default the system will try to use OpenAI gpt-4o-mini model.
After initialization, you can setup and reconfigure the LLM by modifying the `config.yaml` or manually using `configure_llm`.
An other advantage of using dspy is that you can directly use dspy functionalities like `dspy.inspect_history(n)` to display all the previous llm calls.

*Notes on `uv.lock`*:
The `uv.lock` file is included to provide a reproducible environment. It does not affect how users install or use this library—only pyproject.toml defines the library’s install requirements. Assuming you have installed [uv](https://docs.astral.sh/uv/), `uv sync` installs the exact development environment defined in `uv lock`

---

## Initialization and configuration

### Creating a New Knowledge Base

Use `Topg.initialize` to create a new project:

```python
import yaml
from topg import Topg

# Initialize a new KB at the given path
config_path = Topg.initialize(
    base_path="/path/to/my_kb",
    collection_name="MyKnowledgeBase"
)

# Load the generated config / you could manually edit the config before loading
with open(config_path, "r") as f:
    system_config = yaml.safe_load(f)

# Mount the system
system = Topg(config=system_config)
```

If you encounter a dspy.error, please check that the `DSPY_CACHE_DIR` is properly set:

```bash
DSPY_CACHE_DIR="/tmp/dspy"
mkdir -p $DSPY_CACHE_DIR
export DSPY_CACHEDIR=$DSPY_CACHE_DIR
```

### Configuration Parameters

TopG uses a YAML or JSON config file. Key sections:

#### `logger_params`
- **`name`**: project name. (default to `collection_name`)
- **`path`**: path for the logging file. (default to `base_path/logs_and_cache`)
- **`stdout`**: logging in console. (default to `True`)
- **`level`**: logging level. (default to `INFO`)

#### `document_processor_params`
- **`paragraph_separator`**: controls passage chunking. (default "\n\n\n")
- **`paragraph_size_limit`**: controls passage max number of sentences. (default 8)
- **`paragraph_sent_overlap`**: controls the sentence overlap between passages. (default 2)
- **`cache_dir`**: path to the cache. (default to `base_path/logs_and_cache`)
- **`max_workers`**: max number of LLM calls simultaneously. (default to 8)

#### `storage_params`
- **`document_vdb_path`**: path to the vector database for propositions. (default to `base_path/dbs/vdb-docs`)
- **`entities_vdb_path`**: path to the vector database for entities. (default to `base_path/dbs/vdb-entities`)
- **`sqlite_db_path`**: path to the sqlite database for graph storage. (default to `base_path/dbs/sqlite_db.db`)
- **`collection_name`**: name of the project/collection of documents. (default to `collection_name`)
- **`device`**: device for document embeddings. ("cuda" or "cpu")
- **`model_name_documents`**: hf model for proposition embeddings. (default: "BAAI/bge-large-en-v1.5")
- **`model_name_entities`**: hf model for entity embeddings. (default: "BAAI/bge-large-en-v1.5")
- **`encoding_params`**: encoding parameters for generating embeddings with sentenceTransformers. (default to `batch_size=128,convert_to_numpy=True, show_progress_bar=True')
- **`communities_max_cluster_size`**: minimal size for cluster communities. (default to 50)
- **`communities_seed`**: seed for the community detection. (default to 42)

#### `loaders_params`
- **`loading_batch_size`**: size of the batch when inserting items. (default 512)
- **`graph_resolve_synonym`**: should we try to resolve synonym between entities based on their embeddings ?
- **`graph_synonym_max_limit`**: maximum number of synonyms (top-similar) for an entity (default: 10).
- **`graph_synonym_sim_threshold`**: cosine similarity threshold for merging entity names (default: 0.9).

#### `llm_config`
- **`api_base`**: Base URL for the LLM provider (default: "https://api.openai.com/v1").
- **`llm_name`**: Model name (default: "openai/gpt-4o-mini").
- **`max_tokens`**: Max tokens per LLM call (default: 8000).


---

## Indexing Documents

### Inserting a Full Document

To index a complete document (e.g., from a file):

```python
with open("document.txt", "r") as f:
    document_text = f.read()


# Insert with a document ID
system.insert(
    text=document_text,
    doc_id="doc-01"
)

# (Optionally, you can add some metadata)
system.add_document_metadata(doc_id="doc-01", metadata={"Title": "This is a title", "Citation": "...", })

# If no document ID are provided, one will be generated automatically, for instance:
system.insert(text="Paris is the capital of France. It is located on the Seine River.")

# Check indexing statistics
system.store.show_statistics()
```

### Changing The Prompt For Better Extraction

The default prompt break down a passage into proposition according to the following:

```
Your task is to extract all meaningful decontextualized propositions from the given passage, using only the provided Named Entities.
    - Break down the input passage into its basic, minimal propositions that represent distinct pieces of meaning.
    - Propositions are independently interpretable without the context of the initial passage.
Rules for each proposition:
    1. Complete & Standalone: a proposition convey exactly one fact or relationship, including all necessary context.
    2. Entity Usage: Use only the entities from Named Entities; do not introduce any others.
    3. Clear Attribution: Specify precisely which entities participate and their roles in the decontextualised proposition.
    4. Explicit Connections: Make causal, comparative, and temporal relationships explicit, including reference points.
    5. Preserve Time: Integrate temporal information and time markers (eg. a birth date or a date of an event) in the proposition.
    6. Preserve Cause: Keep causal links intact.
    7. Full Coverage: Collect propositions that together capture every meaningful point in the Passage.
    8. Preserve context: add all necessary contextual elements, like key entities, to improve clarity and precision of each propositions without the context of the initial passage.

    The precision and completeness of these propositions directly impact the performance and reliability of downstream tasks, particularly the quality of question answering systems. Therefore, careful adherence to the above rules is critical to ensure factual accuracy, unambiguous interpretation, and full semantic coverage.

    If the input passage is too short, has no entities, lacks context or is not a valid paragraph (like if it is a table of contents for instance), you may return an empty list of propositions.
```

For specific topics (eg. medical, legal, etc.) it is recommend to adapt the prompt. You can do this with:

```python
system.document_processor.hyperpropositionizer.set_custom_instruction_prompt(
    "hyperpropositionizer", "new prompt ..."
)
```
However, the expected inputs ("passage" and "entities") and the output ("hyperpropositions") cannot be changed.


### Using Demonstrations for Better Extraction

Demonstrations are few-shot examples that guide the LLM during proposition extraction. The two expected input for a demonstration for proposition extraction are: "passage" and "entities".
The expected output in a list of propositions along with their associated entities (here after called *hyperpropositions* as the created hyperedges in the graph) as examplified in the next example:

```python
demonstrations = [
    {
        "passage": "Clock tower stands in the historic center of the Grand Bazaar, at a place known as the Field of the Clock. According to the Islamic calendar it was built in the year 1002, which is 1597 in the Gregorian calendar.",
        "entities": ["Clock tower", "Grand Bazaar", "Gjakova", "Field of the Clock", "Islamic calendar", "Gregorian calendar"]
        "hyperpropositions": [
            {
                "proposition": "The clock tower stands in the historic center of the Grand Bazaar of Gjakova in a place known as the “Field of the Clock.”",
                "entities": [
                    "Clock tower",
                    "Grand Bazaar",
                    "Gjakova",
                    "Field of the Clock"
                ]
            },
            {
                "proposition": "According to the Islamic calendar, the clock tower of Gjakova was built in the year 1002.",
                "entities": [
                    "Islamic calendar",
                    "Clock tower",
                    "Gjakova"
                ]
            },
            {
                "proposition": "Year 1002 is the Islamic calendar correspond to 1597 in the Gregorian calendar.",
                "entities": [
                    "Islamic calendar",
                    "Gregorian calendar"
                ]
            }
        ]
    }
]

# Pass demonstrations when initializing or update the config file
system.document_processor.hyperpropositionizer.set_extractor_demonstrations(demonstrations=demonstrations)
```


Similarly to the proposition extraction, you can adapt the prompts and the demonstrations for the NER step with:

```python
system.document_processor.hyperpropositionizer.set_custom_instruction_prompt(
    "ner", "new prompt ner"
)
# and,
system.document_processor.hyperpropositionizer.set_ner_demonstrations(demonstrations=ner_demos)
```

We do not recommend to use numerical value (eg. dates) as entities, particularly if you use synonym resolution for the graph. This will lead to unintended synonym attribution.

## Querying (Naive, Local and Global modes)

TopG supports three retrieval modes: **naive**, **local**, and **global**.

TopG also returns a structured "**memory**" that records what the system did step‑by‑step during retrieval and inference. The memory is a useful inspectable trace for debugging and understanding the model's reasoning.

Key points:
- The memory is returned as the second value from system.query: answer, memory = system.query(...)
- It is a list of pydantic MemoryBlock objects (human‑readable and serializable).
- All query engines are equiped with a `callback_logger`. Everytime the memory is update it can log/emit it using a user-defined `callback_logger` function. This can be particularly useful to log or show progresses over time, for instance when using them as tools for an agent 😉.


See more details for the retriever parameters and the memory feature in [Advanced settings and usage](#advanced-settings-and-usage)

### 1. Naive Mode (Single-Shot)

Retrieves relevant propositions and generates an answer in one LLM call. No graph or Suggestion/Selection involved.

```python
answer, memory = system.query(
    question="Who wrote the novel 'Seven Clocks'?",
    mode="naive",
    top_k=20,
)
```

**Parameters:**
- `question`: The query string.
- `mode`: `"naive"`.
- `top_k`: number of retrieved propositions (default: 20)
- `use_passages`: Integrate the passages associated to the extracted propositions in the final context. (default `False`)
- `context_as_references`. Return all metadata in the `context` attribute for the Inference Memory Block.  (default `False`)
- `callback_logger`: An optional function use as a callback logger everytime the memory is updated. It will takes as `data` argument the current `MemoryBlock`.



### 2. Local Mode (Multi-Hop Reasoning)

Iteratively retrieves facts and passages, building reasoning chains. Ideal for multi-hop questions.

```python
answer, memory = system.query(
    question="What is the population of the city where the Eiffel Tower is located?",
    mode="local",
    max_iter=2,  # Maximum reasoning iterations
    top_k=20,
)
```

**Parameters:**
- `question`: The query string.
- `mode`: `"local"`.
- `top_k`: number of retrieved propositions at each iteration (default: 20)
- `max_iter`: Max reasoning steps (default: `2`).
- `use_passages`: Integrate the passages associated to the extracted propositions in the final context. (default `False`)
- `context_as_references`. Return all metadata in the `context` attribute for the Inference Memory Block.  (default `False`)
- `callback_logger`: An optional function use as a callback logger everytime the memory is updated.


### 3. Global Mode (Community-Based Synthesis)

Decomposes complex questions into subqueries, retrieves across the graph, clusters information into communities, and synthesizes a comprehensive answer. Best for broad, analytical questions. For more parameters see below in [Advanced settings and usage](#advanced-settings-and-usage).

```python
final_response, memory = system.query(
    mode="global",
    question="How do soil health practices mitigate the effects of climate change on agriculture?",
    m=10,          # Number of subqueries to decompose
    min_number_of_facts=100,
)
```

**Parameters:**

- **`m`**: controls the breadth of the exploration.
- **`min_number_of_facts`**: minimum (if possible) number of facts to collect before extracting communities

---

## Advanced Settings And Usage

### Retrieval parameters

When calling the `query` function, you can pass additional parameters to the retriever. See the paper for additional details
- **`seeding_top_k`**: the top-k used in the initial retrieval step when seeding in the graph. (default 20)
- **`lambda_`** $\lambda$ controls the balance between structural and semantic guidance. (default 0.5)
- **`damping`**: the damping factor for the PPR (default: 0.85)
- **`cosine_threshold`**: the cosine similarity threshold (default 0.4)
- **`horizon_top_k`**: the size of the extracted subgraph around the seed node (default 500)
- **`temperature`**: the temperature scaling parameter when creating the transition matrix (default 0.1)

### 📝 Custom Prompts

You can inspect **all** the prompts (all used dspy modules) with:

```python
system.query_manager.list_predictors()
```

Then, you can override default prompts for any mode:

```python
# Set a custom instruction for local mode evaluator
system.query_manager.set_custom_instruction_prompt(
    mode="local",
    predictor_name="evaluator",
    prompt_text="You are a reasoning assistant. Determine if the question is answered."
)
```


### Memory

Each `MemoryBlock` is also automatically sent to the `callback_logger` if one is provided.

#### Memory in naive mode

In naive mode the memory records two main steps produced during a single-shot retrieval + answer generation:

- **`RetrievalMemoryBlock`**
  - question: the original query.
  - facts: list of retrieved proposition texts (final top-k).
  - facts_ids: corresponding proposition IDs.
  - log: human readable note about the retrieval.

- **`InferenceNaiveMemoryBlock`**
  - question: the original query.
  - reasoning: chain-of-thought reasoning produced by the evaluator.
  - response: final answer produced by the evaluator.
  - context: either a simple list of fact texts or a References object (entities, passages, documents) when context_as_references/use_passages is enabled.
  - log: human readable note about the inference step.

*Behavior notes*
- If use_passages=True the evaluator runs with passages and the `InferenceMemoryBlock.context` will include passages (via References).
- If context_as_references=True you get a References object with entities_texts, entities_ids, passages_texts, documents_metadata, etc.


#### Memory in local mode

- **`RetrievalMemoryBlock`**: same as `naive`

- **`InferenceMemoryBlock`**
  - question: the original query.
  - reasoning: chain-of-thought produced by the evaluator.
  - response: answer (if any) produced by the evaluator.
  - context: either a list of fact texts or a References object when context_as_references/use_passages is enabled.
  - is_sufficient: boolean indicating whether the evaluator judged the current facts sufficient.
  - next_questions: list of follow-up questions planned when is_sufficient is False.
  - log: human readable note about the evaluator decision.

Depending on the number of iterations, there can be several `RetrievalMemoryBlock` and `InferenceMemoryBlock`.

#### Memory in global mode

Global mode produces a richer, multi-stage memory trace that reflects decomposition, large‑scale retrieval, community selection, summarization and final synthesis. The memory is an ordered list of typed `MemoryBlock` objects that lets you inspect what happened at each major step.

Main block types (in order of appearance)
- **`DecompositionMemoryBlock`**
  - sub_questions: list of generated sub-queries.
  - log: note about decomposition.
- **`GlobalRetrievalMemoryBlock / SingleRetrievalMemoryBlock`**
  - iteration: iteration index (`GlobalRetrievalMemoryBlock`).
  - retrievals: for each group, facts, facts_ids, optional seed_nodes and selector reasoning.
  - log: note about the retrieval iteration.
- **`CommunityBudgetMemoryBlock`**
  - coverage_ratio, budget_ratio: summary of budget/coverage used for community selection.
- **`IntermediaryResponsesMemoryBlock`**
  - intermediary_responses: generated keypoints / intermediary reports used to build the final answer.
- **`ResponseMemoryBlock`**
  - response: generated (or refined) response text.
  - log: which generation/refinement step produced it.
- **`CommunitiesMemoryBlock`**
  - communities: final list of CommunityOutput objects each containing a References object (entities_texts, facts_texts, passages_texts, documents_metadata, etc.).
  - log: summary about communities used.

*Behavior notes*
- The trace starts with decomposition, continues with multiple retrieval/selection iterations (one or more `GlobalRetrievalMemoryBlock`), then a budget/cover step, community post‑processing, intermediary summaries, the generated/ refined response(s), and ends with the `CommunitiesMemoryBlock` containing the references used.
- Each Retrieval block includes selector reasoning for why facts were chosen; IntermediaryResponses contains the community keypoints used to synthesize the answer.
- callback_logger (if provided) is called every time a memory block is appended so you can stream the trace live.
- References objects in community blocks contain full metadata (entities, passages, documents).


### Export and Import

#### Exporting Data

Export your indexed data to JSON for backup or transfer:

```python
# Export passages
system.store.export_all_passages(output_path="passages.json")

# Export hyperpropositions (with entities and mappings)
system.store.export_all_hyperpropositions(output_path="hyperpropositions.json")

# If you added documents metadata
system.store.export_all_documents(output_path="documents_and_metadata.json")
```

#### Importing Data

Load previously exported data into a new or existing KB:

```python
# Import passages
system.load_passages_from_json(json_path="passages.json")

# Import hyperpropositions
system.load_hyperpropositions_from_json(json_path="hyperpropositions.json")

# Optionally import full documents (if exported separately)
system.load_documents_from_json(json_path="documents.json")

# Reload graphs after import
system.store.load_graphs()
```

---

### Token and Cost Tracking

```python
from topg.utils.llm import get_cost, get_count_tokens

# Count tokens in a string
token_count = get_count_tokens(system.lm)
print(f"Tokens: {token_count}")

# Estimate cost
cost = get_cost(system.lm)
print(f"Cost: ${cost:.4f}")
```


### Global search advanced parameters

Here are some more parameters for the global mode. More details in the paper.

**Parameters:**
- **`budget`**: Token budget for community summarization (default: `12000`).
- **`max_tokens_report`**: maximal number of tokens to use when prompting for generating the final report (default: 3500)
- **`max_tokens_community_chunks`**: maximal number of tokens in a batch when producing  communities' keypoints
- **`passage_token_limit`**: maximal number of tokens dedicated to passages when producing communities' keypoints (default: 500)
- **`max_iter`**: Max refinement iterations (default: `50`).
- **`add_references`**: include references to communities indexes in the response (default: True)
- **`refine_answer`**: Apply the answer refinement step (default: True)
- **`alpha`**: with feedback seetings (see paper) coefficient of the query embeddings (default: 1.0)
- **`beta`**: with feedback seetings (see paper) coefficient of the selected proposition embeddings (default: 0.0)
- **`gamma`**: with feedback seetings (see paper) coefficient of the discarded proposition embeddings (default: 0.0)



## Citation
:page_facing_up:
If you found the paper and/or this repository useful, please consider citing our work: :+1:

```bibtex
@misc{delmas2026navigationalapproachcomprehensiverag,
      title={A Navigational Approach for Comprehensive RAG via Traversal over Proposition Graphs}, 
      author={Maxime Delmas and Lei Xu and André Freitas},
      year={2026},
      eprint={2601.04859},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.04859}, 
}
```
