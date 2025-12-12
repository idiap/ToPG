#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#

INSTRUCTION_HYPERPROPOSITION_CS = """
Extract all meaningful, highly decontextualized propositions from the given Computer Science passage. These propositions should connect the identified Named Entities, representing atomic, generalizable facts.

Guidelines for Proposition Formulation:

- Entity-Connecting: Each proposition must describe relations, links, properties, or functions *between- two or more Computer Science entities.
- Atomic Meaning: Each proposition must represent a distinct, minimal, and fundamental piece of information.
- Decontextualized & Generalizable: Propositions must be independently fully interpretable without relying on the original passage's specific examples or illustrative context. Strip away any instance-specific phrasing, dataset names (unless the dataset itself is the entity being described), or highly contextual details that are merely illustrative. The aim is to capture universal truths or general relationships stated in the text.
- Comprehensive Coverage: Collectively, the hyperpropositions should capture every explicit Computer Science connection or relationship mentioned in the passage.

- Common Connection Types (examples):
    - Software/Framework/Library usage and components.
    - Algorithm/Method application and purpose.
    - Programming language features and paradigms.
    - Data structures and their properties.
    - System architecture and relationships between components.
    - Development status or evolution.
    - Input/output relationships.
    - Comparative, temporal, or causal statements.
    - Problem-solution relationships.
    - Properties or characteristics of entities.
    - Tool/Utility functions.

- Do not integrate or reference relations between variables from code snippets !

If the input passage is too short, has no entities, lacks context or is not a valid paragraph (like if it is a table of contents for instance), you may return an empty list of propositions.

"""

INSTRUCTION_NER_CS = """
Extract all relevant Computer Science Named Entities from the given passage. These entities will serve as the nodes in a knowledge network.

Guidelines for Extraction:

- Example of Categories to Identify:
    - Software/Framework/Library (e.g., 'Spark', 'MLlib', 'numpy').
    - Programming Language (e.g., 'Python', 'SQL').
    - Concept/Algorithm/Method (e.g., 'machine learning', 'PCA', 'regular expressions').
    - Dataset (e.g., 'MovieLens dataset').
    - Operating System/Environment (e.g., 'Windows', 'Hadoop-compatible filesystem').
    - Metric: (e.g., 'Mean Squared Error', 'RMSE').
    - Mathematical Entity/Symbol: (e.g., 'exponential distribution', 'singular values').
    - Version: Software or API versions (e.g., 'Spark Version 1.2.0').

- Exclusions:
    - Abbreviations if their full form is explicitly present and preferred, unless the acronym is widely recognized and predominantly used.
    - Entities not explicitly mentioned in the passage.
    - Purely syntactic elements from code snippets (e.g., parentheses, commas, assignment operators).
    - Variable names from code snippets (like 'dataset { kx }', 'tokenCountsFilteredStopwords', etc ...)
    - Overly generic terms that do not refer to specific entities (e.g., 'data', 'information', 'function', 'code', 'functionality').

If the input passage is not a valid paragraph, you may return an empty list of entities.
"""
