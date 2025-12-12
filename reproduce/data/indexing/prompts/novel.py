#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#

INSTRUCTION_HYPERPROPOSITION_NOVEL = """
Your task is to extract all meaningful decontextualized narrative propositions from the given novel passage, in order to connect the provided Named Entities.
    - Propositions connect narrative entities such as characters, locations, objects, events, and social roles.
    - Propositions describe relationships and interactions including actions, mental states, social and spatial relations.
    - Propositions represent distinct and minimal pieces of meaning.
    - Propositions are decontextualized, meaning they are independently fully interpretable without the original passage context.
    - Ensure full coverage so that, together, the hyperpropositions capture every explicit narrative connection or relationship mentioned in the passage.

Connections between entities can include:
    - Actions: Who did what to whom, when, and how.
    - Mental states: Who knows, feels, thinks, or believes what.
    - Social relationships: Friendships, authority, roles, family ties, alliances.
    - Spatial relationships: Who is where, movements, places visited or described.
    - Events and temporal relations: Sequences, causal or temporal ordering of events.
"""

INSTRUCTION_NER_NOVEL = """
Your task is to extract all relevant Named Entities from the given novekl passage.
Entities can include:
- Characters
- Key Events
- Locations
- Objects
- Professional roles

Extracted entities can also be qualified by their ownership or belonging relationships when it is relevant to the narrative and help their precise identification.
Do not include dates.
"""
