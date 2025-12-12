#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#

## For the medical domain indexing
INSTRUCTION_HYPERPROPOSITION_MEDICAL = """
Your task is to extract all meaningful decontextualized propositions from the given medical passage, in order to connect the provided Biomedical Named Entities.
    - Propositions connect biomedical entities: they describe their relations, links.
    - Propositions represent distinct and minimal pieces of meaning.
    - Propositions are decontextualised, meaning they are independently fully interpretable without the context of the initial passage.
    - Ensure full coverage so that, together, the hyperpropositions capture every explicit biomedical connections/relationships mentioned in the passage.

Connections between biomedical entities can include:
    - Cancer type identification disease staging.
    - Biomarker associations and biological functions.
    - Treatment protocols and procedures.
    - Diagnostic methods and criteria.
    - Risk factors and epidemiological facts.
    - Symptomatology and clinical manifestations.
    - Comparative, temporal, or causal statements directly stated in the passage.

If the passage is too short, lacks context, or has no valid named entities, return an empty list of hyperpropositions.

You will be given some examples.
"""


INSTRUCTION_NER_MEDICAL = """
Your task is to extract all relevant Biomedical Named Entities from the given medical passage.

- Entities can include: Disease (eg. 'Prostate Cancer'), Cell lines (eg. 'HeLa cells'), Gene (eg. 'BRCA1'), Chemical (eg. 'Glucose'), Drugs (eg. 'Lenvatinib'), Species ('Staphylococcus aureus'), Variant (eg. 'p.s2988fs'), Treatment procedures (eg. 'Immunotherapy'), Anatomical structures (eg. 'Head'), Diagnostic methods (eg. 'Magnetic resonance imaging (MRI)'), risk factors (eg. 'Tobacco'), symptoms (eg. 'Bone pain').
- Abbreviations or acronyms should not be included (eg. extract "follicular lymphoma" instead of "follicular lymphoma (FL)").
- Do not introduce new entities not present in the passage.

If the input passage is too short, lacks context, or is not a valid paragraph, you may return an empty list of entities.
"""
