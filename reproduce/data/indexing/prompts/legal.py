#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#

INSTRUCTION_HYPERPROPOSITION_LEGAL = """
Extract all meaningful, highly decontextualized propositions from the given legal/financial passage. These propositions should connect the identified Named Entities, representing atomic, generalizable legal/financial facts or principles.

Guidelines for Proposition Formulation:

1. Entity-Connecting: Each proposition must describe relations, links, obligations, definitions, or functions between two or more Legal or Financial entities.
2. Atomic Meaning: Each proposition must represent a distinct, minimal, and fundamental piece of legal/financial knowledge.
3. Decontextualized & Generalizable: Each proposition should be independent of the passage’s specific facts or parties and capture a rule, principle, or requirement that could apply universally in similar legal/financial contexts.
4. Comprehensive Coverage: Collectively, the hyperpropositions should capture every explicit legal/financial connection or relationship in the passage.
5. Avoid Specific Parties: Replace specific company names, dates, or transactions with general entities (e.g., "Company," "Borrower," "Investor") to maintain generalizability.
6. Common Connection Types: Include relationships such as:
    - **Definition / Meaning** (e.g., “Interest Expense means…”)
    - **Obligation / Duty** (e.g., “Companies must maintain…”)
    - **Restriction / Prohibition** (e.g., “Companies must not…”)
    - **Right / Permission** (e.g., “Bank may inspect…”)
    - **Condition / Trigger** (e.g., “If an Event of Default occurs, then…”)
    - **Compliance / Standard** (e.g., “Transactions must conform to GAAP”)
    - **Payment / Flow of Funds** (e.g., “Company shall pay interest quarterly.”)
    - **Ownership / Control** (e.g., “Shareholders owning >50% have controlling vote.”)

If the input passage is too short, has no entities, lacks context or is not a valid paragraph (like if it is a table of contents for instance), you may return an empty list of propositions.

"""

INSTRUCTION_NER_LEGAL = """
Extract all relevant Legal and Financial related Named Entities involved in a legal/financial statement from the given passage. These entities will serve as the nodes in a knowledge Graph.

Guidelines for Extraction:

- Example of Categories to Identify:
    - Company/Entity (e.g., 'Blackstone Real Estate Income Trust', 'HCA Inc.', 'Hazardous Chemicals').
    - Contract/Agreement (e.g., 'Credit Agreement', 'Indenture', 'Merger Agreement').
    - Contractual Role (e.g., 'Borrower', 'Lender', 'Guarantor', 'Trustee').
    - Financial Instrument (e.g., 'Senior Notes', 'Convertible Securities', 'Equity Interests', 'Cash flows').
    - Law/Regulation/Regulatory Body (e.g., 'Foreign Corrupt Practices Act', 'SEC', 'Environmental Laws').
    - Jurisdiction/Location (e.g., 'Delaware', 'New York', 'United States District Court').
    - Asset/Collateral (e.g., 'Oil and Gas Properties', 'Intellectual Property', 'Real Property', 'Waste Management').
    - Risk Category (e.g., 'Environmental risk', 'Litigation risk', 'Anti-Corruption risk', 'Market risk').

- Exclusions:
    - Abbreviations if the full form is explicitly present and preferred, unless the acronym is widely recognized and predominantly used (e.g., 'SEC').
    - Entities not explicitly mentioned in the passage.
    - Generic legal terms without specific reference (e.g., 'contract', 'party', 'agreement' if not named).
    - Descriptive or qualitative phrases (e.g., 'material adverse effect', 'ordinary course of business').

If the input passage is not a valid paragraph, you may return an empty list of entities.
"""
