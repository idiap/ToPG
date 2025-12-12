#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#


from typing import Any, Callable, List, Optional, Union

from pydantic import BaseModel

# Models for the indexing


class EntitiesList(BaseModel):
    """List of extracted Named Entities.
    Used during the indexing phase to store entities extracted from text.
    """

    entities: List[str]


class HyperProposition(BaseModel):
    """A proposition with its associated entities.

    Represents a single factual statement along with the entities
    mentioned or involved in that proposition.

    Attributes:
        proposition: The text of the proposition/statement
        entities: List of entity names referenced in the proposition
    """

    proposition: str
    entities: List[str]


class HyperPropositionList(BaseModel):
    """
    A list of hyperpropositions.
    """

    hyperpropositions: List[HyperProposition]


# Models for the reasoning and retrieval memory blocks
## Local modes


class MemoryBlock(BaseModel):
    """Base class for memory blocks in the reasoning process.

    Memory blocks track the state and logs of various processing steps.

    Attributes:
        log: Optional logging information for debugging and tracing
    """

    log: str = None


class References(BaseModel):
    """Complete reference information for retrieved content.

    Tracks all levels of references from entities to documents,
    maintaining the full provenance chain of retrieved information.

    Attributes:
        entities_texts: Text content of referenced entities
        entities_ids: Unique identifiers for entities
        facts_texts: Text content of referenced facts/propositions
        facts_ids: Unique identifiers for facts
        passages_texts: Text content of referenced passages
        passages_ids: Unique identifiers for passages
        documents_ids: Unique identifiers for source documents
        documents_metadata: Additional metadata for source documents
    """

    entities_texts: List[str]
    entities_ids: List[str]
    facts_texts: List[str]
    facts_ids: List[str]
    passages_texts: List[str]
    passages_ids: List[str]
    documents_ids: List[str]
    documents_metadata: List[Any]


class RetrievalMemoryBlock(MemoryBlock):
    """Memory block for local / naive retrieval (suggest/select) operations.

    Stores the question and retrieved facts during the retrieval: suggest/select phase.

    Attributes:
        question: The query being processed
        facts: List of retrieved fact texts
        facts_ids: Unique identifiers for the retrieved facts
    """

    question: str
    facts: List[str]
    facts_ids: List[str]


class InferenceMemoryBlock(MemoryBlock):
    """Memory block for inference and reasoning operations in local mode.

    Captures the complete inference process including question, reasoning,
    response, and supporting context. Also tracks sufficiency and follow-up
    questions for iterative reasoning.

    Attributes:
        question: The query being answered
        reasoning: The reasoning process or chain of thought
        response: The generated answer
        context: Supporting references or context strings
        is_sufficient: Whether the answer is sufficient or needs more info
        next_questions: Follow-up questions to explore if needed
    """

    question: str
    reasoning: str
    response: str
    context: Union[References, List[str]]
    is_sufficient: Optional[bool] = None
    next_questions: Optional[List[str]] = None


CallbackLogger = Callable[[dict], None]


# Naive


class InferenceNaiveMemoryBlock(MemoryBlock):
    """Memory block for naive (simple) inference without iterative reasoning.

    Similar to InferenceMemoryBlock but without sufficiency checks
    or follow-up questions. Used for straightforward single-step inference.

    Attributes:
        question: The query being answered
        reasoning: The reasoning process
        response: The generated answer
        context: Supporting references or context strings
    """

    question: str
    reasoning: str
    response: str
    context: Union[References, List[str]]


# Global


class KeyPoint(BaseModel):
    """A key point or important fact with a relevance score.

    Used in global question answering to represent important information
    extracted from communities.

    Attributes:
        description: The text description of the key point
        score: Relevance or importance score
    """

    description: str
    score: int


class IndexedKeyPoint(KeyPoint):
    """A key point with references to the indexed of the used communities."""

    references: List[int]


class SelectorReasoning(BaseModel):
    """Reasoning for fact selection and filtering.

    Captures the proposed facts and the reasoning behind their selection
    during the retrieval process.

    Attributes:
        proposed_facts: The facts being considered
        reasoning: Explanation of why these facts were selected
    """

    proposed_facts: str
    reasoning: str


class DecompositionMemoryBlock(MemoryBlock):
    """Memory block for query decomposition.

    Stores the result of breaking down a complex (abstract) query into
    simpler sub-questions.

    Attributes:
        sub_questions: List of decomposed sub-questions
    """

    sub_questions: List[str]


class SingleRetrievalMemoryBlock(BaseModel):
    """Memory block for a single retrieval (suggest/select) operation.

    Captures facts retrieved in one retrieval (suggest/select) step, along with seed nodes
    used for graph traversal and optional reasoning.

    Attributes:
        facts: Retrieved fact texts
        facts_ids: Unique identifiers for facts
        seed_nodes_ids: Starting node IDs for graph traversal
        seed_nodes_texts: Text content of seed nodes
        reasoning: Optional reasoning about the retrieval
    """

    facts: List[str]
    facts_ids: List[str] = []
    seed_nodes_ids: List[str] = []
    seed_nodes_texts: List[str] = []
    reasoning: Optional[str] = None


class GlobalRetrievalMemoryBlock(MemoryBlock):
    """Memory block for multi-step global retrieval.

    Tracks multiple (`m`) retrieval iterations for complex queries that
    require iterative information gathering.

    Attributes:
        iteration: Current iteration number
        retrievals: List of individual retrieval operations
    """

    iteration: int = 0
    retrievals: List[SingleRetrievalMemoryBlock] = []


class CommunityBudgetMemoryBlock(MemoryBlock):
    """Memory block for tracking community selection budgets.

    Used in global QA to track how much of the community space
    has been covered relative to the allowed budget.

    Attributes:
        coverage_ratio: Proportion of communities covered
        budget_ratio: Proportion of budget consumed
    """

    coverage_ratio: float
    budget_ratio: float


class IntermediaryResponsesMemoryBlock(MemoryBlock):
    """Memory block for intermediate responses during global QA.

    Stores key points extracted from communities before final
    answer synthesis.

    Attributes:
        intermediary_responses: List of indexed key points from communities
    """

    intermediary_responses: List[IndexedKeyPoint]


class ResponseMemoryBlock(MemoryBlock):
    """Memory block for the final response.

    Contains the synthesized response to the user query.

    Attributes:
        response: The final generated response text
    """

    response: str


class CommunityOutput(BaseModel):
    """Output from processing a single community.

    Contains the community identifier and all references
    extracted from that community.

    Attributes:
        community_idx: Index/identifier of the community
        references: Complete reference information from the community
    """

    community_idx: int
    references: References


class CommunitiesMemoryBlock(MemoryBlock):
    """Memory block for multiple community outputs.

    Aggregates outputs from processing multiple communities
    in global question answering.

    Attributes:
        communities: List of outputs from individual communities
    """

    communities: List[CommunityOutput]
