#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#


import asyncio
import logging
import time
from typing import Any, List, Optional

import dspy
import numpy as np

from topg.rag.utils import always_get_event_loop
from topg.utils.models import (
    CallbackLogger,
    InferenceNaiveMemoryBlock,
    References,
    RetrievalMemoryBlock,
)


class SimplePromptEvaluator(dspy.Signature):
    """
    # TASK
    You serve as an intelligent assistant for question answering accross single or multiple facts.
    You will be given a question and a set of collected facts.
    """

    question: str = dspy.InputField(description="The question")
    facts: str = dspy.InputField(description="Some collected facts")
    answer: str = dspy.OutputField(description="The answer")


class SimplePromptEvaluatorWithPassages(dspy.Signature):
    """
    # TASK
    You serve as an intelligent assistant for question answering accross single or multiple facts and passages.
    You will be given a question and a set of collected facts and passages.
    """

    question: str = dspy.InputField(description="The question")
    facts: str = dspy.InputField(description="Some collected facts")
    passages: str = dspy.InputField(description="Some collected passages")
    answer: str = dspy.OutputField(description="The answer")


class NaiveQAProcessor:
    """Single-shot QA: retrieve top facts and answer (optionally with passages)."""

    def __init__(self, retriever: Any, logger=None, **kwargs):
        self.retriever = retriever
        self.evaluator = dspy.ChainOfThought(SimplePromptEvaluator)
        self.evaluator_with_passages = dspy.ChainOfThought(
            SimplePromptEvaluatorWithPassages
        )
        self.logger = logger or logging.getLogger(__name__)
        self._default_retriever_args = {
            "initial_retriever_args": {
                "e_syn_k": 5,
                "e_syn_threshold": 0.80,
                "lambda_mmr": 1.0,
                "top_k": 20,
            },
            "q": 0.5,
            "damping": 0.85,
            "cosine_threshold": 0.4,
            "horizon_top_k": 500,
            "temperature": 0.1,
            "top_k": 20,
        }
        self.demos_evaluator = []

    def get_predictor(self, predictor_name: str):
        """Return a predictor by name (only 'evaluator' supported)."""
        if predictor_name == "evaluator":
            return self.evaluator.predictors()[0]
        else:
            raise ValueError(f"Unknown predictor name '{predictor_name}'")

    async def get_references_from_propositions(
        self, proposition_ids: List[str], proposition_texts: List[str]
    ) -> References:
        """
        Given a list of proposition IDs, return the associated References object
        containing entities, passages, and documents.

        :param proposition_ids: List of proposition IDs
        :return: References object with all associated data
        """
        if not proposition_ids:
            return References(
                entities_texts=[],
                entities_ids=[],
                facts_texts=[],
                facts_ids=[],
                passages_texts=[],
                passages_ids=[],
                documents_ids=[],
                documents_metadata=[],
            )

        loop = asyncio.get_running_loop()

        # Run both DB queries concurrently in thread pool
        all_passages, all_entities = await asyncio.gather(
            loop.run_in_executor(
                None,
                self.retriever.store.sqlite_db.get_passages_by_proposition_ids,
                proposition_ids,
            ),
            loop.run_in_executor(
                None,
                self.retriever.store.sqlite_db.get_entities_by_proposition_ids,
                proposition_ids,
            ),
        )

        # Extract entities
        entities_ids = list(all_entities.keys())
        entities_texts = list(all_entities.values())

        # Extract passages
        passages_ids = list(all_passages.keys())
        passages_texts = [p["page_content"] for p in all_passages.values()]

        # Extract unique document IDs
        all_documents_ids = list(set(p["doc_id"] for p in all_passages.values()))

        # Fetch documents metadata async
        documents_metadata_dict = await loop.run_in_executor(
            None,
            self.retriever.store.sqlite_db.get_documents_by_ids,
            all_documents_ids,
        )

        # Convert dict to lists
        all_documents_ids = list(documents_metadata_dict.keys())
        all_documents_metadata = list(documents_metadata_dict.values())

        return References(
            entities_texts=entities_texts,
            entities_ids=entities_ids,
            facts_texts=proposition_texts,
            facts_ids=proposition_ids,
            passages_texts=passages_texts,
            passages_ids=passages_ids,
            documents_ids=all_documents_ids,
            documents_metadata=all_documents_metadata,
        )

    async def aquery(
        self,
        question: str,
        retriever_args: dict = None,
        use_passages: bool = False,
        context_as_references: bool = False,
        callback_logger: Optional[CallbackLogger] = None,
        *args,
        **kwargs,
    ):
        """
        Async version: Given a question, return the answer.
        """
        memory = []

        if use_passages:
            context_as_references = True

        if retriever_args is None:
            retriever_args = self._default_retriever_args.copy()

        # In naive mode, the final top-k is actually the seeding top-k. So to make this easy for the user, we also overwrite here if different.
        if retriever_args["top_k"] != retriever_args["initial_retriever_args"]["top_k"]:
            retriever_args["initial_retriever_args"]["top_k"] = retriever_args["top_k"]

        # Run retrieval in executor to avoid blocking
        loop = asyncio.get_running_loop()
        retrieval = await loop.run_in_executor(
            None,
            lambda: self.retriever.retrieve(
                queries=[question],
                entities=[[]],
                exclusion_list=[[]],
                **retriever_args["initial_retriever_args"],
            ),
        )

        retrieval_block = RetrievalMemoryBlock(
            question=question,
            facts=retrieval[0]["doc_text"],
            facts_ids=retrieval[0]["doc_ids"],
            log=f"Suggest Naive Retrieval: {len(retrieval[0]['doc_text'])} facts.",
        )
        memory.append(retrieval_block)
        if callback_logger:
            callback_logger(data=retrieval_block)

        retrieval_result = "\n".join(retrieval_block.facts)

        if context_as_references:
            context = await self.get_references_from_propositions(
                proposition_ids=retrieval_block.facts_ids,
                proposition_texts=retrieval_block.facts,
            )
        else:
            context = retrieval_block.facts

        # Run evaluator in executor to avoid blocking
        if use_passages:
            predicted_answer = await loop.run_in_executor(
                None,
                lambda: self.evaluator_with_passages(
                    question=question,
                    facts=retrieval_result,
                    passages="\n".join(
                        context.passages_texts if context_as_references else []
                    ),
                    demos=self.demos_evaluator,
                ),
            )
        else:
            predicted_answer = await loop.run_in_executor(
                None,
                lambda: self.evaluator(
                    question=question,
                    facts=retrieval_result,
                    demos=self.demos_evaluator,
                ),
            )
        response = predicted_answer.answer

        inference_block = InferenceNaiveMemoryBlock(
            question=question,
            reasoning=predicted_answer.reasoning,
            response=response,
            context=context,
            log="Generated answer.",
        )
        memory.append(inference_block)
        if callback_logger:
            callback_logger(data=inference_block)

        return response, memory

    def query(
        self,
        question: str,
        retriever_args: dict = None,
        use_passages: bool = False,
        context_as_references: bool = False,
        callback_logger: Optional[CallbackLogger] = None,
        *args,
        **kwargs,
    ):
        """
        Run the query in naive mode.

        Args:
            - question (str): the user query
            - use_passages (bool, optional): Integrate the passages associated to
                the extracted propositions in the final context. (default `False`)
            - context_as_references (bool, optional): Return all metadata in the
                `context` attribute for the Inference Memory Block.  (default `False`)
            - callback_logger (callable, optional): An optional function use as a
                callback logger everytime the memory is updated. It will takes as `data`
                argument the current `MemoryBlock`.

        Return Answer string and memory blocks
        """

        loop = always_get_event_loop()
        return loop.run_until_complete(
            self.aquery(
                question=question,
                retriever_args=retriever_args,
                use_passages=use_passages,
                context_as_references=context_as_references,
                callback_logger=callback_logger,
                *args,
                **kwargs,
            )
        )
