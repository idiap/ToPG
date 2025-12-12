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
    InferenceMemoryBlock,
    References,
    RetrievalMemoryBlock,
)


class LocalPromptSelector(dspy.Signature):
    """
    You are a retriever agent that has access to a bank of facts.
    You are given a question and a list of retrieved facts formatted as an indexed list.
    Your task is to select facts as relevant facts to help the user answer the question.

    Select facts that make progress toward answering the question, even if they don't provide the complete answer. Look for facts that:
    - Answer the question directly (best case)
    - Answer part of the question or a sub-question
    - Provide relevant context or background information needed for the answer
    - Connect to entities, concepts, or relationships mentioned in the question

    Return the list of the indexes (numbers) of all the selected facts.

    """

    question: str = dspy.InputField(description="The question")
    facts: str = dspy.InputField(description="The indexed retrieved facts")
    selected_facts: List[int] = dspy.OutputField(
        description="The selected facts indexes"
    )


class LocalPromptEvaluator(dspy.Signature):
    """
    # TASK
    You are an agent specialized in complex question answering and reasoning.
    You will be given a question and a set of collected facts (potentially empty).

    # FACTS
    All the relevant facts that have been collected so far.

    # GUIDELINES
    By combining the information from the collected facts, determine if you can answer the question.

    - If YES, return is_sufficient = True and answer the question.
    - If NO, then it means you need more information from the fact bank to answer the question. Return is_sufficient = False and plan the `next questions` for collecting more facts.

    When planning `next_questions`,
    1) Identify what is missing — what do you still need to know - considering the information from the already collected facts ?
    2) What are the most relevant direction to explore ?

    Reason strategically step by step.

    When proposing a question:
        - `entity` refer to the named entity or object onto which the question will apply.
        - `question` formulates the request.
    """

    question: str = dspy.InputField(description="The question")
    facts: str = dspy.InputField(description="Some collected relevant facts")
    is_sufficient: bool = dspy.OutputField(
        description="Whether the facts are sufficient to answer the question"
    )
    answer: str = dspy.OutputField(description="The answer (if sufficient)")
    next_questions: List[str] = dspy.OutputField(
        description="Next retrieval question (if necessary)"
    )


class LocalPromptEvaluatorWithPassages(dspy.Signature):
    """
    # TASK
    You are an agent specialized in complex question answering and reasoning.
    You will be given a question and a set of collected facts and passages (potentially empty).

    # FACTS
    All the relevant facts that have been collected so far.

    # PASSAGES
    All the passages associated to the collected facts.

    # GUIDELINES
    By combining the information from the collected facts and passages, determine if you can answer the question.

    - If YES, return is_sufficient = True and answer the question.
    - If NO, then it means you need more information from the fact bank to answer the question. Return is_sufficient = False and plan the `next questions` for collecting more facts.

    When planning `next_questions`,
    1) Identify what is missing — what do you still need to know - considering the information from the already collected facts ?
    2) What are the most relevant direction to explore ?

    Reason strategically step by step.

    When proposing a question:
        - `entity` refer to the named entity or object onto which the question will apply.
        - `question` formulates the request.
    """

    question: str = dspy.InputField(description="The question")
    facts: str = dspy.InputField(description="Some collected relevant facts")
    passages: str = dspy.InputField(description="Some collected relevant passages")
    is_sufficient: bool = dspy.OutputField(
        description="Whether the facts and passages are sufficient to answer the question"
    )
    answer: str = dspy.OutputField(description="The answer (if sufficient)")
    next_questions: List[str] = dspy.OutputField(
        description="Next retrieval question (if necessary)"
    )


class LocalQAProcessor:
    """Iterative QA over the graph using the suggest/select cycle: retrieve, evaluate sufficiency, plan next steps."""

    def __init__(self, retriever: Any, logger=None, **kwargs):
        self.retriever = retriever
        self.evaluator = dspy.ChainOfThought(LocalPromptEvaluator)
        self.evaluator_with_passages = dspy.ChainOfThought(
            LocalPromptEvaluatorWithPassages
        )
        self.selector = dspy.ChainOfThought(LocalPromptSelector)
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
        """Return a predictor by name (evaluator, evaluator_with_passages, selector)."""
        if predictor_name == "evaluator":
            return self.evaluator.predictors()[0]
        elif predictor_name == "evaluator_with_passages":
            return self.evaluator_with_passages.predictors()[0]
        elif predictor_name == "selector":
            return self.selector.predictors()[0]
        else:
            raise ValueError(f"Unknown predictor name '{predictor_name}'")

    async def _process_group_async(self, question, group_retrieval):
        """
        Async version of _process_group.

        Runs the selector in a thread to avoid blocking.
        """
        retrieved_facts = group_retrieval["doc_text"]
        retrieved_facts_ids = group_retrieval["doc_ids"]

        formatted_facts = "\n".join(
            [f"{idx}: {fact}" for idx, fact in enumerate(retrieved_facts)]
        )

        # Run the potentially blocking selector in a thread
        loop = asyncio.get_running_loop()
        selected_facts_indexes = await loop.run_in_executor(
            None,
            lambda: self.selector(
                question=question, facts=formatted_facts
            ).selected_facts,
        )

        # Check that the selected facts are valid indexes
        n_facts = len(retrieved_facts)
        if any(idx < 0 or idx >= n_facts for idx in selected_facts_indexes):
            self.logger.warning(
                f"Invalid selected indexes in: {selected_facts_indexes}"
            )
            selected_facts_indexes = [
                idx for idx in selected_facts_indexes if 0 <= idx < n_facts
            ]

        selected_facts_ids = [retrieved_facts_ids[j] for j in selected_facts_indexes]
        selected_facts_text = [retrieved_facts[j] for j in selected_facts_indexes]

        return selected_facts_ids, selected_facts_text

    async def _aretrieve(
        self,
        question: str,
        seed_nodes: list[str],
        retriever_args: dict = None,
        collected_facts_ids: list = None,
        callback_logger: Optional[CallbackLogger] = None,
    ):
        """
        Retrieve relevant facts for a SINGLE question.

        Args:
            question: A single question string
            seed_nodes: List of seed node IDs for this question
            collected_facts_ids: List of already collected fact IDs to EXCLUDE (read-only)
            local_memory: Memory block list to append to
            retriever_args: Retriever configuration

        Returns:
            tuple: (selected_facts_ids, selected_facts_text, local_memory)
        """
        if not isinstance(question, str):
            raise ValueError("Question must be a single string.")

        if not retriever_args:
            retriever_args = self._default_retriever_args.copy()

        collected_facts_ids = (
            collected_facts_ids if collected_facts_ids is not None else []
        )
        local_memory = []

        output_facts_ids = []
        output_facts_texts = []

        # Build exclusion list (don't modify collected_facts_ids)
        exclusion_list = list(collected_facts_ids)

        seed_nodes = list(seed_nodes) if seed_nodes else []

        if not seed_nodes:
            self.logger.info(
                f"No seed nodes provided for question: '{question}', performing initial seeding"
            )

            try:
                initial_retrieval = self.retriever.retrieve(
                    queries=[question],
                    entities=[[]],
                    exclusion_list=[[]],
                    **retriever_args["initial_retriever_args"],
                )[0]
            except Exception as e:
                self.logger.error(f"Initial retrieval failed: {e}")
                return [], [], local_memory

            seed_facts_ids, seed_facts_text = await self._process_group_async(
                question=question,
                group_retrieval=initial_retrieval,
            )
            memory_block = RetrievalMemoryBlock(
                question=question,
                facts=seed_facts_text,
                facts_ids=seed_facts_ids,
                log=f"Seeding in the graph with {len(seed_facts_ids)} facts.",
            )
            local_memory.append(memory_block)
            if callback_logger:
                callback_logger(
                    data=memory_block,
                )

            output_facts_ids.extend(seed_facts_ids)
            output_facts_texts.extend(seed_facts_text)
            seed_nodes = seed_facts_ids

            # Update exclusion list for walker (local copy)
            exclusion_list.extend(seed_facts_ids)

        # Get query embedding
        query_embedding = np.array(
            self.retriever.store.document_store._embedding_fn([question])
        )[0]

        # Run walker - This is the suggest step
        if not seed_nodes:
            self.logger.warning(
                f"No seed nodes available after seeding for question: '{question}'. Skipping Suggest."
            )
            return output_facts_ids, output_facts_texts, local_memory

        # Ok, if we have some seed nodes, then we run the walkers
        try:
            local_retrieval = self.retriever.suggest(
                query_embeddings=np.array([query_embedding]),
                seed_nodes=[seed_nodes],
                exclusion_list=exclusion_list,
                max_workers=None,
                groups=np.array([0]),
                **retriever_args,
            )[0]
        except Exception as e:
            self.logger.error(f"Walker failed: {e}")
            return output_facts_ids, output_facts_texts, local_memory

        # This is the select step
        selected_facts_ids, selected_facts_text = await self._process_group_async(
            question=question,
            group_retrieval=local_retrieval,
        )
        memory_block = RetrievalMemoryBlock(
            question=question,
            facts=selected_facts_text,
            facts_ids=selected_facts_ids,
            log=f"Suggest step: {len(selected_facts_ids)} new facts.",
        )
        local_memory.append(memory_block)
        if callback_logger:
            callback_logger(
                data=memory_block,
            )

        output_facts_ids.extend(selected_facts_ids)
        output_facts_texts.extend(selected_facts_text)

        return output_facts_ids, output_facts_texts, local_memory

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
        max_iter: int = 2,
        retriever_args: dict = None,
        use_passages: bool = False,
        context_as_references: bool = False,
        callback_logger: Optional[CallbackLogger] = None,
        *args,
        **kwargs,
    ):
        """
        Given a question, return the answer using iterative retrieval.
        """
        memory = []
        collected_facts_ids = set()

        # If we are going to use the passages, we necessarily going to need this.
        if use_passages:
            context_as_references = True

        if (
            self.retriever.store.g_full is None
            or len(self.retriever.store.g_full.nodes) == 0
        ):
            self.logger.warning("The graph is empty. Cannot process the query.")
            return "I don't know", memory

        original_question = question
        questions = [question]
        is_sufficient = False
        iteration = 0
        answer = ""
        seed_nodes = []

        while not is_sufficient and iteration < max_iter:
            self.logger.debug(
                f"Iteration {iteration + 1}: Processing {len(questions)} question(s)"
            )

            self.logger.debug("Calling the retriever Suggest/Select...")

            tasks = [
                self._aretrieve(
                    question=questions[i],
                    seed_nodes=seed_nodes,
                    collected_facts_ids=list(collected_facts_ids),
                    retriever_args=retriever_args,
                    callback_logger=callback_logger,
                )
                for i in range(len(questions))
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            output_facts_ids = []
            output_facts_texts = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error retrieving for question {i}: {result}")
                    output_facts_ids.append([])
                    output_facts_texts.append([])
                else:
                    facts_ids, facts_texts, local_memory = result
                    memory.extend(local_memory)
                    output_facts_ids.append(facts_ids)
                    output_facts_texts.append(facts_texts)
                    self.logger.debug(f"Question {i}: Retrieved {len(facts_ids)} facts")

            # Collect unique facts while preserving order
            facts_dict = {}

            for block in memory:
                if isinstance(block, RetrievalMemoryBlock):
                    for fact_id, fact in zip(block.facts_ids, block.facts):
                        if fact_id not in facts_dict:
                            facts_dict[fact_id] = fact

            prompt_facts_ids = list(facts_dict.keys())
            prompt_facts = list(facts_dict.values())

            new_collected_facts_ids = set().union(
                *[set(facts) for facts in output_facts_ids]
            )
            collected_facts_ids.update(new_collected_facts_ids)

            self.logger.debug("Calling the evaluator...")

            # Did we ask for the references / or do we use the passages
            if context_as_references:
                context = await self.get_references_from_propositions(
                    proposition_ids=prompt_facts_ids,
                    proposition_texts=prompt_facts,
                )
            else:
                context = prompt_facts

            loop = asyncio.get_running_loop()
            # Now we run the evaluator w/ or w/o passages depending on the flag
            if use_passages:
                evaluator_response = await loop.run_in_executor(
                    None,
                    lambda: self.evaluator_with_passages(
                        question=original_question,
                        facts="\n".join(prompt_facts),
                        passages="\n".join(
                            context.passages_texts if context_as_references else []
                        ),
                        demos=self.demos_evaluator,
                    ),
                )
            else:
                evaluator_response = await loop.run_in_executor(
                    None,
                    lambda: self.evaluator(
                        question=original_question,
                        facts="\n".join(prompt_facts),
                        demos=self.demos_evaluator,
                    ),
                )

            is_sufficient = evaluator_response.is_sufficient
            reasoning = evaluator_response.reasoning
            questions = evaluator_response.next_questions
            answer = evaluator_response.answer

            self.logger.debug(
                f"Evaluator: is_sufficient={is_sufficient}, "
                f"next_questions={len(questions) if not is_sufficient else 'N/A'}"
            )

            memory_block = InferenceMemoryBlock(
                question=original_question,
                context=context,
                reasoning=reasoning,
                response=answer,
                is_sufficient=is_sufficient,
                next_questions=questions if not is_sufficient else None,
                log=f"Evaluator decided is_sufficient={is_sufficient}.",
            )
            memory.append(memory_block)

            self.logger.debug(f"Iteration {iteration}: New questions: {questions}")
            if callback_logger:
                callback_logger(data=memory_block)

            if is_sufficient:
                return answer, memory

            seed_nodes = list(new_collected_facts_ids)
            iteration += 1

        return "I don't know", memory

    def _retrieve(
        self,
        question: str,
        seed_nodes: list[str],
        collected_facts_ids: list = None,
        retriever_args: dict = None,
        callback_logger: Optional[CallbackLogger] = None,
    ):
        loop = always_get_event_loop()
        return loop.run_until_complete(
            self._aretrieve(
                question=question,
                seed_nodes=seed_nodes,
                collected_facts_ids=collected_facts_ids,
                retriever_args=retriever_args,
                callback_logger=callback_logger,
            )
        )

    def query(
        self,
        question: str,
        max_iter: int = 2,
        retriever_args: dict = None,
        use_passages: bool = False,
        context_as_references: bool = False,
        callback_logger: Optional[CallbackLogger] = None,
    ):
        """
        Run the query in local mode.
        Basically, we starting from the query, we use a naive retrieval to seed with some
        propositions in the graph. Then we enter the suggest/select cycle. At the end of
        each cycle we evaluate if we have enought information to answer the question, if not,
        we propose new queries to continue the cycle with.

        Args:
            - question (str): the user query
            - max_iter (int, optional): Max reasoning steps (default: 2).
            - use_passages (bool, optional): Integrate the passages associated
                to the extracted propositions in the final context. (default `False`)
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
                max_iter=max_iter,
                retriever_args=retriever_args,
                use_passages=use_passages,
                context_as_references=context_as_references,
                callback_logger=callback_logger,
            )
        )
