#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#


import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import dspy
import numpy as np
import tiktoken
from sklearn.cluster import AgglomerativeClustering

from topg.rag.utils import always_get_event_loop
from topg.utils.models import (
    CallbackLogger,
    CommunitiesMemoryBlock,
    CommunityBudgetMemoryBlock,
    CommunityOutput,
    DecompositionMemoryBlock,
    GlobalRetrievalMemoryBlock,
    IndexedKeyPoint,
    IntermediaryResponsesMemoryBlock,
    References,
    ResponseMemoryBlock,
    SelectorReasoning,
    SingleRetrievalMemoryBlock,
)


class DecompositionPrompt(dspy.Signature):
    """
    You are an expert question decomposition assistant. Your task is to take an abstract or complex input question and break it down into a set of smaller, precise subquestions.
    Instructions: Decompose the input into N meaningful subquestions (where N is specified by the user).
    Each subquestion must be:
     * Self-contained and unambiguous (stands alone without needing the original question).
     * Focused on one distinct aspect or dimension of the original question.
     * Non-overlapping: avoid redundancy between subquestions.
     * Together, the subquestions should provide a comprehensive, multi-dimensional exploration of the original question.
    """

    question: str = dspy.InputField(description="The input question")
    N: int = dspy.InputField(description="The number of subquestions to generate")
    sub_questions: List[str] = dspy.OutputField(description="The list of subquestions")


class CommunityAnswerPrompt(dspy.Signature):
    """
    You are a helpful assistant responding to questions about content from the provided data reports.

    Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information from the input data reports.

    Each data report is composed of :
    - A set of core entities / topics related to the report
    - A set of propositions (facts) associated with the entities
    - A set of passages from the original documents where the propositions were extracted from

    You should use the data provided by the Reports below as the primary context for generating the response.
    If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

    Each key point in the response should have the following element:
    - Description: A comprehensive description of the point.
    - Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

    Avoid redundancy between key points.
    Do not include information where the supporting evidence for it is not provided.

    Points supported by reports should list the relevant reports IDs in references.
    Key points must be detailed, comprehensive, diverse and should include examples if possible.
    The response should be JSON formatted as follows:
    {{
        "keypoints": [
            {"description": "Comprehensive and detailed description of point 1", "score": score_value, references: [3, 6]},
            {"description": "Comprehensive and detailed description of point 2", "score": score_value,  references: [1]},
            ...
        ]
    }}
    An example:
    {"description": "Person X is the owner of Company Y and subject to many allegations of wrongdoing because [other details...]. He is also CEO of company X.", "score": 85, "references": [3, 6]}

    """

    question: str = dspy.InputField(description="The input question")
    reports: str = dspy.InputField(description="The reports")
    keypoints: List[IndexedKeyPoint] = dspy.OutputField(
        description="The list of key points in JSON format"
    )


class GlobalResponsePrompt(dspy.Signature):
    """
    You are a helpful assistant responding to questions by synthesizing perspectives from multiple experts.
    Your task is to generate a comprehensive response that address the user’s question by summarizing all the insights collected from multiple experts and analysts who focused on different dimensions of the question.

    # Requirements
    - **Faithfulness:** Use only the information provided by the experts. If it lacks sufficient information to fully answer the question, clearly state that. Do not fabricate or infer unsupported facts.
    - **Comprehensiveness:** The final response should remove all irrelevant information from the reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response format. Integrate relevant details and examples from multiple experts' inshights to enrich the answer.
    - **Diversity:** Ensure the response captures the range of perspectives, contexts, conditions, examples, etc. represented in the expert insights.
    - **Integration:** Highlight complementary and contrasting viewpoints across reports. Show how different facets or dimensions connect into a coherent whole. Add sections and commentary to the response as appropriate for the response format.
    - **Actionability:** Where possible, translate findings into implications, recommendations, or next steps that aid decision-making.
    - **Style:**  Style the response in markdown.
    - **References:** The response should also preserve all the data references previously included as follows: "This is an example sentence supported by data references (Reports: 5, 9)". Do not list more than 5 record ids in a single reference, and add '+more' if necessary. Also, do not mention explicitly the role of the 'experts' in the analysis process.
    """

    question: str = dspy.InputField(description="The input question")
    response_format: str = dspy.InputField(
        description="The desired response type (e.g., 'multiple paragraph report')"
    )
    expert_reports: str = dspy.InputField(
        description="The expert reports to use to answer the question"
    )
    response: str = dspy.OutputField(description="The response")


class ResponseRefinement(dspy.Signature):
    """
    You are a helpful assistant tasked with refining and enhancing a synthesized expert response.

    You are given:
    - The original **question**.
    - The **context** and **expert reports** that informed the initial synthesis.
    - The **initial synthesized response** produced from those reports.

    Your task is to refine the response to maximize accuracy, completeness, and clarity, while preserving its grounding in the provided reports.

    # Refinement Objectives
    - **Faithfulness:** Do NOT introduce any new facts or interpretations not supported by the expert reports. If details are missing, clearly acknowledge the gaps.
    - **Precision:** Strengthen factual accuracy, tighten definitions, and clarify ambiguous statements.
    - **Completeness:** Identify missing aspects or expert insights that were overlooked or insufficiently explained in the initial response, and integrate them appropriately.
    - **Integration:** Improve flow and logic; better connect complementary or contrasting viewpoints among experts.
    - **Depth:** Add nuanced reasoning, implications, or examples drawn from the reports to enrich understanding.
    - **Clarity:** Improve readability, structure, and coherence while keeping a professional and analytical tone.
    - **References:** Maintain the original referencing format — e.g., (Reports: 2, 4, 7). Add references when you incorporate new evidence from the provided reports, but do not exceed 5 record IDs per reference (add '+more' if necessary). Also, do not mention explicitly the role of the 'experts' in the analysis process.
    - **Style:** Present the refined answer in well-structured markdown.

    # Output
    Return the improved, polished version of the response that better fulfills the above objectives.
    """

    question: str = dspy.InputField(description="The input question")
    response_format: str = dspy.InputField(
        description="The desired response type (e.g., 'multiple paragraph report')"
    )
    initial_response: str = dspy.InputField(description="The initial response")
    expert_reports: str = dspy.InputField(
        description="The expert reports to use to answer the question"
    )
    refined_response: str = dspy.OutputField(description="The response")


class GlobalPromptSelector(dspy.Signature):
    """
    You are an assistant that selects facts to maximize diversity of perspectives about a given question.

    At each step, you are given:
    - A question,
    - A set of previously selected facts,
    - A new list of candidate facts.

    Your task is to output the indexes of the candidate facts that should be added.
    If none add new diversity, return [].

    # Selection Goal
    Select candidate facts that introduce **new dimensions, contexts, or perspectives** not yet represented in the previously selected facts.
    The aim is to expand the conceptual and informational space of the answer.

    Do not select facts that:
    - Repeat or closely overlap with the ideas or contexts of previously selected facts.
    - Add only surface-level details without introducing a new conceptual or contextual dimension.
    - Are vague or unsupported by factual content.

    Output only the indexes of the candidate facts that should be added.
    """

    question: str = dspy.InputField(description="The original abstract question")
    current_facts: str = dspy.InputField(
        description="The existing abstract answer to refine"
    )
    new_facts: str = dspy.InputField(description="Indexed list of new facts")
    selected_facts: List[int] = dspy.OutputField(
        description="Indexes of the new facts used in the update"
    )


CallbackLogger = Callable[[dict], None]


class GlobalQAProcessor:
    """
    A retrieval system that uses community detection and question decomposition
    to find relevant information clusters from a knowledge graph.
    """

    def __init__(
        self,
        retriever,
        logger=None,
    ):
        """
        Initialize the CommunityRetriever.

        Args:
            retriever: The system retriever object with g_full graph
            max_cluster_size: Maximum cluster size for Leiden algorithm
        """
        self.retriever = retriever

        # logger
        self.logger = logger or logging.getLogger(__name__)

        # Initialize DSPy modules
        self.decomposer = dspy.ChainOfThought(DecompositionPrompt)
        self.selector = dspy.ChainOfThought(GlobalPromptSelector)
        self.community_answer = dspy.ChainOfThought(CommunityAnswerPrompt)
        self.response_generator = dspy.Predict(GlobalResponsePrompt)
        self.response_refinement = dspy.Predict(ResponseRefinement)

        # Control tokens
        self.tokenizer = tiktoken.get_encoding("gpt2")

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

    def get_predictor(self, predictor_name: str):
        """Return a predictor by name (decomposer, selector, community_answer, response_generator, response_refinement)."""
        if predictor_name == "decomposer":
            return self.decomposer.predictors()[0]
        elif predictor_name == "selector":
            return self.selector.predictors()[0]
        elif predictor_name == "community_answer":
            return self.community_answer.predictors()[0]
        elif predictor_name == "response_generator":
            return self.response_generator
        elif predictor_name == "response_refinement":
            return self.response_refinement
        else:
            raise ValueError(f"Unknown predictor name '{predictor_name}'")

    async def _aprocess_group(self, question, current_facts, group_retrieval):
        """
        Select diverse facts for one group.
        This is the `select` step for the global mode.
        """
        retrieved_facts = group_retrieval["doc_text"]
        formatted_facts = "\n".join(
            [f"{idx}: {fact}" for idx, fact in enumerate(retrieved_facts)]
        )
        formated_current_facts = "\n".join([f"- {fact}" for fact in current_facts])

        loop = asyncio.get_running_loop()
        selector_output = await loop.run_in_executor(
            None,
            lambda: self.selector(
                question=question,
                current_facts=formated_current_facts,
                new_facts=formatted_facts,
            ),
        )

        selected_facts_indexes = selector_output.selected_facts
        selector_reasoning = selector_output.reasoning

        # Check that they are valid indexes
        n_facts = len(retrieved_facts)
        if any(idx < 0 or idx >= n_facts for idx in selected_facts_indexes):
            selected_facts_indexes = [
                idx for idx in selected_facts_indexes if 0 <= idx < n_facts
            ]

        return selected_facts_indexes, SelectorReasoning(
            proposed_facts=formatted_facts, reasoning=selector_reasoning
        )

    async def _acall_selectors(
        self,
        question,
        groups,
        all_seed_text_by_group,
        retrieval_output,
        node_id2text: dict,
    ):
        """Run selection across groups concurrently and collect chosen facts."""
        n_groups = len(np.unique(groups))
        all_selected_facts_indexes = [None] * n_groups
        all_selectors_reasonings = [None] * n_groups
        all_selected_facts_ids = [None] * n_groups
        all_selected_facts_text = [None] * n_groups
        all_selected_facts_pids = [None] * n_groups

        tasks = []
        for i, group_retrieval in enumerate(retrieval_output):
            task = self._aprocess_group(
                question,
                all_seed_text_by_group[i],
                group_retrieval,
            )
            tasks.append((i, task))

        results = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        for (i, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing group {i}: {result}")
                all_selected_facts_indexes[i] = []
                all_selectors_reasonings[i] = SelectorReasoning(
                    proposed_facts="", reasoning=""
                )
                all_selected_facts_ids[i] = []
                all_selected_facts_text[i] = []
                all_selected_facts_pids[i] = []
                continue

            selected_facts_text_indexes, selector_reasoning = result

            all_selected_facts_indexes[i] = selected_facts_text_indexes
            all_selectors_reasonings[i] = selector_reasoning
            all_selected_facts_ids[i] = [
                retrieval_output[i]["doc_ids"][j] for j in selected_facts_text_indexes
            ]
            all_selected_facts_text[i] = [
                retrieval_output[i]["doc_text"][j] for j in selected_facts_text_indexes
            ]
            all_selected_facts_pids[i] = [
                retrieval_output[i]["doc_passage_ids"][j]
                for j in selected_facts_text_indexes
            ]

        new_node_id2text = dict(
            zip(
                [_id for ids in all_selected_facts_ids for _id in ids],
                [text for texts in all_selected_facts_text for text in texts],
            )
        )
        node_id2text.update(new_node_id2text)

        return (
            all_selected_facts_indexes,
            all_selected_facts_text,
            all_selected_facts_ids,
            all_selectors_reasonings,
        )

    async def _aprocess_query_iteration(
        self,
        group_selected_indexes,
        grouped_retrieval,
        query_embeddings,
        groups,
        collected_seeds_ids: set,
        m=1,
        alpha=1.0,
        beta=0.7,
        gamma=0.15,
    ):
        """Async version of process_query_iteration"""
        n_groups = len(np.unique(groups))
        assert len(group_selected_indexes) == n_groups
        assert len(grouped_retrieval) == n_groups
        assert query_embeddings.shape[0] == len(groups)

        retrieval_doc_map = {}

        for k, g_retrieval in enumerate(grouped_retrieval):
            g_selected_idx = group_selected_indexes[k]
            n = len(g_retrieval["doc_ids"])
            g_excluded_facts_indexes = [
                idx for idx in range(n) if idx not in g_selected_idx
            ]
            g_excluded_facts_embeddings = np.array(
                [g_retrieval["doc_embeddings"][idx] for idx in g_excluded_facts_indexes]
            )
            if len(g_excluded_facts_embeddings) == 0:
                g_excluded_facts_embeddings = np.zeros_like(
                    g_retrieval["doc_embeddings"][0]
                )
            else:
                g_excluded_facts_embeddings = np.mean(
                    g_excluded_facts_embeddings, axis=0
                )

            for idx in g_selected_idx:
                doc_id = g_retrieval["doc_ids"][idx]
                if not doc_id in retrieval_doc_map:
                    retrieval_doc_map[doc_id] = {}
                    retrieval_doc_map[doc_id]["doc_text"] = g_retrieval["doc_text"][idx]
                    retrieval_doc_map[doc_id]["doc_embeddings"] = g_retrieval[
                        "doc_embeddings"
                    ][idx]

                g_attributed_walker = g_retrieval["attributed_walkers"][idx]
                attributed_walker = np.where(groups == k)[0][g_attributed_walker]
                q_emb = query_embeddings[attributed_walker]
                if "query_embedding" not in retrieval_doc_map[doc_id]:
                    retrieval_doc_map[doc_id]["query_embedding"] = []
                    retrieval_doc_map[doc_id]["query_embedding"].append(q_emb)
                else:
                    retrieval_doc_map[doc_id]["query_embedding"].append(q_emb)

                if "excluded_facts_embeddings" not in retrieval_doc_map[doc_id]:
                    retrieval_doc_map[doc_id]["excluded_facts_embeddings"] = []
                    retrieval_doc_map[doc_id]["excluded_facts_embeddings"].append(
                        g_excluded_facts_embeddings
                    )
                else:
                    retrieval_doc_map[doc_id]["excluded_facts_embeddings"].append(
                        g_excluded_facts_embeddings
                    )

        all_new_seed_nodes = list(retrieval_doc_map.keys())
        collected_seeds_ids.update(all_new_seed_nodes)

        n_seeds = len(all_new_seed_nodes)
        _query_embeddings = np.array(
            [
                np.array(retrieval_doc_map[doc_id]["query_embedding"]).mean(axis=0)
                for doc_id in all_new_seed_nodes
            ]
        )
        selected_doc_embeddings = np.array(
            [
                retrieval_doc_map[doc_id]["doc_embeddings"]
                for doc_id in all_new_seed_nodes
            ]
        )
        excluded_facts_embeddings = np.array(
            [
                np.array(retrieval_doc_map[doc_id]["excluded_facts_embeddings"]).mean(
                    axis=0
                )
                for doc_id in all_new_seed_nodes
            ]
        )

        query_embeddings = (
            alpha * _query_embeddings
            + beta * selected_doc_embeddings
            - gamma * excluded_facts_embeddings
        )

        groups = np.array([0])
        if len(all_new_seed_nodes) > 1:
            model = AgglomerativeClustering(
                n_clusters=min(m, len(all_new_seed_nodes)),
                metric="cosine",
                linkage="average",
            )
            groups = model.fit_predict(selected_doc_embeddings)
        else:
            self.logger.debug("Only one seed node, no clustering performed.")

        seed_nodes = [[all_new_seed_nodes[i]] for i in range(len(all_new_seed_nodes))]

        return query_embeddings, seed_nodes, groups

    def decompose_question(self, question: str, n_subquestions: int = 10) -> List[str]:
        """
        Decompose a complex question into sub-questions.

        Args:
            question: The main question to decompose
            n_subquestions: Number of sub-questions to generate

        Returns:
            List of sub-questions
        """
        initial_decomposition = self.decomposer(question=question, N=n_subquestions)
        sub_questions = initial_decomposition.sub_questions[:n_subquestions]
        return sub_questions

    async def _aretrieve_and_filter(
        self,
        question: str,
        sub_questions: List[str],
        retriever_args: Dict,
        memory: List[Any],
        max_iter=5,
        min_number_of_facts=100,
        m=10,
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        callback_logger: Optional[CallbackLogger] = None,
    ):
        """Async version of retrieve_and_filter"""
        # Local state
        collected_seeds_ids = set()
        node_id2text = {}
        node_id2pid = {}

        n_sub_questions = len(sub_questions)

        # Initial retrieval
        loop = asyncio.get_running_loop()
        initial_retrievals = await loop.run_in_executor(
            None,
            lambda: self.retriever.retrieve(
                queries=sub_questions,
                entities=[[]] * n_sub_questions,
                exclusion_list=[[]] * n_sub_questions,
                **retriever_args["initial_retriever_args"],
            ),
        )

        for i in range(n_sub_questions):
            initial_retrievals[i]["attributed_walkers"] = np.array(
                [0] * len(initial_retrievals[i]["doc_ids"])
            )

        n_groups = n_sub_questions
        groups = np.array(range(n_groups))

        query_embeddings = np.array(
            [retrieval["query_embedding"] for retrieval in initial_retrievals]
        )

        all_seed_text_by_group = [[] for _ in range(n_groups)]
        (
            all_selected_facts_indexes,
            all_selected_facts_text,
            all_selected_facts_ids,
            all_selectors_reasonings,
        ) = await self._acall_selectors(
            question=question,
            groups=groups,
            all_seed_text_by_group=all_seed_text_by_group,
            retrieval_output=initial_retrievals,
            node_id2text=node_id2text,
        )

        if not len(all_selected_facts_ids) or all(
            len(facts) == 0 for facts in all_selected_facts_ids
        ):
            self.logger.warning("No facts selected in the first iteration.")
            return collected_seeds_ids, memory, node_id2text, node_id2pid

        all_retrievals = [
            SingleRetrievalMemoryBlock(
                reasoning=all_selectors_reasonings[i].reasoning,
                facts=all_selected_facts_text[i],
                facts_ids=all_selected_facts_ids[i],
                seed_nodes_ids=[],
                seed_nodes_texts=[],
            )
            for i in range(n_groups)
        ]
        memory_block = GlobalRetrievalMemoryBlock(
            iteration=0, retrievals=all_retrievals, log="Initial retrieval completed."
        )
        memory.append(memory_block)
        if callback_logger:
            callback_logger(data=memory_block)

        query_embeddings, seed_nodes, groups = await self._aprocess_query_iteration(
            group_selected_indexes=all_selected_facts_indexes,
            grouped_retrieval=initial_retrievals,
            query_embeddings=query_embeddings,
            groups=groups,
            collected_seeds_ids=collected_seeds_ids,
            m=m,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        iteration = 0

        while iteration < max_iter and len(collected_seeds_ids) < min_number_of_facts:
            self.logger.info(f"--- Iteration {iteration} ---")
            self.logger.debug(f"Run walkers with {len(seed_nodes)} seed nodes...")

            global_retrieval_output = await loop.run_in_executor(
                None,
                lambda: self.retriever.suggest(
                    query_embeddings=query_embeddings,
                    seed_nodes=seed_nodes,
                    exclusion_list=list(collected_seeds_ids),
                    max_workers=None,
                    groups=groups,
                    **retriever_args,
                ),
            )

            n_groups = len(np.unique(groups))

            all_seed_text_by_group = [[] for _ in range(n_groups)]
            for i, g in enumerate(groups):
                all_seed_text_by_group[g].append(node_id2text[seed_nodes[i][0]])

            (
                all_selected_facts_indexes,
                all_selected_facts_text,
                all_selected_facts_ids,
                all_selectors_reasonings,
            ) = await self._acall_selectors(
                question=question,
                groups=groups,
                all_seed_text_by_group=all_seed_text_by_group,
                retrieval_output=global_retrieval_output,
                node_id2text=node_id2text,
            )

            if not len(all_selected_facts_ids) or all(
                len(facts) == 0 for facts in all_selected_facts_ids
            ):
                self.logger.warning("No facts selected, stopping the iterations.")
                break

            all_retrievals = [
                SingleRetrievalMemoryBlock(
                    reasoning=all_selectors_reasonings[i].reasoning,
                    facts=all_selected_facts_text[i],
                    facts_ids=all_selected_facts_ids[i],
                    seed_nodes_ids=[seed_nodes[j][0] for j in np.where(groups == i)[0]],
                    seed_nodes_texts=all_seed_text_by_group[i],
                )
                for i in range(n_groups)
            ]
            memory_block = GlobalRetrievalMemoryBlock(
                iteration=iteration + 1,
                retrievals=all_retrievals,
                log=f"Iteration {iteration + 1} completed.",
            )
            memory.append(memory_block)
            if callback_logger:
                callback_logger(data=memory_block)

            self.logger.debug("Processing the next queries ...")
            query_embeddings, seed_nodes, groups = await self._aprocess_query_iteration(
                group_selected_indexes=all_selected_facts_indexes,
                grouped_retrieval=global_retrieval_output,
                query_embeddings=query_embeddings,
                groups=groups,
                collected_seeds_ids=collected_seeds_ids,
                m=m,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            self.logger.debug(
                f"Number of collected seed facts ids: {len(collected_seeds_ids)}"
            )
            iteration += 1

        return collected_seeds_ids, memory, node_id2text, node_id2pid

    def get_communities_passages(
        self, communities, passages, node_id2pid, passage_token_limit
    ):
        all_community_passages_text = []
        all_community_passages_ids = []
        for community_nodes in communities:
            passages2score = defaultdict(float)
            for nid in community_nodes:
                pid = node_id2pid[nid]
                passages2score[pid] += 1

            passages_lens = {
                pid: len(self.tokenizer.encode(passages[pid]["page_content"]))
                for pid in passages2score
            }

            for pid in passages2score:
                passages2score[pid] = passages2score[pid] / np.log(passages_lens[pid])

            # Sort passages by score
            sorted_passages = sorted(
                passages2score.items(), key=lambda x: x[1], reverse=True
            )
            selected_passage_texts = []
            selected_passage_ids = []
            max_len = 0
            for pid, score in sorted_passages:
                passage_len = passages_lens[pid]
                if max_len + passage_len < passage_token_limit:
                    selected_passage_texts.append(passages[pid]["page_content"])
                    selected_passage_ids.append(pid)
                    max_len += passage_len
                else:
                    break

            # add to the final list
            all_community_passages_text.append(selected_passage_texts)
            all_community_passages_ids.append(selected_passage_ids)

        return all_community_passages_ids, all_community_passages_text

    def _budgeted_cover(
        self,
        seeds: Set,
        budget: int,
        min_community_size=10,
        max_community_size=100,
    ) -> Tuple[List[int], Set, int]:
        """
        Find communities that cover seeds within budget using greedy algorithm.

        Args:
            seeds: Set of seed node IDs
            budget: Maximum total size of communities

        Returns:
            Tuple of (chosen_communities, covered_seeds, budget_used)
        """
        self.logger.debug("Starting budgeted cover with budget %d", budget)
        self.logger.debug("Number of seeds: %d", len(seeds))
        self.logger.debug(
            f"minimum community size: {min_community_size} and maximum community size: {max_community_size}"
        )

        cm = self.retriever.store.community_manager

        def ancestors(c):
            path = []
            while c in cm.parent_mapping:
                c = cm.parent_mapping[c]
                path.append(c)
            return path

        # Step 1: Get leaf communities
        leaf_comms = {cm.node2community[s][-1] for s in seeds}

        # Step 2: Build candidate communities
        candidates = set()
        for leaf in leaf_comms:
            if cm.community_sizes[leaf] <= budget and (
                cm.community_sizes[leaf] >= min_community_size
                and cm.community_sizes[leaf] <= max_community_size
            ):
                candidates.add(leaf)
            for anc in ancestors(leaf):
                if cm.community_sizes[anc] <= budget and (
                    cm.community_sizes[anc] >= min_community_size
                    and cm.community_sizes[anc] <= max_community_size
                ):
                    candidates.add(anc)

        # Step 3: Map seeds to communities
        seeds_in = defaultdict(set)
        for s in seeds:
            comm_path = cm.node2community[s]
            for c in comm_path:
                if c in candidates:
                    seeds_in[c].add(s)

        # Greedy selection
        covered = set()
        chosen = []
        budget_used = 0

        while covered != seeds and candidates:
            scored = []
            for c in list(candidates):
                uncovered_in_c = seeds_in[c] - covered
                if uncovered_in_c:
                    score = len(uncovered_in_c) / cm.community_sizes[c]
                    scored.append((score, c, uncovered_in_c))

            if not scored:
                break

            scored.sort(reverse=True)
            _, best, uncovered_in_best = scored[0]

            if budget_used + cm.community_sizes[best] <= budget:
                chosen.append(best)
                budget_used += cm.community_sizes[best]
                covered |= uncovered_in_best
                candidates.remove(best)
                for ch in cm.children.get(best, []):
                    if ch in candidates:
                        candidates.remove(ch)
            else:
                candidates.remove(best)
                candidates.update(
                    child
                    for child in cm.children.get(best, [])
                    if (
                        cm.community_sizes[child] >= min_community_size
                        and cm.community_sizes[child] <= max_community_size
                    )
                )

        percent_coverage = (len(covered) / len(seeds)) if seeds else 0.0
        budget_consumed_ratio = (budget_used / budget) if budget > 0 else 0.0
        self.logger.info(f"percentage of covered seeds: {percent_coverage:.2%}")

        self.logger.info(
            f"Budget used: {budget_used} ({budget_consumed_ratio:.2%} of total budget)"
        )

        # get the uncovered seeds
        uncovered_seeds = seeds - covered

        return chosen, covered, budget_used, uncovered_seeds

    async def _post_process_communities(
        self,
        chosen_communities: List[int],
        passage_token_limit=500,
    ) -> List[List]:
        cm = self.retriever.store.community_manager

        # Merge small communities
        final_communities_nodes = [
            cm.community2node[community_id] for community_id in chosen_communities
        ]

        # Step 5: Extract propositions and entities
        communities_propositions_nodes = []
        communities_entities_nodes = []
        list_of_propositions_nodes = []
        list_of_entities_nodes = []

        for c_idx, c_nodes in enumerate(final_communities_nodes):
            communities_propositions = [
                node for node in c_nodes if node in cm.propositions_nodes
            ]
            communities_entities = [
                node for node in c_nodes if node in cm.entities_nodes
            ]
            list_of_propositions_nodes.extend(communities_propositions)
            list_of_entities_nodes.extend(communities_entities)

            communities_propositions_nodes.append(communities_propositions)
            communities_entities_nodes.append(communities_entities)

        list_of_propositions_nodes = list(set(list_of_propositions_nodes))
        list_of_entities_nodes = list(set(list_of_entities_nodes))

        # Step 6: Fetch data from stores
        propositions_data = self.retriever.store.document_store.get(
            ids=list_of_propositions_nodes, include=["documents", "metadatas"]
        )
        propositions_id2text = dict(
            zip(propositions_data["ids"], propositions_data["documents"])
        )

        entities_data = self.retriever.store.entity_store.get(
            ids=list_of_entities_nodes, include=["documents"]
        )
        entities_id2text = dict(zip(entities_data["ids"], entities_data["documents"]))

        node_id2pid = dict(
            zip(
                propositions_data["ids"],
                [md["passage_id"] for md in propositions_data["metadatas"]],
            )
        )
        all_passages_ids = list(
            set(metadata["passage_id"] for metadata in propositions_data["metadatas"])
        )

        # Query the database and prepare the passages
        loop = asyncio.get_running_loop()
        passages = await loop.run_in_executor(
            None,
            lambda: self.retriever.store.sqlite_db.get_passages_by_ids(
                passage_ids=all_passages_ids
            ),
        )
        passages = {passage.pop("passage_id"): passage for passage in passages}

        community_selected_passages_ids, community_selected_passages_text = (
            self.get_communities_passages(
                communities_propositions_nodes,
                passages,
                node_id2pid,
                passage_token_limit=passage_token_limit,
            )
        )

        # For metadata purposes, traces the documents
        passage_id2doc_id = {
            passage_id: passage["doc_id"] for passage_id, passage in passages.items()
        }

        all_documents_ids = list(
            set(passage["doc_id"] for passage in passages.values())
        )

        documents_metadata_dict = await loop.run_in_executor(
            None,
            self.retriever.store.sqlite_db.get_documents_by_ids,
            all_documents_ids,
        )
        all_community_passages_ids = [
            list(
                set(
                    [
                        node_id2pid[node_id]
                        for node_id in communities_propositions_nodes[i]
                    ]
                )
            )
            for i in range(len(communities_propositions_nodes))
        ]
        all_community_documents_ids = [
            list(set([passage_id2doc_id[pid] for pid in all_community_passages_ids[i]]))
            for i in range(len(communities_propositions_nodes))
        ]

        # Step 7: Build final communities
        list_of_communities = []

        for i in range(len(final_communities_nodes)):
            # Prepare the community-reference content
            refs = References(
                entities_texts=[
                    entities_id2text[node] for node in communities_entities_nodes[i]
                ],
                entities_ids=[node for node in communities_entities_nodes[i]],
                facts_texts=[
                    propositions_id2text[node]
                    for node in communities_propositions_nodes[i]
                ],
                facts_ids=[node for node in communities_propositions_nodes[i]],
                passages_texts=community_selected_passages_text[i],
                passages_ids=community_selected_passages_ids[i],
                documents_ids=all_community_documents_ids[i],
                documents_metadata=[
                    documents_metadata_dict[doc_id]
                    for doc_id in all_community_documents_ids[i]
                ],
            )
            community_output = CommunityOutput(
                community_idx=chosen_communities[i], references=refs
            )
            list_of_communities.append(community_output)

        return list_of_communities

    async def _agenerate_community_summaries(
        self,
        question,
        communities_batches,
    ):
        async def _acall_community_answer(question, context):
            loop = asyncio.get_running_loop()
            community_points = await loop.run_in_executor(
                None,
                lambda: self.community_answer(
                    question=question,
                    reports=context,
                ),
            )
            return community_points.keypoints

        self.logger.info(f"Generating summaries for {len(communities_batches)} batches")

        tasks = [
            _acall_community_answer(question, context)
            for context in communities_batches
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_communities_kps = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error generating summary for batch {i}: {result}")
                all_communities_kps.append([])
            else:
                all_communities_kps.append(result)
                self.logger.debug(f"Community Batch {i} done.")

        return all_communities_kps

    async def aextract_communities(
        self,
        question: str,
        memory: List[Any],
        m: int = 5,
        budget: int = 1000,
        retriever_args: Dict = None,
        max_iter: int = 5,
        min_number_of_facts: int = 100,
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        passage_token_limit=500,
        min_community_size=10,
        max_community_size=100,
        callback_logger: Optional[CallbackLogger] = None,
    ):
        """Async version of extract_communities"""
        if retriever_args is None:
            retriever_args = self._default_retriever_args

        # Step 1: Decompose question
        loop = asyncio.get_running_loop()
        sub_questions = await loop.run_in_executor(
            None, lambda: self.decompose_question(question, m)
        )

        memory_block = DecompositionMemoryBlock(
            sub_questions=sub_questions,
            log=f"Initial question decomposed in {m} sub-questions.",
        )
        memory.append(memory_block)

        sub_questions_str = "\n".join(memory_block.sub_questions)
        self.logger.debug(
            f"Decomposed question into sub-questions: {sub_questions_str}"
        )
        if callback_logger:
            callback_logger(data=memory_block)

        # Step 2: Retrieve and filter
        all_seeds, memory, node_id2text, node_id2pid = await self._aretrieve_and_filter(
            question,
            sub_questions,
            retriever_args,
            memory,
            max_iter,
            min_number_of_facts,
            m,
            alpha,
            beta,
            gamma,
            callback_logger=callback_logger,
        )

        # Step 3: Budgeted cover
        chosen, covered, budget_used, uncovered_seeds = self._budgeted_cover(
            all_seeds,
            budget,
            min_community_size=min_community_size,
            max_community_size=max_community_size,
        )

        coverage_ratio = (len(covered) / len(all_seeds)) * 100 if all_seeds else 0.0
        budget_ratio = (budget_used / budget) * 100 if budget else 0.0
        self.logger.info(
            "Identified %d communities covering %.2f%% of seeds.",
            len(chosen),
            coverage_ratio,
        )
        memory_block = CommunityBudgetMemoryBlock(
            log=f"Communities have been selected. Now generating communities responses",
            coverage_ratio=coverage_ratio,
            budget_ratio=budget_ratio,
        )
        memory.append(memory_block)
        if callback_logger:
            callback_logger(data=memory_block)

        self.logger.info("Budget used: %d / %d", budget_used, budget)

        # Cast choen as a list
        chosen = list(chosen)
        if not chosen:
            self.logger.warning("No communities were chosen within the budget.")
            return [], memory

        # Step 4: Post-process communities
        list_of_communities = await self._post_process_communities(
            chosen,
            passage_token_limit=passage_token_limit,
        )

        return (
            list_of_communities,
            memory,
        )

    async def aquery(
        self,
        question: str,
        m: int = 5,
        budget: int = 1000,
        min_community_size: int = 10,
        max_community_size: int = 100,
        retriever_args: Dict = None,
        max_iter: int = 5,
        min_number_of_facts: int = 100,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        passage_token_limit: int = 500,
        max_tokens_report: int = 3500,
        max_tokens_community_chunks: int = 2500,
        response_format: str = "multiple paragraphs long report",
        refine_answer: bool = True,
        add_references: bool = True,
        callback_logger: Optional[CallbackLogger] = None,
    ):
        """
        Query in global mode.
        Basically, we decompose the question, retrieve and filter facts
        until we reach `min_number_of_facts`. Then, we extract communities,
        generate community summaries, and finally generate the final answer.

        Args:
            - question (str): the user query
            - m (int): controls the breadth of the exploration (default 5).
            - min_number_of_facts (int): minimum (if possible) number of facts to collect
                before extracting communities (default 100).
            - budget (int, optional): Token budget for community summarization (default: `12000`).
            - max_tokens_report (int, optional): maximal number of tokens to use when prompting
                for generating the final report (default: 3500)
            - max_tokens_community_chunks (int, optional): maximal number of tokens in a batch
                when producing  communities' keypoints
            - passage_token_limit (int, optional): maximal number of tokens dedicated to passages
                when producing communities' keypoints (default: 500)
            - max_iter (int, optional): Max refinement iterations (default: `50`).
            - add_references (bool, optional): include references to communities indexes in the
                response (default: True)
            - refine_answer (bool, optional): Apply the answer refinement step (default: True)
            - alpha (float, optional): with feedback seetings (see paper) coefficient of the query
                embeddings (default: 1.0)
            - beta (float, optional): with feedback seetings (see paper) coefficient of the selected
                proposition embeddings (default: 0.0)
            - gamma (float, optional): with feedback seetings (see paper) coefficient of the discarded
                proposition embeddings (default: 0.0)

        Return Answer string and memory blocks
        -------
        """

        if (
            self.retriever.store.g_full is None
            or len(self.retriever.store.g_full.nodes) == 0
        ):
            self.logger.warning("The graph is empty. Cannot process the query.")
            return "I don't know", {}

        memory = []

        (
            list_of_communities,
            memory,
        ) = await self.aextract_communities(
            question,
            memory,
            m=m,
            budget=budget,
            retriever_args=retriever_args,
            max_iter=max_iter,
            min_number_of_facts=min_number_of_facts,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            passage_token_limit=passage_token_limit,
            min_community_size=min_community_size,
            max_community_size=max_community_size,
            callback_logger=callback_logger,
        )

        if not list_of_communities:
            self.logger.warning("No communities extracted. Returning default response.")

            memory_block = IntermediaryResponsesMemoryBlock(
                log="No communities extracted, no intermediary responses generated.",
                intermediary_responses=[],
            )
            memory.append(memory_block)
            if callback_logger:
                callback_logger(data=memory_block)

            default_response = "I'm sorry, I couldn't find relevant information to answer your question."
            memory_block = ResponseMemoryBlock(
                log="No communities extracted, returning default response.",
                response=default_response,
            )
            memory.append(memory_block)
            if callback_logger:
                callback_logger(data=memory_block)

            memory_block = CommunitiesMemoryBlock(
                log="No communities extracted to generate the response.",
                communities=[],
            )
            memory.append(memory_block)
            if callback_logger:
                callback_logger(data=memory_block)

            return default_response, memory

        batched_communities_contents = []
        current_batch = ""
        current_size = 0
        all_communities_original_ids = [
            commu.community_idx for commu in list_of_communities
        ]

        for c_idx, community in enumerate(list_of_communities):
            commu_content = community.references
            pasted_passages = "\n\n".join(
                [
                    f"Passage {idx + 1}: {passage}"
                    for idx, passage in enumerate(commu_content.passages_texts)
                ]
            )
            content = (
                f"####### Report Data ID {c_idx} #######\n\n"
                + "**Entities:**\n"
                + ", ".join(commu_content.entities_texts)
                + "\n\n**Facts:**\n"
                + "\n".join(commu_content.facts_texts)
                + (
                    "\n\n**Passages:**\n" + pasted_passages
                    if pasted_passages.strip()
                    else ""
                )
                + "\n\n"
            )

            content_size = len(self.tokenizer.encode(content))

            # Check if adding this would exceed limit AND we have existing content
            if (
                current_size + content_size > max_tokens_community_chunks
                and current_batch
            ):
                # Save current batch before starting new one
                batched_communities_contents.append(current_batch)
                current_batch = content
                current_size = content_size
            else:
                # Add to current batch (handles first iteration and normal additions)
                current_batch += content
                current_size += content_size

        # Don't forget the last batch
        if current_batch:
            batched_communities_contents.append(current_batch)

        communities_reports = await self._agenerate_community_summaries(
            question=question,
            communities_batches=batched_communities_contents,
        )

        all_community_reports = [
            kp for community_kps in communities_reports for kp in community_kps
        ]

        all_community_reports.sort(key=lambda kp: kp.score, reverse=True)

        expert_reports = []
        total_tokens = 0
        for kp in all_community_reports:
            if add_references:
                expert_reports.append(
                    f"{kp.description} (Report: {[all_communities_original_ids[i] for i in kp.references]})"
                )
            else:
                expert_reports.append(kp.description)
            total_tokens += len(self.tokenizer.encode(kp.description))
            if total_tokens > max_tokens_report:
                break

        self.logger.info(f"Total tokens in expert reports: {total_tokens}")

        # Here we store the intermediary responses
        all_reports_with_std_community_idx = [
            IndexedKeyPoint(
                description=kp.description,
                score=kp.score,
                references=[all_communities_original_ids[j] for j in kp.references],
            )
            for kp in all_community_reports
        ]

        memory_block = IntermediaryResponsesMemoryBlock(
            log=f"Generated {len(all_reports_with_std_community_idx)} intermediary responses.",
            intermediary_responses=all_reports_with_std_community_idx,
        )

        memory.append(memory_block)
        if callback_logger:
            callback_logger(data=memory_block)

        list1 = [p for i, p in enumerate(expert_reports) if i % 2 == 1]
        list2 = [p for i, p in enumerate(expert_reports) if i % 2 == 0]

        organized = list1 + list2[::-1]
        formatted_expert_reports = "\n".join(organized)

        self.logger.info("Generating the initial response...")
        loop = asyncio.get_running_loop()
        final_response = await loop.run_in_executor(
            None,
            lambda: self.response_generator(
                question=question,
                response_format=response_format,
                expert_reports=formatted_expert_reports,
            ),
        )

        response = final_response.response
        memory_block = ResponseMemoryBlock(
            log="Generated response from the community reports.",
            response=response,
        )
        memory.append(memory_block)
        if callback_logger:
            callback_logger(data=memory_block)

        if refine_answer:
            self.logger.info("Refining the response...")
            organized_refiner = list1[::-1] + list2
            formatted_expert_reports_refinement = "\n".join(organized_refiner)
            refined_response = await loop.run_in_executor(
                None,
                lambda: self.response_refinement(
                    question=question,
                    response_format=response_format,
                    initial_response=final_response.response,
                    expert_reports=formatted_expert_reports_refinement,
                ),
            )
            response = refined_response.refined_response

            memory_block = ResponseMemoryBlock(
                log="Generated refined response.",
                response=response,
            )
            memory.append(memory_block)
            if callback_logger:
                callback_logger(data=memory_block)

        # We finally add the references at the end, always the last memory element.
        last_memory_block = CommunitiesMemoryBlock(
            log=f"{len(list_of_communities)} communities used to generate the response.",
            communities=list_of_communities,
        )
        memory.append(last_memory_block)
        if callback_logger:
            callback_logger(data=last_memory_block)

        return response, memory

    def query(
        self,
        question: str,
        m: int = 5,
        budget: int = 1000,
        min_community_size: int = 10,
        max_community_size: int = 100,
        retriever_args: Dict = None,
        max_iter: int = 5,
        min_number_of_facts: int = 100,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        passage_token_limit: int = 500,
        max_tokens_report: int = 3500,
        max_tokens_community_chunks: int = 2500,
        response_format: str = "multiple paragraphs long report",
        refine_answer: bool = True,
        add_references: bool = True,
        callback_logger: Optional[CallbackLogger] = None,
    ):
        """Synchronous wrapper for aquery"""
        loop = always_get_event_loop()
        return loop.run_until_complete(
            self.aquery(
                question=question,
                m=m,
                budget=budget,
                min_community_size=min_community_size,
                max_community_size=max_community_size,
                retriever_args=retriever_args,
                max_iter=max_iter,
                min_number_of_facts=min_number_of_facts,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                passage_token_limit=passage_token_limit,
                max_tokens_report=max_tokens_report,
                max_tokens_community_chunks=max_tokens_community_chunks,
                response_format=response_format,
                refine_answer=refine_answer,
                add_references=add_references,
                callback_logger=callback_logger,
            )
        )
