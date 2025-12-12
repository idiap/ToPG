#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#

import json
import os
import time
from collections import Counter, defaultdict
from typing import List

import dspy
import networkx as nx
import numpy as np
from scipy.sparse import diags

from topg.rag.utils import mmr


class SelectorPrompt(dspy.Signature):
    """
    Your task is to filter a list of facts based on their relevance to a given query.
    Select only the facts that are most useful for reasoning toward the answer — they may not necessarily answer it directly.
    Each fact has an associated index. In your output, return only the indices of the selected facts — not the fact texts themselves.
    You need to be very critical and selective in your choices, as the selected facts will impact the final results.
    You must select at least 1 fact and no more than 5.
    """

    query: str = dspy.InputField(description="The input query")
    facts: str = dspy.InputField(
        description="A list of facts with their indexes to filter based on their relevance to the query."
    )
    selected_facts_indexes: List[int] = dspy.OutputField(
        description="The indexes of the selected facts."
    )


def trw_for_seeds_batch(
    G,
    seed_nodes,
    number_of_steps=25000,
    seed_value=42,
    restart_prob=0.15,
    top_k=500,
):
    """Random walk with restarts to build a proposition horizon around seeds."""
    # init
    all_counts = defaultdict(int)
    n_seeds = len(seed_nodes)

    # Select a random initial seed
    current = seed_nodes[np.random.randint(0, n_seeds)]
    for _ in range(number_of_steps):
        # Count the current node
        all_counts[current] += 1

        # Restart with some probability
        if np.random.random() < restart_prob:
            current = seed_nodes[np.random.randint(0, n_seeds)]
            continue

        # Otherwise move to a random neighbor
        neighbors = list(G.neighbors(current))
        current = neighbors[np.random.randint(0, len(neighbors))]

    # So, Get only proposition nodes
    all_nodes_ids = set(
        [node[0] for node in G.nodes(data=True) if node[1]["type"] == "p"]
    )

    # filter counts to keep only passage nodes
    all_counts_propositions = {
        node: count for node, count in all_counts.items() if node in all_nodes_ids
    }

    # Get the top_k propositions nodes + their neighbors
    counts = Counter(all_counts_propositions)
    top_nodes = counts.most_common(top_k)
    horizon_nodes = [node for node, _ in top_nodes]

    # Just check that all seeds are in
    for seed in seed_nodes:
        if seed not in horizon_nodes:
            horizon_nodes.append(seed)

    horizon_nodes_neighbors = list(
        set([n for node in horizon_nodes for n in G.neighbors(node)])
    )

    # Create subgraph
    horizon_graph = G.subgraph(horizon_nodes + horizon_nodes_neighbors).copy()

    # Prepare the proposition 2 proposition transition matrix
    A = nx.adjacency_matrix(horizon_graph)
    g_node_list = list(horizon_graph.nodes)

    # Get indices of propositions and entities in the adjacency matrix
    index_of_propositions = [g_node_list.index(node) for node in horizon_nodes]
    index_of_ep = [g_node_list.index(node) for node in horizon_nodes_neighbors]
    ordered_propositions_nodes = [g_node_list[i] for i in index_of_propositions]

    # Extract submatrices
    a_p_eP = A[index_of_propositions, :][:, index_of_ep]
    a_eP_p = a_p_eP.T

    # normalize
    row_sums_a_p_eP = np.array(a_p_eP.sum(axis=1)).flatten()
    # Create inverse diagonal matrix

    row_sums_a_p_eP[row_sums_a_p_eP == 0] = 1  # Handle zeros
    D_inv_a_p_eP = diags(1.0 / row_sums_a_p_eP)
    a_p_eP_normalized = D_inv_a_p_eP @ a_p_eP
    row_sums_a_eP_p = np.array(a_eP_p.sum(axis=1)).flatten()

    # Create inverse diagonal matrix
    row_sums_a_eP_p[row_sums_a_eP_p == 0] = 1  # Handle zeros
    D_inv_a_eP_p = diags(1.0 / row_sums_a_eP_p)
    a_eP_p_normalized = D_inv_a_eP_p @ a_eP_p

    # Final proposition to proposition transition matrix
    T_U = a_p_eP_normalized @ a_eP_p_normalized

    # Put the diagonal to zero
    T_U.setdiag(0)

    # Normalize rows again
    row_sums_T_U = np.array(T_U.sum(axis=1)).flatten()
    row_sums_T_U[row_sums_T_U == 0] = 1  # Handle zeros
    D_inv_T_U = diags(1.0 / row_sums_T_U)
    T_U = D_inv_T_U @ T_U

    # Create graph
    final_horizon_graph = nx.from_scipy_sparse_array(T_U, create_using=nx.DiGraph)

    # Relabel nodes to original proposition ids
    node_mapping = {
        i: ordered_propositions_nodes[i] for i in range(len(ordered_propositions_nodes))
    }
    final_horizon_graph = nx.relabel_nodes(final_horizon_graph, node_mapping)

    return final_horizon_graph


def get_walker_ns_matrix(
    q_embedding,
    embeddings_matrix,
    q,
    damping,
    transition_symbolic,
    restart_vector,
    temperature=0.1,
    threshold=0.4,
):
    """
    Blend symbolic transitions and neural similarities into a walk matrix.
    In the paper, this is when we build M with the QueryAwareTransition
    """
    # get the symbolic_masks
    symbolic_masks = np.where(transition_symbolic > 0, 1, 0)

    # Compute the cosine similarities with the direction given by q_embedding
    cosines_t = np.dot(q_embedding, embeddings_matrix.T)

    # Apply the trheshold
    cosines_mask = cosines_t >= threshold

    # Apply the temperature scaling and the mask
    exp_cosines_t = np.exp(cosines_t / temperature)
    cosines_matrix = exp_cosines_t * symbolic_masks

    cosine_threshold_mask_matrix = cosines_mask * symbolic_masks

    # check there rows where there is no transition - because of the cosine threshold. For these rows, we will automatilly restart.
    row_sums = cosine_threshold_mask_matrix.sum(axis=1, keepdims=True)
    no_transition = np.array(row_sums.squeeze() == 0)

    row_sums_cosines = cosines_matrix.sum(axis=1, keepdims=True)
    row_sums_cosines[row_sums_cosines == 0] = 1  # Avoid division by zero

    neuro_matrix = cosines_matrix / row_sums_cosines

    neuro_symbolic_matrix = q * transition_symbolic + (1 - q) * neuro_matrix
    neuro_symbolic_matrix[no_transition] = restart_vector

    ppr_matrix = damping * neuro_symbolic_matrix + (1 - damping) * restart_vector

    return ppr_matrix


def compute_stationary_proba(M, restart_vector, tol=1e-6, max_iter=1000):
    """Compute stationary distribution for the PPR by power iteration."""
    # Ensure restart_vector is normalized
    stationary_proba = restart_vector

    for _ in range(max_iter):
        # Update stationary probability vector
        new_stationary_proba = stationary_proba @ M

        # Check for convergence
        if np.linalg.norm(new_stationary_proba - stationary_proba, ord=1) < tol:
            return new_stationary_proba

        stationary_proba = new_stationary_proba

    # Return the final stationary probability vector
    return stationary_proba


class HybridRetrieval:
    """
    Hybrid retrieval over graph and vectors: seed, walk, rank, and suggest.
    """

    def __init__(self, store, logger=None):
        self.store = store
        self.logger = logger or logging.getLogger(__name__)

    def retrieve(
        self,
        queries: List[str],
        entities: List[List[str]],
        exclusion_list: List[List[str]],
        e_syn_k=3,
        e_syn_threshold=0.80,
        lambda_mmr=1.0,
        top_k=10,
        direct_k_multiplier=3.0,
    ):
        """
        Retrieve top-k propositions per query via entity seeding and/or direct search.

        Args:
        queries (List[str]): List of input queries.
        entities (List[List[str]]): List of entity lists per query.
        exclusion_list (List[List[str]]): List of proposition IDs to exclude per query.
        e_syn_k (int): Maximum number of entity synonyms to retrieve for expansion.
        e_syn_threshold (float): Similarity threshold for entity synonym selection.
        lambda_mmr (float): MMR lambda parameter for diversity in selection.
        top_k (int): Number of top propositions to retrieve per query.
        direct_k_multiplier (float): Multiplier for top-k in direct retrieval when we have lambda_mmr < 1.0.
        Returns:
            List[dict]: Retrieval results per query with keys:
                - "query_embedding": np.array of the query embedding
                - "doc_text": List[str] of retrieved document texts
                - "doc_ids": List[str] of retrieved document IDs
                - "doc_embeddings": List[np.array] of document embeddings
                - "doc_cosines": List[float] of cosine similarities
        """
        # how many queries to process ?
        n_queries = len(queries)
        if n_queries == 0:
            return []

        # Prepare the retrieval output structure
        retrieval_output = [
            {
                "query_embedding": np.array([]),
                "doc_text": [],
                "doc_ids": [],
                "doc_embeddings": [],
                "doc_cosines": [],
            }
            for j in range(n_queries)
        ]

        # Separate queries into entity-based and direct queries
        entity_based_indices = []
        direct_query_indices = []

        for i in range(n_queries):
            if len(entities[i]) > 0:
                entity_based_indices.append(i)
            else:
                direct_query_indices.append(i)

        if len(direct_query_indices) > 0:
            self.logger.debug(
                f"Queries at indexes {direct_query_indices} don't have associated entities, apply direct proposition retrieval."
            )

        # Process entity-based queries (existing logic)
        if len(entity_based_indices) > 0:
            entity_based_results = self._process_entity_based_queries(
                queries,
                entities,
                exclusion_list,
                entity_based_indices,
                e_syn_k,
                e_syn_threshold,
                lambda_mmr,
                top_k,
            )

            # Update retrieval_output with entity-based results
            for idx, result in zip(entity_based_indices, entity_based_results):
                retrieval_output[idx] = result

        # Process direct queries (if any)
        if len(direct_query_indices) > 0:
            direct_results = self._process_direct_queries(
                queries,
                exclusion_list,
                direct_query_indices,
                lambda_mmr,
                top_k,
                direct_k_multiplier,
            )

            # Update retrieval_output with direct results
            for idx, result in zip(direct_query_indices, direct_results):
                retrieval_output[idx] = result

        return retrieval_output

    def _process_entity_based_queries(
        self,
        queries,
        entities,
        exclusion_list,
        query_indices,
        e_syn_k,
        e_syn_threshold,
        lambda_mmr,
        top_k,
    ):
        """Process queries with entities: expand seeds, embed, MMR select."""

        # Extract only the queries and entities we're processing
        selected_queries = [queries[i] for i in query_indices]
        selected_entities = [entities[i] for i in query_indices]
        selected_exclusions = [exclusion_list[i] for i in query_indices]

        # Existing entity processing logic (slightly modified)
        flat_entities = [item for sublist in selected_entities for item in sublist]
        flat_entities_index = np.array(
            [
                i
                for i in range(len(selected_entities))
                for _ in range(len(selected_entities[i]))
            ]
        )

        # In total how many entities we have
        n_entities = len(flat_entities)

        # So we encode the entities
        entities_embeddings = self.store.entity_store._embedding_fn(flat_entities)

        # We retrieve the top k entities
        entities_retrieval = self.store.entity_store.query(
            query_embeddings=entities_embeddings,
            n_results=e_syn_k,
            include=["distances"],
        )

        entity_ids = entities_retrieval["ids"]
        entity_cosines = np.clip(1 - np.array(entities_retrieval["distances"]), 0, 1)

        # So we also filter the entities that do not pass a similarity threshold in the top-k
        entity_top = np.sum(entity_cosines >= e_syn_threshold, axis=1)

        # However, if nothing pass we still get the top 1
        entity_top = np.where(entity_top == 0, 1, entity_top)

        # Here we extract the ids of the selected entities
        selected_seed_entities = [
            entity_ids[i][: entity_top[i]] for i in range(n_entities)
        ]

        # Same as before, we flatten the list of selected entities before requesting the propositions
        flatten_selected_seed_entities = [
            item for sublist in selected_seed_entities for item in sublist
        ]

        # We keep track of the entity index for each selected entity to then combine the results
        entity_mask = np.array(
            [
                i
                for i in range(n_entities)
                for _ in range(len(selected_seed_entities[i]))
            ]
        )

        # Here we get the list of propositions for each selected entity
        entities2propositions = self.store.sqlite_db.get_propositions_by_entity_ids(
            flatten_selected_seed_entities
        )
        # Format
        entities2propositions = [
            entities2propositions[entity_id]
            for entity_id in flatten_selected_seed_entities
        ]

        propositions_per_queries = [[] for _ in range(len(selected_queries))]
        for idx, propositions in enumerate(entities2propositions):
            # Get the index of the query for the current entity
            q_idx = flat_entities_index[entity_mask[idx]]
            # Add the propositions to the corresponding query
            propositions_per_queries[q_idx].extend(propositions)

        # If we don't expand, we just keep the original propositions
        expanded_propositions_per_queries = propositions_per_queries

        # make a warning if id not in the graph
        for i in range(len(selected_queries)):
            for _id in expanded_propositions_per_queries[i]:
                if not self.store.g_full.has_node(_id):
                    self.logger.warning(
                        f"Proposition ID {_id} is not connected in the graph nodes and will not be selected."
                    )

        # For each query, we remove the exclude ids
        for i in range(len(selected_queries)):
            expanded_propositions_per_queries[i] = [
                _id
                for _id in expanded_propositions_per_queries[i]
                if _id not in selected_exclusions[i] and self.store.g_full.has_node(_id)
            ]

        # Continue with document store querying and MMR
        queries_to_continue = np.where(
            np.array([len(ids) > 0 for ids in expanded_propositions_per_queries])
        )[0]

        results = []

        if len(queries_to_continue) == 0:
            # Return empty results for all queries
            return [
                {
                    "query_embedding": np.array([]),
                    "doc_text": [],
                    "doc_ids": [],
                    "doc_embeddings": [],
                    "doc_passage_ids": [],
                    "doc_cosines": [],
                    "doc_entities": [],
                }
                for _ in range(len(selected_queries))
            ]

        final_selected_queries = np.array(selected_queries)[queries_to_continue]
        selected_ids = [
            expanded_propositions_per_queries[i] for i in queries_to_continue
        ]

        # Query embeddings for selected queries
        query_embeddings = self.store.document_store._embedding_fn(
            final_selected_queries
        )

        proposition_retrieval = [
            self.store.document_store.query(
                query_embeddings=query_embeddings[i],
                include=["documents", "embeddings", "distances", "metadatas"],
                ids=selected_ids[i],
                n_results=len(selected_ids[i]),
            )
            for i in range(len(queries_to_continue))
        ]

        doc_embeddings = [
            np.array(item["embeddings"][0]) for item in proposition_retrieval
        ]
        doc_ids = [item["ids"][0] for item in proposition_retrieval]
        doc_text = [item["documents"][0] for item in proposition_retrieval]
        cosine_similarities = [
            1 - np.array(item["distances"][0]) for item in proposition_retrieval
        ]
        doc_entities = [
            [json.loads(item["entities"]) for item in q_item["metadatas"][0]]
            for q_item in proposition_retrieval
        ]
        doc_passage_ids = [
            [item["passage_id"] for item in q_item["metadatas"][0]]
            for q_item in proposition_retrieval
        ]

        # Initialize results with empty dictionaries
        results = [
            {
                "query_embedding": np.array([]),
                "doc_text": [],
                "doc_ids": [],
                "doc_embeddings": [],
                "doc_passage_ids": [],
                "doc_cosines": [],
                "doc_entities": [],
            }
            for _ in range(len(selected_queries))
        ]

        # Apply MMR and populate results
        for j in range(len(queries_to_continue)):
            q_docs_embeddings = doc_embeddings[j]
            q_doc_cosines = cosine_similarities[j]

            # Get the top k documents based on MMR
            mmr_indices = mmr(
                q_docs_embeddings, q_doc_cosines, lambda_param=lambda_mmr, k=top_k
            )

            query_idx = queries_to_continue[j]
            results[query_idx] = {
                "query_embedding": query_embeddings[j],
                "doc_text": [doc_text[j][idx] for idx in mmr_indices],
                "doc_ids": [doc_ids[j][idx] for idx in mmr_indices],
                "doc_embeddings": [q_docs_embeddings[idx] for idx in mmr_indices],
                "doc_passage_ids": [doc_passage_ids[j][idx] for idx in mmr_indices],
                "doc_cosines": [q_doc_cosines[idx] for idx in mmr_indices],
                "doc_entities": [doc_entities[j][idx] for idx in mmr_indices],
            }

        return results

    def _process_direct_queries(
        self,
        queries,
        exclusion_list,
        query_indices,
        lambda_mmr,
        top_k,
        direct_k_multiplier,
    ):
        """Process queries without entities via direct vector search + MMR."""

        # Extract queries for direct processing
        direct_queries = [queries[i] for i in query_indices]
        direct_exclusions = [exclusion_list[i] for i in query_indices]

        # Calculate k for direct queries (larger to account for exclusions and MMR)
        direct_k = (
            max(top_k, int(top_k * direct_k_multiplier)) if lambda_mmr < 1.0 else top_k
        )

        self.logger.debug(f"Using direct_k = {direct_k} for direct queries")

        # Get query embeddings
        query_embeddings = self.store.document_store._embedding_fn(direct_queries)

        results = []

        for i, (query_emb, exclusions) in enumerate(
            zip(query_embeddings, direct_exclusions)
        ):
            # Query the document store directly
            retrieval_result = self.store.document_store.query(
                query_embeddings=query_emb,
                include=["documents", "embeddings", "distances", "metadatas"],
                n_results=direct_k,
            )

            # Extract results
            doc_ids = retrieval_result["ids"][0]
            doc_text = retrieval_result["documents"][0]
            doc_embeddings = np.array(retrieval_result["embeddings"][0])
            cosine_similarities = 1 - np.array(retrieval_result["distances"][0])

            # Extract metadata if available
            try:
                doc_entities = [
                    json.loads(item["entities"])
                    for item in retrieval_result["metadatas"][0]
                ]
            except (KeyError, json.JSONDecodeError):
                doc_entities = [[] for _ in doc_ids]

            doc_passage_ids = [
                item["passage_id"] for item in retrieval_result["metadatas"][0]
            ]
            # Filter out excluded documents and those not in graph nodes
            valid_indices = []
            filtered_doc_ids = []
            filtered_doc_text = []
            filtered_doc_embeddings = []
            filtered_cosines = []
            filtered_entities = []
            filtered_passage_ids = []

            for idx, doc_id in enumerate(doc_ids):
                if doc_id not in exclusions and self.store.g_full.has_node(doc_id):
                    valid_indices.append(idx)
                    filtered_doc_ids.append(doc_id)
                    filtered_doc_text.append(doc_text[idx])
                    filtered_doc_embeddings.append(doc_embeddings[idx])
                    filtered_cosines.append(cosine_similarities[idx])
                    filtered_entities.append(doc_entities[idx])
                    filtered_passage_ids.append(doc_passage_ids[idx])

            if len(filtered_doc_embeddings) == 0:
                # No valid documents found
                result = {
                    "query_embedding": query_emb,
                    "doc_text": [],
                    "doc_ids": [],
                    "doc_embeddings": [],
                    "doc_passage_ids": [],
                    "doc_cosines": [],
                    "doc_entities": [],
                }
            else:
                # Apply MMR to filtered results
                filtered_doc_embeddings = np.array(filtered_doc_embeddings)
                filtered_cosines = np.array(filtered_cosines)

                mmr_indices = mmr(
                    filtered_doc_embeddings,
                    filtered_cosines,
                    lambda_param=lambda_mmr,
                    k=min(top_k, len(filtered_doc_embeddings)),
                )

                result = {
                    "query_embedding": query_emb,
                    "doc_text": [filtered_doc_text[idx] for idx in mmr_indices],
                    "doc_ids": [filtered_doc_ids[idx] for idx in mmr_indices],
                    "doc_embeddings": [
                        filtered_doc_embeddings[idx] for idx in mmr_indices
                    ],
                    "doc_passage_ids": [
                        filtered_passage_ids[idx] for idx in mmr_indices
                    ],
                    "doc_cosines": [filtered_cosines[idx] for idx in mmr_indices],
                    "doc_entities": [filtered_entities[idx] for idx in mmr_indices],
                }

            results.append(result)

        return results

    def get_horizon(
        self,
        initial_seed_nodes,
        use_passage_links=True,
        number_of_steps=20000,
        seed=42,
        restart_prob=0.15,
        top_k=500,
        groups=None,
    ):
        """
        Build per-group proposition horizons around merged seed nodes.
        Basicall, given a set of seed nodes, the goal of the method in to extract a subgraph G*
        from the main graph G, where G* contains the most relevant nodes around the seeds.
        This is done via a truncated random walk with restarts (TRW) from the seed nodes.

        This corresponds to the GExtract step in the paper.

        Args:
            initial_seed_nodes (List[List[str]]): List of seed node lists per query.
            use_passage_links (bool): Whether to use passage links in the graph.
            number_of_steps (int): Number of steps for the random walk.
            seed (int): Random seed for reproducibility.
            restart_prob (float): Probability of restarting the walk at a seed node.
            top_k (int): The size of the return subgraph.
                This is the number of top nodes to include in the horizon.
            groups (List[int]): List of group IDs per query for grouping seed nodes.
        """
        n = np.unique(groups).shape[0]
        all_horizon_graphs = [None] * n

        assert len(initial_seed_nodes) == len(groups), (
            "Number of seed node lists must match the len of groups"
        )
        # Merge all seed nodes form the same group

        merged_seed_nodes = [[] for _ in range(n)]
        for i, group_id in enumerate(groups):
            merged_seed_nodes[group_id].extend(initial_seed_nodes[i])

        max_workers = min(n, os.cpu_count() or 1)

        # Do we use the links between passages ?
        if use_passage_links:
            working_graph = self.store.g_full
        else:
            # Use only the entity graph
            working_graph = self.store.g_e

        start_time = time.time()

        for i, batch_seeds in enumerate(merged_seed_nodes):
            horizon_graph = trw_for_seeds_batch(
                working_graph,
                batch_seeds,
                number_of_steps,
                seed,
                restart_prob,
                top_k,
            )
            all_horizon_graphs[i] = horizon_graph

        end_time = time.time()
        self.logger.debug(f"TRW time: {end_time - start_time:.2f} seconds")

        return all_horizon_graphs

    def _run_walkers(
        self,
        query_embeddings,
        seed_nodes: List[List[str]],
        q=0.5,
        damping=0.5,
        temperature=0.1,
        cosine_threshold=0.4,
        horizon_top_k=500,
        max_workers=None,  # New parameter for controlling parallelism
        use_passage_links=True,  # Use passage links by default
        groups=None,
    ):
        """Run random walkers per group and return stationary scores."""
        assert query_embeddings.shape[0] == len(seed_nodes), (
            "Number of queries must match number of seed nodes"
        )

        # Get the horizon subgraph
        horizon_subgraphs = self.get_horizon(
            initial_seed_nodes=seed_nodes,
            top_k=horizon_top_k,
            use_passage_links=use_passage_links,
            groups=groups,
        )

        transitions_symbolic = [
            nx.adjacency_matrix(horizon_subgraph).todense()
            for horizon_subgraph in horizon_subgraphs
        ]
        n_nodes = [ts.shape[0] for ts in transitions_symbolic]

        # Ensure the symbolic transition matrix and embeddings match in terms of nodes
        horizon_subgraphs_nodes = [
            list(horizon_subgraph.nodes) for horizon_subgraph in horizon_subgraphs
        ]

        all_nodes = list(set().union(*horizon_subgraphs_nodes))

        # Extract the text of the associated proposition nodes
        all_nodes_meta = self.store.document_store.get(
            ids=all_nodes, include=["documents", "embeddings", "metadatas"]
        )

        # Get text and pid
        node2text = dict(
            zip(
                all_nodes_meta["ids"],
                all_nodes_meta["documents"],
            )
        )
        node2pid = dict(
            zip(
                all_nodes_meta["ids"],
                [meta["passage_id"] for meta in all_nodes_meta["metadatas"]],
            )
        )

        horizon_subgraphs_nodes_texts = [
            {k: node2text[k] for k in horizon_subgraph_nodes}
            for horizon_subgraph_nodes in horizon_subgraphs_nodes
        ]

        horizon_subgraphs_nodes_pids = [
            {k: node2pid[k] for k in horizon_subgraph_nodes}
            for horizon_subgraph_nodes in horizon_subgraphs_nodes
        ]

        # Reorder the embeddings
        indexes = [
            np.array(
                [all_nodes_meta["ids"].index(node) for node in horizon_subgraph_nodes]
            )
            for horizon_subgraph_nodes in horizon_subgraphs_nodes
        ]
        embeddings_matrixes = [
            np.array(all_nodes_meta["embeddings"][_idxs]) for _idxs in indexes
        ]

        # Prepare data for parallel processing
        n_queries = query_embeddings.shape[0]

        # measure the time to run
        start_time = time.time()

        all_stationary_probas = []

        for i in range(n_queries):
            # Prepare arguments for this query
            q_seed_nodes = seed_nodes[i]
            q_embs = query_embeddings[i]
            group_id = groups[i]
            horizon_subgraph_nodes = horizon_subgraphs_nodes[group_id]
            n_node = n_nodes[group_id]
            embeddings_matrix = embeddings_matrixes[group_id]
            transition_symbolic = transitions_symbolic[group_id]

            # Create restart vector for this query
            restart_vector = np.zeros(n_node)
            seed_index = np.array(
                [horizon_subgraph_nodes.index(node) for node in q_seed_nodes]
            )

            restart_vector[seed_index] = 1.0
            restart_vector = restart_vector / np.sum(restart_vector)

            # Get the ns matrix
            M = get_walker_ns_matrix(
                q_embs,
                embeddings_matrix,
                q,
                damping,
                transition_symbolic,
                restart_vector,
                temperature=temperature,
                threshold=cosine_threshold,
            )

            # Compute stationary probability
            stationary_proba = compute_stationary_proba(M, restart_vector)
            all_stationary_probas.append(stationary_proba)

        end_time = time.time()
        self.logger.debug(
            f"Walkers run completed in {end_time - start_time:.2f} seconds (sequential)."
        )

        return (
            all_stationary_probas,
            horizon_subgraphs_nodes,
            horizon_subgraphs_nodes_texts,
            horizon_subgraphs_nodes_pids,
            embeddings_matrixes,
        )

    def suggest(
        self,
        query_embeddings,
        seed_nodes,
        groups=None,
        exclusion_list=[],
        top_k=20,
        q=0.5,
        damping=0.85,
        horizon_top_k=500,
        temperature=0.1,
        cosine_threshold=0.4,
        use_passage_links=True,
        **kwargs,
    ):
        """
        Suggest top-k propositions per group using cumulative walker scores.
        Basically, each set of seed nodes is asign with a query embedding.
        We run a random walk with restarts from those seed nodes,
        considering both symbolic transitions and neural similarities guided by the query embedding.
        Then, for each group of queries, we sum the walker stationary probabilities
        across all queries in that group to get a cumulative score per node.
        We then select the top-k nodes based on these cumulative scores, excluding any nodes
        present in the exclusion list.
        Args:
            query_embeddings (np.array): Array of query embeddings.
            seed_nodes (List[List[str]]): List of seed node lists per query.
            groups (List[int]): List of group IDs per query for grouping seed nodes.
            exclusion_list (List[str]): List of proposition IDs to exclude from suggestions.
            top_k (int): Number of top propositions to suggest per group.
            q (float): Blending factor between symbolic and neural transitions.
            damping (float): Damping factor for the random walk.
            horizon_top_k (int): Size of the horizon subgraph per group.
            temperature (float): Temperature for neural transition scaling.
            cosine_threshold (float): Cosine similarity threshold for neural transitions.
            use_passage_links (bool): Whether to use passage links (g_p) in the graph.
        Returns:
            List[dict]: Suggestion results per group with keys:
                - "doc_text": List[str] of suggested document texts
                - "doc_ids": List[str] of suggested document IDs
                - "doc_passage_ids": List[str] of suggested document passage IDs
                - "p_walkers": List[float] of cumulative walker probabilities
                - "attributed_walkers": List[int] of indices of queries attributing to each node
                - "doc_embeddings": List[np.array] of document embeddings
        """
        # If no groups are provided, assume all belong to the same group
        if groups is None:
            groups = np.zeros(len(seed_nodes), dtype=int)

        # Validate groups array
        assert len(groups) == len(seed_nodes), (
            "Groups array length must match number of queries"
        )
        assert len(groups) == query_embeddings.shape[0], (
            "Groups array length must match query embeddings"
        )

        # Get the walker results
        (
            all_stationary_probas,
            horizon_subgraphs_nodes,
            horizon_subgraphs_nodes_texts,
            horizon_subgraphs_nodes_pids,
            embeddings_matrixes,
        ) = self._run_walkers(
            query_embeddings=query_embeddings,
            seed_nodes=seed_nodes,
            q=q,
            damping=damping,
            horizon_top_k=horizon_top_k,
            temperature=temperature,
            cosine_threshold=cosine_threshold,
            use_passage_links=use_passage_links,
            groups=groups,
        )

        # Get unique groups and sort them to maintain order
        unique_groups = np.unique(groups)

        # Prepare results list - one result dict per group
        results_by_group = []

        # Process each group independently
        for group_id in unique_groups:
            # Find all query indices belonging to this group
            group_mask = groups == group_id
            group_indices = np.where(group_mask)[0]

            # Get the query specific results
            horizon_subgraph_nodes = horizon_subgraphs_nodes[group_id]
            horizon_subgraph_nodes_texts = horizon_subgraphs_nodes_texts[group_id]
            horizon_subgraph_nodes_pids = horizon_subgraphs_nodes_pids[group_id]
            embeddings_matrix = embeddings_matrixes[group_id]

            # Extract stationary probabilities for this group only
            group_stationary_probas = np.array(
                [all_stationary_probas[_idx] for _idx in group_indices]
            )

            # Compute cumulative walkers for this group (sum across group queries)
            group_cumulative_walkers = group_stationary_probas.sum(axis=0)

            # Compute attribution for this group (argmax across group queries)
            group_attributed_walkers = np.argmax(group_stationary_probas, axis=0)

            # Filter out nodes in the exclusion list
            filtered_indexes = [
                idx
                for idx in range(len(horizon_subgraph_nodes))
                if horizon_subgraph_nodes[idx] not in exclusion_list
            ]

            # Handle empty case
            if len(filtered_indexes) == 0:
                group_result = {
                    "doc_text": [],
                    "doc_ids": [],
                    "doc_passage_ids": [],
                    "p_walkers": [],
                    "attributed_walkers": [],
                    "doc_embeddings": [],
                }
                results_by_group.append(group_result)
                continue

            # Filter cumulative walkers and attributed walkers based on filtered indexes
            filtered_group_cumulative_walkers = group_cumulative_walkers[
                filtered_indexes
            ]
            filtered_group_attributed_walkers = group_attributed_walkers[
                filtered_indexes
            ]

            # Extract the top-k indexes from the filtered cumulative walkers for this group
            top_k_indexes = np.argsort(filtered_group_cumulative_walkers)[::-1][:top_k]

            # Map top-k indexes to nodes and their corresponding data
            top_k_nodes = [
                horizon_subgraph_nodes[filtered_indexes[idx]] for idx in top_k_indexes
            ]
            # Text
            top_k_nodes_texts = [
                horizon_subgraph_nodes_texts[node] for node in top_k_nodes
            ]
            # Node ids
            top_k_nodes_pids = [
                horizon_subgraph_nodes_pids[node] for node in top_k_nodes
            ]
            # Their associated proba
            top_k_walkers = filtered_group_cumulative_walkers[top_k_indexes]

            # Their associated embeddings
            top_k_node_embeddings = np.array(
                embeddings_matrix[filtered_indexes][top_k_indexes]
            )

            # Their attributed walkers
            top_k_attributed_walkers = filtered_group_attributed_walkers[top_k_indexes]

            # Prepare the result for this group
            group_result = {
                "doc_text": top_k_nodes_texts,
                "doc_ids": top_k_nodes,
                "doc_passage_ids": top_k_nodes_pids,
                "p_walkers": top_k_walkers,
                "attributed_walkers": top_k_attributed_walkers,
                "doc_embeddings": top_k_node_embeddings,
            }

            results_by_group.append(group_result)

        # Return list of results, one per group, in group order
        return results_by_group
