#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#

import hashlib
import json
import logging
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any, List, Mapping

import numpy as np
from dateutil.parser import parse
from langchain_core.documents.base import Document


def is_date(string):
    try:
        parse(string, fuzzy=False)
        return True
    except ValueError:
        return False


def from_dict_to_adjacency_matrix(mapping, row_to_index, col_to_index):
    adj_matrix = np.zeros((len(row_to_index), len(col_to_index)), dtype=int)

    # Step 4: Populate the adjacency matrix
    for prop, entities in mapping.items():
        row_idx = row_to_index[prop]
        for entity in entities:
            col_idx = col_to_index[entity]
            adj_matrix[row_idx, col_idx] = 1

    return adj_matrix


class HybridBaseLoader:
    """
    Loads and indexes documents, passages, propositions, and entities into hybrid storage.

    Coordinates batch insertion into vector databases (ChromaDB) and graph database (SQLite).
    Handles entity synonym resolution and maintains entity-proposition mappings.
    """

    def __init__(
        self,
        storage,
        config: Mapping[str, Any] = {},
        logger=None,
    ):
        self.storage = storage
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def generate_id(self, entity: str) -> str:
        """Generate SHA-256 hash ID for an entity."""
        return hashlib.sha256(entity.encode("utf-8")).hexdigest()

    def index_document(self, doc_id: str, metadata: Mapping[str, Any] = {}) -> None:
        """Insert document metadata into SQLite."""
        str_metadata = json.dumps(metadata)
        self.storage.sqlite_db.insert_document(doc_id, str_metadata)

    def index_passages(self, passages: List[Document]) -> None:
        """Insert passages into SQLite."""
        self.storage.sqlite_db.insert_passages(
            passages, self.storage.document_store.name()
        )

    def index_propositions(
        self,
        input_propositions: List[Document],
    ) -> None:
        """
        Index propositions with their entities into vector and graph storage.

        Processes propositions in batches: embeds texts, resolves entity synonyms,
        creates entity-proposition mappings, and updates the graph table  emap.
        """
        self.logger.info("Loading propositions.")
        propositions = deepcopy(
            input_propositions
        )  # important to deepcopy. But we could fix this later

        n = len(propositions)
        loading_batch_size = self.config.get("loading_batch_size", 512)
        for i in range(0, n, loading_batch_size):
            index_start = i
            index_end = min(i + loading_batch_size, n)
            self.logger.info(f"Loading propositions from {index_start} to {index_end}.")

            # Create a batch of documents
            batch_propositions = propositions[index_start:index_end]

            # extract the ids
            batch_proposition_ids = [
                proposition.metadata["id"] for proposition in batch_propositions
            ]

            # Extract the propositions text
            self.logger.debug(
                "Calling embeddings function on the propositions.",
            )
            batch_propositions_texts = [
                proposition.page_content for proposition in batch_propositions
            ]
            batch_propositions_passages_ids = [
                proposition.metadata["passage_id"] for proposition in batch_propositions
            ]

            # Extract the entities
            batch_dict_of_entities = defaultdict(list)
            all_propositions_entities = [
                proposition.metadata.get("entities")
                for proposition in batch_propositions
            ]

            # Convert entities to jsonlist
            for proposition in batch_propositions:
                proposition.metadata["entities"] = json.dumps(
                    proposition.metadata.get("entities")
                )
            # Collect all the propositions in the current batch that use this entity
            for i, entities in enumerate(all_propositions_entities):
                for entity in entities:
                    batch_dict_of_entities[entity].append(i)

            batch_entities = list(batch_dict_of_entities.keys())

            # To make the id unique, we add the chroma_collection_entities.name to the id
            batch_entities_ids = [self.generate_id(entity) for entity in batch_entities]
            n_entities = len(batch_entities)

            # we store all the info of ALL the entities, even those that not cannot added in the collection because already present
            batch_emap_infos = [
                {
                    "entity_id": batch_entities_ids[i],
                    "propositions_id": [
                        batch_proposition_ids[k]
                        for k in batch_dict_of_entities[batch_entities[i]]
                    ],
                    "passages_id": [
                        batch_propositions_passages_ids[k]
                        for k in batch_dict_of_entities[batch_entities[i]]
                    ],
                }
                for i in range(n_entities)
            ]
            self.logger.debug(f"Emap size: {len(batch_emap_infos)}")

            # We store separately the ids of the entities and their labels, because we need to insert them in the collection
            batch_entities_infos = [
                {
                    "entity_id": batch_entities_ids[i],
                    "entity_label": batch_entities[i],
                }
                for i in range(n_entities)
            ]

            ## INSERTIONS

            # Now we need to insert the propositions in the collection
            batch_propositions_texts_embeddings = (
                self.storage.document_store._embedding_fn(batch_propositions_texts)
            )
            self.logger.debug("Embeddings done.")
            self.logger.debug("Inserting propositions in the collection.")
            self.storage.document_store.add(
                documents=batch_propositions_texts,
                embeddings=batch_propositions_texts_embeddings,
                metadatas=[proposition.metadata for proposition in batch_propositions],
                ids=batch_proposition_ids,
            )
            collection_new_count = self.storage.document_store.count()
            self.logger.info(f"Proposition collection new size: {collection_new_count}")

            # Now insert the entities in the collection
            to_insert_entities_labels = [
                e_info["entity_label"] for e_info in batch_entities_infos
            ]
            to_insert_entities_ids = [
                e_info["entity_id"] for e_info in batch_entities_infos
            ]
            # Now first we insert the new entities
            self.logger.debug("Calling embeddings function on the entities.")
            to_insert_entities_embeddings = self.storage.entity_store._embedding_fn(
                to_insert_entities_labels
            )
            self.logger.debug("Embeddings done.")

            self.logger.debug("Inserting entities in the collection")
            previous_count = self.storage.entity_store.count()
            self.storage.entity_store.add(
                documents=to_insert_entities_labels,
                embeddings=to_insert_entities_embeddings,
                ids=to_insert_entities_ids,
            )

            self.storage.sqlite_db.insert_propositions(
                batch_propositions, self.storage.document_store.name()
            )

            # Synonym for emap
            resolve_synonym = self.config.get("graph_resolve_synonym", True)
            synonym_max_limit = self.config.get("graph_synonym_max_limit", 10)
            synonym_sim_threshold = self.config.get("graph_synonym_sim_threshold", 0.9)

            if resolve_synonym:
                synonyms = self.storage.entity_store.query(
                    query_embeddings=to_insert_entities_embeddings,
                    n_results=synonym_max_limit,
                    include=["distances"],
                )

                matches_ids = synonyms["ids"]
                cosines = 1 - np.array(synonyms["distances"])
                matches = np.where(cosines >= synonym_sim_threshold)

                # map as a dict of synonyms
                list_of_indexes = [[] for _ in range(cosines.shape[0])]
                for row, col in zip(*matches):
                    list_of_indexes[row].append(int(col))

                list_of_synonyms = [
                    [
                        matches_ids[i][j]
                        for j in list_of_indexes[i]
                        if matches_ids[i][j] != to_insert_entities_ids[i]
                    ]
                    for i in range(len(matches_ids))
                ]

                # Create the synonym dict
                syn_dict = {}
                for i, l_syn in enumerate(list_of_synonyms):
                    for syn in l_syn:
                        if not syn in syn_dict:
                            syn_dict[syn] = {
                                "propositions_id": [
                                    batch_proposition_ids[k]
                                    for k in batch_dict_of_entities[
                                        to_insert_entities_labels[i]
                                    ]
                                ],
                                "passages_id": [
                                    batch_propositions_passages_ids[k]
                                    for k in batch_dict_of_entities[
                                        to_insert_entities_labels[i]
                                    ]
                                ],
                            }
                        else:
                            for k in batch_dict_of_entities[
                                to_insert_entities_labels[i]
                            ]:
                                if (
                                    batch_proposition_ids[k]
                                    not in syn_dict[syn]["propositions_id"]
                                ):
                                    syn_dict[syn]["propositions_id"].append(
                                        batch_proposition_ids[k]
                                    )
                                if (
                                    batch_propositions_passages_ids[k]
                                    not in syn_dict[syn]["passages_id"]
                                ):
                                    syn_dict[syn]["passages_id"].append(
                                        batch_propositions_passages_ids[k]
                                    )

                # Complete the batch_emap_infos with the synonyms info
                for entity_id, info in syn_dict.items():
                    batch_emap_infos.append(
                        {
                            "entity_id": entity_id,
                            "propositions_id": info["propositions_id"],
                            "passages_id": info["passages_id"],
                        }
                    )
                self.logger.debug(
                    f"Emap size after synonym resolution: {len(batch_emap_infos)}"
                )

            self.storage.sqlite_db.insert_entities(
                batch_entities_infos,
                batch_emap_infos,
                self.storage.document_store.name(),
            )

            self.logger.debug("Loading done.")
            self.logger.info("Now we need to update the graph.")
