#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#


import json
import logging
import os
from typing import Any, List, Mapping, Optional, Union, cast

import chromadb
from chromadb import Collection, Documents, Embeddings
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

from topg.builders.communities import CommunityManager
from topg.builders.sql_db import MapDB


class ChromaEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        encoding_params: dict = {},
        device: str = "cpu",
        **kwargs: Any,
    ):
        self.model = SentenceTransformer(model_name, device=device, **kwargs)
        self.encoding_params = encoding_params

    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, **self.encoding_params).tolist()

    def __call__(self, input: Documents) -> Embeddings:
        """
        Callable interface required by ChromaDB's EmbeddingFunction.

        Args:
            input (Documents): Documents to embed (ChromaDB format).

        Returns:
            Embeddings: Embeddings in ChromaDB format.
        """
        return cast(Embeddings, self.encode(input))


class ChromaEmbeddingStorage:
    """
    Wrapper for ChromaDB persistent storage with custom embedding functions.

    This class manages a ChromaDB collection for storing and retrieving vector
    embeddings. It provides methods for adding, retrieving, and querying documents
    with their embeddings, using HNSW indexing for efficient similarity search.

    Attributes:
        path (str): File system path to the ChromaDB persistent storage.
        collection_name (str): Name of the ChromaDB collection.
        _embedding_fn: The embedding function used for this collection.
        logger (logging.Logger): Logger instance.
        _collection (Collection): The underlying ChromaDB collection.
    """

    def __init__(
        self,
        path: str,
        collection_name: str,
        embedding_fn,
        logger: Optional[logging.Logger] = None,
    ):
        self.path = path
        self.collection_name = collection_name
        self._embedding_fn = embedding_fn
        self.logger = logger or logging.getLogger(__name__)
        self._collection = self._init_collection()

    def _init_collection(self) -> Collection:
        """
        Initialize or load the ChromaDB collection.

        Creates the directory if it doesn't exist and initializes a persistent
        ChromaDB client with HNSW indexing configuration optimized for cosine similarity.

        Returns:
            Collection: The initialized ChromaDB collection.
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.logger.debug(f"Created Chroma DB directory: {self.path}")

        client = chromadb.PersistentClient(
            path=self.path,
            settings=Settings(allow_reset=False),
        )

        self.logger.debug(f"Getting or creating collection: {self.collection_name}")
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 128,
                "hnsw:search_ef": 128,
                "hnsw:M": 16,
            },
        )

    def name(self) -> str:
        """
        Return the name of the ChromaDB collection.
        """
        return self._collection.name

    def count(self) -> int:
        """
        Return the number of documents in the collection.
        """
        return self._collection.count()

    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """
        Add documents and their embeddings to the collection.
        """
        self.logger.debug(
            f"Adding {len(documents)} documents to collection '{self.name()}'"
        )
        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def get(self, ids: List[str], include: Optional[List[str]] = None) -> dict:
        """
        Retrieve documents by their IDs from the collection.
        """
        self.logger.debug(
            f"Retrieving {len(ids)} documents from collection '{self.name()}'"
        )
        return self._collection.get(ids=ids, include=include)

    def query(
        self,
        query_embeddings: Union[List[float], List[List[float]]],
        n_results: int = 5,
        include: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
    ) -> dict:
        """
        Wrapper to the chromadb query method.
        """
        if isinstance(query_embeddings[0], float):
            query_embeddings = [query_embeddings]
        self.logger.debug(
            f"Querying collection '{self.name()}' for top {n_results} results"
        )
        return self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include,
            ids=ids,
        )


class HybridStore:
    """
    Unified storage system combining vector databases and graph storage.

    HybridStore manages the complete storage infrastructure for TopG, including:
    - Vector storage for document propositions (ChromaDB)
    - Vector storage for entities (ChromaDB)
    - Graph storage for relationships between entities and propositions (SQLite)
    - Community detection and clustering for global query support

    It coordinates between different storage backends and provides a unified
    interface for data access and statistics.

    Attributes:
        config (Mapping[str, Any]): Configuration dictionary for storage parameters.
        document_store (ChromaEmbeddingStorage): Vector store for hyperpropositions.
        entity_store (ChromaEmbeddingStorage): Vector store for entities.
        sqlite_db (MapDB): SQLite database for graph and metadata storage.
        g_e (networkx.Graph): Graph where propositions are connected through entities.
        g_p (networkx.Graph): Graph where propositions are connected to their original passages.
        g_full (networkx.Graph): Complete graph: g_e + g_p.
        community_manager (CommunityManager): Manages hierarchical community detection.
        logger (logging.Logger): Logger instance.
    """

    def __init__(self, config: Mapping[str, Any], logger=None):
        self.config = config
        self.document_store = None
        self.entity_store = None
        self.sqlite_db = None
        self.g_e = None
        self.g_p = None
        self.g_full = None
        self.logger = logger or logging.getLogger(__name__)

        self.community_manager = CommunityManager(
            max_cluster_size=self.config.get("communities_max_cluster_size", 50),
            seed=self.config.get("communities_seed", 42),
            logger=self.logger,
        )

        self.init_storage()

    def init_storage(self):
        """
        Initialize all storage components: vector stores for documents and entities,
        and the SQLite database for graph storage.
        """
        self.logger.info("Initializing HybridStore")

        # ----- Embedding Parameters -----
        default_encoding_params = {
            "batch_size": 128,
            "convert_to_numpy": True,
            "show_progress_bar": True,
        }

        # ----- Embedding Functions -----
        self.logger.info("Initializing embedding functions for documents and entities")
        document_embedding_fn = ChromaEmbeddingFunction(
            model_name=self.config.get(
                "model_name_documents", "BAAI/bge-large-en-v1.5"
            ),
            encoding_params=self.config.get("encoding_params", default_encoding_params),
            device=self.config.get("device", "cpu"),
            trust_remote_code=True,
        )

        entity_embedding_fn = ChromaEmbeddingFunction(
            model_name=self.config.get(
                "model_name_entities", "sentence-transformers/all-mpnet-base-v2"
            ),
            encoding_params=self.config.get("encoding_params", default_encoding_params),
            device=self.config.get("device", "cpu"),
        )

        # ----- Chroma Document Store -----
        self.logger.info("Initializing Chroma document store")
        self.document_store = ChromaEmbeddingStorage(
            path=self.config.get("document_vdb_path", "./hybrid-db/vdb-docs"),
            collection_name=self.config.get("collection_name", "test"),
            embedding_fn=document_embedding_fn,
            logger=self.logger,
        )

        # ----- Chroma Entity Store -----
        self.logger.info("Initializing Chroma entity store")
        self.entity_store = ChromaEmbeddingStorage(
            path=self.config.get("entities_vdb_path", "./hybrid-db/vdb-entities"),
            collection_name=self.config.get("collection_name", "test"),
            embedding_fn=entity_embedding_fn,
            logger=self.logger,
        )

        # ----- SQLite DB -----
        self.logger.info("Initializing SQLite database")
        # TODO: MapDB should also be abstracted to allow different implementations
        self.sqlite_db = MapDB(
            db_path=self.config.get("sqlite_db_path", "./hybrid-db/sqlite_db.db"),
            logger=self.logger,
        )

        self.logger.info("Laoding Graphs database")
        self.load_graphs()

        self.logger.info("HybridStore initialized successfully")

    def show_statistics(self) -> None:
        """
        Show the stats from the different stores (propositions and entities)
        """
        self.logger.info(f"Propositions collection size: {self.document_store.count()}")
        self.logger.info(f"Entities collection size: {self.entity_store.count()}")

    def export_all_documents(self, output_path: str) -> None:
        """
        Exports all documents from the document store with parsed metadata.

        :param output_path: Path to save the exported JSON file.
        """
        self.logger.info("Exporting all documents from the store")

        # Get all documents from SQLite
        documents_metadata = self.sqlite_db.get_all_documents()

        # Save as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents_metadata, f, ensure_ascii=False, indent=4)

        self.logger.info(
            f"Exported {len(documents_metadata)} documents to {output_path}"
        )

    def export_all_passages(self, output_path: str) -> None:
        """
        Exports all passages from the document store.
        """
        self.logger.info("Exporting all passages from the store")
        passages = self.sqlite_db.get_all_passages()

        # Save as json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(passages, f, ensure_ascii=False, indent=4)

        self.logger.info(f"Exported {len(passages)} passages to {output_path}")

    def export_all_hyperpropositions(self, output_path: str) -> None:
        """
        Exports all hyperpropositions (propositions + entities) from the document store.
        """
        self.logger.info("Exporting all hyperpropositions from the store")
        hyperpropositions = self.document_store._collection.get()

        all_hyperpropositions = [
            {"id": _id, "page_content": doc, "metadata": meta}
            for _id, doc, meta in zip(
                hyperpropositions["ids"],
                hyperpropositions["documents"],
                hyperpropositions["metadatas"],
            )
        ]
        for hyperproposition in all_hyperpropositions:
            hyperproposition["metadata"]["entities"] = json.loads(
                hyperproposition["metadata"].get("entities", "[]")
            )
        # Save as json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_hyperpropositions, f, ensure_ascii=False, indent=4)

        self.logger.info(
            f"Exported {len(all_hyperpropositions)} passages to {output_path}"
        )

    def extract_sub_graph(self, node_lists):
        """
        Given a list of nodes, extract the corresponding subgraph from g_full
        """
        _a, _b, graph = self.sqlite_db.get_graphs()

        # get the subgraph
        subgraph = graph.subgraph(node_lists).copy()

        # prepare the nodes
        all_nodes = list(subgraph.nodes(data=True))
        all_edges = list(subgraph.edges())

        return {"nodes": all_nodes, "edges": all_edges}

    def load_graphs(self):
        """
        Utilitary function use to reload all the graphs
        """
        self.g_e, self.g_p, self.g_full = self.sqlite_db.get_graphs()
        self.logger.debug(
            f"Full Graph has {self.g_full.number_of_nodes()} nodes and {self.g_full.number_of_edges()} edges."
        )
        self.logger.debug(
            f"Entity Graph has {self.g_e.number_of_nodes()} nodes and {self.g_e.number_of_edges()} edges."
        )
        self.logger.debug(
            f"Proposition Graph has {self.g_p.number_of_nodes()} nodes and {self.g_p.number_of_edges()} edges."
        )
        self.logger.debug("Graph loaded successfully.")

        self.community_manager.compute(self.g_full)

    def get_statistics(self):
        """
        Return all the stats of the graph
        """
        n_nodes = self.g_full.number_of_nodes()
        n_edges = self.g_full.number_of_edges()
        n_passages = self.sqlite_db.get_number_of_passages()
        n_propositions = self.document_store.count()
        n_entities = self.entity_store.count()
        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_passages": n_passages,
            "n_propositions": n_propositions,
            "n_entities": n_entities,
        }
