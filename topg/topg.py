#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#


import json
import os
import random
from logging import INFO
from typing import Any, Mapping

import numpy as np
import yaml
from langchain_core.documents.base import Document

from topg.builders.chunking import GeneralDocumentProcessor, ParagraphSplitter
from topg.builders.hybrid_vector_concept_db import HybridBaseLoader
from topg.builders.storage import HybridStore
from topg.rag.hybrid_rag import HybridRetrieval
from topg.rag.query_manager import QueryManager
from topg.utils.llm import configure_llm, get_cost
from topg.utils.logger import get_std_logger


class Topg:
    """
    Main TopG system class that manages document indexing and querying.

    TopG (Think On Proposition Graph) is a hybrid RAG framework that builds a
    graph from passages, entities, and propositions (facts). It supports three query modes:
    - naive: single-shot retrieval
    - local: iterative multi-hop reasoning
    - global: community-based decomposition and synthesis

    Attributes:
        config (dict): Configuration dictionary containing parameters for all components.
        logger (logging.Logger): Logger instance for system operations.
        document_processor (GeneralDocumentProcessor): Processes documents into passages and propositions. Used in the document processing pipeline.
        store (HybridStore): Manages vector and graph storage for documents, entities, and propositions.
        loader (HybridBaseLoader): Indexes documents, passages, and propositions into storage.
        retriever (HybridRetrieval): Retrieves relevant information from the knowledge base. This contains for instance the `suggest` method used for the suggest/select cycles.
        query_manager (QueryManager): Manages different query modes and routing.
        lm (dspy.LM): Configured language model instance.
    """

    def __init__(
        self,
        config: dict,
        seed: int = 42,
    ):
        self.config = config
        self.logger = get_std_logger(**self.config.get("logger_params", {}))
        self.logger.info("Initializing Topg with configuration")

        self.document_processor = GeneralDocumentProcessor(
            config=self.config.get("document_processor_params", {}),
            logger=self.logger,
        )
        self.store = HybridStore(
            config=self.config.get("storage_params", {}), logger=self.logger
        )

        self.logger.info("Initializing the base loader")
        self.loader = HybridBaseLoader(
            storage=self.store,
            config=self.config.get("loaders_params", {}),
            logger=self.logger,
        )
        self.logger.info("Initializing the retriever")
        self.retriever = HybridRetrieval(store=self.store, logger=self.logger)

        self.logger.info("Initializing the query manager")
        self.query_manager = QueryManager(
            retriever=self.retriever,
            logger=self.logger,
        )

        # Load the LLM
        self.logger.info("Loading the LLM config")
        self.lm = configure_llm(llm_config=self.config.get("llm_config", {}))

        self.logger.info("All initializations ok!")

        self.set_global_seed(seed)

    def add_document_metadata(
        self, doc_id: str, metadata: Mapping[str, Any] = {}
    ) -> None:
        """
        Add metadata for a document without indexing its content.

        This method registers document-level metadata in the knowledge base,
        which can be associated with passages and propositions later.
        This is useful when you want to add things like a citation, some authors or any additional metadata
        you would like to keep associated with the documents from which originate the passages and propositions.

        Args:
            doc_id (str): Unique identifier for the document.
            metadata (Mapping[str, Any], optional): Arbitrary metadata dictionary.
                Can include fields like title, source, date, etc. Defaults to {}.

        Returns:
            None
        """
        self.loader.index_document(doc_id, metadata)

    def insert(self, text: str, doc_id: str = None, **kwargs) -> None:
        """
        Insert a docunent text into the knowledge base.

        This method processes the input text through the complete indexing pipeline:
        1. Splits text into passages using configured chunking strategy
        2. Extracts propositions (facts) along with named entities from passages
        3. Indexes passages, propositions, entities, and relationships into storage


        Args:
            text (str): The document text to be indexed.
            doc_id (str, optional): Unique identifier for the document. If None,
                a unique ID will be generated. Defaults to None.
            **kwargs: Additional arguments passed to the document processor.

        Returns:
            None

        Note:
            This method will log the total cost of LLM calls made during processing.
        """
        passages = self.document_processor.get_passages(text, doc_id=doc_id)

        # Index the passages
        self.loader.index_passages(passages)

        # Extract hyperpropositions from passages
        hyperpropositions = self.document_processor.get_hyperpropositions(passages)

        # While duplicated ids will not be added - we should prevent duplicated ids in the same batch insert ! - We keep the first one
        all_ids = []
        filtered_hyperpropositions = []
        for hyperproposition in hyperpropositions:
            if hyperproposition.metadata["id"] not in all_ids:
                all_ids.append(hyperproposition.metadata["id"])
                filtered_hyperpropositions.append(hyperproposition)

        # Index the hyperpropositions
        self.loader.index_propositions(filtered_hyperpropositions)

        # self.store.show_statistics()

        cost = get_cost(self.lm)
        self.logger.info(f"Total cost of LLM calls: {cost:.4f} USD")

    def query(
        self,
        question: str,
        mode: str = "local",
        *args,
        **kwargs,
    ) -> Any:
        """
        Query the knowledge base to answer a question.

        Args:
            question (str): The question to answer.
            mode (str, optional): Query mode to use. Options:
                - "naive": Single-shot retrieval and answering
                - "local": Iterative multi-hop reasoning with entity-centric graph traversal
                - "global": Community-based question decomposition and synthesis
                Defaults to "local".
            *args: Additional positional arguments passed to the query manager.
            **kwargs: Additional keyword arguments. Please check the documentation of each query mode for specific parameters.

        Returns:
            Any: The answer result. Structure depends on the query mode used.
        """
        answer = self.query_manager.query(
            question=question,
            mode=mode,
            *args,
            **kwargs,
        )
        return answer

    def load_config(self, config_path: str) -> Mapping[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            Mapping[str, Any]: Configuration dictionary loaded from the file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the file contains invalid YAML syntax.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def set_global_seed(self, seed: int = 42) -> None:
        """
        Set the global random seed for reproducibility.

        This method sets the random seed for both Python's random module and NumPy
        to ensure reproducible results across multiple runs.

        Args:
            seed (int, optional): The random seed value. Defaults to 42.

        Returns:
            None
        """
        random.seed(seed)

        # Set seed for NumPy's global random state
        np.random.seed(seed)

    @staticmethod
    def initialize(base_path: str, collection_name: str = "test") -> str:
        """
        Initialize a new TopG project structure with default configuration.

        This static method creates the necessary directory structure and generates
        a default configuration file for a new TopG knowledge base project. The
        created structure includes:
        - Project root directory
        - dbs/ directory for vector and SQL databases
        - logs_and_cache/ directory for logging and caching
        - config.yaml with default parameters

        Args:
            base_path (str): The directory where the project should be created.
                If it doesn't exist, it will be created.
            collection_name (str, optional): Name of the collection (project name), used as the
                default logger name and storage collection identifier.
                Defaults to "test".

        Returns:
            str: Absolute path to the generated config.yaml file.

        Raises:
            FileExistsError: If a config.yaml already exists in the base_path,
                indicating the project has already been initialized.

        Example:
            >>> config_path = Topg.initialize(
            ...     base_path="/path/to/my_kb",
            ...     collection_name="MyKnowledgeBase"
            ... )
            >>> with open(config_path, "r") as f:
            ...     config = yaml.safe_load(f)
            >>> system = Topg(config=config)
        """
        project_dir = os.path.join(base_path)
        dbs_dir = os.path.join(project_dir, "dbs")
        logs_dir = os.path.join(project_dir, "logs_and_cache")

        if os.path.exists(os.path.join(project_dir, "config.yaml")):
            raise FileExistsError(
                f"Project directory '{project_dir}' already exists. Please choose a different name or if you want to create an other db."
            )

        # Create directories
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(dbs_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Default configuration
        config: Dict[str, Any] = {
            "logger_params": {
                "name": collection_name,
                "path": logs_dir,
                "stdout": True,
                "level": "INFO",
            },
            "document_processor_params": {
                "paragraph_separator": "\n\n\n",
                "paragraph_size_limit": 8,
                "paragraph_sent_overlap": 2,
                "cache_dir": logs_dir,
                "max_workers": 8,
            },
            "storage_params": {
                "document_vdb_path": os.path.join(dbs_dir, "vdb-docs"),
                "entities_vdb_path": os.path.join(dbs_dir, "vdb-entities"),
                "sqlite_db_path": os.path.join(dbs_dir, "sqlite_db.db"),
                "collection_name": collection_name,
                "device": "cuda",
                "model_name_documents": "BAAI/bge-large-en-v1.5",
                "model_name_entities": "BAAI/bge-large-en-v1.5",
                "encoding_params": {
                    "batch_size": 128,
                    "convert_to_numpy": True,
                    "show_progress_bar": True,
                },
                "communities_max_cluster_size": 50,
                "communities_seed": 42,
            },
            "loaders_params": {
                "loading_batch_size": 512,
                "graph_resolve_synonym": True,
                "graph_synonym_max_limit": 10,
                "graph_synonym_sim_threshold": 0.9,
            },
            "llm_config": {
                "api_base": None,
                "llm_name": "openai/gpt-4o-mini",
                "max_tokens": 2048,
            },
        }

        # Save config to YAML
        config_path = os.path.join(project_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return config_path

    def load_passages_from_json(self, json_passages: str):
        """
        Load and index pre-chunked passages from a JSON file.

        This method allows bulk loading of passages that have been previously
        processed and saved to JSON format. Useful for re-indexing or loading
        cached data without re-processing documents.

        Args:
            json_passages (str): Path to JSON file containing passages.
                Expected format: List of dicts with keys:
                    - page_content (str): The passage text
                    - passage_id (str): Unique passage identifier
                    - doc_id (str): Parent document identifier
                    - collection (str): Collection name

        Example:
            {
                "passage_id": "passage_001",
                "page_content": "This is an example passage text.",
                "doc_id": "doc-001",
                "collection": "test"
            }
        Returns:
            None

        Note:
            This method logs the number of passages loaded and shows storage statistics.
        """
        with open(json_passages, "r") as f:
            passages = [
                Document(
                    page_content=passage["page_content"],
                    metadata={
                        "passage_id": passage["passage_id"],
                        "doc_id": passage["doc_id"],
                        "collection": passage["collection"],
                    },
                )
                for passage in json.load(f)
            ]
        self.logger.info(f"Number of passages to insert: {len(passages)}")
        self.loader.index_passages(passages)
        self.logger.info("Passages indexed.")
        self.store.show_statistics()

    def load_hyperpropositions_from_json(self, json_hyperpropositions: str):
        """
        Load and index pre-extracted hyperpropositions from a JSON file.

        This method allows bulk loading of hyperpropositions (atomic facts with entities)
        that have been previously extracted and saved to JSON format.

        Args:
            json_hyperpropositions (str): Path to JSON file containing hyperpropositions.
            Expected format: List of dicts with keys:
                - page_content (str): The proposition text
                - metadata (dict): Containing:
                    - id (str): Unique proposition identifier
                    - passage_id (str): Parent passage identifier
                    - entities (List[str]): List of entity names in the proposition

        Example:
        {
            "id": "proposition_001",
            "page_content": "This is an example proposition text.",
            "metadata": {
                "entities": [
                    "entity_text_1",
                    "entity_text_2",
                    "entity_text_3"
                ],
                "passage_id": "passage_001",
                "id": "proposition_001"
            }
        }

        Returns:
            None

        Note:
            This method logs the number of hyperpropositions loaded and shows storage statistics.
        """
        with open(json_hyperpropositions, "r") as f:
            new_hyperpropositions = [
                Document(
                    page_content=hyperproposition["page_content"],
                    metadata={
                        "id": hyperproposition["metadata"]["id"],
                        "passage_id": hyperproposition["metadata"]["passage_id"],
                        "entities": hyperproposition["metadata"]["entities"],
                    },
                )
                for hyperproposition in json.load(f)
            ]

        self.logger.info(
            f"Number of hyperpropositions to insert: {len(new_hyperpropositions)}"
        )
        self.loader.index_propositions(new_hyperpropositions)
        self.logger.info("Hyperpropositions indexed.")
        self.store.show_statistics()

    def load_documents_from_json(self, json_documents: str):
        """
        Load and index document metadata from a JSON file.

        This method allows bulk loading of document-level metadata without
        loading the actual document content. Useful for maintaining document
        information in the knowledge base.

        Args:
            json_documents (str): Path to JSON file containing document metadata.
                Expected format: Dict mapping doc_id (str) to metadata dict (dict).
                The metadata can contain arbitrary fields like title, source, date, etc.

        Returns:
            None

        Note:
            This method logs the number of documents indexed.
        """
        with open(json_documents, "r") as f:
            documents_metadata = json.load(f)

        for doc_id, doc_metadata in documents_metadata.items():
            self.loader.index_document(doc_id, doc_metadata)

        self.logger.info(f"{len(documents_metadata)} documents indexed.")
