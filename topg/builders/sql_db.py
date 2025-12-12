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
import sqlite3
import threading
from collections import defaultdict
from typing import List, Tuple

import networkx as nx
from tqdm import tqdm


class MapDB:
    """
    SQLite database manager for storing passages, propositions, entities, and their relationships.

    Maintains the graph structure with tables for documents, passages, propositions,
    entities, and entity-proposition mappings (emap).
    """

    def __init__(self, db_path, logger=None, chunk_size=60000, verbose=False):
        """
        Initialize the database connection, create tables if they do not exist,
        and set up logging.

        :param db_path: Path to the SQLite database file.
        :param log_file_path: Path to the log file (default is 'db_operations.log').
        """
        self.db_path = db_path
        self.verbose = verbose
        self.chunk_size = chunk_size
        # Check that sqlite_db_path directory exists
        sqlite_db_dir = os.path.dirname(db_path)
        if not os.path.exists(sqlite_db_dir):
            os.makedirs(sqlite_db_dir)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Add thread lock for cursor operations
        self._lock = threading.RLock()

        # Set up logging with the provided log file path
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info(f"Connecting to database at {db_path}")

        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            self.logger.info("Creating tables if they do not exist.")

            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    metadata TEXT NOT NULL
                )
                """
            )

            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS passages (
                passage_id TEXT PRIMARY KEY,
                page_content TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                collection TEXT NOT NULL
            );
            """
            )

            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS propositions (
                id TEXT PRIMARY KEY,
                passage_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                FOREIGN KEY (passage_id) REFERENCES passages (passage_id)
            );
            """
            )

            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                collection TEXT NOT NULL
            );
            """
            )

            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS emap (
                emap_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                proposition_id TEXT NOT NULL,
                passage_id TEXT NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES entities (entity_id),
                FOREIGN KEY (proposition_id) REFERENCES propositions (id),
                FOREIGN KEY (passage_id) REFERENCES passages (passage_id)
            );
            """
            )

            self.conn.commit()
            self.logger.info("Tables are created or already exist.")
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    def insert_document(self, doc_id: str, str_metadata: str) -> None:
        """Insert or ignore a document entry with its metadata."""
        try:
            with self._lock:
                self.cursor.execute(
                    """
                    INSERT OR IGNORE INTO documents (doc_id, metadata)
                    VALUES (?, ?)
                    """,
                    (doc_id, str_metadata),
                )
                self.conn.commit()
            self.logger.info(f"Document {doc_id} inserted successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting document {doc_id}: {e}")
            raise

    def insert_passages(self, passages, collection):
        """Insert multiple passages in batches."""

        insert_data = [
            (
                passage.metadata["passage_id"],
                passage.page_content,
                passage.metadata["doc_id"],
                collection,
            )
            for passage in passages
        ]

        self.logger.info(f"Inserting {len(insert_data)} passages into the database.")
        try:
            with self._lock:
                with self.conn:  # Ensures commit/rollback automatically
                    for batch in tqdm(
                        self.chunk_list(insert_data, self.chunk_size),
                        desc="Inserting passages",
                        unit="batch",
                    ):
                        self.cursor.executemany(
                            """
                            INSERT OR IGNORE INTO passages (passage_id, page_content, doc_id, collection)
                            VALUES (?, ?, ?, ?)
                            """,
                            batch,
                        )
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting passages: {e}")
            raise
        self.logger.info("All passages inserted successfully.")

    def insert_propositions(self, propositions, collection):
        """Insert propositions linking them to passages."""

        insert_data = [
            (prop.metadata["id"], prop.metadata["passage_id"], collection)
            for prop in propositions
        ]

        self.logger.info(
            f"Inserting {len(insert_data)} propositions into the database."
        )
        try:
            with self._lock:
                with self.conn:  # Ensures commit or rollback as needed
                    for batch in tqdm(
                        self.chunk_list(insert_data, self.chunk_size),
                        desc="Inserting propositions",
                        unit="batch",
                    ):
                        self.cursor.executemany(
                            """
                            INSERT OR IGNORE INTO propositions (id, passage_id, collection)
                            VALUES (?, ?, ?)
                            """,
                            batch,
                        )
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting propositions: {e}")
            raise

        self.logger.info("All propositions inserted successfully.")

    def insert_single_entities(self, entities, collection):
        """Insert entity labels with their IDs in batches."""

        insert_data = [
            (entity["entity_id"], entity["entity_label"], collection)
            for entity in entities
        ]

        self.logger.info(f"Inserting {len(insert_data)} entities.")
        try:
            with self._lock:
                with self.conn:  # Ensures all operations are in a single transaction
                    for batch in tqdm(
                        self.chunk_list(insert_data, self.chunk_size),
                        desc="Inserting entities",
                        unit="batch",
                    ):
                        self.cursor.executemany(
                            """
                            INSERT OR IGNORE INTO entities (entity_id, label, collection)
                            VALUES (?, ?, ?)
                            """,
                            batch,
                        )
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting entities: {e}")
            raise
        self.logger.info("All entities inserted successfully.")

    def insert_emap(self, entities):
        """Insert entity-proposition-passage mappings. This is the main table used for the graph."""

        # Prepare all rows in memory before inserting
        insert_data = [
            (
                f"{entity['entity_id']}{prop_id}",
                entity["entity_id"],
                prop_id,
                passage_id,
            )
            for entity in entities
            for prop_id, passage_id in zip(
                entity["propositions_id"], entity["passages_id"]
            )
        ]

        self.logger.info(f"Inserting {len(insert_data)} entity mappings.")
        try:
            with self._lock:
                with self.conn:  # Transaction block
                    for batch in tqdm(
                        self.chunk_list(insert_data, self.chunk_size),
                        desc="Inserting entity mapping",
                        unit="batch",
                    ):
                        self.cursor.executemany(
                            """
                            INSERT OR IGNORE INTO emap (emap_id, entity_id, proposition_id, passage_id)
                            VALUES (?, ?, ?, ?)
                            """,
                            batch,
                        )
        except AssertionError as ae:
            self.logger.error(f"Data shape error: {ae}")
            raise
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting entity mappings: {e}")
            raise
        self.logger.info("All entity mappings inserted successfully.")

    def insert_entities(self, entities, emap, collection):
        """Insert entities and their proposition mappings."""
        # insert data in the entities table
        self.insert_single_entities(entities, collection)
        # insert data in the emap table
        self.insert_emap(emap)

    def chunk_list(self, lst, size):
        """Yield successive n-sized chunks from a list."""
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    def get_documents_by_ids(self, doc_ids: List[str]) -> List[dict]:
        """Fetch documents by their IDs with metadata."""
        if not doc_ids:
            self.logger.warning("No document IDs provided.")
            return []

        result = {doc_id: {} for doc_id in doc_ids}
        batch_iter = self.chunk_list(doc_ids, self.chunk_size)

        for batch in tqdm(batch_iter, desc="Fetching documents by IDs", unit="batch"):
            placeholders = ",".join("?" for _ in batch)
            query = f"""
                SELECT doc_id, metadata
                FROM documents
                WHERE doc_id IN ({placeholders})
            """

            try:
                with self._lock:
                    self.cursor.execute(query, batch)
                    rows = self.cursor.fetchall()

                result.update(
                    {doc_id: json.loads(metadata) for doc_id, metadata in rows}
                )

            except sqlite3.Error as e:
                self.logger.error(f"Error fetching documents by IDs: {e}")
                raise

        fetched_count = sum(1 for v in result.values() if v)
        self.logger.info(
            f"Fetched {fetched_count}/{len(doc_ids)} available documents metadata."
        )

        return result

    def get_all_documents(self) -> dict:
        """Fetch all documents with their metadata."""
        try:
            self.logger.info("Fetching all documents from the database.")
            query = """
                SELECT doc_id, metadata
                FROM documents
            """
            with self._lock:
                self.cursor.execute(query)
                rows = self.cursor.fetchall()

            result = {doc_id: json.loads(metadata) for doc_id, metadata in rows}

            self.logger.info(f"Fetched {len(rows)} documents from the database.")
            return result

        except sqlite3.Error as e:
            self.logger.error(f"Error fetching documents: {e}")
            raise

    def get_all_entities_edges(self) -> List[Tuple[str, str, float]]:
        """
        Retrieve all edges from the 'edge_neighbors' table.

        :return: A list of tuples where each tuple represents an edge (node_id, neighbor_id, weight).
        """
        try:
            self.logger.info("Fetching all edges from the database.")
            query = """
                SELECT proposition_id, entity_id
                FROM emap
            """
            with self._lock:
                self.cursor.execute(query)
                edges = self.cursor.fetchall()

            self.logger.info(f"Fetched {len(edges)} p-e edges from the database.")
            return edges

        except sqlite3.Error as e:
            self.logger.error(f"Error fetching edges: {e}")
            raise

    def get_passages_by_proposition_ids(self, proposition_ids):
        """Fetch passages linked to propositions using batch queries."""
        if not proposition_ids:
            self.logger.warning("No proposition ids provided.")
            return {}

        result = {}

        batch_iter = self.chunk_list(proposition_ids, self.chunk_size)

        # Iterate over chunks to handle large lists of IDs
        for batch in tqdm(batch_iter, desc="Fetching passages", unit="batch"):
            placeholders = ",".join("?" for _ in batch)
            query = f"""
                SELECT DISTINCT p.passage_id, p.page_content, p.doc_id
                FROM propositions pr
                JOIN passages p ON p.passage_id = pr.passage_id
                WHERE pr.id IN ({placeholders})
            """

            try:
                with self._lock:
                    # Execute the query with the current batch of proposition_ids
                    self.cursor.execute(query, batch)
                    rows = self.cursor.fetchall()

                # Use dict update with dict comprehension for efficiency
                result.update(
                    {
                        passage_id: {
                            "page_content": page_content,
                            "doc_id": doc_id,
                        }
                        for passage_id, page_content, doc_id in rows
                    }
                )
            except sqlite3.Error as e:
                self.logger.error(f"Error fetching passages: {e}")
                return {}

        self.logger.info(
            f"Fetched {len(result)} passages for {len(proposition_ids)} proposition ids."
        )
        return result

    def get_passages_by_ids(self, passage_ids: List[str]) -> List[dict]:
        """Fetch passages by their IDs with metadata."""
        if not passage_ids:
            self.logger.warning("No passage IDs provided.")
            return []

        result = []
        batch_iter = self.chunk_list(passage_ids, self.chunk_size)

        for batch in tqdm(batch_iter, desc="Fetching passages by IDs", unit="batch"):
            placeholders = ",".join("?" for _ in batch)
            query = f"""
                SELECT passage_id, page_content, doc_id, collection
                FROM passages
                WHERE passage_id IN ({placeholders})
            """

            try:
                with self._lock:
                    self.cursor.execute(query, batch)
                    rows = self.cursor.fetchall()

                result.extend(
                    {
                        "passage_id": row[0],
                        "page_content": row[1],
                        "doc_id": row[2],
                        "collection": row[3],
                    }
                    for row in rows
                )
            except sqlite3.Error as e:
                self.logger.error(f"Error fetching passages by IDs: {e}")
                raise

        self.logger.info(f"Fetched {len(result)} passages for the provided IDs.")
        return result

    def get_propositions_by_entity_ids(self, entity_ids):
        """
        Fetch all proposition IDs related to a list of entity IDs, using batching for large requests.

        :param entity_ids: List of entity IDs to fetch proposition IDs for.
        :return: A dictionary where keys are entity IDs and values are lists of proposition IDs.
        """
        if not entity_ids:
            self.logger.warning("No entity IDs provided.")
            return {}

        result = defaultdict(set)
        batch_iter = self.chunk_list(entity_ids, self.chunk_size)

        for batch in tqdm(batch_iter, desc="Fetching propositions", unit="batch"):
            placeholders = ",".join("?" for _ in batch)
            query = f"""
                SELECT entity_id, proposition_id
                FROM emap
                WHERE entity_id IN ({placeholders})
            """

            try:
                with self._lock:
                    self.cursor.execute(query, batch)
                    rows = self.cursor.fetchall()

                for entity_id, proposition_id in rows:
                    result[entity_id].add(proposition_id)

            except sqlite3.Error as e:
                self.logger.error(f"Error fetching propositions: {e}")
                return {}

        # Convert sets to lists and defaultdict to a normal dictionary
        result = {
            entity_id: list(propositions) for entity_id, propositions in result.items()
        }

        self.logger.info(f"Fetched propositions for {len(result)} unique entity IDs.")

        return result

    def get_entities_by_proposition_ids(self, proposition_ids):
        """
        Fetch entities from proposition idss.
        """
        if not proposition_ids:
            self.logger.warning("No proposition IDs provided.")
            return {}

        result = {}
        batch_iter = self.chunk_list(proposition_ids, self.chunk_size)

        for batch in tqdm(batch_iter, desc="Fetching entities", unit="batch"):
            placeholders = ",".join("?" for _ in batch)
            query = f"""
                SELECT DISTINCT e.entity_id, en.label
                FROM emap e
                JOIN entities en ON e.entity_id = en.entity_id
                WHERE e.proposition_id IN ({placeholders})
            """

            try:
                with self._lock:
                    self.cursor.execute(query, batch)
                    rows = self.cursor.fetchall()

                result.update({entity_id: label for entity_id, label in rows})

            except sqlite3.Error as e:
                self.logger.error(f"Error fetching entities: {e}")
                return {}

        self.logger.info(f"Fetched entities for {len(result)} unique proposition IDs.")
        return result

    def close(self):
        """Close the database connection."""
        try:
            self.logger.info("Closing database connection.")
            self.conn.close()
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")
            raise

    def get_table_row_count(self, table_name):
        """
        Get the row count for a given table.

        :param table_name: The name of the table to query.
        :return: The number of rows in the table.
        """
        try:
            with self._lock:
                self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = self.cursor.fetchone()[0]

            return row_count
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving row count for table {table_name}: {e}")
            raise

    def get_db_size(self):
        """
        Get the size of the database file.

        :return: The size of the database in bytes.
        """
        try:
            db_size = os.path.getsize(self.db_path)
            return db_size
        except OSError as e:
            self.logger.error(f"Error getting size of database file: {e}")
            raise

    def show_db_statistics(self):
        """
        Display basic statistics about the database: table row counts and file size.
        """
        try:
            self.logger.info("Fetching database statistics...")

            # List of all tables to fetch statistics for
            tables = ["passages", "propositions", "entities", "emap"]

            # Fetch row counts for each table
            table_stats = {}
            for table in tables:
                try:
                    table_stats[table] = self.get_table_row_count(table)
                except Exception as e:
                    self.logger.warning(
                        f"Could not fetch row count for table '{table}': {e}"
                    )
                    table_stats[table] = "Error"

            # Get database size
            db_size = self.get_db_size()

            # Display the statistics
            self.logger.info(f"Database Size: {db_size / 1024 / 1024:.2f} MB")
            for table, count in table_stats.items():
                self.logger.info(f"Total Rows in '{table}': {count}")

        except Exception as e:
            self.logger.error(f"Error fetching database statistics: {e}")
            raise

    def get_g_p2e(self):
        # Create a set of unique node names
        edges = self.get_all_entities_edges()

        g_p2e = nx.Graph()

        # Add edges with weights
        for src, tgt in edges:
            g_p2e.add_edge(src, tgt)

        nx.set_node_attributes(g_p2e, {src: {"type": "p"} for src, tgt in edges})
        nx.set_node_attributes(g_p2e, {tgt: {"type": "e"} for src, tgt in edges})

        return g_p2e

    def get_g_p2P(self):
        # Get all passage-proposition links
        tuples_list = self.get_all_passage_proposition_links()

        g_p2P = nx.Graph()

        for passage_id, proposition_id in tuples_list:
            g_p2P.add_edge(passage_id, proposition_id)

        nx.set_node_attributes(
            g_p2P,
            {passage_id: {"type": "P"} for passage_id, proposition_id in tuples_list},
        )
        nx.set_node_attributes(
            g_p2P,
            {
                proposition_id: {"type": "p"}
                for passage_id, proposition_id in tuples_list
            },
        )

        return g_p2P

    def get_all_passages(self) -> List[dict]:
        """
        Retrieve all passages from the 'passages' table.

        :return: A list of dictionaries where each dictionary represents a passage with its metadata.
        """
        try:
            self.logger.info("Fetching all passages from the database.")
            query = """
                SELECT passage_id, page_content, doc_id, collection
                FROM passages
            """
            with self._lock:
                self.cursor.execute(query)
                rows = self.cursor.fetchall()

            passages = [
                {
                    "passage_id": row[0],
                    "page_content": row[1],
                    "doc_id": row[2],
                    "collection": row[3],
                }
                for row in rows
            ]
            self.logger.info(f"Fetched {len(passages)} passages from the database.")
            return passages
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching passages: {e}")
            raise

    def get_all_passage_proposition_links(self) -> List[Tuple[str, str]]:
        """
        Retrieve all links between passage IDs and proposition IDs from the 'propositions' table.
        :return: A list of tuples where each tuple represents a link (passage_id, proposition_id).
        """
        try:
            self.logger.info(
                "Fetching all passage-proposition links from the database."
            )
            query = """
                SELECT passage_id, id
                FROM propositions
            """
            with self._lock:
                self.cursor.execute(query)
                links = self.cursor.fetchall()
            self.logger.info(
                f"Fetched {len(links)} passage-proposition links from the database."
            )
            return links
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching passage-proposition links: {e}")
            raise

    def get_graphs(self):
        g_p2e = self.get_g_p2e()
        self.logger.info(
            f"Graph size via entities: {g_p2e.number_of_nodes()} vertices, {g_p2e.number_of_edges()} edges"
        )

        g_p2P = self.get_g_p2P()
        self.logger.info(
            f"Graph size via passage: {g_p2P.number_of_nodes()} vertices, {g_p2P.number_of_edges()} edges"
        )
        # Create merged graph by copying the first graph
        g_p2e2P = nx.compose(g_p2e, g_p2P)

        return g_p2e, g_p2P, g_p2e2P

    def get_number_of_passages(self) -> int:
        """
        Get the total number of passages in the 'passages' table.

        :return: The total count of passages.
        """
        try:
            self.logger.info("Fetching the total number of passages.")
            query = "SELECT COUNT(*) FROM passages"
            with self._lock:
                self.cursor.execute(query)
                count = self.cursor.fetchone()[0]
            self.logger.info(f"Total number of passages: {count}")
            return count
        except sqlite3.Error as e:
            self.logger.error(f"Error fetching the number of passages: {e}")
            raise
