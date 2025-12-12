#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from graspologic.partition import hierarchical_leiden


class CommunityManager:
    """
    Manages hierarchical community detection for in the graph of the knowledge base.

    Attributes:
        max_cluster_size (int): Maximum size of clusters at the lowest level.
        seed (int): Random seed for reproducibility.
        logger (logging.Logger): Logger for logging information.
    """

    def __init__(
        self,
        max_cluster_size: int = 50,
        seed: int = 42,
        logger: Optional[logging.Logger] = None,
    ):
        self.max_cluster_size = max_cluster_size
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)

        # Community data structures
        self.node_id_to_community_map: Dict[int, Dict[str, int]] = {}
        self.parent_mapping: Dict[int, int] = {}
        self.community2node: Dict[int, List[str]] = {}
        self.node2community: Dict[str, List[int]] = defaultdict(list)
        self.community_sizes: Dict[int, int] = {}
        self.children: Dict[int, List[int]] = defaultdict(list)
        self.levels: List[int] = []

        # Node type caches
        self.propositions_nodes: Set[str] = set()
        self.entities_nodes: Set[str] = set()

        self._is_computed: bool = False

    def compute(self, graph: nx.Graph) -> None:
        """
        Compute communities for the given graph.

        Args:
            graph: NetworkX graph to compute communities for
        """
        if graph is None or len(graph.nodes) == 0:
            self.logger.warning("The graph is empty. Cannot compute communities.")
            self._clear()
            return

        self.logger.info("Computing Leiden communities...")

        # Extract node types
        self.propositions_nodes = {
            node[0] for node in graph.nodes(data=True) if node[1].get("type") == "p"
        }
        self.entities_nodes = {
            node[0] for node in graph.nodes(data=True) if node[1].get("type") == "e"
        }

        # Compute Leiden communities
        self.node_id_to_community_map, self.parent_mapping = (
            self._compute_leiden_communities(graph, self.max_cluster_size, self.seed)
        )

        self.levels = sorted(self.node_id_to_community_map.keys())

        # Build community2node mapping
        self.community2node = {}
        for level in self.levels:
            for node_id, community_id in self.node_id_to_community_map[level].items():
                if community_id not in self.community2node:
                    self.community2node[community_id] = []
                self.community2node[community_id].append(node_id)

        # Build node2community mapping
        self.node2community = defaultdict(list)
        for community_map in self.node_id_to_community_map.values():
            for node_id, community_id in community_map.items():
                self.node2community[node_id].append(community_id)

        # Add root community
        for node_id in self.node2community:
            self.node2community[node_id].insert(0, -1)

        # Compute community sizes
        self.community_sizes = {
            cid: len(nodes) for cid, nodes in self.community2node.items()
        }
        self.community_sizes[-1] = len(graph.nodes)

        # Build children mapping
        self.children = defaultdict(list)
        for child, parent in self.parent_mapping.items():
            self.children[parent].append(child)

        self._is_computed = True

        self.logger.info(
            f"Computed {len(self.community2node)} communities across {len(self.levels)} levels"
        )

    def _clear(self) -> None:
        """Clear all community data."""
        self.node_id_to_community_map = {}
        self.parent_mapping = {}
        self.community2node = {}
        self.node2community = defaultdict(list)
        self.community_sizes = {}
        self.children = defaultdict(list)
        self.levels = []
        self.propositions_nodes = set()
        self.entities_nodes = set()
        self._is_computed = False

    @staticmethod
    def _compute_leiden_communities(
        graph: nx.Graph | nx.DiGraph,
        max_cluster_size: int,
        seed: int | None = None,
    ) -> Tuple[Dict[int, Dict[str, int]], Dict[int, int]]:
        """Return Leiden root communities and their hierarchy mapping."""
        community_mapping = hierarchical_leiden(
            graph, max_cluster_size=max_cluster_size, random_seed=seed
        )
        results: Dict[int, Dict[str, int]] = {}
        hierarchy: Dict[int, int] = {}

        for partition in community_mapping:
            results[partition.level] = results.get(partition.level, {})
            results[partition.level][partition.node] = partition.cluster
            hierarchy[partition.cluster] = (
                partition.parent_cluster if partition.parent_cluster is not None else -1
            )

        return results, hierarchy

    def is_computed(self) -> bool:
        """Check if communities have been computed."""
        return self._is_computed

    def get_statistics(self) -> Dict:
        """Get statistics about computed communities."""
        if not self._is_computed:
            return {
                "n_communities": 0,
                "n_levels": 0,
                "n_propositions": 0,
                "n_entities": 0,
                "largest_community": 0,
                "smallest_community": 0,
            }

        return {
            "n_communities": len(self.community2node),
            "n_levels": len(self.levels),
            "n_propositions": len(self.propositions_nodes),
            "n_entities": len(self.entities_nodes),
            "largest_community": max(self.community_sizes.values())
            if self.community_sizes
            else 0,
            "smallest_community": min(v for v in self.community_sizes.values() if v > 0)
            if self.community_sizes
            else 0,
        }
