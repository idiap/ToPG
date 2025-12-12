#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#


import logging
from typing import Any, List, Literal, Mapping, Optional

import dspy
import numpy as np

from topg.rag.global_query import GlobalQAProcessor
from topg.rag.local_query import LocalQAProcessor
from topg.rag.naive_query import NaiveQAProcessor

QA_MODE = Literal["naive", "local", "global"]


class QueryManager:
    """
    Manages different query modes and routes queries to appropriate processing engines.

    QueryManager acts as a central dispatcher for handling questions in different modes:
    - naive: Single-shot retrieval and direct answering
    - local: Iterative multi-hop reasoning with entity-centric graph traversal
    - global: Community-based question decomposition and synthesis

    It maintains separate processor instances for each mode and provides utilities
    for customizing prompts, demonstrations, and inspecting available predictors.

    Attributes:
        logger (logging.Logger): Logger instance for debugging and info messages.
        _engines (Mapping[QA_MODE, Any]): Dictionary mapping mode names to processor instances.
        _default_mode (QA_MODE): Default query mode when none is specified.
        _last_mode (QA_MODE): The mode used in the most recent query.
    """

    def __init__(
        self,
        retriever,
        default_mode: QA_MODE = "local",
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self._engines: Mapping[QA_MODE, Any] = {
            "naive": NaiveQAProcessor(
                retriever=retriever,
                logger=self.logger,
            ),
            "local": LocalQAProcessor(
                retriever=retriever,
                logger=self.logger,
            ),
            "global": GlobalQAProcessor(
                retriever=retriever,
                logger=self.logger,
            ),
        }
        if default_mode not in self._engines:
            raise ValueError(f"Unknown default_mode '{default_mode}'")

        self._default_mode: QA_MODE = default_mode
        self._last_mode: QA_MODE = default_mode

        self.logger.debug(f"QueryManager initialised with default mode {default_mode}.")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def query(
        self,
        question,
        mode: Optional[QA_MODE] = None,
        # Optional retriever args
        seeding_e_syn_k: int = 5,
        seeding_e_syn_threshold: float = 0.80,
        seeding_lambda_mmr: float = 1.0,
        seeding_top_k: int = 20,
        lambda_: float = 0.5,
        damping: float = 0.85,
        cosine_threshold: float = 0.4,
        horizon_top_k: int = 500,
        temperature: float = 0.1,
        top_k: int = 20,
        *args,
        **kwargs,
    ):
        """
        Route a question to the appropriate QA engine and return the answer.

        This method delegates question answering to the selected engine (naive, local,
        or global) with the specified retrieval parameters. If no mode is specified,
        uses the default mode configured during initialization.

        Args:
            question (str): The question to answer.
            mode (Optional[QA_MODE], optional): Query mode override for this call.
                If None, uses the default mode. Defaults to None.

            -- Retriever arguments --

            seeding_e_syn_k (int, optional): Number of similar entities to consider
                during initial seeding. Defaults to 5.
            seeding_e_syn_threshold (float, optional): Cosine similarity threshold
                for entity synonyms during seeding. Defaults to 0.80.
            seeding_lambda_mmr (float, optional): MMR lambda parameter for diversity
                vs relevance tradeoff in initial retrieval (1.0 = pure relevance).
                Defaults to 1.0.
            seeding_top_k (int, optional): Number of propositions to retrieve during
                initial seeding. Defaults to 20.
            lambda_ (float, optional): Interpolation weight between vector similarity
                and graph structure in random walk. Defaults to 0.5.
            damping (float, optional): PageRank-style damping factor for random walk.
                Defaults to 0.85.
            cosine_threshold (float, optional): Minimum cosine similarity for
                propositions to be included. Defaults to 0.4.
            horizon_top_k (int, optional): Maximum number of propositions to keep
                in the active horizon. Defaults to 500.
            temperature (float, optional): Temperature parameter for sampling
                during retrieval. Defaults to 0.1.
            top_k (int, optional): Final number of propositions to use for answering.
                Defaults to 20.
            *args: Additional positional arguments passed to the engine.
            **kwargs: Additional keyword arguments passed to the engine.

            -- naive mode arguments --

            - use_passages (bool, optional): Integrate the passages associated to
                the extracted propositions in the final context. (default `False`)
            - context_as_references (bool, optional): Return all metadata in the
                `context` attribute for the Inference Memory Block.  (default `False`)
            - callback_logger (callable, optional): An optional function use as a
                callback logger everytime the memory is updated. It will takes as `data`
                argument the current `MemoryBlock`.

            -- local mode arguments --

            - max_iter (int, optional): Max reasoning steps (default: 2).
            - use_passages (bool, optional): Integrate the passages associated
                to the extracted propositions in the final context. (default `False`)
            - context_as_references (bool, optional): Return all metadata in the
                `context` attribute for the Inference Memory Block.  (default `False`)
            - callback_logger (callable, optional): An optional function use as a
                callback logger everytime the memory is updated. It will takes as `data`
                argument the current `MemoryBlock`.

            -- global mode arguments --
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

        Returns:
            Any: The answer result, structure depends on the query mode used.
        """
        engine_mode: QA_MODE = mode or self._default_mode
        engine = self._get_engine(engine_mode)

        # Build the retriever args dict
        retriever_args = {
            "initial_retriever_args": {
                "e_syn_k": seeding_e_syn_k,
                "e_syn_threshold": seeding_e_syn_threshold,
                "lambda_mmr": seeding_lambda_mmr,
                "top_k": seeding_top_k,
            },
            "q": lambda_,
            "damping": damping,
            "cosine_threshold": cosine_threshold,
            "horizon_top_k": horizon_top_k,
            "temperature": temperature,
            "top_k": top_k,
        }

        self.logger.debug("Routing query to %s", engine_mode)
        result = engine.query(
            question=question,
            retriever_args=retriever_args,
            *args,
            **kwargs,
        )

        self._last_mode = engine_mode
        return result

    def _get_engine(self, mode: QA_MODE) -> Any:
        """
        Retrieve the QA processor engine for the specified mode.

        Args:
            mode (QA_MODE): The query mode ('naive', 'local', or 'global').

        Returns:
            Any: The corresponding QA processor instance.

        Raises:
            ValueError: If the mode is not recognized.
        """
        try:
            return self._engines[mode]
        except KeyError:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes: {list(self._engines)}"
            ) from None

    def set_custom_instruction_prompt(
        self, mode: QA_MODE, predictor_name: str, instruction_prompt: str
    ):
        """
        Set a custom instruction prompt for a specific predictor in a query mode.

        This method allows fine-tuning the behavior of individual predictors by
        modifying their instruction prompts. Useful for domain adaptation or
        specialized reasoning patterns.

        Args:
            mode (QA_MODE): The query mode containing the predictor. One of
                ['naive', 'local', 'global'].
            predictor_name (str): The name of the predictor to update. Use
                list_predictors() to see available predictor names for each mode.
            instruction_prompt (str): The new instruction prompt text to set.

        Raises:
            ValueError: If the mode is invalid or the predictor name is not found.

        Example:
            >>> manager.set_custom_instruction_prompt(
            ...     mode="local",
            ...     predictor_name="reasoner",
            ...     instruction_prompt="Analyze the medical evidence carefully..."
            ... )
        """
        if mode not in self._engines:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes are {list(self._engines.keys())}"
            )

        query_engine = self._engines[mode]
        try:
            predictor = query_engine.get_predictor(predictor_name)
            self.logger.info(
                "Setting instruction prompt for '%s' in mode '%s'",
                predictor_name,
                mode,
            )
            predictor.signature.instructions = instruction_prompt
        except ValueError:
            self.logger.error(
                f"Predictor '{predictor_name}' in mode '{mode}' does not support custom instructions."
            )

    def set_evaluator_demonstrations(self, mode: QA_MODE, demonstrations: List[dict]):
        """
        Set demonstration examples for the evaluator agent in a specific mode.

        Demonstrations provide few-shot examples to guide the evaluator's behavior
        when assessing whether sufficient information has been gathered to answer
        a question. Only applicable to 'naive' and 'local' modes.

        Args:
            mode (QA_MODE): The query mode. Must be 'naive' or 'local'.
            demonstrations (List[dict]): List of demonstration examples. Each dict
                should contain:
                    - question (str): The example question
                    - facts (List[str]): Retrieved facts
                    - answer (str): Expected answer or evaluation
                    - passages (List[str], optional): Source passages

        Raises:
            ValueError: If the mode is not 'naive' or 'local'.

        Example:
            >>> demos = [{
            ...     "question": "What is the capital of France?",
            ...     "facts": "Paris is the capital of France.",
            ...     "answer": "Paris"
            ... }]
            >>> manager.set_evaluator_demonstrations(mode="naive", demonstrations=demos)
        """
        if mode not in ["naive", "local"]:
            raise ValueError(
                f"Evaluator demonstrations can only be set for 'naive' or 'local' modes. Given mode: '{mode}'"
            )

        query_engine = self._engines[mode]

        if not demonstrations:
            self.logger.info(
                "No demonstrations provided. Clearing evaluator demonstrations."
            )
            query_engine.demos_evaluator = []

        # Check if passages are included in the demonstrations
        if "passages" in demonstrations[0]:
            query_engine.demos_evaluator = [
                dspy.Example(**demo).with_inputs("question", "facts", "passages")
                for demo in demonstrations
            ]
        else:
            query_engine.demos_evaluator = [
                dspy.Example(**demo).with_inputs("question", "facts")
                for demo in demonstrations
            ]
        self.logger.info(
            f"Evaluator demonstrations set: {len(query_engine.demos_evaluator)} examples."
        )

    def list_predictors(self, mode: Optional[QA_MODE] = None) -> dict:
        """
        List available predictors and their signatures for query mode(s).

        This introspection method helps identify which predictors are available
        for customization in each query mode, along with their types and signature
        information (instructions, inputs, outputs).

        Args:
            mode (Optional[QA_MODE], optional): Specific mode to inspect. If None,
                returns information for all available modes. Defaults to None.

        Returns:
            dict: Dictionary mapping mode names to lists of predictor descriptors.
                Each descriptor contains:
                    - name (str): Predictor identifier
                    - attribute (str): Engine attribute name
                    - type (str): Predictor class name (e.g., 'ChainOfThought', 'Predict')
                    - signature (dict): Signature info with instructions, inputs, outputs
                    - error (str, optional): Error message if inspection failed

        Example:
            >>> predictors = manager.list_predictors(mode="local")
            >>> for pred in predictors["local"]:
            ...     print(f"{pred['name']}: {pred['type']}")
        """
        modes = [mode] if mode else list(self._engines.keys())
        out = {}
        for m in modes:
            if m not in self._engines:
                out[m] = {"error": "unknown mode"}
                continue
            engine = self._engines[m]
            predictors_info = []
            # Inspect engine attributes for dspy predictors (ChainOfThought, Predict)
            for attr, val in vars(engine).items():
                try:
                    if isinstance(val, dspy.ChainOfThought):
                        preds = val.predictors()
                        for idx, p in enumerate(preds):
                            name = attr if idx == 0 else f"{attr}_{idx}"
                            sig = getattr(p, "signature", None)
                            sig_info = {}
                            if sig is not None:
                                sig_info["instructions"] = getattr(
                                    sig, "instructions", None
                                )
                                sig_info["inputs"] = getattr(sig, "inputs", None)
                                sig_info["outputs"] = getattr(sig, "outputs", None)
                                # fallback to string repr for anything else
                                if not any(sig_info.values()):
                                    sig_info = str(sig)
                            else:
                                sig_info = None
                            predictors_info.append(
                                {
                                    "name": name,
                                    "attribute": attr,
                                    "type": type(p).__name__,
                                    "signature": sig_info,
                                }
                            )
                    elif isinstance(val, dspy.Predict):
                        p = val
                        sig = getattr(p, "signature", None)
                        sig_info = {}
                        if sig is not None:
                            sig_info["instructions"] = getattr(
                                sig, "instructions", None
                            )
                            sig_info["inputs"] = getattr(sig, "inputs", None)
                            sig_info["outputs"] = getattr(sig, "outputs", None)
                            if not any(sig_info.values()):
                                sig_info = str(sig)
                        else:
                            sig_info = None
                        predictors_info.append(
                            {
                                "name": attr,
                                "attribute": attr,
                                "type": type(p).__name__,
                                "signature": sig_info,
                            }
                        )
                except Exception as e:
                    # Don't fail the whole listing if one attribute inspection errors
                    predictors_info.append(
                        {
                            "name": attr,
                            "attribute": attr,
                            "error": f"inspecting attribute failed: {e}",
                        }
                    )
            out[m] = predictors_info
            self.logger.debug(
                f"Listed {len(predictors_info)} predictors for mode '{m}'"
            )
        return out
