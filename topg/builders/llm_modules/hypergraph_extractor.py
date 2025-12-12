#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#

import json
import logging
import random
from typing import List

import dspy
from langchain_core.documents.base import Document

from topg.utils.models import EntitiesList, HyperProposition, HyperPropositionList


class PromptHyperPropositionizerWithNER(dspy.Signature):
    """Your task is to extract all meaningful decontextualized propositions from the given passage, using only the provided Named Entities.
    - Break down the input passage into its basic, minimal propositions that represent distinct pieces of meaning.
    - Propositions are independently interpretable without the context of the initial passage.
    Rules for each proposition:
    1. Complete & Standalone: a proposition convey exactly one fact or relationship, including all necessary context.
    2. Entity Usage: Use only the entities from Named Entities; do not introduce any others.
    3. Clear Attribution: Specify precisely which entities participate and their roles in the decontextualised proposition.
    4. Explicit Connections: Make causal, comparative, and temporal relationships explicit, including reference points.
    5. Preserve Time: Integrate temporal information and time markers (eg. a birth date or a date of an event) in the proposition.
    6. Preserve Cause: Keep causal links intact.
    7. Full Coverage: Collect propositions that together capture every meaningful point in the Passage.
    8. Preserve context: add all necessary contextual elements, like key entities, to improve clarity and precision of each propositions without the context of the initial passage.

    The precision and completeness of these propositions directly impact the performance and reliability of downstream tasks, particularly the quality of question answering systems. Therefore, careful adherence to the above rules is critical to ensure factual accuracy, unambiguous interpretation, and full semantic coverage.

    If the input passage is too short, has no entities, lacks context or is not a valid paragraph (like if it is a table of contents for instance), you may return an empty list of propositions.
    """

    passage = dspy.InputField(description="The input passage")
    entities: EntitiesList = dspy.InputField(
        description="A list of entities to use for the extraction"
    )
    hyperpropositions: List[HyperProposition] = dspy.OutputField(
        description="A JSON list of propositions and their associated entities"
    )


class PromptNER(dspy.Signature):
    """
    Your task is to extract all relevant entities from the Passage.
    - An entity can be a named entity, an idea, a concept, or an object central to the statement.
    - Include named entities, objects, phenomena, concepts, places, people, organizations, and multi-word terms.
    - Do not include redundant or nested variants of the same concept.

    If the input passage is not a valid paragraph, you may return an empty list of entities.
    """

    passage = dspy.InputField(description="The input passage")
    entities: List[str] = dspy.OutputField(description="A list of entities")


class HyperPropositionizerWithNER(dspy.Module):
    """
    Two-stage LLM pipeline for extracting atomic propositions with entities from passages.

    First extracts named entities using NER, then decomposes passages into atomic
    propositions annotated with those entities. Implements retry logic with temperature
    adjustment for robustness.
    """

    def __init__(self, logger=None, **kwargs):
        super().__init__()
        self.ner = dspy.Predict(PromptNER)
        self.hyperpropositionizer = dspy.Predict(PromptHyperPropositionizerWithNER)
        self.extractor_demonstrations = []
        self.ner_demonstrations = []

        self.logger = logger or logging.getLogger(__name__)

    def set_custom_instruction_prompt(
        self, predictor_name: str, instruction_prompt: str
    ):
        """Set custom instruction prompt for a predictor (ner or hyperpropositionizer)."""

        # Materialize named predictors
        matched = False
        for full_name, predictor in self.named_predictors():
            base_name = full_name.split(".")[0]  # Extract base predictor name
            self.logger.debug(
                f"Checking predictor: {base_name} against {predictor_name}"
            )
            if base_name == predictor_name:
                if hasattr(predictor, "signature") and hasattr(
                    predictor.signature, "instructions"
                ):
                    self.logger.info(
                        f"Setting instruction prompt for {predictor_name} in mode",
                    )
                    predictor.signature.instructions = instruction_prompt
                    matched = True
                    break
                else:
                    self.logger.warning(
                        f"Predictor '{predictor_name}' in mode does not support custom instructions."
                    )

        if not matched:
            self.logger.error(f"Predictor '{predictor_name}' not found.")

    def extract_propositions(
        self,
        passage: Document,
        retries: int = 3,
    ):
        """
        Extract hyperpropositions from a passage with retry logic.

        First extracts entities via NER, then decomposes the passage into atomic
        propositions. Retries with increasing temperature on failure.

        Returns:
            HyperPropositionList: Extracted propositions or empty list if all retries fail.
        """
        passage_content = passage.page_content

        # Always be sure that first attemp is done with temperature 0.0
        dspy.settings.lm.kwargs["temperature"] = 0.0

        # Retry until success or max attempts
        success = False
        for attempt in range(retries):
            if attempt > 0:
                # It is the first retry, we gonna try increasing the temperature by sampling a temperature between 0.0 and 1.0
                temperature = random.uniform(0.0, 1.0)
                dspy.settings.lm.kwargs["temperature"] = temperature

            # call the LLM
            try:
                llm_entities = self.ner(
                    passage=passage_content, demos=self.ner_demonstrations
                )
                llm_response = self.hyperpropositionizer(
                    passage=passage_content,
                    entities=llm_entities.entities,
                    demos=self.extractor_demonstrations,
                )

                hyperproposition_list = llm_response.hyperpropositions
                success = True
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} with temperature {dspy.settings.lm.kwargs['temperature']} failed: {e}"
                )
                success = False

        # If all attempts failed, return an empty list
        if success:
            return HyperPropositionList(**{"hyperpropositions": hyperproposition_list})
        else:
            self.logger.error(
                "All retry attempts failed. Returning an empty PropositionList."
            )
            return HyperPropositionList(**{"hyperpropositions": []})

    def set_ner_demonstrations(self, demonstrations: List[dict]):
        """Set few-shot examples for the NER predictor."""
        demonstrations = [
            dspy.Example(**demo).with_inputs("passage") for demo in demonstrations
        ]
        self.ner_demonstrations = demonstrations
        self.logger.info(
            f"NER demonstrations set: {len(self.ner_demonstrations)} examples."
        )

    def set_extractor_demonstrations(self, demonstrations: List[dict]):
        """Set few-shot examples for the hyperproposition extractor."""
        demonstrations = [
            dspy.Example(**demo).with_inputs("passage", "entities")
            for demo in demonstrations
        ]
        self.extractor_demonstrations = demonstrations
        self.logger.info(
            f"Extractor demonstrations set: {len(self.extractor_demonstrations)} examples."
        )
