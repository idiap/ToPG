#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#

import hashlib
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from diskcache import Cache
from langchain_core.documents.base import Document
from tqdm import tqdm

from topg.builders.llm_modules.hypergraph_extractor import HyperPropositionizerWithNER
from topg.utils.models import HyperPropositionList


class NLTKSentenceTextSplitter:
    """
    Text splitter that uses NLTK's Punkt sentence tokenizer for sentence segmentation.

    This class wraps NLTK's sentence tokenizer to split text and documents into
    individual sentences while preserving metadata. Useful for preprocessing before
    creating larger passages.

    Attributes:
        sentence_spliter (PunktSentenceTokenizer): NLTK sentence tokenizer instance.
    """

    def __init__(self, *args, **kwargs) -> None:
        # params for the nltk pukt spliter
        try:
            from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

            st_params = PunktParameters()
            sentence_tokenizer = PunktSentenceTokenizer(st_params)
            self.sentence_spliter = sentence_tokenizer

        # from the langchin NLTK tokenizer
        except ImportError:
            raise ImportError(
                "NLTK is not installed, please install it with `pip install nltk`."
            )

    def split_text(self, text: str) -> List[Document]:
        """
        Split text into individual sentences.

        Args:
            text (str): Input text to split into sentences.

        Returns:
            List[Document]: List of Document objects, each containing one sentence.
        """
        # split the text into sentences
        # return a list of documents
        sentence = self.sentence_spliter.tokenize(text)
        return [Document(page_content=sent) for sent in sentence]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into sentences, preserving metadata.

        Args:
            documents (List[Document]): Input documents to split.

        Returns:
            List[Document]: List of Document objects.
        """
        new_documents = []
        for doc in tqdm(
            documents,
            desc="NLTK Sentence Splitter - Splitting documents",
            unit="documents",
        ):
            # get the original document text and metadata
            doc_content = doc.page_content
            document_metadata = doc.metadata

            # get the list of text chunks
            splited_doc = self.split_text(doc_content)

            for sentence in splited_doc:
                new_documents.append(
                    Document(
                        page_content=sentence.page_content,
                        metadata={**document_metadata},
                    )
                )
        return new_documents


class ParagraphSplitter:
    """
    Split documents into passages with controlled sentence count and overlap.

    This class creates passages (paragraphs) from documents by:
    1. Splitting on paragraph separators
    2. Further splitting long paragraphs into chunks with specified sentence limits
    3. Adding sentence overlap between consecutive chunks for context continuity

    Attributes:
        paragraph_separator (str): String to split on to identify paragraph boundaries.
        sentence_spliter (NLTKSentenceTextSplitter): Sentence tokenizer for further splitting.
        paragraph_size_limit (int): Maximum number of sentences per passage.
        paragraph_sent_overlap (int): Number of sentences to overlap between passages.
    """

    def __init__(
        self,
        paragraph_separator: str = "\n",
        paragraph_size_limit: int = 5,
        paragraph_sent_overlap: int = 2,
    ) -> None:
        if paragraph_size_limit <= paragraph_sent_overlap:
            raise ValueError(
                f"`paragraph_size_limit` ({paragraph_size_limit}) must be greater than "
                f"`paragraph_sent_overlap` ({paragraph_sent_overlap})"
            )
        self.paragraph_separator = paragraph_separator
        self.sentence_spliter = NLTKSentenceTextSplitter()
        self.paragraph_size_limit = paragraph_size_limit
        self.paragraph_sent_overlap = paragraph_sent_overlap

    def split_document(self, document: Document) -> List[Document]:
        """
        Split a single document into multiple passage documents.

        This method:
        1. Splits the document on paragraph_separator
        2. For each paragraph, splits into sentences
        3. If paragraph exceeds sentence limit, creates overlapping chunks
        4. Preserves all original metadata in each resulting passage

        Args:
            document (Document): Input document to split into passages.

        Returns:
            List[Document]: List of passage documents with original metadata preserved.
        """
        new_documents = []

        doc_content = document.page_content
        document_metadata = document.metadata
        doc_title = document.metadata.get("title", None)

        splited_paragraphs = doc_content.split(self.paragraph_separator)
        for paragraph in splited_paragraphs:
            list_of_sentences = self.sentence_spliter.split_text(paragraph)
            if len(list_of_sentences) > self.paragraph_size_limit:
                for i in range(
                    0,
                    len(list_of_sentences),
                    self.paragraph_size_limit - self.paragraph_sent_overlap,
                ):
                    new_paragraph = " ".join(
                        [
                            sent.page_content
                            for sent in list_of_sentences[
                                i : i + self.paragraph_size_limit
                            ]
                        ]
                    )

                    new_documents.append(
                        Document(
                            page_content=new_paragraph.strip(),
                            metadata={**document_metadata},
                        )
                    )
                    if i + self.paragraph_size_limit >= len(list_of_sentences):
                        break
            else:
                new_documents.append(
                    Document(
                        page_content=paragraph.strip(),
                        metadata={**document_metadata},
                    )
                )
        return new_documents


class IdGenerator:
    """
    A class that generates unique identifiers for documents and assigns them
    to a specified metadata key.
    Attributes:
        key_name (str): The name of the metadata key where the generated
                        unique identifier will be stored.
    Methods:
        modify(documents: List[Document]) -> None:
            Iterates over a list of documents and assigns a unique identifier
            to each document's metadata under the specified key.
    """

    def __init__(self, key_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_name = key_name

    def modify(self, documents: List[Document]) -> None:
        for i, doc in tqdm(
            enumerate(documents), desc="Generating ids", unit="documents"
        ):
            normalized_text = doc.page_content.strip()
            unique_key_id = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
            doc.metadata[self.key_name] = unique_key_id


class PassagesToHyperpropositions:
    """
    Parallel processor for extracting hyperpropositions from passages using LLM.

    Later, we plan to integrate this in async instead.

    This class manages the extraction of atomic propositions (hyperpropositions) from
    passages using an LLM. It implements:
    - Disk-based caching to avoid redundant LLM calls for the same passage
    - Parallel processing with ThreadPoolExecutor for efficiency
    - Automatic retry for empty results
    - Cache export/import functionality

    The extraction process decomposes each passage into self-contained atomic facts,
    each annotated with the entities it mentions.

    Attributes:
        hyperpropositionizer: LLM-based extractor for propositions and entities.
        passage_id_key (str): Metadata key to identify passages for caching.
        hyperpropositionizer_args (dict): Additional arguments for the extractor.
        cache (Cache): Disk-based cache for storing extraction results.
        max_workers (int): Number of parallel threads for LLM calls.
        logger (logging.Logger): Logger instance for logging messages.
    """

    def __init__(
        self,
        hyperpropositionizer,
        passage_id_key="passage_id",
        cache_dir="cache/propositions",
        max_workers=10,
        hyperpropositionizer_args={},
        logger=None,
    ):
        self.hyperpropositionizer = hyperpropositionizer
        self.passage_id_key = passage_id_key
        self.hyperpropositionizer_args = hyperpropositionizer_args
        self.cache = Cache(cache_dir)
        self.max_workers = max_workers
        self.logger = logger or logging.getLogger(__name__)

    def _call_llm(self, doc):
        """
        Internal method to call the LLM for hyperproposition extraction.

        Args:
            doc (Document): Document containing the passage to extract from.

        Returns:
            HyperPropositionList: Extracted propositions with entities.
        """
        return self.hyperpropositionizer.extract_propositions(
            doc, **self.hyperpropositionizer_args
        )

    def _process_doc(self, doc):
        """
        Process a single document with caching support.

        Checks the cache first. If a valid cached result exists (non-empty),
        returns it. Otherwise, calls the LLM and caches the result.

        Args:
            doc (Document): Document to process, must have passage_id in metadata.

        Returns:
            tuple: (passage_id, result_dict) where result_dict contains
                hyperpropositions extracted from the passage.
        """
        passage_id = doc.metadata[self.passage_id_key]
        if passage_id in self.cache:
            # Cache hit
            cached_data = self.cache[passage_id]
            # We also check that it is not empty - if it is we want to retry
            if len(cached_data["hyperpropositions"]):
                return passage_id, cached_data

        # Cache miss: call LLM
        result = self._call_llm(doc)
        dumped = result.model_dump()
        self.cache[passage_id] = dumped
        return passage_id, dumped

    def split_documents(self, documents):
        """
        Extract hyperpropositions from multiple passages in parallel.

        This method processes all documents concurrently using ThreadPoolExecutor,
        leveraging the cache to avoid redundant LLM calls. Each passage is
        decomposed into atomic propositions, and each proposition becomes a
        separate document with entity annotations.

        Args:
            documents (List[Document]): List of passage documents to process.
                Each must have the passage_id_key in its metadata.

        Returns:
            List[Document]: List of hyperproposition documents, where each contains:
                - page_content (str): The atomic proposition text
                - metadata (dict): Contains:
                    - entities (List[str]): Entity names in the proposition
                    - passage_id_key (str): Parent passage identifier

        Note:
            Errors during processing of individual documents are caught and logged,
            but do not stop processing of other documents. Progress is shown with tqdm.
        """
        new_documents = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_doc, doc): doc for doc in documents
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing documents"
            ):
                doc = futures[future]
                try:
                    passage_id, result_data = future.result()
                    hyperpropositions_list = HyperPropositionList(**result_data)

                    for h in hyperpropositions_list.hyperpropositions:
                        new_documents.append(
                            Document(
                                page_content=h.proposition,
                                metadata={
                                    "entities": h.entities,
                                    self.passage_id_key: passage_id,
                                },
                            )
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error processing doc {doc.metadata[self.passage_id_key]}: {e}"
                    )

        return new_documents


class GeneralDocumentProcessor:
    """
    Complete document processing pipeline for TopG indexing.

    This class orchestrates the entire document processing workflow:
    1. Splits documents into passages using ParagraphSplitter
    2. Generates unique IDs for passages
    3. Extracts hyperpropositions (atomic facts with entities) using LLM
    4. Generates unique IDs for hyperpropositions
    5. Manages LLM call caching to avoid redundant processing

    The processor uses parallel processing for LLM calls and disk-based caching
    to enable resuming interrupted processing jobs.

    Attributes:
        logger (logging.Logger): Logger instance.
        paragraph_spliter (ParagraphSplitter): Splits documents into passages.
        passage_id_generator (IdGenerator): Generates SHA-256 based passage IDs.
        proposition_id_generator (IdGenerator): Generates SHA-256 based proposition IDs.
        hyperpropositionizer (HyperPropositionizerWithNER): LLM-based extractor.
        passages2hyperpropositions (PassagesToHyperpropositions): Cached LLM processor.
    """

    def __init__(self, config, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.paragraph_spliter = ParagraphSplitter(
            paragraph_separator=config.get("paragraph_separator", "\n\n\n"),
            paragraph_size_limit=config.get("paragraph_size_limit", 12),
            paragraph_sent_overlap=config.get("paragraph_sent_overlap", 3),
        )
        self.passage_id_generator = IdGenerator(key_name="passage_id")
        self.proposition_id_generator = IdGenerator(key_name="id")

        self.hyperpropositionizer = HyperPropositionizerWithNER(logger=self.logger)
        # Setting the llm modules into the corresponding document modifiers.
        self.passages2hyperpropositions = PassagesToHyperpropositions(
            hyperpropositionizer=self.hyperpropositionizer,
            cache_dir=config.get("cache_dir", "cache/propositions"),
            max_workers=config.get("max_workers", 5),
            logger=self.logger,
        )

    def get_passages(self, text: str, doc_id: str = None) -> List[Document]:
        """
        Split input text into passages with unique IDs.

        Args:
            text (str): The document text to process.
            doc_id (str, optional): Document identifier. If None, generates a UUID.
                Defaults to None.

        Returns:
            List[Document]: List of passage documents, each with:
                - page_content (str): The passage text
                - metadata (dict): Contains doc_id and passage_id (SHA-256 hash of content)
        """
        # Create a Document object with the provided text

        if doc_id is None:
            doc_id = f"{uuid.uuid4()}"

        # Include kwargs as metadata
        metadata = {"doc_id": doc_id}

        document = Document(page_content=text, metadata=metadata)

        # Split the document into passages
        passages = self.paragraph_spliter.split_document(document)

        # Modify the passage IDs
        self.passage_id_generator.modify(passages)

        return passages

    def get_hyperpropositions(self, passages: List[Document]) -> List[Document]:
        """
        Extract hyperpropositions (atomic facts) from passages using LLM.

        This method uses an LLM to decompose each passage into atomic propositions
        with associated entities. Results are cached to avoid redundant LLM calls.
        Processing happens in parallel for efficiency.

        Args:
            passages (List[Document]): Passages to extract propositions from.
                Each passage must have a passage_id in its metadata.

        Returns:
            List[Document]: List of hyperproposition documents, each with:
                - page_content (str): The proposition text
                - metadata (dict): Contains:
                    - id (str): Unique SHA-256 hash of the proposition
                    - passage_id (str): Parent passage identifier
                    - entities (List[str]): List of entity names mentioned

        Note:
            Uses disk-based caching keyed by passage_id. Cached results are reused
            to avoid repeated LLM calls for the same passage.
        """
        # Apply the passage to hyperproposition conversion
        hyperpropositions = self.passages2hyperpropositions.split_documents(passages)

        # Modify the hyperproposition IDs
        self.proposition_id_generator.modify(hyperpropositions)

        return hyperpropositions
