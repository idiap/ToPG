#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Maxime Delmas <maxime.delmas@idiap.ch>
#
# SPDX-License-Identifier: LicenseRef-Idiap
#


import os
from typing import Any, Mapping

import dspy
from dotenv import load_dotenv

load_dotenv()


def configure_llm(llm_config: Mapping[str, Any]):
    """
    Configure and initialize a dspy language model with JSONAdapter.

    This function sets up a dspy.LM instance for making LLM calls throughout
    the TopG system. It supports various providers (OpenAI, Anthropic, Ollama, etc.)
    through dspy's unified interface.

    Args:
        llm_config (Mapping[str, Any]): Configuration dictionary with keys:
            - api_base (str, optional): API endpoint URL. Defaults to OpenAI.
            - llm_name (str, optional): Model identifier (e.g., "gpt-4o-mini",
                "anthropic/claude-3-5-sonnet"). Defaults to "gpt-4o-mini".
            - max_tokens (int, optional): Maximum tokens in response. Defaults to 4096.
            - timeout (int, optional): Request timeout in seconds. Defaults to 120.
            - api_key (str, optional): API key. If not provided, reads from
                LLM_API_KEY environment variable.

    Returns:
        dspy.LM: Configured language model instance ready for use.

    Raises:
        ValueError: If no API key is found in config or environment.

    Example:
        >>> config = {
        ...     "llm_name": "openai/gpt-4o-mini",
        ...     "max_tokens": 2048
        ... }
        >>> lm = configure_llm(config)
        >>> dspy.configure(lm=lm)
    """
    api_base = llm_config.get("api_base", "https://api.openai.com/v1")
    llm_name = llm_config.get("llm_name", "gpt-4o-mini")
    max_tokens = llm_config.get("max_tokens", 4096)
    timeout = llm_config.get("timeout", 120)

    # Check if there is an api_key in the config or environment variables
    api_key = llm_config.get("api_key", os.getenv("LLM_API_KEY"))
    if api_key is None:
        raise ValueError(
            "API key for LLM is not provided in config (not recommended) or environment variables: LLM_API_KEY (recommended)."
        )

    lm = dspy.LM(
        llm_name,
        api_base=api_base,
        max_tokens=max_tokens,
        api_key=api_key,
        timeout=timeout,
    )

    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

    return lm


def get_cost(lm):
    """
    Calculate the total cost of all LLM calls made so far.

    Sums up the cost field from the language model's call history.
    Useful for tracking expenses during document processing and querying.

    Args:
        lm (dspy.LM): The configured language model instance.

    Returns:
        float: Total cost in USD across all LLM calls.

    Example:
        >>> cost = get_cost(lm)
        >>> print(f"Total spent: ${cost:.4f}")
    """
    cost = sum([x["cost"] for x in lm.history if x["cost"] is not None])
    return cost


def get_count_tokens(lm):
    """
    Get token usage statistics from the language model's history.

    Aggregates token counts across all LLM calls to provide usage metrics.

    Args:
        lm (dspy.LM): The configured language model instance.

    Returns:
        dict: Dictionary containing:
            - total_tokens (int): Sum of all tokens used (input + output)
            - completion_tokens (int): Sum of all output tokens generated
            - prompt_tokens (int): Sum of all input tokens sent

    Example:
        >>> tokens = get_count_tokens(lm)
        >>> print(f"Total tokens used: {tokens['total_tokens']}")
        >>> print(f"Input: {tokens['prompt_tokens']}, Output: {tokens['completion_tokens']}")
    """
    total_tokens = sum(
        [x["usage"]["total_tokens"] for x in lm.history if "total_tokens" in x["usage"]]
    )
    completion_tokens = sum(
        [
            x["usage"]["completion_tokens"]
            for x in lm.history
            if "completion_tokens" in x["usage"]
        ]
    )
    prompt_tokens = sum(
        [
            x["usage"]["prompt_tokens"]
            for x in lm.history
            if "prompt_tokens" in x["usage"]
        ]
    )
    all_counts = {
        "total_tokens": total_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
    }
    return all_counts
