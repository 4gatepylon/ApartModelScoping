from __future__ import annotations
from typing import List, Dict, Union
import litellm
import openai

"""
Provide the methods to basically just query an LLM API to get judgements for LLMAsJudge.

It includes the `Prompter` functionality since that is primarily meant for use in
LLMAsJudge via LiteLLM. This part is meant to hydrate prompt templates. It enforces
seperation of prompt declaration and prompt usage for easier management of prompts
"""


################################ GLOBAL HELPER FUNCTIONS ################################
def api_generate(
    prompts: Union[List[str], List[List[Dict[str, str]]]],
    model: str,
    num_retries: int = 4,
    batch_size: int = 16,
    max_new_tokens=128,
    tqdm_enabled: bool = False,  # Nawwww
) -> List[str]:
    """
    This is a helper function to make it easy to generate using various LLM APIs
    (e.g. OpenAI, Anthropic, etc.) with built in error-handling.

    prompts can be either a list of string prompts, or it can be a list of multi-turn
    conversations in huggingface format:
        [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response},
            {"role": "user", "content": user_input1},
            ...
        ]
    """

    # If we pass a list of prompts, convert to message format
    if isinstance(prompts[0], str):
        prompts = [[{"role": "user", "content": p}] for p in prompts]

    try:
        # Attempt batched completion call with litellm
        responses = []
        for i in range(0, len(prompts), batch_size):
            r = litellm.batch_completion(
                model=model,
                messages=prompts[i : i + batch_size],
                max_tokens=max_new_tokens,
                num_retries=num_retries,
            )
            responses.extend(r)
        new_texts = [r.choices[0].message.content for r in responses]

    except openai.OpenAIError as e:
        # Error handling
        should_retry = litellm._should_retry(e.status_code)
        print("Error: API failed to respond.", e, f"should_retry: {should_retry}")
        new_texts = []

    return new_texts