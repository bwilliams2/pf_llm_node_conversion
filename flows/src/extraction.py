import time
import logging

from promptflow.core import tool
from promptflow.connections import  AzureOpenAIConnection
from promptflow.tools import template_rendering
from promptflow.tools.common import parse_chat

from openai import AzureOpenAI, RateLimitError

def call_with_fallback(
    paygo_connection: AzureOpenAIConnection,
    ptu_connection: AzureOpenAIConnection,
    completions_kwargs: dict = {},
    ptu_max_wait: int = 4000, 
    max_openai_retries: int = 0,
):
    # https://github.com/placerda/retry-logic/blob/main/smart_retry.py
    
    paygo_client = AzureOpenAI(
       api_key=paygo_connection["api_key"], 
       api_version=paygo_connection["api_version"],
       azure_endpoint=paygo_connection["api_base"],
       max_retries=max_openai_retries,
    )
    ptu_client = AzureOpenAI(
       api_key=ptu_connection["api_key"], 
       api_version=ptu_connection["api_version"],
       azure_endpoint=ptu_connection["api_base"],
       max_retries=max_openai_retries,
    )
    
    # Record the start time to measure latency
    start_time = time.time()

    while True:
        try:
            # Try calling the PTU model
            return ptu_client.chat.completions.create(**completions_kwargs)
        except RateLimitError as e:
            # If a rate limit error occurs, calculate the retry wait time
            retry_after_ms = int(e.response.headers['retry-after-ms'])        
            elapsed_time = (time.time() - start_time) * 1000
            # Check if the total wait time exceeds the maximum allowed latency
            if elapsed_time + retry_after_ms > ptu_max_wait:
                # If it does, log and switch to the standard model
                logging.info(f"Latency {elapsed_time + retry_after_ms} ms exceeds threshold. Switching to paygo.")
                return paygo_client.chat.completions.create(**completions_kwargs)
            else:
                # If not, log and wait for the retry time before retrying the PTU model
                logging.info(f"retrying PTU after {retry_after_ms} ms")
                time.sleep(e.retry_after / 1000)

@tool
def extraction(
    chat_history: list[dict[str, str]],
    prompt_template: str, 
    deployment_name: str,
    max_tokens: int,
    temperature: int,
    function_call: str,
    functions: dict,
    paygo_connection: AzureOpenAIConnection,
    ptu_connection: AzureOpenAIConnection,
) -> dict:

    with open(prompt_template, "r") as f:
        render = template_rendering.render_template_jinja2(
            f.read(),
            chat_history=chat_history,
        )

    messages = parse_chat(render)

    completions_kwargs = {
        "model": deployment_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "function_call": function_call,
        "functions": functions,
    }

    res = call_with_fallback(paygo_connection, ptu_connection, completions_kwargs)

    return res.choices[0].message.model_dump()