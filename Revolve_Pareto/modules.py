"""
Various stages of individual generation, training, and evaluation:
1. Reward Function Generation
2. Policy Training
3. Policy Evaluation
"""
import multiprocessing
import traceback
import concurrent.futures
import json
import os
import hydra
import time
from typing import List, Tuple, Optional, Dict
from hydra.core.global_hydra import GlobalHydra
from typing import Callable, List
import numpy as np
import openai
from openai import OpenAI
import os
import re

import absl.logging as logging
#from rl_agent.generate_scores import generate_behaviour


from utils import parse_llm_output, serialize_dict, format_human_feedback#, extract_code

def extract_code(text: str) -> str:
    """
    Extracts the first fenced code block. Falls back to raw text if none found.
    """
    m = re.search(r"```(?:python)?\s*(.+?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()

openai_api_key= 'YOUR_KEY'
client = OpenAI(api_key=openai_api_key)

MODEL = "gpt-5"  # or "gpt-4o"
MAX_TOKENS = 10000
# generates reward functions

class RewardFunctionGeneration:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.llm = MODEL

    def query_llm(self, in_context_prompt: str) -> Tuple[str, int, int]:
        response = client.chat.completions.create(
            model=self.llm,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": in_context_prompt},
            ],
            max_completion_tokens=MAX_TOKENS,
        )
        content = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        return content, prompt_tokens, completion_tokens

    @staticmethod
    def prepare_in_context_prompt(
        in_context_samples: Optional[List[Tuple[str, float]]],
        operator_prompt: str,
        evolve: bool,
        baseline: str,
    ) -> str:
        if not evolve:
            return ""

        # Accumulators for all policies and metrics (handles mutation=1 or crossover=2+)
        all_policies = []
        all_metrics = []

        for filename, fitness_score in in_context_samples:
            # ---- POLICY CODE ----
            policy_code = "\n```python\n"
            policy_code += open(filename, "r").read()
            policy_code += "\n```"
            all_policies.append(policy_code)
           # print("inside prepare in context policy_code",policy_code)

            # ---- METRICS ----
            metrics_dir = os.path.dirname(filename).rsplit("policies", 1)[0] + "policy_history"
            metrics_file = os.path.join(metrics_dir, os.path.basename(filename).replace(".txt", ".json"))

            # old version (kept for reference)
            # metrics_file = os.path.dirname(filename).replace("policies", "policy_history")
            # metrics_file = os.path.join(metrics_file, os.path.basename(filename).replace(".txt", ".json"))
            # metrics_file = filename.replace("policies", "policy_history").replace(".txt", ".json")

            metrics_dict = json.load(open(metrics_file, "r"))

            # old version (kept for reference)
            # in_context_samples_str += f"fitness score: {metrics_dict.get('fitness', fitness_score)}\n"
            # in_context_samples_str += f"{serialize_dict(metrics_dict)}"

            metrics_str = serialize_dict(metrics_dict)
            all_metrics.append(metrics_str)

            if "auto" not in baseline:
                # human feedback (if used)
                human_feedback_file = filename.replace(
                    "policies", "human_feedback"
                ).replace(".txt", ".txt")
                try:
                    human_feedback = open(human_feedback_file, "r").read()
                    human_feedback = format_human_feedback(human_feedback)
                    # Append human feedback to last metrics
                    all_metrics[-1] += f"\nhuman feedback: {human_feedback}"
                except FileNotFoundError:
                    pass

        # Join policies and metrics so crossover includes both
        policies_str = "\n\n".join(all_policies)
        metrics_str  = "\n\n".join(all_metrics)

        # Replace placeholders in operator prompt
        operator_prompt = operator_prompt.replace("<Policy>", policies_str)
        operator_prompt = operator_prompt.replace("<Metrics>", metrics_str)

        #print("operator_prompt before end", operator_prompt)
        return operator_prompt



    def generate_rf(self, in_context_prompt: str) -> str:
        parsed_function_str = None
        while True:
            try:
                raw_llm_output, _, _ = self.query_llm(in_context_prompt)
              #  print("raw_llm_output",raw_llm_output)
                #parsed_function_str = parse_llm_output(raw_llm_output)
                parsed_function_str = extract_code(raw_llm_output)
                
                
           #     print("parsed_function_str",parsed_function_str)
                # parsed_function_str = "\n".join(
                # l for l in parsed_function_str.splitlines()
                # if l.strip() not in ("```", "```python"))
              #  print("parsed_function_str after strip cleaning?",parsed_function_str)

                break
            # except openai.RateLimitError or openai.APIError or openai.Timeout:
            except openai.RateLimitError or openai.APIError or openai.Timeout:
                time.sleep(10)
                continue
        # parsed_function_str = open("test_heuristic", "r").read()
        return parsed_function_str
