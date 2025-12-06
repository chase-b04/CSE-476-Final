#!/usr/bin/env python3

#Student: Chase Bulkin
#ID: 1224681913

#MAKE SURE TO CONNECT TO sslvpn.asu.edu/2fa ON SSL
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import os, json, textwrap, re, time
import requests
from collections import Counter, deque 


INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")  
MODEL    = os.getenv("MODEL_NAME", "bens_model")              

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def chain_of_thought(prompt, system): #Zero-shot-cot
    return call_model_chat_completions(prompt, system, model=MODEL, temperature=0.0)

def self_consistency(prompt, num_samples):
    responses = []
    for i in range(num_samples):
        response = chain_of_thought(prompt, num_samples)
        responses.append(response["text"])

    count = Counter(responses)
    return count.most_common(1)[0][0] #most common response

def generate_thoughts(prompt, num_samples):
    thoughts = []
    for i in range(num_samples):
        response = call_model_chat_completions(prompt, model=MODEL, temperature=0.0)
        thoughts.append(response["text"])
    return thoughts

def evaluate_thoughts(thoughts: List[str]) -> List[Dict]:
    evaluated_thoughts_list = []
    for thought in thoughts:
        response = call_model_chat_completions(thought["prompt"], temperature=0.0)
        evaluated_thoughts_list.append(response["text"])
    return evaluated_thoughts_list

def tree_of_thought(prompt, num_samples):
    thoughts = generate_thoughts(prompt, num_samples)
    evaluated_thoughts_list = evaluate_thoughts(thoughts)
    best_thought = evaluated_thoughts_list[0]["thought"]

    return best_thought


def make_prompt(question):
    prompt = f"""For the given question:

    {question}

    Solve step-by-step and show your reasoning and analysis for your answer.
"""
    return prompt

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    synonyms = {
        "unchanged": "stay the same",
        "no change": "stay the same",
        "same": "stay the same",
        "second place": "second",
        "2nd": "second",
        "first place": "first",
        "third place": "third",
    }
    return synonyms.get(s, s)

def agent_loop(question):
    prompt = make_prompt(question)
    cot = chain_of_thought(prompt)
    num_samples = 5
    self_con = self_consistency(prompt, num_samples)
    tot = tree_of_thought(prompt, num_samples)
    combined_answers = [normalize_text(cot["text"]), normalize_text(self_con["text"]), normalize_text(tot["text"])]
    count = Counter(combined_answers)
    best_answer = count.most_common(1)[0][0]

    if best_answer == normalize_text(tot["text"]):
        return tot
    elif best_answer == normalize_text(self_con["text"]):
        return self_con
    else:
        return cot

def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers = []
    for idx, question in enumerate(questions, start=1):
        # Example: assume you have an agent loop that produces an answer string.
        real_answer = agent_loop(question["input"])
        answers.append({"output": real_answer})
        # placeholder_answer = f"Placeholder answer for question {idx}"
        # answers.append({"output": placeholder_answer})
    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


def main() -> None:
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions)

    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )


if __name__ == "__main__":
    main()

'''
NOTES:
Have to pick 3 different technologies with <20 system calls per question.
You must not hardcode a full delegation to an external tool (e.g., google_search(input_problem)). 
Coded is vscode not cursor. No GPT calls or calls of any other AI model.
Top ranked by performance get EC.

techniques and time-inference algorithms (must select 3):
-Chain-of-Thought (CoT)
-Decoding/Generation?
-Diffusion Model Alignment?
-Mixture of Experts (MoE)?
-RAG?
-Self-Consistency
-Tree-of-thought / X-of-thought
-Analogical Reasoning


Making good prompts:
Follow this guide: https://www.sophiehundertmark.com/en/new-prompting-rules-for-the-use-of-reasoning-models-deep-research/
1. Direct and simple
this guide says to avoid chain of thought, be I actually want to use it here for my methods
'''