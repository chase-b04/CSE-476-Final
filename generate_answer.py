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


def chain_of_thought(prompt) -> str: #Zero-shot-cot
    response = call_model_chat_completions(prompt, model=MODEL, temperature=0.0)
    if response:
        return extract_answer(response["text"])
    return "Error, no response pulled for chain of thought method"

def self_consistency(prompt, num_samples):
    responses = []
    for i in range(num_samples):
        response = chain_of_thought(prompt, num_samples, temperature=0.7)
        answer = extract_answer(response["text"])
        responses.append(normalize_text(answer))

    count = Counter(responses)
    return count.most_common(1)[0][0] #most common response


def generate_thoughts(prompt, num_samples) -> str:
    thoughts = []
    for i in range(num_samples):
        response = call_model_chat_completions(prompt, model=MODEL, temperature=0.7)
        thoughts.append(response["text"])
    return thoughts

def evaluate_thoughts(thoughts: List[str]) -> List[Dict]:
    evaluated_thoughts_list = []
    for thought in thoughts:
        response = call_model_chat_completions(thought["prompt"], temperature=0.0)
        score_num = extract_number(response["text"])
        score = max(1.0, min(10.0, float(score_num)))
        evaluated_thoughts_list.append({"thought": thought, "score": score})
    return evaluated_thoughts_list

def tree_of_thought(prompt, num_samples):
    thoughts = generate_thoughts(prompt, num_samples)
    evaluated_thoughts_list = evaluate_thoughts(thoughts)
    best_thought = evaluated_thoughts_list[0]["thought"]
    finish_prompt = f"""Question: {prompt}
    Good reasoning approach: {best_thought}

    only post the final answer and your reasoning
    """
    response = call_model_chat_completions(finish_prompt, model=MODEL, temperature=0.0)
    if response["ok"]:
        return extract_answer(response["text"])
    return extract_answer(best_thought)


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
    num_samples = 3
    self_con = self_consistency(prompt, num_samples)
    tot = tree_of_thought(prompt, num_samples)
    combined_answers = [normalize_text(cot), normalize_text(self_con), normalize_text(tot)]
    count = Counter(combined_answers)
    best_answer = count.most_common(1)[0][0]

    if best_answer == normalize_text(tot):
        return tot
    elif best_answer == normalize_text(self_con):
        return self_con
    else:
        return cot

def extract_number(s: str):
    if not s:
        return None
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return m.group(0) if m else None

def extract_answer(s: str):
    patterns = [] #Add to this later when answer is fully figured out
    for pattern in patterns:
        m = re.search(pattern, s)
        if m:
            return m.group(1).strip()

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