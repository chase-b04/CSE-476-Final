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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")

# INPUT_PATH = Path("testing_questions.json")
# OUTPUT_PATH = Path("testing_answers.json")

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")  
MODEL    = os.getenv("MODEL_NAME", "bens_model")        

# api_calls = 0
question_number = 1

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    url = f"{API_BASE}/chat/completions" #COULD REMOVE TIMEOUTS IF DESPERATE, BUT IT MIGHT MAKE THINGS MESSY
    # global api_calls
    # api_calls += 1
    # print(api_calls)
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
    } #SHRINK MAX TOKENS FROM 128 TO 64 IF NEEDED. IT WILL RUIN ACCURACY BUT MAKE CODE FASTER.

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
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def chain_of_thought(prompt) -> str: #Zero-shot-cot
    new_prompt = f"""QUESTION: {prompt} 

    Let's think step-by-step. At the end, state your final answer.
    """
    response = call_model_chat_completions(new_prompt, model=MODEL, temperature=0.0)
    return extract_answer(response["text"])

def self_consistency(prompt, num_samples):
    responses = []
    lock = threading.Lock()

    def get_sample(i):
        temps = [0.0, 0.1, 0.5, 0.7]
        temp = temps[i] if i < len(temps) else 0.7
        new_prompt = f"""QUESTION: {prompt} 

        Let's think step-by-step. At the end, state your final answer.
        """
        response = call_model_chat_completions(new_prompt, model=MODEL, temperature=temp)
        answer = extract_answer(response["text"])
        if not response["ok"] or not response["text"]:
            return
        if answer:
            with lock:
                responses.append(normalize_text(answer))

    with ThreadPoolExecutor(max_workers=num_samples) as executor:
        futures = [executor.submit(get_sample, i) for i in range(num_samples)]
        for future in as_completed(futures):
            future.result()

    if responses:
        count = Counter(responses)
        return count.most_common(1)[0][0] #most common response
    return "Error: LLM could not produce an answer."

def few_shot_prompting(prompt):
    new_prompt = f"""You will be asked a question, here some examples of how you should respond:

    Example 1: 
    Question: Which of these processes could cause grass to be wet? A. condensation B. evaporation
    Final Answer: A. condensation

    Example 2:
    Question: What is (0.30 + 0.10) * 1500?
    Final Answer: 600

    Now, solve this problem:

    Question: {prompt}
    Final Answer:
    """
    response = call_model_chat_completions(new_prompt, model=MODEL, temperature=0.0)
        #print(response["text"]) #Debug
    return extract_answer(response["text"])


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
    global question_number
    prompt = make_prompt(question)
    results = {}
    lock = threading.Lock()
    def run_cot(): #I made three different functions inside the agent loop function for parallelism, 3x faster :3
        cot = chain_of_thought(prompt)
        with lock:
            results['cot'] = cot
    def run_self_con():
        num_samples = 2
        self_con = self_consistency(prompt, num_samples)
        with lock:
            results['self_con'] = self_con
    def run_few():
        few = few_shot_prompting(prompt)
        with lock:
            results['few'] = few
    #print(f"Answers:\n-Chain-of-Though =     {cot},\n-Self-Consistency =     {self_con},\n-Few-Shot-Prompting = {few}\n") #more readable for me

    with ThreadPoolExecutor(max_workers=3) as executor: #Use parallel threads run all three programs at the same time 
        futures = [
            executor.submit(run_cot),
            executor.submit(run_self_con),
            executor.submit(run_few)
        ]
        for future in as_completed(futures):
            future.result()

    cot = results.get('cot', '') #convert the results of each program to text to normalize
    self_con = results.get('self_con', '')
    few = results.get('few', '')
    combined_answers = [normalize_text(cot), normalize_text(self_con), normalize_text(few)]
    count = Counter(combined_answers)
    best_answer = count.most_common(1)[0][0] #return the best/most common answer of the three
    if not best_answer:
        print(f"Question Number {question_number}\nBest Answer: {cot}\nBest method was: Chain-of-Thought\n")
        question_number += 1
        return cot

    best_return = None
    best_method = ""
    if best_answer == normalize_text(self_con):
        best_return = self_con
        best_method = "Self-Consistency"
    elif best_answer == normalize_text(few):
        best_return = few
        best_method = "Few-Shot Prompting"
    else:
        best_return = cot
        best_method = "Chain-of-Thought"
    print(f"Question Number {question_number}\nBest Answer: {best_return}\nBest method was: {best_method}\n")
    question_number += 1
    return best_return
    

def extract_number(s: str):
    if not s:
        return None
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return m.group(0) if m else None

def extract_answer(s: str):
    if not s:
        return "There is no answer found"
    
    patterns = [r"[Ff]inal\s+[Aa]nswer\s*:?\s*(.+)",
        r"[Ff]inal\s+[Aa]nswer:?\s*(.+?)(?:\n|$)",
        r"[Ff]inal\s+[Aa]nswer\s*:?\s*\n(.+)",
        r"[Aa]nswer:?\s*(.+?)(?:\n|$)",
        r"[Aa]nswer\s*:?\s*\n(.+)",
        r"FINAL ANSWER:?\s*(.+?)(?:\n|$)",
        r"ANSWER:?\s*(.+?)(?:\n|$)",
        r"[Tt]he\s+[Ff]inal\s+[Aa]nswer\s+[Ii]s?\s*(.+?)(?:\n|$)",
        r"[Tt]he\s+[Aa]nswer\s+[Ii]s?\s*(.+?)(?:\n|$)"] #Catch any possible answers since it mocks my messy writing.
    #Have versions for same line and new line.
    for pattern in patterns:
        m = re.search(pattern, s)
        if m:
            return m.group(1).strip()
    lines = [l.strip() for l in s.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return "There is no answer found"

def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers = [None] * len(questions)
    def process_question(idx, question):
        answer = agent_loop(question["input"])
        return idx, {"output": answer}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_question, idx, q) 
            for idx, q in enumerate(questions)
        ]
        
        for future in as_completed(futures):
            idx, answer = future.result()
            answers[idx] = answer
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

    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r", encoding="utf-8") as fp:
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
Follow this guide for CHAIN OF THOUGHT prompts: https://www.promptingguide.ai/techniques/cot
1. Direct and simple
this guide says to avoid chain of thought, be I actually want to use it here for my methods

For extract answer with print(response) on:
Final Answer: A. condensation**
Answers: Chain-of-Though = The grass being wet suggests that water is present on its surface.
**Total earnings** = $750 + $250 = **$1000**

Final Answer: **$1000**
Answers: Chain-of-Though = - Total earnings: $ 300 + 150 = 4

Final Answer:** Warner Bros. Records

So I need: Final Answer:, Answers:, 


"A marketing company pays its employees on a commission-based salary system. If you sell goods worth $1000, you earn a 30% commission. Sales over $1000 get you an additional 10% commission. Calculate the amount of money Antonella earned if she sold goods worth $2500."
"input": "A student walks to school one morning and notices the grass is wet but the streets are dry. Which of these processes most likely caused the grass to be wet? A. condensation B. erosion C. evaporation D. precipitation"
'''