"""Optional: fill the judge_correct column of an annotation CSV with an LLM judge.

Works with any OpenAI-compatible chat endpoint, e.g. a local vLLM server:
    vllm serve swiss-ai/Apertus-70B-Instruct-2509 --port 8000
    JUDGE_BASE_URL=http://localhost:8000/v1 JUDGE_MODEL=swiss-ai/Apertus-70B-Instruct-2509 \
        python -m groundqa.judge --csv results/annotation.csv

Then `python -m groundqa.annotate agree` reports judge-vs-human agreement next to
verifier-vs-human agreement.
"""

import argparse
import csv
import json
import os
import re
import urllib.request

JUDGE_PROMPT = """You are grading an answer to a question about a context.

Context:
{context}

Question: {question}

Reference answer: {reference}

Model answer: {prediction}

The model answer is CORRECT if either
- the question is answerable from the context and the model answer is consistent with the reference and states nothing that the context does not support, or
- the question is not answerable from the context (reference says "(unanswerable)") and the model declines to answer.
Otherwise it is INCORRECT.

Reply with one word: CORRECT or INCORRECT."""


def ask(prompt: str, base_url: str, model: str, api_key: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def parse_verdict(text: str) -> str:
    m = re.search(r"\b(INCORRECT|CORRECT)\b", text.upper())
    return "" if not m else ("y" if m.group(1) == "CORRECT" else "n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    base_url = os.environ["JUDGE_BASE_URL"]
    model = os.environ["JUDGE_MODEL"]
    api_key = os.environ.get("JUDGE_API_KEY", "none")

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    for i, row in enumerate(rows):
        if row["judge_correct"].strip():
            continue
        row["judge_correct"] = parse_verdict(ask(JUDGE_PROMPT.format(**row), base_url, model, api_key))
        print(f"{i + 1}/{len(rows)} {row['id']}: {row['judge_correct'] or '?'}")
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
