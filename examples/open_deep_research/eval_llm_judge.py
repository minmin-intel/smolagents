import requests
import os
import pandas as pd
import json

JUDGE_PROMPT = """" \
You are an impartial judge. Score the answer using the rubrics below.

Score 1 if the answer is the same as gold truth. Same list of items with different order is considered the same. \
    If the answer is a number, it should be considered correct if the difference between the answer and the gold truth very small. \
Score 0.5 if the answer is partially correct. 
Score 0 if the answer is not correct.

The question is: {question}
The answer is: {answer}
The gold truth is: {gold_truth}

Output your score in JSON format:
```json
{{score: <your score here>}}
```
The score should be 0, 0.5, or 1.
"""

def run_llm(question, answer, gold_truth):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        gold_truth=gold_truth,
        score="{score}",
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 1000,
    }
    response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
    try:
        output = response.json()["choices"][0]["message"]["content"]
        useage = response.json()["usage"]
        print(f"*** Prompt tokens: {useage['prompt_tokens']}, Completion tokens: {useage['completion_tokens']}")
    except Exception:
        raise Exception(f"Response format unexpected: {response.json()}")
    return output

def parse_score(output):
    if "```json" in output:
        output = output.split("```json")[1].split("```")[0].strip()
    elif "```" in output:
        output = output.split("```")[1].split("```")[0].strip()
    try:
        score = json.loads(output)
        score = score["score"]
    except Exception:
        score = output
    return score

def load_data(args):
    data_folder = args.data_folder
    filename = args.filename
    df = pd.read_csv(os.path.join(data_folder, filename))
    df = df[["question", "prediction", "true_answer"]]
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LLM answers.")
    parser.add_argument("--data_folder", type=str, help="Path to the folder containing the data.")
    parser.add_argument("--filename", type=str, help="Name of the CSV file containing the data.")
    args = parser.parse_args()

    df = load_data(args)
    scores = []
    for index, row in df.iterrows():
        question = row["question"]
        answer = row["prediction"]
        gold_truth = row["true_answer"]
        score = run_llm(question, answer, gold_truth)
        score = parse_score(score)
        scores.append(score)
        print(f"Score: {score}")

    df["score"] = scores

    print(f"Average score: {sum(scores) / len(scores)}")

    output_file = os.path.join(args.data_folder, args.filename.replace(".csv", "_scores.csv"))
    df.to_csv(output_file, index=False)