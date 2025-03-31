import argparse
import os
import threading
import datasets
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from huggingface_hub import login
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
    DownloadTool
)
from scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    # HfApiModel,
    LiteLLMModel,
    OpenAIServerModel,
    ToolCallingAgent,
)


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]
load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=int, help="0,1,2")
    parser.add_argument("--model-id", type=str, default="openai/gpt-4o")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--agent_type", type=str, default="code", help="code or toolcalling")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--datapath", type=str, default="data/gaia")
    parser.add_argument("--test_type", type=str, default="text", help="text or image")
    parser.add_argument("--answer_file", type=str, default="answers.jsonl")
    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent(args):
    model_id=args.model_id
    
    
    if args.provider == "openai":
        model_params = {
            "model_id": model_id,
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": 8192,
        }
        if model_id == "o1":
            model_params["reasoning_effort"] = "high"
        model = LiteLLMModel(**model_params)
    elif args.provider == "deepseek":
        model_params = {
            "model_id": model_id,
            "custom_role_conversions": custom_role_conversions,
            "max_completion_tokens": 8192,
            "api_base":"https://api.deepseek.com",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "flatten_messages_as_text": True,
        }
        model = OpenAIServerModel(**model_params)
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
        DownloadTool(browser),
        visualizer,
    ]

    if args.agent_type == "code":
        text_webbrowser_agent = CodeAgent(
            model=model,
            tools=WEB_TOOLS,
            max_steps=args.max_steps,
            verbosity_level=2,
            planning_interval=4,
            name="search_agent",
            additional_authorized_imports=AUTHORIZED_IMPORTS,
        )
    elif args.agent_type == "toolcalling":
        text_webbrowser_agent = ToolCallingAgent(
            model=model,
            tools=WEB_TOOLS,
            max_steps=args.max_steps,
            verbosity_level=2,
            planning_interval=4,
            name="search_agent",
        )
    # text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    # If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    # Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    return text_webbrowser_agent


def load_gaia_dataset(args):
    DATAPATH=args.datapath
    SET = "validation"
    df = pd.read_json(f"{DATAPATH}/{SET}/metadata_filtered_{args.test_type}_sampled.jsonl", lines=True)
    print(f"Loaded dataset: {df.shape[0]} questions")
    print(df["Level"].value_counts())
    df.rename(columns={"Question": "question", "Final answer": "true_answer", "Level": "task"}, inplace=True)
    for _, row in df.iterrows():
        if len(row["file_name"]) > 0:
            row["file_name"] = f"{DATAPATH}/{SET}/" + row["file_name"]
            #check if file exists
            if not os.path.exists(row["file_name"]):
                print(f"File {row['file_name']} does not exist!")
            else:
                print(f"File {row['file_name']} exists!")

    return df

def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert os.path.exists(jsonl_file), "File not found!"
    print("Answer exported to file:", jsonl_file.resolve())


from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.reformulator import prepare_response

def answer_single_question(args, example, answers_file, visual_inspection_tool=None):
    agent = create_agent(args)
    model = agent.model

    # document_inspection_tool = TextInspectorTool(model, 100000)

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!
Here is the task:
""" + example["question"]

    if example["file_name"]:
        ### HF run_gaia.py has the following code:
        ### that preprocess files with document_inspection_tool
        ### and visual_inspection_tool
        ### which may boost agent performance
        # if ".zip" in example["file_name"]:
        #     prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
        #     prompt_use_files += get_zip_description(
        #         example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
        #     )
        # else:
        #     prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
        #     prompt_use_files += get_single_file_description(
        #         example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
        #     )
        #################################################
        
        # Here we will test the scenario where agent needs to figure out how to deal with the file
        prompt_use_files = f"\n\nTo solve the task above, you will have to use this attached file: {example['file_name']}"
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Run agent
        final_result = agent.run(augmented_question)

        agent_memory = agent.write_memory_to_messages(summary_mode=True)

        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)

        output = str(final_result)
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = [str(step) for step in agent.memory.steps]

        # Check for parsing errors which indicate the LLM failed to follow the required format
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False

        # check if iteration limit exceeded
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        print("Error on ", augmented_question, e)
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "task": example["task"],
        "task_id": example["task_id"],
        "true_answer": example["true_answer"],
    }
    append_answer(annotated_example, answers_file)


def get_examples_to_answer(answers_file, df) -> List[dict]:
    print(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        print(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        # print("Error when loading records: ", e)
        print("No usable records! Starting new.")
        done_questions = []
    # filter df and get the subset that is not in done_questions
    df = df[~df["question"].isin(done_questions)]
    return df



def test_web_agent():
    args = parse_args()

    agent = create_agent()

    if args.question == 0:
        question = "Weather in San Francisco March 1, 2025"
    elif args.question == 1:
        question = "In the Scikit-Learn July 2017 changelog, what other predictor base command received a bug fix? Just give the name, not a path."
    elif args.question == 2:
        question = "The photograph in the Whitney Museum of American Art's collection with accession number 2022.128 shows a person holding a book. Which military unit did the author of this book join in 1813? Answer without using articles."
    answer = agent.run(question)

    print(f"Got this answer: {answer}")

def save_as_csv(answers_file):
    df = pd.read_json(answers_file, lines=True)
    df.to_csv(answers_file.replace(".jsonl", ".csv"), index=False)
    print(f"Saved answers to {answers_file.replace('.jsonl', '.csv')}")

if __name__ == "__main__":
    args= parse_args()
    print(args)

    df = load_gaia_dataset(args)

    answers_file = args.answer_file
    
    tasks_to_run = get_examples_to_answer(answers_file, df)

    for i, example in tasks_to_run.iterrows():
        print(f"Processing task {i}/{len(tasks_to_run)}")
        print(f"Question: {example['question']}")
        answer_single_question(args, example, answers_file)
        if i+1 == 1:
            break

    save_as_csv(answers_file)
    print("All tasks done!")

    # with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
    #     futures = [
    #         exe.submit(answer_single_question, args, example, answers_file)
    #         for example in tasks_to_run
    #     ]
    #     for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
    #         f.result()

