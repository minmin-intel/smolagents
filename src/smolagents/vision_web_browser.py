import argparse
from io import BytesIO
from time import sleep

import helium
import PIL.Image
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, WebSearchTool, tool
from smolagents.agents import ActionStep
from smolagents.cli import load_model
import os
import time




def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a web browser automation script with a specified model.")
    # parser.add_argument(
    #     "prompt",
    #     type=str,
    #     nargs="?",  # Makes it optional
    #     default=search_request,
    #     help="The prompt to run with the agent",
    # )
    parser.add_argument(
        "--model-type",
        type=str,
        default="LiteLLMModel",
        help="The model type to use (e.g., OpenAIServerModel, LiteLLMModel, TransformersModel, InferenceClientModel)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-4o",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="The inference provider to use for the model",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        # default="https://api.together.xyz/v1",
        help="The base URL for the model",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the test data file (CSV or JSONL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output file for saving results",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with a single question",
    )
    return parser.parse_args()

WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'datasets/webarena/test_smolagents')
if not os.path.exists(DATAPATH):
    os.makedirs(DATAPATH)

def capture_screenshot():
    """
    Save a screenshot of the current page.
    """
    print("Saving screenshot...")
    driver = helium.get_driver()
    # Get the current page title
    # Create a filename based on the title
    filename = f"{time.time()}.png"
    # Save the screenshot
    png_bytes = driver.get_screenshot_as_png()
    image = PIL.Image.open(BytesIO(png_bytes))
    image.save(os.path.join(DATAPATH, filename))
    print(f"Screenshot saved as {filename}")
    return image


def reset():
    print("Starting Chrome and go to login page...")
    url = "http://10.7.4.57:9083/admin"

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")

    driver = helium.start_chrome(url, headless=True, options=chrome_options)

    username = os.getenv('SHOPPING_ADMIN_USERNAME')
    password = os.getenv('SHOPPING_ADMIN_PASSWORD')
    print("Entering login info...")
    helium.write(username, into='Username')
    helium.write(password, into='Password')

    print("Clicking sign in...")
    helium.click('Sign in')
    time.sleep(3)
    image = capture_screenshot()
    return driver, image

def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = PIL.Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!
        image.save(os.path.join(DATAPATH, f"step_{current_step}.png"))

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
    return


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()

# from test_helium import reset
def initialize_driver():
    """Initialize the Selenium WebDriver."""
    # chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument("--force-device-scale-factor=1")
    # chrome_options.add_argument("--window-size=1000,1350")
    # chrome_options.add_argument("--disable-pdf-viewer")
    # chrome_options.add_argument("--window-position=0,0")
    # return helium.start_chrome(headless=True, options=chrome_options)
    driver, login_screenshot = reset()
    return driver, login_screenshot

def initialize_agent(model):
    """Initialize the CodeAgent with the specified model."""
    return CodeAgent(
        tools=[go_back, close_popups, search_item_ctrl_f],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )


helium_instructions = """You are a web browser agent that helps users achieve their goals.
You are given the screenshot of the homepage of the website that you are browsing. Only navigate within this website.
Use helium to navigate the website.
Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
Code:
```py
go_to('github.com/trending')
```<end_code>

You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>

If it's a link:
Code:
```py
click(Link("Top products"))
```<end_code>

If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.

To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
Code:
```py
scroll_down(num_pixels=1200) # This will scroll one viewport down
```<end_code>

When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
```py
close_popups()
```<end_code>

You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>

Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
Code:
```py
final_answer("YOUR_ANSWER_HERE")
```<end_code>

If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To list elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually, or use your tool search_item_ctrl_f.
Of course, you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url.
But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states.
Don't kill the browser.
When you have modals or cookie banners on screen, you should get rid of them before you can click anything else.
"""



def run_webagent(prompt: str, model_type: str, model_id: str, api_base:str, api_key:str) -> None:
    # Load environment variables
    load_dotenv()

    # Initialize the model based on the provided arguments
    model = load_model(model_type, model_id, api_base=api_base, api_key=api_key)

    global driver
    driver, first_screenshot = initialize_driver()
    agent = initialize_agent(model)

    # Run the agent with the provided prompt
    agent.python_executor("from helium import *")
    answer = agent.run(task=prompt + helium_instructions, images=[first_screenshot])
    return answer


import pandas as pd
import os
def get_test_data(args):
    if args.input is not None:
        if args.input.endswith(".csv"):
            df = pd.read_csv(args.input)
        elif args.input.endswith(".jsonl"):
            df = pd.read_json(args.input, lines=True)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSONL file.")
        
        if os.path.exists(args.output):
            tested = pd.read_json(args.output, lines=True)
            df = df[~df["Question"].isin(tested["question"])]
        print(f"Loaded {len(df)} questions from {args.input}")
        return df
    else:
        raise ValueError("No input file provided. Please specify a CSV or JSONL file.")

def append_to_output_file(output_file, data):
    import json
    with open(output_file, "a") as f:
        f.write(json.dumps(data) + "\n")

def save_as_csv(answers_file):
    df = pd.read_json(answers_file, lines=True)
    df.to_csv(answers_file.replace(".jsonl", ".csv"), index=False)
    print(f"Saved answers to {answers_file.replace('.jsonl', '.csv')}")


def main() -> None:
    from datetime import datetime
    # Parse command line arguments
    args = parse_arguments()
    df = get_test_data(args)
    # df = df.head(1)

    api_base = args.api_base
    if args.model_type == "OpenAIServerModel":
        api_key = os.getenv("TOGETHER_API_KEY")
    elif args.model_type == "LiteLLMModel":
        api_key = os.getenv("OPENAI_API_KEY")

    if args.quick_test:
        # Run a quick test with a single question
        prompt = "What are the top-3 best-selling product in **Jan 2023**? Today is May 2025."
        print(f"Running quick test with prompt: {prompt}")
        response = run_webagent(prompt, args.model_type, args.model_id, api_base, api_key)
        print(f"Response: {response}")
        return None


    for i, row in df.iterrows():
        # Extract the prompt from the DataFrame
        prompt = row["Question"]
        print(f"Processing question {i}: {prompt}")
        # Run the web agent with the extracted prompt
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = run_webagent(prompt, args.model_type, args.model_id, api_base, api_key)
        print(f"Response: {response}")
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "question": prompt,
            "prediction": response,
            "true_answer": row["Final answer"],
            "task": row["Level"],
            "start_time": start_time,
            "end_time": end_time,
        }
        # Append the result to the output file
        append_to_output_file(args.output, data)
        print("="*50)

    save_as_csv(args.output)


if __name__ == "__main__":
    main()
