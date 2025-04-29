import json

# log_path = "results_text_deepseek-reasoner_singlecodeagent.jsonl"
# log_path = "results_text_deepseek_singlecodeagent_code.jsonl"
# log_path = "log_image_deepseek_singlecodeagent_code.log"
# log_path = "log_text_deepseek-reasoner_singlecodeagent.log"
log_path = "log_text_deepseek-chat_browseruse.log"

def find_number_after_keyword(line, keyword):
    """
    Find the number that follows a specific keyword in a line.
    """
    start = line.find(keyword) + len(keyword)
    end = line.find(",", start)
    return line[start:end].strip()


input_tokens_list = []
output_tokens_list = []
reasoning_tokens_list = []
vlm_input_tokens_list = []
vlm_output_tokens_list = []
search_llm_completion_tokens = []
search_llm_prompt_tokens = []
with open(log_path, "r") as f:
    lines = f.readlines()
    print(f"Number of lines: {len(lines)}")
    if log_path.endswith(".jsonl"):
        for line in lines:
            # find all instances of "completion_tokens="
            # and "prompt_tokens=" in the line
            steps = json.loads(line)["intermediate_steps"]
            for step in steps:
                if "completion_tokens" in step:
                    completion_tokens = find_number_after_keyword(step, "completion_tokens=")       
                    prompt_tokens = find_number_after_keyword(step, "prompt_tokens=")
                    input_tokens_list.append(int(prompt_tokens))
                    output_tokens_list.append(int(completion_tokens))

                    if "reasoner" in log_path:
                        reasoning_tokens = find_number_after_keyword(step, "reasoning_tokens=")
                        reasoning_tokens_list.append(int(reasoning_tokens)) 
                    

    else:
        for line in lines:
            if line.startswith("*** response:"):
                #find the number following completion_tokens
                completion_tokens = find_number_after_keyword(line, "completion_tokens=")
                # print(completion_tokens)
                prompt_tokens = find_number_after_keyword(line, "prompt_tokens=")
                # print(prompt_tokens)
                input_tokens_list.append(int(prompt_tokens))
                output_tokens_list.append(int(completion_tokens))
            if line.startswith("*** Used VLM:"):
                #find the number following completion_tokens
                completion_tokens = find_number_after_keyword(line, "Completion tokens: ")
                # print(completion_tokens)
                prompt_tokens = find_number_after_keyword(line, "Prompt tokens:")
                # print(prompt_tokens)
                vlm_input_tokens_list.append(int(prompt_tokens))
                vlm_output_tokens_list.append(int(completion_tokens))
            if line.startswith("Search web - Prompt tokens:"):
                search_llm_prompt_tokens.append(int(find_number_after_keyword(line, "Search web - Prompt tokens: ")))
            if line.startswith("Search web - Completion tokens:"):
                search_llm_completion_tokens.append(int(find_number_after_keyword(line, "Search web - Completion tokens: ")))

print("input_tokens_list", input_tokens_list)
print("output_tokens_list", output_tokens_list)
print("reasoning_tokens_list", reasoning_tokens_list)
print("vlm_input_tokens_list", vlm_input_tokens_list)
print("vlm_output_tokens_list", vlm_output_tokens_list)
print("search_llm_prompt_tokens", search_llm_prompt_tokens)
print("search_llm_completion_tokens", search_llm_completion_tokens)

entire_output_tokens_list = []
if reasoning_tokens_list:
    for i in range(len(output_tokens_list)):
        entire_output_tokens_list.append(output_tokens_list[i] + reasoning_tokens_list[i])
    
print("entire_output_tokens_list", entire_output_tokens_list)

def get_median_and_max(tokens_list):
    """
    Get the median and maximum of a list of tokens.
    """
    if len(tokens_list) == 0:
        return 0, 0
    # avg = sum(tokens_list) / len(tokens_list)
    tokens_list.sort()
    median = tokens_list[len(tokens_list) // 2]
    max_tokens = max(tokens_list)
    return median, max_tokens

avg_input_tokens, max_input_tokens = get_median_and_max(input_tokens_list)
avg_output_tokens, max_output_tokens = get_median_and_max(output_tokens_list)
avg_entire_output_tokens, max_entire_output_tokens = get_median_and_max(entire_output_tokens_list)
avg_vlm_input_tokens, max_vlm_input_tokens = get_median_and_max(vlm_input_tokens_list)
avg_vlm_output_tokens, max_vlm_output_tokens = get_median_and_max(vlm_output_tokens_list)
avg_search_llm_prompt_tokens, max_search_llm_prompt_tokens = get_median_and_max(search_llm_prompt_tokens)
avg_search_llm_completion_tokens, max_search_llm_completion_tokens = get_median_and_max(search_llm_completion_tokens)
print("avg_input_tokens", avg_input_tokens)
print("max_input_tokens", max_input_tokens)
print("avg_output_tokens", avg_output_tokens)
print("max_output_tokens", max_output_tokens)
print("avg_entire_output_tokens", avg_entire_output_tokens)
print("max_entire_output_tokens", max_entire_output_tokens)
print("avg_vlm_input_tokens", avg_vlm_input_tokens)
print("max_vlm_input_tokens", max_vlm_input_tokens)
print("avg_vlm_output_tokens", avg_vlm_output_tokens)
print("max_vlm_output_tokens", max_vlm_output_tokens)
print("avg_search_llm_prompt_tokens", avg_search_llm_prompt_tokens)
print("max_search_llm_prompt_tokens", max_search_llm_prompt_tokens)
print("avg_search_llm_completion_tokens", avg_search_llm_completion_tokens)
print("max_search_llm_completion_tokens", max_search_llm_completion_tokens)

print(f"Total # of steps: {len(input_tokens_list)}")
print(f"Total # of input tokens: {sum(input_tokens_list)}")
print(f"Total # of output tokens: {sum(output_tokens_list)}")
print(f"Avg # of steps per question: {len(input_tokens_list)/15}")
print(f"Avg # of input tokens per question: {sum(input_tokens_list)/15}")
print(f"Avg # of output tokens per question: {sum(output_tokens_list)/15}")
