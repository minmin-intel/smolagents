
logfile="log_gpt4o_visionbrowser_run3.log"
prompt_tokens = []
completion_tokens = []

 
with open(logfile, "r") as f:
    lines = f.readlines()
    num_questions = int(lines[0].split("Loaded")[-1].split("questions")[0].strip())
    print(f"Number of questions: {num_questions}")
    for line in lines:
        if "**Prompt tokens" in line:
            token = line.split(":")[1].strip()
            prompt_tokens.append(token)
        elif "**Completion tokens" in line:
            token = line.split(":")[1].strip()
            completion_tokens.append(token)

total_steps = len(prompt_tokens)
print(f"Total steps: {total_steps}")
avg_steps = total_steps / num_questions
print(f"Average steps per question: {avg_steps}")

total_prompt_tokens = sum([int(token) for token in prompt_tokens])
print(f"Total prompt tokens: {total_prompt_tokens}")
avg_prompt_tokens = total_prompt_tokens / num_questions
print(f"Average prompt tokens per question: {avg_prompt_tokens}")

total_completion_tokens = sum([int(token) for token in completion_tokens])
print(f"Total completion tokens: {total_completion_tokens}")
avg_completion_tokens = total_completion_tokens / num_questions
print(f"Average completion tokens per question: {avg_completion_tokens}")