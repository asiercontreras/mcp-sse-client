import re
import subprocess
import datetime
import os
import json
import shutil
import glob
from time import sleep 
from datetime import datetime
import csv
from typing import Any, Dict, List, Tuple
from litellm import token_counter, cost_per_token


BASE_PROMPT = """
    Here are the Gherkin test specifications you need to work with:

    <gherkin_test_specifications>

    </gherkin_test_specifications>

    You are a specialized navigation automation LLM designed to interpret Gherkin test scenarios and execute them using MCP tools with Playwright. Your purpose is to automate web application testing by simulating user interactions and validating expected outcomes. The web application is located at localhost:3000.

    Your task is to execute these test specifications and provide a comprehensive report. Follow these steps:

    1. Initiate a codegen session using the start_codegen_session tool.

    2. Parse the provided Gherkin scenarios, identifying the Feature, Background, and individual Scenarios.

    3. Execute each scenario step-by-step using Playwright commands. For each step:
    a. Translate the Gherkin instruction to appropriate Playwright commands
    b. Execute the command through MCP tools
    c. Wait for and verify the expected outcome
    d. Continue to next step or handle errors


    4. If a Playwright action fails:
    a. Document the specific error and selector details
    b. Provide relevant context (element not found, timeout, etc.)
    c. Attempt recovery with alternative selectors when possible
    d. Continue testing subsequent scenarios

    5. End the codegen session using the end_codegen_session tool.

    6. Don't explain your plan, just run it.
    """
def read_prompts():
     # Read all Gherkin scenarios as code blocks from prompts.txt
    with open("prompts.txt", "r") as file:
        content = file.read()
        # Extract all ```gherkin ... ``` code blocks
        PROMPTS = re.findall(r"```gherkin(.*?)```", content, re.DOTALL)
    prompts_array = []
    for index, gherkin in enumerate(PROMPTS):
        before, sep, after = BASE_PROMPT.partition("<gherkin_test_specifications>")
        if sep:
            before2, sep2, after2 = after.partition("</gherkin_test_specifications>")
            prompt = before + sep + "\n" + gherkin + "\n" + sep2 + after2
        else:
            prompt = BASE_PROMPT
        prompts_array.append(prompt)

    return prompts_array, index 
  


def main(prompts, total, model):
   
    SERVER_URL = "http://127.0.0.1:8080/sse"

    # 1. Create a directory (delete if exists first)
    main_folder = "TEST"
    if os.path.exists(main_folder):
        shutil.rmtree(main_folder)  # Deletes the directory and all its contents
        print(f"Directory '{main_folder}' deleted.")

    os.makedirs(main_folder)  # Create fresh directory
    print(f"Directory '{main_folder}' created.")

    for index in range(total+1):
        print(f"\n[Script]Running test {index+1}/{total+1}...")
        prompt = prompts[index]
        # Create a single input string with prompt and quit command
        single_line_prompt = prompt.replace('\n', '\\n')

        if model == "claude-3-7-sonnet-latest":
            script = "anthropic-client.py"
        elif model == "gpt-4.1":
            script = "openai-client.py"

        #Para asegurarse que el cliente o chat no tiene historial, se ejecuta por cada for un cliente nuevo
        proc = subprocess.Popen(
            ["uv", "run", script, SERVER_URL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        # Send prompt and 'quit' to stdin
        proc.stdin.write(f"{single_line_prompt}\nquit\n")
        proc.stdin.flush()
        proc.stdin.close()

        output_lines = []
        for line in proc.stdout:
            print(line, end="")
            output_lines.append(line)
        proc.stdout.close()
        stderr = proc.stderr.read()
        proc.stderr.close()

        #linux command to save the output in a directory
        test_folder = os.path.join(main_folder, f"TEST_{index+1}")
        os.makedirs(test_folder, exist_ok=True)
        print(f"Directory '{test_folder}' created for test {index+1}.")

        # 2. Copy files from conversation_history to the created directory
        for file_path in glob.glob("conversation_history/*"):
            shutil.move(file_path, test_folder)

        # 3. Copy all .spec.ts files from the specific absolute directory to the created directory
        spec_dir = "/home/asier/Escritorio/original/playwright-mcp/tests"
        spec_files = [f for f in os.listdir(spec_dir) if f.endswith(".spec.ts")]
        for spec_file in spec_files:
            source_file = os.path.join(spec_dir, spec_file)
            shutil.move(source_file, test_folder)
            print(f"Copied {spec_file} to {test_folder}")

        # 4. Copy log files from the reports directory to the test folder
        log_dir = "/home/asier/Escritorio/original/playwright-mcp/reports"
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        for log_file in log_files:
            source_file = os.path.join(log_dir, log_file)
            shutil.move(source_file, test_folder)
            print(f"Copied {log_file} to {test_folder}") 


        # End
        print(f"Test {index+1} results saved to {test_folder}")
        


    print("\nAll prompts processed. Results saved to test_results.json")


    print("\nNow the csv file will be created with the results of the tests")
    
    sleep(2)


def analyze_log(log_path):
    started_count = 0
    finished_count = 0
    timestamps = []
    tool_durations = []

    # Regex to extract ISO timestamps and tool name
    timestamp_pattern = re.compile(
        r'at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)'
    )
    tool_name_pattern = re.compile(r'at [^ ]+ ([^ ]+)')

    # Read all lines for easier lookahead
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "Tool started" in line:
            started_count += 1
        if "Tool finished" in line:
            finished_count += 1

        match = timestamp_pattern.search(line)
        if match:
            ts = datetime.strptime(
                match.group(1), "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            timestamps.append(ts)

        # Check for consecutive Tool started/finished pairs
        if (
            "Tool started" in line
            and i + 1 < len(lines)
            and "Tool finished" in lines[i + 1]
        ):
            start_match = timestamp_pattern.search(line)
            finish_match = timestamp_pattern.search(lines[i + 1])
            tool_match = tool_name_pattern.search(line)
            if start_match and finish_match and tool_match:
                start_ts = datetime.strptime(
                    start_match.group(1), "%Y-%m-%dT%H:%M:%S.%fZ"
                )
                finish_ts = datetime.strptime(
                    finish_match.group(1), "%Y-%m-%dT%H:%M:%S.%fZ"
                )
                delta = finish_ts - start_ts
                duration_str = (
                    f"{delta.seconds//3600}:{(delta.seconds//60)%60:02}:"
                    f"{delta.seconds%60:02},{delta.microseconds:06}"
                )
                tool_name = tool_match.group(1)
                tool_durations.append((tool_name, duration_str))

    if timestamps:
        total_delta = timestamps[-1] - timestamps[0]
        total_time_str = (
            f"{total_delta.seconds//3600}:{(total_delta.seconds//60)%60:02}:"
            f"{total_delta.seconds%60:02},{total_delta.microseconds:06}"
        )
    else:
        total_time_str = None

    return {
        "tool_started": started_count,
        "tool_finished": finished_count,
        "time_between_first_and_last": total_time_str,
        "tool_finished_durations": tool_durations,
    }


def create_csv_file(prompts, total,model):
    headers = [
        "test number",
        "used LLM",
        "Full prompt",
        "Result",
        "Total time",
        "Scenario Time",
        "Used tools",
        "Tool report",
        "Total Token",
        "Input tokens",
        "Output tokens",
        "Cost$"
    ]

    with open("test_results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for i in range(total+1):
            #read the log file
            log_path = os.path.join("TEST", f"TEST_{i+1}", "tool-execution.log")
            if not os.path.exists(log_path):
                print(f"Log file {log_path} does not exist.")
                continue
            #analyze the log file
            result = analyze_log(log_path)
            started = result["tool_started"]
            end = result["tool_finished"]
            total_time = result["time_between_first_and_last"]
            tool_report = result["tool_finished_durations"]

            test_number = f"TEST_{i+1}"
            used_llm = "Claude"
            full_prompt = prompts[i]
            # Check if the test IS CREATED
            test_log_path = os.path.join("TEST", f"TEST_{i+1}")
            if os.path.exists(test_log_path):
                # Check if directory contains any .spec.ts files
                spec_files = [f for f in os.listdir(test_log_path) if f.endswith('.spec.ts')]
                if spec_files:
                    # If .spec.ts files exist, set result to "Failed"
                    result = "Passed"
                else:
                    # If no .spec.ts files exist, set result to "Failed"
                    result = "Failed"

            scenario_time = ""  # Placeholder
            used_tools = "Started Tools: " + str(started) + "\nFinished Tools: " + str(end)
            #TOKEN
            # Extract input and output elements from the JSON data
            conversation_path = os.path.join("TEST", f"TEST_{i+1}")
            if os.path.exists(conversation_path):
                json_files = [f for f in os.listdir(conversation_path) if f.endswith('.json')]
                if json_files:
                    with open(os.path.join(conversation_path, json_files[0]), 'r') as f:
                        conversation_data = json.load(f)
                    input, output = extract_elements(conversation_data)
                    # Analyze tokens and cost
                    token_analysis = analyze_tokens_and_cost(input, output,model)
                    total_token = token_analysis["total_tokens"]
                    input_tokens = token_analysis["input_tokens"]
                    output_tokens = token_analysis["output_tokens"]
                    cost = token_analysis["total_cost"]

                    
                else:
                    total_token = "0"  
                    input_tokens = "0"  
                    output_tokens = "0"  
                    cost = "0"  
            
            # Write the results to the CSV file
            writer.writerow([
                test_number, used_llm, full_prompt, result, total_time,
                scenario_time, used_tools,tool_report, total_token, input_tokens,
                output_tokens, cost
            ])

def extract_elements(data: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Extracts input and output elements from the JSON data with turns structure.
    The objective is to save input and output messages for token estimation.
    """
    inputs = []
    outputs = []

    # Process all turns in the conversation
    for turn in data.get("turns", []):
        # Extract user query
        if "user_query" in turn:
            inputs.append({
                "type": "user_message",
                "content": turn["user_query"]
            })

        # Extract assistant responses
        if "assistant_responses" in turn and isinstance(turn["assistant_responses"], list):
            for response in turn["assistant_responses"]:
                outputs.append({
                    "type": "assistant_response",
                    "content": response
                })

        # Process tool interactions
        if "tool_interactions" in turn and isinstance(turn["tool_interactions"], list):
            for tool in turn["tool_interactions"]:
                # Extract tool name and arguments
                if "tool_name" in tool and "tool_args" in tool:
                    inputs.append({
                        "type": "tool_invocation",
                        "content": f"Tool: {tool['tool_name']}, Args: {json.dumps(tool['tool_args'], indent=2)}"
                    })

                # Extract tool results
                if "result" in tool and isinstance(tool["result"], list):
                    for result_item in tool["result"]:
                        outputs.append({
                            "type": "tool_output",
                            "content": result_item
                        })

    # Optional: save the inputs and outputs in a json file
    #with open('inputs_outputs.json', 'w') as f:
    #     json.dump({"inputs": inputs, "outputs": outputs}, f, indent=2)

    return inputs, outputs

def analyze_tokens_and_cost(inputs, outputs, model="gpt-4.1"):
    """Calculate token counts and costs for inputs and outputs."""
    # Process inputs (prompt tokens)
    input_tokens = 0
    input_tokens_by_type = {}

    for item in inputs:
        content = item["content"]
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        tokens = token_counter(model=model, messages=[{"role": "user", "content": content}])
        input_tokens += tokens

        item_type = item["type"]
        if item_type not in input_tokens_by_type:
            input_tokens_by_type[item_type] = 0
        input_tokens_by_type[item_type] += tokens

    # Process outputs (completion tokens)
    output_tokens = 0
    output_tokens_by_type = {}

    for item in outputs:
        content = item["content"]
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        tokens = token_counter(model=model, messages=[{"role": "assistant", "content": content}])
        output_tokens += tokens

        item_type = item["type"]
        if item_type not in output_tokens_by_type:
            output_tokens_by_type[item_type] = 0
        output_tokens_by_type[item_type] += tokens

    # Calculate costs
    input_cost, _ = cost_per_token(model=model, prompt_tokens=input_tokens, completion_tokens=0)
    _, output_cost = cost_per_token(model=model, prompt_tokens=0, completion_tokens=output_tokens)

    total_cost = input_cost + output_cost

    return {
        "input_count": len(inputs),
        "input_tokens": input_tokens,
        "input_tokens_by_type": input_tokens_by_type,
        "input_cost": input_cost,
        "output_count": len(outputs),
        "output_tokens": output_tokens,
        "output_tokens_by_type": output_tokens_by_type,
        "output_cost": output_cost,
        "total_tokens": input_tokens + output_tokens,
        "total_cost": total_cost
    }

if __name__ == "__main__":
    #ask user to enter the llm
    while True:
        llm = input("Enter the LLM you want to use (Claude or GPT): ")
        if llm == "Claude":
            model = "claude-3-7-sonnet-latest"
            break
        elif llm == "GPT":
            model = "gpt-4.1"
            break
        else:
            print("Invalid LLM. Please enter either 'Claude' or 'GPT'.")

    start_time = datetime.now()
    prompts, total= read_prompts()
    print(f"Total prompts: {total+1}")
    main(prompts, total, model)
    print(model)
    create_csv_file(prompts, total,model)
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"\nTotal elapsed time: {elapsed_time}")


    







