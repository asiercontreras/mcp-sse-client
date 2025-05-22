# MCP Client Automation Suite for Playwright MCP

This repository provides an automation framework for running Gherkin-based end-to-end tests on a web application using LLMs (Claude or GPT-4) and MCP tools with Playwright. The suite supports prompt-driven test execution, tool invocation, and result analysis.

## Project Structure

- `anthropic-client.py`: Client for running tests using Anthropic's Claude model.
- `openai-client.py`: Client for running tests using OpenAI's GPT-4 model.
- `start_test.py`: Orchestrates batch test execution, collects results, and generates CSV reports.
- `prompts.txt`: Contains Gherkin test scenarios as code blocks.
- `conversation_history/`: Stores conversation logs for each test run.
- `TEST/`: Stores test artifacts, logs, and generated test files.
- `test_results.csv`: Aggregated results of all test runs.
- `.env`: Environment variables (API keys, etc.).
- `pyproject.toml`: Python project dependencies.

## Setup

1. **Install dependencies**  
   Ensure you have Python 3.13+ and [uv](https://github.com/astral-sh/uv) installed.  
   Install dependencies:
   ```sh
   # Create virtual environment
   uv venv
   
   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On Unix or MacOS:
   source .venv/bin/activate
   uv add mcp anthropic python-dotenv litellm openai
   ```

2. **Configure environment**  
   Copy `.env.example` to `.env` and fill in required API keys and settings.

## Usage

### 1. Prepare Prompts

Add your Gherkin test scenarios to `prompts.txt` using fenced code blocks:
<pre>
```gherkin
Feature: Example
  Scenario: ...
```
</pre>

### 2. Run Batch Tests

Start the test runner:
```sh
python start_test.py
```
- You will be prompted to select the LLM (`Claude` or `GPT`).
- The script will run each prompt, invoke the appropriate client, and save results in the `TEST/` directory.

### 3. Review Results

- Conversation logs and generated test files are saved in `TEST/TEST_N/`.
- Aggregated results are available in `test_results.csv`.

## Scripts Overview

- [`anthropic-client.py`](anthropic-client.py):  
  Handles interactive and batch queries using Claude, manages tool calls, and saves conversation history.

- [`openai-client.py`](openai-client.py):  
  Similar to the above, but uses OpenAI's GPT-4 model and function-calling API.

- [`start_test.py`](start_test.py):  
  Reads prompts, runs tests in isolation, collects logs, and generates a CSV report with token/cost analysis.

## Notes

- Ensure the MCP server is running and accessible at the configured URL before starting tests.
- The suite moves/copies generated files and logs for each test for easy traceability.

---

For more details, see the docstrings in each script or open an issue for help.
