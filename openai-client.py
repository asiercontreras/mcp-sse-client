import asyncio
import sys
import os
import json
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from datetime import datetime
from litellm import token_counter, cost_per_token
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI()

        # Add conversation tracking
        self.conversation_history = {
            "timestamp": datetime.now().isoformat(),
            "turns": []
        }

    async def connect_to_sse_server(self, server_url: str):
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

  
    async def process_query(self, query: str, messages=None) -> list:
        """Process a query using OpenAI GPT-4 and available tools (functions), supporting multiple tool calls."""

        # Get available tools (functions)
        response = await self.session.list_tools()
        available_functions = [{
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema  # OpenAI expects 'parameters' for function schema
        } for tool in response.tools]

        # Initialize messages if not provided
        if messages is None:
            messages = []
        # Add the current query to messages
        messages.append({
            "role": "user",
            "content": query
        })

        # Track this conversation turn
        current_turn = {
            "user_query": query,
            "assistant_responses": [],
            "tool_interactions": []
        }

        while True:
            # OpenAI API call (async)
            response = await self.openai.chat.completions.create(
                model="gpt-4-1106-preview",  # or your preferred function-calling model
                messages=messages,
                functions=available_functions,
                function_call="auto",
                max_tokens=1000,
            )

            message = response.choices[0].message
            tool_used = False
            assistant_text = ""

            if message.content:
                assistant_text = message.content
                print(assistant_text, end="", flush=True)
                if assistant_text.strip():
                    current_turn["assistant_responses"].append(assistant_text)

            if message.function_call:
                tool_used = True
                tool_name = message.function_call.name
                tool_args = json.loads(message.function_call.arguments)
                tool_id = f"{tool_name}_{len(current_turn['tool_interactions'])}"

                # Track tool interaction
                tool_interaction = {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_id": tool_id,
                    "result": None
                }

                # Output structured data about the tool call
                tool_call_data = {
                    "event": "tool_call",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_id": tool_id
                }
                print(json.dumps(tool_call_data), flush=True)

                # Output structured data about tool running
                tool_running_data = {
                    "event": "tool_running",
                    "tool_name": tool_name,
                    "tool_id": tool_id
                }
                print(json.dumps(tool_running_data), flush=True)

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                print(f"\n[Tool Result from {tool_name}]: {result.content}\n", flush=True)

                # Convert result to serializable format
                if hasattr(result.content, 'text'):
                    result_content = result.content.text
                elif isinstance(result.content, list):
                    result_content = [item.text if hasattr(item, 'text') else str(item) for item in result.content]
                else:
                    result_content = str(result.content)

                # Update tool interaction with result
                tool_interaction["result"] = result_content
                current_turn["tool_interactions"].append(tool_interaction)

                # Prepare detailed result output
                detailed_result = {
                    "event": "tool_completed",
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "result": result_content,
                    "pastTenseMessage": {
                        "value": f"Ran `{tool_name}`",
                        "isTrusted": False,
                        "supportThemeIcons": False,
                        "supportHtml": False
                    },
                    "isConfirmed": True,
                    "isComplete": True,
                    "resultDetails": {
                        "input": tool_args,
                        "output": result_content
                    }
                }
                print(json.dumps(detailed_result), flush=True)

                # Add the assistant's function call message
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args)
                    }
                })
                # Add the tool result as a function response
                messages.append({
                    "role": "function",
                    "name": tool_name,
                    "content": json.dumps(result_content)
                })
                # Continue the loop to see if the assistant wants to call another tool
                continue

            # If no tool was used, add the assistant message to history and break
            if assistant_text:
                messages.append({
                    "role": "assistant",
                    "content": assistant_text
                })
            break  # No more tool calls, exit loop

        # Add the completed turn to conversation history
        self.conversation_history["turns"].append(current_turn)

        print()  # Ensure newline at end
        return messages


    async def save_conversation(self, filename=None):
        """Save the conversation history to a JSON file"""
        folder = os.path.join(os.getcwd(), "conversation_history")

        os.makedirs(folder,exist_ok=True)
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        filepath = os.path.join(folder, filename)

        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        print(f"\nConversation saved to {filename}")
        return filename

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries, 'save' to save the conversation, or 'quit' to exit.")

        # Maintain conversation history across turns
        messages = []

        while True:
            try:
                query = input("\nQuery: ").strip()
                if '\\n' in query:
                    query = query.replace('\\n', '\n')

                if query.lower() == 'quit':
                    # Save conversation before exiting
                    await self.save_conversation()
                    break

                if query.lower() == 'save':
                    await self.save_conversation()
                    continue

                # Update the message history with each interaction
                messages = await self.process_query(query, messages)

            except Exception as e:
                print(f"\nError: {str(e)}")
                await self.save_conversation()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(sys.argv[1])
        await client.chat_loop()
    except Exception as e:
        print(f"An error occurred: {e}")
    
    



# Example usage inside chat_loop (for demonstration):
if __name__ == "__main__":
    asyncio.run(main())