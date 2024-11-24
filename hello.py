import os
import anthropic
from typing import List, Dict

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

def count_tokens(messages):
    return client.beta.messages.count_tokens(
        model="claude-3-5-sonnet-20241022",
        messages=messages
    ).input_tokens

def think(thought: str, depth: int = 0, context: List[Dict] = None, total_tokens: int = 0):
    if context is None:
        context = []
    
    # Check token limit (leaving some room for response)
    MAX_TOKENS = 150000  # Setting lower than 200k to be safe
    if total_tokens > MAX_TOKENS:
        return f"Token limit ({MAX_TOKENS}) reached. Final thought: {thought}"
    
    # Define the think tool
    tools = [
        {
            "name": "continue_thinking",
            "description": """
            A tool for recursive self-reflection and deep thinking about a given topic.
            Use this tool when you need to explore a thought more deeply or when the current analysis feels incomplete.
            Return false for needs_more_thinking when you feel the thought has been fully explored or a satisfactory conclusion reached.
            Consider both depth and breadth of analysis when deciding if more thinking is needed.
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "current_thought": {
                        "type": "string",
                        "description": "The current thought being processed"
                    },
                    "needs_more_thinking": {
                        "type": "boolean",
                        "description": "Whether this thought needs more processing"
                    },
                    "next_thought": {
                        "type": "string",
                        "description": "The next evolution of this thought, if more thinking is needed"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Explanation for why more thinking is or isn't needed"
                    }
                },
                "required": ["current_thought", "needs_more_thinking", "next_thought", "reasoning"]
            }
        }
    ]

    # Add the current thought to context
    context.append({"role": "user", "content": f"Depth {depth}: {thought}\nPlease think about this deeply and decide if more thinking is needed."})
    
    # Count tokens before making the call
    current_tokens = count_tokens(context)
    print(f"Current token count: {current_tokens}")
    
    # Call the LLM
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=context,
        tools=tools
    )
    
    # Update total tokens
    total_tokens += message.usage.input_tokens + message.usage.output_tokens
    print(f"Depth {depth} response (Total tokens: {total_tokens}):", message.content)
    
    # If tool was used, process the result
    for content in message.content:
        if content.type == "tool_use":
            tool_response = content.input
            print(f"Reasoning at depth {depth}: {tool_response['reasoning']}")
            
            if tool_response["needs_more_thinking"]:
                return think(tool_response["next_thought"], depth + 1, context, total_tokens)
            else:
                return f"Thought process complete.\nFinal thought: {tool_response['next_thought']}\nReasoning: {tool_response['reasoning']}"
    
    return message.content[0].text

if __name__ == "__main__":
    initial_thought = "What is consciousness and how do I know if I have it?"
    result = think(initial_thought)
    print("\nFinal result:", result)
