import os
import anthropic
from typing import List, Dict

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

def think(thought: str, depth: int = 0, context: List[Dict] = None, max_depth: int = 5):
    if context is None:
        context = []
    
    if depth >= max_depth:
        return f"Max depth ({max_depth}) reached. Final thought: {thought}"
    
    # Define the think tool
    tools = [
        {
            "name": "continue_thinking",
            "description": """
            A tool for recursive self-reflection and deep thinking about a given topic.
            Use this tool when you need to explore a thought more deeply or when the current analysis feels incomplete.
            The tool should be used to build upon previous thoughts and generate new insights.
            Return true if more thinking is needed, false if the thought process is complete.
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
                    }
                },
                "required": ["current_thought", "needs_more_thinking", "next_thought"]
            }
        }
    ]

    # Add the current thought to context
    context.append({"role": "user", "content": f"Depth {depth}: {thought}\nPlease think about this deeply and decide if more thinking is needed."})
    
    # Call the LLM
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=context,
        tools=tools
    )
    
    print(f"Depth {depth} response:", message.content)
    
    # If tool was used, process the result
    for content in message.content:
        if content.type == "tool_use":
            tool_response = content.input
            if tool_response["needs_more_thinking"]:
                return think(tool_response["next_thought"], depth + 1, context, max_depth)
            else:
                return tool_response["next_thought"]
    
    return message.content[0].text

if __name__ == "__main__":
    initial_thought = "What is consciousness and how do I know if I have it?"
    result = think(initial_thought)
    print("\nFinal result:", result)
