import os
import anthropic
from typing import List, Dict
from datetime import datetime
import json
from pathlib import Path

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

class ThoughtProcess:
    def __init__(self, initial_thought: str):
        self.initial_thought = initial_thought
        self.thoughts = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(f"thought_sessions/{self.timestamp}")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
    def add_thought(self, depth: int, thought: Dict):
        self.thoughts.append({
            "depth": depth,
            "thought": thought["current_thought"],
            "needs_more_thinking": thought["needs_more_thinking"],
            "next_thought": thought["next_thought"],
            "reasoning": thought["reasoning"],
            "tokens_at_depth": thought.get("tokens_at_depth", 0),
            "total_tokens": thought.get("total_tokens", 0)
        })
        
    def save_session(self):
        session_data = {
            "initial_thought": self.initial_thought,
            "thoughts": self.thoughts,
            "total_depths": len(self.thoughts),
            "final_tokens": self.thoughts[-1]["total_tokens"] if self.thoughts else 0
        }
        
        with open(self.session_dir / "session.json", "w") as f:
            json.dump(session_data, f, indent=2)
            
    def generate_summary(self):
        summary = f"Thought Evolution Analysis\n"
        summary += f"Initial Question: {self.initial_thought}\n\n"
        
        for t in self.thoughts:
            summary += f"Depth {t['depth']}:\n"
            summary += f"Tokens: {t['tokens_at_depth']} (Total: {t['total_tokens']})\n"
            summary += f"Thought: {t['thought'][:100]}...\n"
            summary += f"Reasoning: {t['reasoning'][:100]}...\n"
            summary += "-" * 80 + "\n"
            
        with open(self.session_dir / "summary.txt", "w") as f:
            f.write(summary)
        
        return summary

def count_tokens(messages):
    return client.beta.messages.count_tokens(
        model="claude-3-5-sonnet-20241022",
        messages=messages
    ).input_tokens

def think(initial_thought: str):
    thought_process = ThoughtProcess(initial_thought)
    
    def recursive_think(thought: str, depth: int = 0, context: List[Dict] = None, total_tokens: int = 0):
        if context is None:
            context = []
        
        # Check token limit (leaving some room for response)
        MAX_TOKENS = 150000  # Setting lower than 200k to be safe
        if total_tokens > MAX_TOKENS:
            return f"Token limit ({MAX_TOKENS}) reached. Final thought: {thought}"
        
        tools = [
            {
                "name": "continue_thinking",
                "description": """
                A tool for recursive self-reflection and deep thinking about a given topic.
                Use this tool when you need to explore a thought more deeply or when the current analysis feels incomplete.
                Return false for needs_more_thinking when you feel the thought has been fully explored or a satisfactory conclusion reached.
                Consider both depth and breadth of analysis when deciding if more thinking is needed.
                Track key concepts and their evolution through the thinking process.
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
                        },
                        "key_concepts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Main concepts explored in this thought"
                        }
                    },
                    "required": ["current_thought", "needs_more_thinking", "next_thought", "reasoning", "key_concepts"]
                }
            }
        ]

        context.append({"role": "user", "content": f"Depth {depth}: {thought}\nPlease think about this deeply and decide if more thinking is needed."})
        
        current_tokens = count_tokens(context)
        print(f"Current token count: {current_tokens}")
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=context,
            tools=tools
        )
        
        total_tokens += message.usage.input_tokens + message.usage.output_tokens
        print(f"Depth {depth} response (Total tokens: {total_tokens}):", message.content)
        
        for content in message.content:
            if content.type == "tool_use":
                tool_response = content.input
                tool_response["tokens_at_depth"] = message.usage.input_tokens + message.usage.output_tokens
                tool_response["total_tokens"] = total_tokens
                
                print(f"Reasoning at depth {depth}: {tool_response['reasoning']}")
                print(f"Key concepts at depth {depth}: {', '.join(tool_response['key_concepts'])}")
                
                thought_process.add_thought(depth, tool_response)
                
                if tool_response["needs_more_thinking"]:
                    return recursive_think(tool_response["next_thought"], depth + 1, context, total_tokens)
                else:
                    thought_process.save_session()
                    summary = thought_process.generate_summary()
                    print("\nThought Evolution Summary:")
                    print(summary)
                    return f"Thought process complete.\nFinal thought: {tool_response['next_thought']}\nReasoning: {tool_response['reasoning']}"
        
        return message.content[0].text

    return recursive_think(initial_thought)

if __name__ == "__main__":
    initial_thought = "What is free will and do we really have it?"
    result = think(initial_thought)
    print("\nFinal result:", result)
