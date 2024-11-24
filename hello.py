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
    
    # Add the current thought to context
    context.append({"role": "user", "content": f"Depth {depth}: {thought}\nPlease think about this and decide if more thinking is needed."})
    
    # Call the LLM
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=context
    )
    
    new_thought = message.content[0].text
    print(f"Depth {depth}: {new_thought}")
    
    # Simple way to decide if more thinking is needed
    # We could make this more sophisticated
    if "needs more thinking" in new_thought.lower():
        return think(new_thought, depth + 1, context, max_depth)
    
    return new_thought

if __name__ == "__main__":
    initial_thought = "What is consciousness and how do I know if I have it?"
    result = think(initial_thought)
    print("\nFinal result:", result)
