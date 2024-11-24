import os
import anthropic
from typing import List, Dict
from datetime import datetime
import json
from pathlib import Path
import glob

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

class ThoughtLibrary:
    def __init__(self, base_dir="thought_sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def list_sessions(self):
        """List all available thinking sessions"""
        sessions = []
        for session_dir in self.base_dir.glob("*"):
            if session_dir.is_dir():
                session_file = session_dir / "session.json"
                if session_file.exists():
                    with open(session_file) as f:
                        data = json.load(f)
                        sessions.append({
                            "timestamp": session_dir.name,
                            "initial_thought": data["initial_thought"],
                            "total_depths": data["total_depths"],
                            "final_tokens": data["final_tokens"]
                        })
        return sessions
    
    def get_session(self, timestamp):
        """Retrieve a specific thinking session"""
        session_file = self.base_dir / timestamp / "session.json"
        if session_file.exists():
            with open(session_file) as f:
                return json.load(f)
        return None
    
    def find_related_sessions(self, query):
        """Find sessions related to a given query/topic"""
        # We could use the LLM to help find related sessions
        sessions = self.list_sessions()
        
        messages = [
            {"role": "user", "content": f"""Given this query: '{query}'
             Please analyze these previous thinking sessions and identify which might be relevant:
             {json.dumps(sessions, indent=2)}
             Return your response as a JSON array of timestamp strings."""}
        ]
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages
        )
        
        try:
            related_timestamps = json.loads(response.content[0].text)
            return [self.get_session(ts) for ts in related_timestamps]
        except:
            return []

class ThoughtProcess:
    def __init__(self, initial_thought: str, library: ThoughtLibrary = None):
        self.initial_thought = initial_thought
        self.thoughts = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(f"thought_sessions/{self.timestamp}")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.library = library or ThoughtLibrary()
        
        # Find related previous sessions
        self.related_sessions = self.library.find_related_sessions(initial_thought)
        
    def add_thought(self, depth: int, thought: Dict):
        self.thoughts.append({
            "depth": depth,
            "thought": thought["current_thought"],
            "needs_more_thinking": thought["needs_more_thinking"],
            "next_thought": thought["next_thought"],
            "reasoning": thought["reasoning"],
            "tokens_at_depth": thought.get("tokens_at_depth", 0),
            "total_tokens": thought.get("total_tokens", 0),
            "key_concepts": thought.get("key_concepts", [])
        })
        
    def get_context_from_related_sessions(self):
        """Extract relevant context from related sessions"""
        if not self.related_sessions:
            return ""
            
        context = "Related previous thoughts on this topic:\n"
        for session in self.related_sessions:
            context += f"\nFrom session {session['timestamp']}:\n"
            context += f"Initial thought: {session['initial_thought']}\n"
            # Add final thought and key concepts from the session
            if session['thoughts']:
                final_thought = session['thoughts'][-1]
                context += f"Final conclusion: {final_thought['thought']}\n"
                context += f"Key concepts: {', '.join(final_thought.get('key_concepts', []))}\n"
        
        return context

def think(initial_thought: str, library: ThoughtLibrary = None):
    if library is None:
        library = ThoughtLibrary()
    
    thought_process = ThoughtProcess(initial_thought, library)
    previous_context = thought_process.get_context_from_related_sessions()
    
    def recursive_think(thought: str, depth: int = 0, context: List[Dict] = None, total_tokens: int = 0):
        if context is None:
            context = []
            # Add previous session context if available
            if previous_context:
                context.append({
                    "role": "user", 
                    "content": f"Before you start thinking about this topic, here is relevant context from previous thinking sessions:\n{previous_context}"
                })
        
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
    library = ThoughtLibrary()
    
    # List available sessions
    print("Available thinking sessions:")
    for session in library.list_sessions():
        print(f"Timestamp: {session['timestamp']}")
        print(f"Topic: {session['initial_thought']}")
        print(f"Depths: {session['total_depths']}")
        print("-" * 40)
    
    # Start new thinking process
    initial_thought = "What is free will and do we really have it?"
    result = think(initial_thought, library)
    print("\nFinal result:", result)
