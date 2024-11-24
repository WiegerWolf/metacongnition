import os
import anthropic
from typing import List, Dict
from datetime import datetime
import json
from pathlib import Path

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

class ThoughtLibrary:
    def __init__(self, base_dir="thought_sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def list_sessions(self):
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
    
    def find_related_sessions(self, query):
        """Find sessions related to a given query/topic"""
        sessions = self.list_sessions()
        if not sessions:
            return []
        
        messages = [
            {"role": "user", "content": f"""Given this query: '{query}'
            Please analyze these previous thinking sessions and identify which might be relevant.
            Return your response as a JSON array of timestamp strings.
            Previous sessions:
            {json.dumps(sessions, indent=2)}"""}
        ]
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages
        )
        
        try:
            related_timestamps = json.loads(response.content[0].text)
            related_sessions = []
            for ts in related_timestamps:
                session = self.get_session(ts)
                if session:
                    # Add timestamp to the session data
                    session['timestamp'] = ts
                    related_sessions.append(session)
            return related_sessions
        except Exception as e:
            print(f"Error finding related sessions: {e}")
            return []

    def get_session(self, timestamp):
        session_file = self.base_dir / timestamp / "session.json"
        if session_file.exists():
            with open(session_file) as f:
                return json.load(f)
        return None

class ThoughtProcess:
    def __init__(self, initial_thought: str, library: ThoughtLibrary = None):
        self.initial_thought = initial_thought
        self.thoughts = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(f"thought_sessions/{self.timestamp}")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.library = library or ThoughtLibrary()
        self.referenced_sessions = {}
        
    def add_thought(self, depth: int, thought: Dict):
        self.thoughts.append({
            "depth": depth,
            "thought": thought["current_thought"],
            "needs_more_thinking": thought["needs_more_thinking"],
            "next_thought": thought["next_thought"],
            "reasoning": thought["reasoning"],
            "tokens_at_depth": thought.get("tokens_at_depth", 0),
            "total_tokens": thought.get("total_tokens", 0),
            "key_concepts": thought.get("key_concepts", []),
            "referenced_insights": thought.get("referenced_insights", [])
        })
        
    def save_session(self):
        session_data = {
            "initial_thought": self.initial_thought,
            "thoughts": self.thoughts,
            "total_depths": len(self.thoughts),
            "final_tokens": self.thoughts[-1]["total_tokens"] if self.thoughts else 0,
            "referenced_sessions": self.referenced_sessions
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
            if t['referenced_insights']:
                summary += "Referenced Insights:\n"
                for ref in t['referenced_insights']:
                    summary += f"- From {ref['session']}: {ref['insight']}\n"
            summary += "-" * 80 + "\n"
        
        with open(self.session_dir / "summary.txt", "w") as f:
            f.write(summary)
        
        return summary

def think(initial_thought: str, library: ThoughtLibrary = None):
    if library is None:
        library = ThoughtLibrary()
        
    thought_process = ThoughtProcess(initial_thought, library)
    
    # Get related sessions
    related_sessions = library.find_related_sessions(initial_thought)
    
    def recursive_think(thought: str, depth: int = 0, context: List[Dict] = None, total_tokens: int = 0):
        # Add safety limits
        MAX_TOKENS = 100000  # Lower than previous to be safe
        MAX_DEPTH = 15      # Reasonable depth limit
        
        if total_tokens > MAX_TOKENS:
            print(f"\nReached token limit ({MAX_TOKENS})")
            return f"Stopping due to token limit. Final thought: {thought}"
            
        if depth >= MAX_DEPTH:
            print(f"\nReached maximum depth ({MAX_DEPTH})")
            return f"Stopping due to depth limit. Final thought: {thought}"

        if context is None:
            context = []
            # Add context about related sessions
            if related_sessions:
                context_text = "Before thinking about this topic, consider these relevant previous sessions:\n"
                for session in related_sessions:
                    context_text += f"\nFrom session {session['timestamp']}:\n"
                    context_text += f"Topic: {session['initial_thought']}\n"
                    if session.get('thoughts'):
                        final_thought = session['thoughts'][-1]
                        context_text += f"Final conclusion: {final_thought['thought']}\n"
                        context_text += f"Key concepts: {', '.join(final_thought.get('key_concepts', []))}\n"
                
                context.append({"role": "user", "content": context_text})

        # Modify tool description to encourage concrete conclusions
        tools = [
            {
                "name": "continue_thinking",
                "description": """
                A tool for recursive self-reflection and deep thinking about a given topic.
                Use this tool when you need to explore a thought more deeply or when the current analysis feels incomplete.
                Consider insights from previous thinking sessions when relevant.
                
                Guidelines for deciding if more thinking is needed:
                1. Are there concrete, practical aspects still unexplored?
                2. Would additional thinking lead to actionable insights?
                3. Has the current line of thought reached a natural conclusion?
                4. Are we moving from abstract to concrete understanding?
                
                Return needs_more_thinking=False when:
                1. A comprehensive practical understanding has been reached
                2. Further exploration would be redundant
                3. The key questions have been answered with actionable insights
                4. The analysis has moved from theory to practical application
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
                        },
                        "referenced_insights": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "session": {"type": "string"},
                                    "insight": {"type": "string"}
                                }
                            },
                            "description": "Insights from previous sessions that informed this thought"
                        },
                        "concrete_applications": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Practical applications or implications of this thought"
                        }
                    },
                    "required": [
                        "current_thought", 
                        "needs_more_thinking", 
                        "next_thought", 
                        "reasoning", 
                        "key_concepts", 
                        "referenced_insights",
                        "concrete_applications"
                    ]
                }
            }
        ]

        # Manage context window by keeping only recent depths
        CONTEXT_WINDOW = 5  # Keep only last 5 depths of context
        if len(context) > CONTEXT_WINDOW + 1:  # +1 for the initial related sessions context
            context = [context[0]] + context[-CONTEXT_WINDOW:]

        # Add the current thought to context
        context.append({
            "role": "user", 
            "content": f"""Depth {depth}: {thought}
            Please think about this deeply and decide if more thinking is needed.
            Focus on reaching practical, actionable conclusions rather than purely theoretical exploration."""
        })
        
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=context,
                tools=tools
            )
            
            total_tokens += message.usage.input_tokens + message.usage.output_tokens
            print(f"\nDepth {depth} response (Total tokens: {total_tokens}):")
            
            # Handle the response
            for content in message.content:
                if content.type == "tool_use":
                    tool_response = content.input
                    tool_response["tokens_at_depth"] = message.usage.input_tokens + message.usage.output_tokens
                    tool_response["total_tokens"] = total_tokens
                    
                    print(f"\nReasoning at depth {depth}: {tool_response['reasoning']}")
                    print(f"Key concepts at depth {depth}: {', '.join(tool_response['key_concepts'])}")
                    if tool_response.get('referenced_insights'):
                        print(f"Referenced insights at depth {depth}:")
                        for ref in tool_response['referenced_insights']:
                            print(f"- From {ref['session']}: {ref['insight']}")
                    
                    thought_process.add_thought(depth, tool_response)
                    
                    if tool_response["needs_more_thinking"]:
                        return recursive_think(tool_response["next_thought"], depth + 1, context, total_tokens)
                    else:
                        thought_process.save_session()
                        summary = thought_process.generate_summary()
                        print("\nThought Evolution Summary:")
                        print(summary)
                        return f"Thought process complete.\nFinal thought: {tool_response['next_thought']}\nReasoning: {tool_response['reasoning']}"
                elif content.type == "text":
                    print(f"Text response: {content.text}")
            
            # If we get here without returning, something went wrong
            print("Warning: No tool response received")
            return "Thinking process incomplete - no tool response received"
            
        except Exception as e:
            print(f"Error during thinking process: {e}")
            return f"Error occurred: {str(e)}"

    return recursive_think(initial_thought)

if __name__ == "__main__":
    library = ThoughtLibrary()
    
    print("\nAvailable thinking sessions:")
    for session in library.list_sessions():
        print(f"Timestamp: {session['timestamp']}")
        print(f"Topic: {session['initial_thought']}")
        print(f"Depths: {session['total_depths']}")
        print("-" * 40)
    
    initial_thought = "Can artificial intelligence have free will?"
    result = think(initial_thought, library)
    if result:
        print("\nFinal result:", result)
    else:
        print("\nError: No result generated")