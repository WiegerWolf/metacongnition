import os
import anthropic
from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

@dataclass
class ThinkingMetrics:
    depth: int
    abstractness_score: float  # 0-1, lower is more concrete
    novelty_score: float      # 0-1, higher means more new concepts
    coherence_score: float    # 0-1, higher means better connection to previous thoughts
    practical_score: float    # 0-1, higher means more actionable insights
    
    def __str__(self):
        return f"""Depth {self.depth} Metrics:
        Abstractness: {self.abstractness_score:.2f} (lower is better)
        Novelty: {self.novelty_score:.2f}
        Coherence: {self.coherence_score:.2f}
        Practical Value: {self.practical_score:.2f}
        Overall Quality: {self.overall_quality:.2f}"""
    
    @property
    def overall_quality(self) -> float:
        return np.mean([
            1 - self.abstractness_score,  # Convert to concreteness
            self.novelty_score,
            self.coherence_score,
            self.practical_score
        ])

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
             Return only a JSON array of timestamp strings, nothing else.
             Previous sessions:
             {json.dumps(sessions, indent=2)}"""}
        ]
        
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=messages
            )
            
            # Clean the response text to ensure it's valid JSON
            response_text = response.content[0].text.strip()
            # Remove any additional text before or after the JSON array
            if '[' in response_text and ']' in response_text:
                json_str = response_text[response_text.find('['):response_text.rfind(']')+1]
                related_timestamps = json.loads(json_str)
                return [self.get_session(ts) for ts in related_timestamps]
            return []
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
        self.metrics = []
        self.concept_history = set()

    def calculate_metrics(self, thought: Dict, depth: int) -> ThinkingMetrics:
        """Calculate metrics based on the thought content directly"""
        try:
            # Calculate abstractness based on presence of concrete examples
            abstractness = 0.8 if not thought.get('concrete_applications') else 0.4
            
            # Calculate novelty based on new key concepts
            current_concepts = set(thought['key_concepts'])
            new_concepts = current_concepts - self.concept_history
            self.concept_history.update(current_concepts)
            novelty = len(new_concepts) / max(1, len(current_concepts))
            
            # Calculate coherence based on referenced insights
            coherence = min(1.0, len(thought.get('referenced_insights', [])) * 0.25)
            
            # Calculate practical value based on concrete applications
            practical = min(1.0, len(thought.get('concrete_applications', [])) * 0.2)
            
            metrics = ThinkingMetrics(
                depth=depth,
                abstractness_score=abstractness,
                novelty_score=novelty,
                coherence_score=coherence,
                practical_score=practical
            )
            
            self.metrics.append(metrics)
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
    
    def generate_metrics_summary(self) -> str:
        if not self.metrics:
            return "No metrics available"
            
        summary = "Thinking Quality Metrics:\n\n"
        
        # Overall progression
        depths = [m.depth for m in self.metrics]
        qualities = [m.overall_quality for m in self.metrics]
        
        summary += "Quality Progression:\n"
        for d, q in zip(depths, qualities):
            summary += f"Depth {d}: {'=' * int(q * 20)} {q:.2f}\n"
        
        # Averages
        avg_metrics = {
            "Abstractness": np.mean([m.abstractness_score for m in self.metrics]),
            "Novelty": np.mean([m.novelty_score for m in self.metrics]),
            "Coherence": np.mean([m.coherence_score for m in self.metrics]),
            "Practical Value": np.mean([m.practical_score for m in self.metrics])
        }
        
        summary += "\nAverage Metrics:\n"
        for metric, value in avg_metrics.items():
            summary += f"{metric}: {value:.2f}\n"
        
        return summary
    
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
            
            for content in message.content:
                if content.type == "tool_use":
                    tool_response = content.input
                    tool_response["tokens_at_depth"] = message.usage.input_tokens + message.usage.output_tokens
                    tool_response["total_tokens"] = total_tokens
                    
                    # Calculate metrics without additional API call
                    metrics = thought_process.calculate_metrics(tool_response, depth)
                    if metrics:
                        print(f"\nMetrics for depth {depth}:")
                        print(metrics)
                    
                    print(f"\nReasoning at depth {depth}: {tool_response['reasoning']}")
                    print(f"Key concepts at depth {depth}: {', '.join(tool_response['key_concepts'])}")
                    if tool_response.get('concrete_applications'):
                        print(f"Concrete applications at depth {depth}: {', '.join(tool_response['concrete_applications'])}")
                    
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
    
    # Try a new question that builds on both previous sessions
    initial_thought = "How does artificial intelligence change our understanding of consciousness?"
    
    print("\nAvailable thinking sessions:")
    for session in library.list_sessions():
        print(f"Timestamp: {session['timestamp']}")
        print(f"Topic: {session['initial_thought']}")
        print(f"Depths: {session['total_depths']}")
        print("-" * 40)
    
    result = think(initial_thought, library)
    if result:
        print("\nFinal result:", result)
    else:
        print("\nError: No result generated")