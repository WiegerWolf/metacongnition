import os
import anthropic
from datetime import datetime
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Set, Optional
import networkx as nx

client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

@dataclass
class Concept:
    name: str
    first_appearance: str  # timestamp
    sessions: Set[str]     # timestamps where this concept appears
    related_concepts: Set[str]
    definition: str

class KnowledgeGraph:
    def __init__(self, base_dir: Path):
        self.graph = nx.DiGraph()
        self.base_dir = base_dir
        self.concepts_file = base_dir / "concepts.json"
        self.load_graph()
    
    def load_graph(self):
        if self.concepts_file.exists():
            with open(self.concepts_file) as f:
                concepts_data = json.load(f)
                for concept_name, data in concepts_data.items():
                    self.graph.add_node(concept_name, **data)
    
    def save_graph(self):
        concepts_data = {
            node: self.graph.nodes[node] 
            for node in self.graph.nodes
        }
        with open(self.concepts_file, 'w') as f:
            json.dump(concepts_data, f, indent=2)
    
    def add_concept(self, concept: Concept):
        self.graph.add_node(
            concept.name,
            first_appearance=concept.first_appearance,
            sessions=list(concept.sessions),
            related_concepts=list(concept.related_concepts),
            definition=concept.definition
        )
        
        # Add edges to related concepts
        for related in concept.related_concepts:
            if related in self.graph:
                self.graph.add_edge(concept.name, related)
                self.graph.add_edge(related, concept.name)
    
    def get_related_concepts(self, concept_name: str, depth: int = 2) -> Set[str]:
        if concept_name not in self.graph:
            return set()
        
        related = set()
        current_level = {concept_name}
        
        for _ in range(depth):
            next_level = set()
            for concept in current_level:
                neighbors = set(self.graph.neighbors(concept))
                next_level.update(neighbors - related - {concept_name})
            related.update(next_level)
            current_level = next_level
            
        return related

class ThoughtSession:
    def __init__(self, timestamp: str, initial_thought: str):
        self.timestamp = timestamp
        self.initial_thought = initial_thought
        self.thoughts = []
        self.status = "active"  # active, paused, completed
        self.branches = []  # list of related thought sessions
        self.parent_session = None  # timestamp of parent session if this is a branch
        
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "initial_thought": self.initial_thought,
            "thoughts": self.thoughts,
            "status": self.status,
            "branches": self.branches,
            "parent_session": self.parent_session
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ThoughtSession':
        session = cls(data["timestamp"], data["initial_thought"])
        session.thoughts = data["thoughts"]
        session.status = data.get("status", "completed")
        session.branches = data.get("branches", [])
        session.parent_session = data.get("parent_session")
        return session

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

class EnhancedThoughtLibrary:
    def __init__(self, base_dir="thought_sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_graph = KnowledgeGraph(self.base_dir)
        self.active_sessions = {}  # timestamp -> ThoughtSession
        
    def create_session(self, initial_thought: str, parent_session: str = None) -> ThoughtSession:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = ThoughtSession(timestamp, initial_thought)
        if parent_session:
            session.parent_session = parent_session
            parent = self.get_session(parent_session)
            if parent:
                parent.branches.append(timestamp)
                self.save_session(parent)
        
        self.active_sessions[timestamp] = session
        return session
    
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
            if '[' in response_text and ']' in response_text:
                json_str = response_text[response_text.find('['):response_text.rfind(']')+1]
                related_timestamps = json.loads(json_str)
                related_sessions = []
                for ts in related_timestamps:
                    session = self.get_session(ts)
                    if session:  # Only add if session was found
                        related_sessions.append(session)
                return related_sessions
            return []
        except Exception as e:
            print(f"Error finding related sessions: {e}")
            print(f"Response text was: {response_text}")
            return []
    
    def find_related_concepts(self, query: str) -> List[str]:
        """Find concepts related to a query using the knowledge graph"""
        # Get all concepts that appear in sessions related to the query
        related_sessions = self.find_related_sessions(query)
        related_concepts = set()
        
        for session in related_sessions:
            for thought in session.get('thoughts', []):
                related_concepts.update(thought.get('key_concepts', []))
        
        return list(related_concepts)
    
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
                            "total_depths": data.get("total_depths", 0),
                            "final_tokens": data.get("final_tokens", 0),
                            "status": data.get("status", "completed")
                        })
        return sessions
    
    def save_session(self, session: ThoughtSession):
        """Save a session to disk"""
        session_dir = self.base_dir / session.timestamp
        session_dir.mkdir(parents=True, exist_ok=True)
        
        with open(session_dir / "session.json", "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def get_recent_concepts(self, n: int = 5) -> List[str]:
        """Get the n most recently added concepts"""
        concepts = sorted(
            self.knowledge_graph.graph.nodes(),
            key=lambda x: self.knowledge_graph.graph.nodes[x]['first_appearance'],
            reverse=True
        )
        return concepts[:n]
    
    def get_session(self, timestamp):
        """Retrieve a specific thinking session"""
        session_file = self.base_dir / timestamp / "session.json"
        if session_file.exists():
            with open(session_file) as f:
                data = json.load(f)
                data['timestamp'] = timestamp
                return data
        return None
    
    def pause_session(self, timestamp: str):
        session = self.active_sessions.get(timestamp)
        if session:
            session.status = "paused"
            self.save_session(session)
    
    def resume_session(self, timestamp: str) -> Optional[ThoughtSession]:
        session_file = self.base_dir / timestamp / "session.json"
        if session_file.exists():
            with open(session_file) as f:
                data = json.load(f)
                if data.get("status") == "paused":
                    session = ThoughtSession.from_dict(data)
                    session.status = "active"
                    self.active_sessions[timestamp] = session
                    return session
        return None
    
    def suggest_new_thoughts(self, context: str = None) -> List[str]:
        """Let the model suggest new thinking threads based on knowledge graph"""
        # Get recent concepts and their relationships
        recent_concepts = self.knowledge_graph.get_recent_concepts(5)
        related_concepts = set()
        for concept in recent_concepts:
            related_concepts.update(
                self.knowledge_graph.get_related_concepts(concept)
            )
        
        prompt = f"""Based on these recent concepts and their relationships:
        Recent concepts: {', '.join(recent_concepts)}
        Related concepts: {', '.join(related_concepts)}
        
        Please suggest 3-5 new questions or topics that would be valuable to explore.
        Consider:
        1. Unexplored connections between concepts
        2. Practical applications not yet discussed
        3. Potential implications that need deeper analysis
        
        Return your suggestions as a JSON array of strings.
        """
        
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return json.loads(response.content[0].text)
        except Exception as e:
            print(f"Error suggesting new thoughts: {e}")
            return []

    def _get_concept_definition(self, concept: str, context: str) -> str:
        """Get a definition for a concept from the model"""
        prompt = f"""Based on this context:
        {context}
        
        Please provide a clear, concise definition of the concept: {concept}
        Focus on how this concept is being used in the given context.
        """
        
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error getting concept definition: {e}")
            return ""

class ThoughtProcess:
    def __init__(self, initial_thought: str, library: EnhancedThoughtLibrary = None):
        self.initial_thought = initial_thought
        self.thoughts = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(f"thought_sessions/{self.timestamp}")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.library = library or EnhancedThoughtLibrary()
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
        # Create a properly structured thought with all necessary fields
        structured_thought = {
            "depth": depth,
            "thought": thought["current_thought"],
            "needs_more_thinking": thought["needs_more_thinking"],
            "next_thought": thought["next_thought"],
            "reasoning": thought["reasoning"],
            "tokens_at_depth": thought.get("tokens_at_depth", 0),
            "total_tokens": thought.get("total_tokens", 0),
            "key_concepts": thought.get("key_concepts", []),  # Make sure this is captured
            "referenced_insights": thought.get("referenced_insights", []),
            "concrete_applications": thought.get("concrete_applications", [])
        }
        
        # Print debug information
        print(f"\nAdding thought at depth {depth}:")
        print(f"Key concepts: {structured_thought['key_concepts']}")
        print(f"Concrete applications: {structured_thought['concrete_applications']}")
        
        self.thoughts.append(structured_thought)
        
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

def update_knowledge_graph(library: EnhancedThoughtLibrary, session: ThoughtSession):
    """Separate function to handle knowledge graph updates"""
    timestamp = session.timestamp
    all_concepts = set()
    concept_definitions = {}
    concept_relationships = {}
    
    print("\nExtracting concepts from session...")
    print(f"Number of thoughts: {len(session.thoughts)}")
    
    # Debug print of thought structure
    for i, thought in enumerate(session.thoughts):
        print(f"\nThought {i} structure:")
        for key, value in thought.items():
            print(f"{key}: {value}")
    
    # First pass: collect all concepts and their contexts
    for thought in session.thoughts:
        depth = thought.get('depth', 0)
        # Get key_concepts from the thought content
        key_concepts = []
        
        # Try to get key_concepts from different possible locations
        if 'key_concepts' in thought:
            key_concepts = thought['key_concepts']
        elif isinstance(thought.get('thought'), dict) and 'key_concepts' in thought['thought']:
            key_concepts = thought['thought']['key_concepts']
            
        print(f"\nProcessing thought at depth {depth}:")
        print(f"Key concepts found: {', '.join(key_concepts)}")
        
        # Update all_concepts set
        all_concepts.update(key_concepts)
        
        # Build relationships between concepts in the same thought
        for concept in key_concepts:
            if concept not in concept_relationships:
                concept_relationships[concept] = set()
            concept_relationships[concept].update(set(key_concepts) - {concept})
            
            # Get definition if we don't have it yet
            if concept not in concept_definitions:
                print(f"Getting definition for: {concept}")
                definition = library._get_concept_definition(
                    concept,
                    thought.get('thought', '')
                )
                concept_definitions[concept] = definition
    
    print(f"\nTotal concepts found: {len(all_concepts)}")
    print(f"Concepts: {', '.join(all_concepts)}")
    
    # Second pass: create or update concepts in the graph
    for concept in all_concepts:
        if concept not in library.knowledge_graph.graph:
            print(f"\nAdding new concept: {concept}")
            new_concept = Concept(
                name=concept,
                first_appearance=timestamp,
                sessions={timestamp},
                related_concepts=concept_relationships.get(concept, set()),
                definition=concept_definitions.get(concept, "")
            )
            library.knowledge_graph.add_concept(new_concept)
        else:
            print(f"\nUpdating existing concept: {concept}")
            node = library.knowledge_graph.graph.nodes[concept]
            node['sessions'] = list(set(node['sessions']) | {timestamp})
            node['related_concepts'] = list(
                set(node['related_concepts']) | 
                concept_relationships.get(concept, set())
            )
    
    # Save the updated graph
    library.knowledge_graph.save_graph()
    return library.knowledge_graph

def think(initial_thought: str, library: EnhancedThoughtLibrary = None):
    if library is None:
        library = EnhancedThoughtLibrary()
        
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
                    # Add error checking for session structure
                    timestamp = session.get('timestamp', 'Unknown timestamp')
                    initial_thought = session.get('initial_thought', 'Unknown topic')
                    thoughts = session.get('thoughts', [])
                    
                    context_text += f"\nFrom session {timestamp}:\n"
                    context_text += f"Topic: {initial_thought}\n"
                    if thoughts:  # Only add if there are thoughts
                        final_thought = thoughts[-1]
                        context_text += f"Final conclusion: {final_thought.get('thought', 'No conclusion available')}\n"
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
                    
                    # Store the complete thought data
                    thought_data = {
                        "depth": depth,
                        "thought": tool_response["current_thought"],
                        "needs_more_thinking": tool_response["needs_more_thinking"],
                        "next_thought": tool_response["next_thought"],
                        "reasoning": tool_response["reasoning"],
                        "tokens_at_depth": tool_response["tokens_at_depth"],
                        "total_tokens": tool_response["total_tokens"],
                        "key_concepts": tool_response["key_concepts"],
                        "referenced_insights": tool_response["referenced_insights"],
                        "concrete_applications": tool_response["concrete_applications"]
                    }
                    
                    # Add the thought to the process
                    thought_process.thoughts.append(thought_data)
                    
                    # Calculate metrics
                    metrics = thought_process.calculate_metrics(tool_response, depth)
                    if metrics:
                        print(f"\nMetrics for depth {depth}:")
                        print(metrics)
                    
                    print(f"\nReasoning at depth {depth}: {tool_response['reasoning']}")
                    print(f"Key concepts at depth {depth}: {', '.join(tool_response['key_concepts'])}")
                    if tool_response.get('concrete_applications'):
                        print(f"Concrete applications at depth {depth}: {', '.join(tool_response['concrete_applications'])}")
                    
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
            
            print("Warning: No tool response received")
            return "Thinking process incomplete - no tool response received"
            
        except Exception as e:
            print(f"Error during thinking process: {e}")
            return f"Error occurred: {str(e)}"

    result = recursive_think(initial_thought)
    return result, thought_process  # Return both the result and the thought process

if __name__ == "__main__":
    library = EnhancedThoughtLibrary()
    
    print("\nAvailable thinking sessions:")
    for session in library.list_sessions():
        print(f"Timestamp: {session['timestamp']}")
        print(f"Topic: {session['initial_thought']}")
        print(f"Status: {session['status']}")
        print(f"Depths: {session['total_depths']}")
        print("-" * 40)
    
    # Create a new thinking session
    initial_thought = "How does artificial intelligence change our understanding of consciousness?"
    
    # Find related concepts before starting new session
    print("\nFinding related concepts from previous sessions...")
    related_concepts = library.find_related_concepts(initial_thought)
    if related_concepts:
        print("Related concepts:", ", ".join(related_concepts))    
    # Run thinking process and get both result and thought_process
    result, thought_process = think(initial_thought, library)
    
    if result:
        print("\nFinal result:", result)
        print("\nUpdating knowledge graph...")
        
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a new ThoughtSession with the thoughts
        session = ThoughtSession(timestamp, initial_thought)
        session.thoughts = thought_process.thoughts  # Now this should contain the thoughts
        
        # Save the session
        library.save_session(session)
        
        print("\nExtracting concepts from session...")
        print(f"Number of thoughts: {len(session.thoughts)}")
        
        # Update knowledge graph
        graph = update_knowledge_graph(library, session)
        
        # Print graph statistics
        print("\nKnowledge Graph Statistics:")
        print(f"Number of concepts: {len(graph.graph.nodes)}")
        print(f"Number of relationships: {len(graph.graph.edges)}")
        
        if graph.graph.nodes:
            print("\nExample Concepts:")
            for node in list(graph.graph.nodes)[:5]:
                print(f"\nConcept: {node}")
                node_data = graph.graph.nodes[node]
                print(f"Definition: {node_data.get('definition', 'No definition available')}")
                print(f"Related concepts: {', '.join(node_data.get('related_concepts', []))}")
                print(f"Appears in sessions: {', '.join(node_data.get('sessions', []))}")