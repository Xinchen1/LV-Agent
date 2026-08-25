"""
Advanced Memory System - Knowledge Graph + Context Compression
Long-term memory for agents with semantic retrieval
"""

from __future__ import annotations

import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import logging

from pydantic import BaseModel, Field


# ============ Types ============

@dataclass
class MemoryNode:
    """
    A node in the knowledge graph representing an entity or concept
    """
    label: str = "entity"  # entity type (person, organization, concept, task, etc.)
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def access(self):
        """Update access timestamp"""
        self.last_accessed = datetime.now().isoformat()


@dataclass
class MemoryEdge:
    """
    An edge representing a relationship between two nodes
    """
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0  # strength of relationship 0-1
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class MemoryRecord:
    """
    A complete memory entry (like an episode or experience)
    """
    content: str
    summary: str
    task_type: str
    nodes: List[str] = field(default_factory=list)  # linked node IDs
    edges: List[str] = field(default_factory=list)  # linked edge IDs
    importance: float = 0.5  # 0-1
    emotional_valence: float = 0.0  # -1 to 1 (negative to positive)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


# ============ Knowledge Graph ============

class KnowledgeGraph:
    """
    Persistent knowledge graph with vector search capabilities
    """

    def __init__(
        self,
        storage_path: str = "./data/kg_store",
        device: str = "cpu"
    ):
        self.storage_path = storage_path
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, MemoryEdge] = {}
        self.node_index: Dict[str, Set[str]] = defaultdict(set)  # label -> set of node IDs
        self.edge_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)  # (relation, source) -> edge IDs

        self._mode: str = "memory"

    def add_node(self, node: MemoryNode) -> str:
        """Add a node to the graph"""
        self.nodes[node.id] = node

        # Index by label
        self.node_index[node.label].add(node.id)

        return node.id

    def add_edge(self, edge: MemoryEdge) -> str:
        """Add an edge to the graph"""
        # Validate nodes exist
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_id} does not exist")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_id} does not exist")

        self.edges[edge.id] = edge

        # Index by relation and source
        self.edge_index[(edge.relation, edge.source_id)].add(edge.id)
        self.edge_index[(edge.relation, edge.target_id)].add(edge.id)

        return edge.id

    def find_nodes(
        self,
        query: Optional[str] = None,
        label: Optional[str] = None,
        k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Tuple[MemoryNode, float]]:
        """Find nodes by keyword/label matching."""
        query_lower = query.lower() if query else ""
        candidates = []
        for node in self.nodes.values():
            if label and node.label != label:
                continue
            if query:
                text = f"{node.label} {json.dumps(node.properties)}".lower()
                if query_lower in text:
                    candidates.append((node, 0.8))
            else:
                candidates.append((node, 1.0))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def get_connected_nodes(
        self,
        node_id: str,
        relation: Optional[str] = None,
        direction: str = "outgoing"  # "outgoing", "incoming", "both"
    ) -> List[Tuple[MemoryEdge, MemoryNode]]:
        """
        Get nodes connected to given node

        Returns:
            List of (edge, connected_node)
        """
        results = []

        # Get outgoing edges
        if direction in ("outgoing", "both"):
            for edge in self.edges.values():
                if edge.source_id == node_id:
                    if relation is None or edge.relation == relation:
                        if edge.target_id in self.nodes:
                            results.append((edge, self.nodes[edge.target_id]))

        # Get incoming edges
        if direction in ("incoming", "both"):
            for edge in self.edges.values():
                if edge.target_id == node_id:
                    if relation is None or edge.relation == relation:
                        if edge.source_id in self.nodes:
                            results.append((edge, self.nodes[edge.source_id]))

        return results

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 5
    ) -> Optional[List[Tuple[MemoryEdge, MemoryNode]]]:
        """
        Find shortest path between two nodes using BFS
        Returns list of (edge, node) tuples representing path
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        if source_id == target_id:
            return []

        # BFS
        queue = deque([(source_id, [])])  # (current_node, path)
        visited = {source_id}

        for _ in range(max_hops):
            if not queue:
                break

            current, path = queue.popleft()

            # Get neighbors
            for edge_id, edge in self.edges.items():
                neighbor = None
                if edge.source_id == current and edge.target_id not in visited:
                    neighbor = edge.target_id
                elif edge.target_id == current and edge.source_id not in visited:
                    neighbor = edge.source_id

                if neighbor:
                    new_path = path + [(edge, self.nodes[neighbor])]
                    if neighbor == target_id:
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))

        return None

    def suggest_related(
        self,
        node_id: str,
        max_results: int = 5
    ) -> List[Tuple[MemoryNode, float]]:
        """Suggest related nodes based on graph connectivity."""
        neighbor_scores = defaultdict(float)
        for edge, n in self.get_connected_nodes(node_id, direction="both"):
            neighbor_scores[n.id] += edge.weight
        ranked = sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True)[:max_results]
        return [(self.nodes[nid], score) for nid, score in ranked if nid in self.nodes]

    def prune_by_utility(
        self,
        threshold_days: float = 30.0,
        keep_min_nodes: int = 100
    ) -> int:
        """
        Remove rarely accessed old nodes
        Returns number of nodes removed
        """
        if len(self.nodes) <= keep_min_nodes:
            return 0

        cutoff_date = datetime.now() - timedelta(days=threshold_days)
        to_remove = []

        for node in self.nodes.values():
            last_access = datetime.fromisoformat(node.last_accessed)
            if (len(self.nodes) - len(to_remove) > keep_min_nodes and
                last_access < cutoff_date):
                # Check if node has many connections (important)
                connections = self.get_connected_nodes(node.id)
                if len(connections) < 3:  # not highly connected
                    to_remove.append(node.id)

        # Remove edges first
        for node_id in to_remove:
            edges_to_remove = [
                edge_id for edge_id, edge in self.edges.items()
                if edge.source_id == node_id or edge.target_id == node_id
            ]
            for edge_id in edges_to_remove:
                del self.edges[edge_id]
            del self.nodes[node_id]

        logging.info(f"KnowledgeGraph: Pruned {len(to_remove)} nodes")
        return len(to_remove)

    def save(self):
        """Save graph to disk"""
        data = {
            'nodes': [asdict(n) for n in self.nodes.values()],
            'edges': [asdict(e) for e in self.edges.values()],
            'saved_at': datetime.now().isoformat()
        }
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        with open(f"{self.storage_path}/graph.json", 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self):
        """Load graph from disk"""
        path = Path(f"{self.storage_path}/graph.json")
        if not path.exists():
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self.nodes.clear()
        self.edges.clear()
        self.node_index.clear()
        self.edge_index.clear()

        for node_data in data.get('nodes', []):
            node = MemoryNode(**node_data)
            self.nodes[node.id] = node
            self.node_index[node.label].add(node.id)

        for edge_data in data.get('edges', []):
            edge = MemoryEdge(**edge_data)
            self.edges[edge.id] = edge
            self.edge_index[(edge.relation, edge.source_id)].add(edge.id)
            self.edge_index[(edge.relation, edge.target_id)].add(edge.id)


# ============ Episodic Memory ============

class EpisodicMemory:
    """
    Store and retrieve episodic memories (experiences/trajectories)
    """

    def __init__(
        self,
        storage_path: str = "./data/episodic_store",
        max_episodes: int = 10000
    ):
        self.storage_path = storage_path
        self.max_episodes = max_episodes
        self.episodes: Dict[str, MemoryRecord] = {}
        self._mode: str = "memory"

        self.load()

    def store(
        self,
        content: str,
        summary: str,
        task_type: str,
        linked_nodes: List[str] = None,
        importance: float = 0.5,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store a new episodic memory"""
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            content=content,
            summary=summary,
            task_type=task_type,
            nodes=linked_nodes or [],
            importance=importance,
            metadata=metadata or {}
        )

        self.episodes[record.id] = record

        if len(self.episodes) > self.max_episodes:
            self._prune_old()

        self.save()
        return record.id

    def retrieve_similar(
        self,
        query: str,
        task_type: Optional[str] = None,
        k: int = 10,
        min_similarity: float = 0.6
    ) -> List[Tuple[MemoryRecord, float]]:
        """Retrieve similar episodes (keyword matching)."""
        query_lower = query.lower()
        results = []
        for ep in self.episodes.values():
            if task_type and ep.task_type != task_type:
                continue
            if query_lower in ep.summary.lower() or query_lower in ep.content.lower():
                results.append((ep, 0.8))
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]

    def retrieve_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve by ID"""
        return self.episodes.get(memory_id)

    def retrieve_recent(
        self,
        n: int = 10,
        task_type: Optional[str] = None
    ) -> List[MemoryRecord]:
        """Get most recent episodes"""
        filtered = list(self.episodes.values())
        if task_type:
            filtered = [e for e in filtered if e.task_type == task_type]

        # Sort by created_at descending
        filtered.sort(key=lambda x: x.created_at, reverse=True)
        return filtered[:n]

    def _prune_old(self):
        """Remove oldest episodes to stay under limit"""
        if len(self.episodes) <= self.max_episodes:
            return

        sorted_by_date = sorted(
            self.episodes.values(),
            key=lambda x: x.created_at
        )
        to_remove = sorted_by_date[:len(self.episodes) - self.max_episodes]

        for ep in to_remove:
            del self.episodes[ep.id]

    def save(self):
        """Save to disk"""
        if self._mode != "vector":
            path = Path(self.storage_path)
            path.mkdir(parents=True, exist_ok=True)
            data = [asdict(ep) for ep in self.episodes.values()]
            with open(path / "episodes.json", 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self):
        """Load from disk"""
        if self._mode != "vector":
            path = Path(self.storage_path) / "episodes.json"
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                for item in data:
                    ep = MemoryRecord(**item)
                    self.episodes[ep.id] = ep


# ============ Context Compressor ============

class ContextCompressor:
    """
    Compress long context using various techniques
    """

    @staticmethod
    def extractive_summary(text: str, target_tokens: int = 256) -> str:
        """
        Simple extractive summarization (take important sentences)
        """
        import re
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) <= 3:
            return " ".join(sentences)

        # Score sentences by length and position
        scores = []
        for i, s in enumerate(sentences):
            s_clean = s.strip()
            if not s_clean:
                scores.append((i, 0))
                continue
            score = len(s_clean) * (1.0 if i < len(sentences) * 0.5 else 0.5)
            scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Select top sentences up to target
        selected = sorted([i for i, _ in scores[:min(5, len(sentences))]])
        summary_sentences = [sentences[i].strip() for i in selected if sentences[i].strip()]

        return " ".join(summary_sentences)

    @staticmethod
    def sliding_window(
        text: str,
        window_size: int = 512,
        overlap: int = 64
    ) -> List[str]:
        """
        Split text into overlapping windows
        """
        words = text.split()
        if len(words) <= window_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + window_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += window_size - overlap
            if start >= end:
                break

        return chunks

    @staticmethod
    def hierarchical_compress(
        text: str,
        levels: int = 2,
        tokens_per_level: List[int] = None
    ) -> List[str]:
        """
        Hierarchical compression: keep summaries at multiple abstraction levels
        """
        if tokens_per_level is None:
            tokens_per_level = [1024, 256, 64]

        current = text
        summaries = [text]  # level 0 = original

        for level in range(1, levels):
            target = tokens_per_level[level] if level < len(tokens_per_level) else tokens_per_level[-1]
            summary = ContextCompressor.extractive_summary(current, target_tokens=target)
            summaries.append(summary)
            current = summary

        return summaries


# ============ Unified Memory Manager ============

class MemoryManager:
    """
    Unified memory system combining knowledge graph, episodic memory, and context compression
    """

    def __init__(
        self,
        kg_storage: str = "./data/kg_store",
        episodic_storage: str = "./data/episodic_store",
    ):
        self.kg = KnowledgeGraph(storage_path=kg_storage)
        self.episodic = EpisodicMemory(storage_path=episodic_storage)
        self.compressor = ContextCompressor()
        self.logger = logging.getLogger("MemoryManager")

    def store_interaction(
        self,
        task: str,
        trajectory: Dict[str, Any],
        outcome: str,
        success: bool,
        entities: List[str] = None
    ):
        """Store an agent interaction as episodic memory"""
        content = json.dumps(trajectory, indent=2)
        summary = f"Task: {task[:100]}... Outcome: {outcome[:100]} Success: {success}"

        self.episodic.store(
            content=content,
            summary=summary,
            task_type="agent_interaction",
            importance=1.0 if success else 0.3,
            metadata={
                'success': success,
                'task': task,
                'entities': entities or []
            }
        )

    def extract_and_store_entities(
        self,
        text: str,
        entity_types: List[str] = None
    ) -> List[str]:
        """
        Extract named entities and store them in knowledge graph
        Simplified: use regex for demo. Production would use NER model.
        """
        if entity_types is None:
            entity_types = ['person', 'organization', 'location', 'concept', 'tool']

        import re
        extracted = []

        # Simple extraction for capitalized phrases (proper nouns)
        # This is a stub - real implementation would use a proper NER
        matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        for match in matches:
            if len(match) > 2 and len(match) < 50:
                node = MemoryNode(
                    label='concept',
                    properties={'name': match, 'source_text': text[:200]}
                )
                if any(kw in match.lower() for kw in ['api', 'tool', 'library', 'function', 'command']):
                    node.label = 'tool'
                node_id = self.kg.add_node(node)
                extracted.append(node_id)

        self.logger.info(f"Extracted {len(extracted)} entities")
        return extracted

    def link_memory_to_nodes(
        self,
        memory_id: str,
        node_ids: List[str],
        relation: str = "related_to"
    ):
        """Link an episodic memory to knowledge graph nodes"""
        memory = self.episodic.retrieve_by_id(memory_id)
        if memory:
            memory.nodes.extend(node_ids)
            self.episodic.save()

    def get_relevant_context(
        self,
        query: str,
        max_episodes: int = 3,
        max_nodes: int = 5,
        include_related: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query
        """
        context = {
            'episodes': [],
            'knowledge_nodes': [],
            'related_nodes': []
        }

        # Get similar episodes
        episodes = self.episodic.retrieve_similar(query, k=max_episodes)
        context['episodes'] = [
            {
                'summary': ep.summary,
                'task_type': ep.task_type,
                'id': ep.id
            }
            for ep, _ in episodes
        ]

        # Get knowledge nodes
        nodes = self.kg.find_nodes(query=query, k=max_nodes)
        context['knowledge_nodes'] = [
            {
                'label': node.label,
                'properties': node.properties,
                'id': node.id,
                'score': score
            }
            for node, score in nodes
        ]

        # Get related nodes from episodes
        if include_related and episodes:
            for ep, _ in episodes:
                for node_id in ep.nodes[:3]:  # limit
                    related = self.kg.suggest_related(node_id, max_results=2)
                    context['related_nodes'].extend([
                        {
                            'label': n.label,
                            'properties': n.properties,
                            'score': s,
                            'via_node': node_id
                        }
                        for n, s in related
                    ])

        return context

    def augment_prompt_with_memory(
        self,
        base_prompt: str,
        query: str,
        max_tokens: int = 1024
    ) -> str:
        """
        Augment a prompt with relevant memory context
        """
        context = self.get_relevant_context(query, max_episodes=2, max_nodes=3)

        augmented = base_prompt

        if context['episodes']:
            augmented += "\n\n## Relevant Past Experiences:\n"
            for ep in context['episodes'][:2]:
                augmented += f"\n- {ep['summary'][:150]}...\n"

        if context['knowledge_nodes']:
            augmented += "\n## Relevant Knowledge:\n"
            for node in context['knowledge_nodes'][:3]:
                props = ', '.join(f"{k}: {v}" for k, v in list(node['properties'].items())[:3])
                augmented += f"\n- [{node['label']}] {props}\n"

        return augmented

    def compress_and_store(
        self,
        text: str,
        task: str,
        max_tokens: int = 256
    ) -> str:
        """
        Compress text and store in memory
        Returns compressed version suitable for prompts
        """
        # Compress
        hierarchical = self.compressor.hierarchical_compress(
            text,
            levels=2,
            tokens_per_level=[max_tokens * 2, max_tokens]
        )

        compressed = hierarchical[-1]  # most compressed

        # Store
        self.episodic.store(
            content=text,
            summary=compressed,
            task_type="compressed_context",
            importance=0.3
        )

        return compressed


# Convenience functions
def create_memory_manager(
    kg_storage: str = "./data/kg_store",
    episodic_storage: str = "./data/episodic_store",
) -> MemoryManager:
    """Create memory manager with custom settings"""
    return MemoryManager(
        kg_storage=kg_storage,
        episodic_storage=episodic_storage,
    )


def quick_store(
    task: str,
    content: str,
    summary: str = None,
    success: bool = True
):
    """Quick function to store interaction"""
    manager = create_memory_manager()
    manager.store_interaction(
        task=task,
        trajectory={'content': content},
        outcome=summary or content[:100],
        success=success
    )


if __name__ == "__main__":
    # Quick demo
    manager = MemoryManager()

    # Store something
    manager.store_interaction(
        task="Calculate fibonacci(10)",
        trajectory={"steps": ["think", "use calculator", "get result"]},
        outcome="Result: 55",
        success=True
    )

    # Search
    results = manager.get_relevant_context("fibonacci math")
    print(f"Found {len(results['episodes'])} relevant episodes")
    print(f"Found {len(results['knowledge_nodes'])} knowledge nodes")

    print("\033[2m  Memory system initialized\033[0m")


# ============ Planning (co-located for backward-compatible imports) ============
# NOTE: agent.py imports Planner, PlanningStrategy and create_simple_plan from
# this module. Keep these symbols available even though the bulk of this file is
# the memory subsystem.

class PlanningStrategy(Enum):
    """Supported planning strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


@dataclass
class PlanNode:
    """A single node in an execution plan."""
    node_id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    assigned_loops: int = 2
    tool_hint: Optional[str] = None
    expected_output: str = ""


class Plan:
    """Simple DAG-based plan."""

    def __init__(self, nodes: Dict[str, PlanNode]):
        self.nodes = nodes

    def topological_sort(self) -> List[str]:
        """Return node IDs in dependency order."""
        visited: Set[str] = set()
        order: List[str] = []

        def visit(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            node = self.nodes.get(nid)
            if node:
                for dep in node.dependencies:
                    visit(dep)
            order.append(nid)

        for nid in self.nodes:
            visit(nid)
        return order

    def to_markdown(self) -> str:
        lines = ["## Plan"]
        for idx, nid in enumerate(self.topological_sort(), 1):
            node = self.nodes[nid]
            deps = f" (after: {', '.join(node.dependencies)})" if node.dependencies else ""
            lines.append(f"{idx}. [{nid}] {node.description}{deps}")
        return "\n".join(lines)


class Planner:
    """Lightweight LLM-based task planner."""

    def __init__(self, model_backend: Any, tokenizer: Any = None, config: Any = None):
        self.model = model_backend
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logging.getLogger("Planner")

    def create_plan(
        self,
        task: str,
        strategy: PlanningStrategy = PlanningStrategy.ADAPTIVE,
        optimize: bool = True,
        max_subtasks: int = 10
    ) -> Plan:
        """Create a plan by asking the LLM to decompose the task."""
        prompt = self._build_plan_prompt(task, strategy, max_subtasks)
        try:
            raw = self.model.generate(
                prompt=prompt,
                n_loops=1,
                temperature=0.2,
                max_tokens=1024,
            ).strip()
        except Exception as e:
            self.logger.warning(f"LLM plan generation failed: {e}")
            return self._fallback_plan(task)

        nodes = self._parse_plan_text(raw, max_subtasks)
        if not nodes:
            return self._fallback_plan(task)

        if optimize:
            nodes = self._optimize(nodes, strategy)

        return Plan(nodes)

    def _build_plan_prompt(self, task: str, strategy: PlanningStrategy, max_subtasks: int) -> str:
        strategy_desc = {
            PlanningStrategy.SEQUENTIAL: "Break the task into sequential steps that must run in order.",
            PlanningStrategy.PARALLEL: "Identify independent subtasks that can be executed in parallel.",
            PlanningStrategy.HIERARCHICAL: "Create a high-level plan with nested sub-goals.",
            PlanningStrategy.ADAPTIVE: "Choose sequential or parallel decomposition based on the task.",
        }.get(strategy, "Break the task into clear, actionable steps.")

        return (
            "You are a task planner. Decompose the user's task into a small number of concrete steps.\n\n"
            f"Task: {task}\n\n"
            f"Strategy: {strategy.value}\n"
            f"{strategy_desc}\n\n"
            "Rules:\n"
            "- Output ONLY a JSON object with key 'steps' containing a list of objects.\n"
            "- Each step object must have 'description' (string), 'dependencies' (list of step ids), and optional 'tool_hint' (string).\n"
            "- Use step ids like 'task_0', 'task_1', etc. task_0 is the root step.\n"
            f"- Limit to at most {max_subtasks} steps.\n"
            "- Do not include any explanation outside the JSON.\n\n"
            "Example:\n"
            '{"steps": [{"id": "task_0", "description": "Locate the target file or directory", "dependencies": []}, '
            '{"id": "task_1", "description": "Read and analyze the relevant contents", "dependencies": ["task_0"]}]}'
        )

    def _parse_plan_text(self, text: str, max_subtasks: int) -> Dict[str, PlanNode]:
        """Parse plan from LLM output (JSON or numbered list)."""
        # Try JSON first.
        try:
            start = text.find('{')
            if start != -1:
                payload = json.loads(text[start:])
                steps = payload.get("steps") or payload.get("tasks") or payload.get("plan")
                if isinstance(steps, list):
                    nodes: Dict[str, PlanNode] = {}
                    for i, s in enumerate(steps[:max_subtasks]):
                        sid = s.get("id") if s.get("id") else f"task_{i}"
                        nodes[sid] = PlanNode(
                            node_id=sid,
                            description=s.get("description", s.get("task", s.get("name", "unknown"))),
                            dependencies=[str(d) for d in s.get("dependencies", []) if d],
                            assigned_loops=max(1, min(8, int(s.get("loops", 2)))),
                            tool_hint=s.get("tool_hint") or s.get("tool"),
                            expected_output=s.get("expected_output", ""),
                        )
                    return nodes
        except Exception as e:
            self.logger.debug(f"Plan JSON parse failed: {e}")

        # Fallback: parse numbered lines.
        nodes: Dict[str, PlanNode] = {"task_0": PlanNode("task_0", f"Complete: {text[:120]}")}
        line_re = re.compile(r"^\s*(?:\d+|[\-\*])\.\s*(.+)$")
        count = 1
        prev_ids: List[str] = []
        for line in text.splitlines():
            m = line_re.match(line)
            if not m:
                continue
            sid = f"task_{count}"
            nodes[sid] = PlanNode(
                node_id=sid,
                description=m.group(1).strip(),
                dependencies=prev_ids[-1:] if prev_ids else [],
                assigned_loops=2,
            )
            prev_ids.append(sid)
            count += 1
            if count > max_subtasks:
                break
        return nodes

    def _optimize(self, nodes: Dict[str, PlanNode], strategy: PlanningStrategy) -> Dict[str, PlanNode]:
        """Basic optimization: cap loops and ensure root exists."""
        if "task_0" not in nodes:
            root_desc = "Overall task execution"
            nodes["task_0"] = PlanNode(
                node_id="task_0",
                description=root_desc,
                dependencies=[],
                assigned_loops=2,
            )

        for nid, node in nodes.items():
            # Keep loop assignments modest.
            node.assigned_loops = max(1, min(node.assigned_loops, 6))
            # Sequential fallback for unknown strategies.
            if strategy == PlanningStrategy.SEQUENTIAL and node.dependencies:
                pass
        return nodes

    def _fallback_plan(self, task: str) -> Plan:
        """Minimal fallback plan when LLM planning fails."""
        return Plan({
            "task_0": PlanNode("task_0", task, assigned_loops=2),
            "task_1": PlanNode("task_1", "Execute the task and verify the result", dependencies=["task_0"], assigned_loops=2),
        })


def create_simple_plan(task: str) -> Plan:
    """Create a lightweight plan for file/folder analysis tasks without LLM."""
    task_lower = task.lower()
    nodes: Dict[str, PlanNode] = {
        "task_0": PlanNode("task_0", f"Locate and identify the target for: {task[:120]}", assigned_loops=1),
        "task_1": PlanNode("task_1", "Gather relevant files and context", dependencies=["task_0"], assigned_loops=2),
        "task_2": PlanNode("task_2", "Analyze contents and synthesize the answer", dependencies=["task_1"], assigned_loops=2),
    }

    if any(kw in task_lower for kw in ["fix", "修改", "修复", "apply", "patch"]):
        nodes["task_3"] = PlanNode(
            "task_3",
            "Apply the required changes and verify syntax",
            dependencies=["task_2"],
            assigned_loops=2,
            tool_hint="file_ops",
        )
    elif any(kw in task_lower for kw in ["test", "测试", "run", "执行"]):
        nodes["task_3"] = PlanNode(
            "task_3",
            "Run tests or commands to verify behavior",
            dependencies=["task_2"],
            assigned_loops=2,
            tool_hint="bash_exec",
        )

    return Plan(nodes)


# Backwards-compatible aliases used by older tests and external callers.
TaskNode = PlanNode
