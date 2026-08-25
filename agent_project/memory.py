"""
Advanced Memory System - Knowledge Graph + Context Compression
Long-term memory for agents with semantic retrieval
"""

from __future__ import annotations

import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

from pydantic import BaseModel, Field

from .terminal import style as _style


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
        storage_path: str = "./data/kg_store"
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

        # Prune if over limit
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
    Unified memory system combining knowledge graph, episodic memory,
    file memory, and SQLite session memory.
    """

    def __init__(
        self,
        kg_storage: str = "./data/kg_store",
        episodic_storage: str = "./data/episodic_store",
        file_memory_path: str = "./data/memory.md",
        user_memory_path: str = "./data/user.md",
        sqlite_session_path: str = "./data/sessions.db",
        project_root: Optional[str] = None,
    ):
        self.kg = KnowledgeGraph(storage_path=kg_storage)
        self.episodic = EpisodicMemory(storage_path=episodic_storage)
        self.compressor = ContextCompressor()
        self.logger = logging.getLogger("MemoryManager")

        # Optional file + session layers; failures degrade gracefully.
        self.file_memory: Optional[Any] = None
        self.session_store: Optional[Any] = None
        try:
            from .file_memory import FileMemoryManager
            self.file_memory = FileMemoryManager(
                memory_path=file_memory_path,
                user_path=user_memory_path,
                project_root=project_root,
            )
        except Exception as e:
            self.logger.warning(f"File memory unavailable: {e}")

        try:
            from .sqlite_memory import SQLiteSessionStore
            self.session_store = SQLiteSessionStore(db_path=sqlite_session_path)
        except Exception as e:
            self.logger.warning(f"SQLite session store unavailable: {e}")

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

    # ------------------------------------------------------------------
    # Unified interface for the agent loop
    # ------------------------------------------------------------------

    def remember_turn(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        """Persist a conversation turn to SQLite session memory."""
        if self.session_store is None:
            return False
        ok = True
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            if not self.session_store.add_turn(session_id, role, content, metadata=msg.get("metadata")):
                ok = False
        return ok

    def remember_file(
        self,
        topic: str,
        content: str,
        source: str = "memory",
        append: bool = True,
    ) -> bool:
        """Persist a topic to file memory (MEMORY.md / USER.md)."""
        if self.file_memory is None:
            return False
        return self.file_memory.write(topic, content, source=source, append=append)

    def recall_all(
        self,
        query: str,
        session_id: str = "",
        k: int = 5,
    ) -> Dict[str, Any]:
        """Aggregate recall from KG, episodic, file, and session memory."""
        context: Dict[str, Any] = {
            "episodes": [],
            "knowledge_nodes": [],
            "file_snippets": [],
            "session_turns": [],
        }
        try:
            episodes = self.episodic.retrieve_similar(query, k=k)
            context["episodes"] = [
                {"summary": ep.summary, "task_type": ep.task_type, "id": ep.id}
                for ep, _ in episodes
            ]
        except Exception as e:
            self.logger.debug(f"Episodic recall failed: {e}")

        try:
            nodes = self.kg.find_nodes(query=query, k=k)
            context["knowledge_nodes"] = [
                {
                    "label": node.label,
                    "properties": node.properties,
                    "id": node.id,
                    "score": score,
                }
                for node, score in nodes
            ]
        except Exception as e:
            self.logger.debug(f"KG recall failed: {e}")

        if self.file_memory is not None:
            try:
                snippets = self.file_memory.search(query, source="both", k=k)
                context["file_snippets"] = [
                    {"file": f, "text": text[:300]} for f, text in snippets
                ]
            except Exception as e:
                self.logger.debug(f"File memory recall failed: {e}")

        if self.session_store is not None:
            try:
                turns = self.session_store.search(query, session_id=session_id, k=k)
                context["session_turns"] = [
                    {"role": t.role, "content": t.content[:300], "created_at": t.created_at}
                    for t in turns
                ]
            except Exception as e:
                self.logger.debug(f"Session recall failed: {e}")

        return context

    def augment_prompt(
        self,
        base_prompt: str,
        query: str,
        session_id: str = "",
        max_tokens: int = 1024,
    ) -> str:
        """Inject aggregated memory context into a prompt."""
        context = self.recall_all(query, session_id=session_id, k=3)
        parts = [base_prompt]

        if context["file_snippets"]:
            parts.append("## File Memory")
            for snippet in context["file_snippets"][:3]:
                parts.append(f"[{snippet['file']}] {snippet['text']}")

        if context["session_turns"]:
            parts.append("## Recent Relevant Conversation")
            for turn in context["session_turns"][:3]:
                parts.append(f"{turn['role']}: {turn['content']}")

        if context["episodes"]:
            parts.append("## Relevant Past Experiences")
            for ep in context["episodes"][:2]:
                parts.append(f"- {ep['summary'][:150]}...")

        if context["knowledge_nodes"]:
            parts.append("## Relevant Knowledge")
            for node in context["knowledge_nodes"][:3]:
                props = ", ".join(f"{k}: {v}" for k, v in list(node["properties"].items())[:3])
                parts.append(f"- [{node['label']}] {props}")

        joined = "\n\n".join(parts)
        # Simple budget guard: truncate to roughly max_tokens * 4 chars.
        budget = max_tokens * 4
        if len(joined) > budget:
            joined = joined[:budget] + "\n... [memory truncated]"
        return joined


# Convenience functions
def create_memory_manager(
    kg_storage: str = "./data/kg_store",
    episodic_storage: str = "./data/episodic_store",
    llm_client: Optional[Any] = None,
    file_memory_path: str = "./data/memory.md",
    user_memory_path: str = "./data/user.md",
    sqlite_session_path: str = "./data/sessions.db",
    project_root: Optional[str] = None,
) -> MemoryManager:
    """Create memory manager with custom settings"""
    return MemoryManager(
        kg_storage=kg_storage,
        episodic_storage=episodic_storage,
        file_memory_path=file_memory_path,
        user_memory_path=user_memory_path,
        sqlite_session_path=sqlite_session_path,
        project_root=project_root,
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

    print(_style("  Memory system initialized", "2"))
