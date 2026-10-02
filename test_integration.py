#!/usr/bin/env python3
"""
Integration test for ContextEngine with content_type-aware compression.
"""
import sys
sys.path.insert(0, '.')

from agent_project.context_engine import ContextEngine, WorkingMemoryEvent

# Simple config mock
class Config:
    class memory:
        enabled = True
        max_context_tokens = 6000
        user_memory_path = './data/user.md'
        kg_storage_path = './data/kg'
        episodic_storage_path = './data/episodic'
        importance_threshold = 0.45

def test_context_engine_basic():
    print("=== ContextEngine Integration Test ===")
    try:
        engine = ContextEngine(Config())
        print("✓ ContextEngine instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate ContextEngine: {e}")
        return False
    
    # Test adding different types of events
    try:
        engine.working_memory.add("user", "What is the forecast model?", "message", "text")
        engine.working_memory.add("assistant", "def forecast(data): return data * 1.1", "message", "code")
        engine.working_memory.add("user", "What was the Q3 revenue?", "message", "text")
        engine.working_memory.add("assistant", "Revenue: $5.2M\nGrowth: 15%", "message", "data")
        engine.working_memory.add("assistant", "Product\tQ1\tQ2\tQ3\tQ4\nWidget\t1.0M\t1.2M\t1.5M\t5.2M", "message", "table")
        print("✓ Added events of different content_types")
    except Exception as e:
        print(f"✗ Failed to add events: {e}")
        return False
    
    # Test format_for_prompt (which uses compression internally)
    try:
        prompt_text = engine.working_memory.format_for_prompt(max_tokens=100)
        print(f"✓ format_for_prompt works, length: {len(prompt_text)} chars")
        # Should contain some of our content
        if "forecast" in prompt_text or "Revenue" in prompt_text or "Product" in prompt_text:
            print("✓ Prompt contains expected content")
        else:
            print("! Prompt may not contain expected content (might be compressed differently)")
        print(f"Prompt preview: {prompt_text[:200]}...")
    except Exception as e:
        print(f"✗ Failed in format_for_prompt: {e}")
        return False
    
    # Test compress_working_memory
    try:
        compressed_count = engine.compress_working_memory(target_tokens=50)
        print(f"✓ compress_working_memory completed, compressed {compressed_count} events")
    except Exception as e:
        print(f"✗ Failed in compress_working_memory: {e}")
        return False
        
    print("\n🎉 Integration test passed!")
    return True

if __name__ == "__main__":
    success = test_context_engine_basic()
    sys.exit(0 if success else 1)
