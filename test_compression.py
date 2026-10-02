#!/usr/bin/env python3
"""
Test script for content-type-aware compression in ContextEngine.
"""
import sys
sys.path.insert(0, '.')

from agent_project.context_engine import ContextCompressor, WorkingMemoryEvent

def test_compression():
    compressor = ContextCompressor(llm_client=None)  # No LLM client for testing
    
    # Create sample events of different types
    events = [
        WorkingMemoryEvent(role="user", content="What is the capital of France?", event_type="message", content_type="text"),
        WorkingMemoryEvent(role="assistant", content="The capital of France is Paris.", event_type="message", content_type="text"),
        WorkingMemoryEvent(role="user", content="Show me a Python function to add two numbers.", event_type="message", content_type="text"),
        WorkingMemoryEvent(role="assistant", content="def add(a, b):\\n    return a + b", event_type="message", content_type="code"),
        WorkingMemoryEvent(role="user", content="What was the revenue in Q3?", event_type="message", content_type="text"),
        WorkingMemoryEvent(role="assistant", content="Revenue: $1,250,000\\nProfit: $250,000", event_type="message", content_type="data"),
        WorkingMemoryEvent(role="user", content="Show sales data table.", event_type="message", content_type="text"),
        WorkingMemoryEvent(role="assistant", content="Quarter\\tRevenue\\nQ1\\t1000\\nQ2\\t1200\\nQ3\\t1250\\nQ4\\t1300", event_type="message", content_type="table"),
    ]
    
    print("Testing compression with target_tokens=100")
    result = compressor.compress_events(events, target_tokens=100)
    print("Result:")
    print(result)
    print("\n---\n")
    
    # Test with only code events
    code_events = [e for e in events if e.content_type == "code"]
    print("Testing code-only compression:")
    result_code = compressor.compress_events(code_events, target_tokens=50)
    print(result_code)
    print("\n---\n")
    
    # Test with only data events
    data_events = [e for e in events if e.content_type == "data"]
    print("Testing data-only compression:")
    result_data = compressor.compress_events(data_events, target_tokens=50)
    print(result_data)
    print("\n---\n")
    
    # Test with only table events
    table_events = [e for e in events if e.content_type == "table"]
    print("Testing table-only compression:")
    result_table = compressor.compress_events(table_events, target_tokens=50)
    print(result_table)
    print("\n---\n")

if __name__ == "__main__":
    test_compression()
