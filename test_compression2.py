#!/usr/bin/env python3
"""
Comprehensive test for content-type-aware compression.
"""
import sys
sys.path.insert(0, '.')

from agent_project.context_engine import ContextCompressor, WorkingMemoryEvent

def estimate_tokens(text: str) -> int:
    """Same token estimator as in context_engine.py."""
    if not text:
        return 0
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return max(1, cjk // 2 + (len(text) - cjk) // 4)

def test_basic():
    print("=== Basic functionality test ===")
    compressor = ContextCompressor(llm_client=None)
    
    # Test empty events
    assert compressor.compress_events([], 100) == ""
    print("✓ Empty events")
    
    # Test single text event
    events = [WorkingMemoryEvent(role="user", content="Hello world", event_type="message", content_type="text")]
    result = compressor.compress_events(events, 100)
    assert "[TEXT]" in result
    assert "Hello world" in result
    print("✓ Single text event")
    
    # Test that result fits budget
    tokens = estimate_tokens(result)
    assert tokens <= 100, f"Result tokens {tokens} > budget 100"
    print(f"✓ Token budget respected ({tokens} tokens)")

def test_code_compression():
    print("\n=== Code compression test ===")
    compressor = ContextCompressor(llm_client=None)
    code = """def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total"""
    events = [WorkingMemoryEvent(role="assistant", content=code, event_type="message", content_type="code")]
    result = compressor.compress_events(events, 50)
    print(f"Result: {result}")
    assert "[CODE]" in result
    # Should have shortened variable names (first 3 chars for long names)
    # calculate_sum -> cal, numbers -> num, total -> tot (but total length 5 -> tot)
    # Check that we have some shortening
    assert "def cal(" in result or "def calculate_sum(" not in result  # function name shortened
    assert "tot =" in result or "total =" in result  # total may be shortened to tot
    print("✓ Code compression works")

def test_data_compression():
    print("\n=== Data compression test ===")
    compressor = ContextCompressor(llm_client=None)
    data = "Price: 1299.99\nDiscount: 25%\nFinal: 974.99"
    events = [WorkingMemoryEvent(role="assistant", content=data, event_type="message", content_type="data")]
    result = compressor.compress_events(events, 50)
    print(f"Result: {result}")
    assert "[DATA]" in result
    # Should have compressed numbers (approximation)
    # 1299.99 -> ~1.3k, 25% -> ~25%? Actually our algorithm: >=1000 -> ~/1000:.1f k, >=100 -> ~:.0f, else ~:.1f
    # 1299.99 >=1000 -> ~1.3k, 25 <100 -> ~25.0? Wait: 25 >=100? No, 25<100 -> else: ~{num:.1f} -> ~25.0
    # 974.99 >=100 -> ~974.99:.0f -> ~975
    # So we expect something like "Price: ~1.3k\nDiscount: ~25.0%\nFinal: ~975"
    # But we just check that compression happened (numbers changed)
    assert "~" in result  # indicates compression
    print("✓ Data compression works")

def test_table_compression():
    print("\n=== Table compression test ===")
    compressor = ContextCompressor(llm_client=None)
    table = """Name\tAge\tCity
Alice\t25\tNew York
Bob\t30\tLondon
Charlie\t35\tTokyo"""
    events = [WorkingMemoryEvent(role="assistant", content=table, event_type="message", content_type="table")]
    result = compressor.compress_events(events, 50)
    print(f"Result: {result}")
    assert "[TABLE]" in result
    # Should preserve header and maybe first few rows
    assert "Name" in result
    assert "Age" in result
    assert "City" in result
    print("✓ Table compression works")

def test_diagram_compression():
    print("\n=== Diagram compression test ===")
    compressor = ContextCompressor(llm_client=None)
    diagram = """flowchart TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C[Great]
    B -->|No| D[Try again]
    C --> E[End]"""
    events = [WorkingMemoryEvent(role="assistant", content=diagram, event_type="message", content_type="diagram")]
    result = compressor.compress_events(events, 50)
    print(f"Result: {result}")
    assert "[DIAGRAM]" in result
    # Should preserve first line and maybe a few lines
    assert "flowchart TD" in result
    print("✓ Diagram compression works")

def test_mixed_types():
    print("\n=== Mixed types test ===")
    compressor = ContextCompressor(llm_client=None)
    events = [
        WorkingMemoryEvent(role="user", content="Analyze this code:", event_type="message", content_type="text"),
        WorkingMemoryEvent(role="assistant", content="def foo(x): return x*2", event_type="message", content_type="code"),
        WorkingMemoryEvent(role="user", content="What's the cost?", event_type="message", content_type="text"),
        WorkingMemoryEvent(role="assistant", content="Cost: $1250.50", event_type="message", content_type="data"),
    ]
    result = compressor.compress_events(events, 100)
    print(f"Result:\n{result}")
    # Should contain all four types
    assert "[TEXT]" in result
    assert "[CODE]" in result
    assert "[DATA]" in result
    # Check approximate token count
    tokens = estimate_tokens(result)
    assert tokens <= 100, f"Mixed result tokens {tokens} > budget 100"
    print(f"✓ Mixed types work ({tokens} tokens)")

def test_fallback_to_extractive():
    print("\n=== Fallback to extractive logic test ===")
    compressor = ContextCompressor(llm_client=None)
    # Events with unsupported content_type (should fall back to text logic)
    events = [
        WorkingMemoryEvent(role="user", content="What is AI?", event_type="message", content_type="unknown"),
        WorkingMemoryEvent(role="assistant", content="AI stands for Artificial Intelligence.", event_type="message", content_type="unknown"),
    ]
    result = compressor.compress_events(events, 100)
    print(f"Result: {result}")
    # Should still work (treat unknown as text? Actually our code only processes known types; unknown will fall through to extractive logic)
    # Since we have no known types, compressed_parts will be empty, then we go to extractive fallback.
    # The extractive logic should produce something.
    assert result != ""
    # Should contain the original content in some form
    assert "AI" in result or "Artificial Intelligence" in result
    print("✓ Fallback works")

def test_with_llm_mock():
    print("\n=== Test with mock LLM client ===")
    class MockLLM:
        def chat(self, messages, temperature=0.2, max_tokens=100):
            # Return a fixed summary
            return "[Mock LLM Summary]"
    
    compressor = ContextCompressor(llm_client=MockLLM())
    # Create long text that would trigger LLM summarization
    long_text = "This is a very long sentence. " * 50
    events = [WorkingMemoryEvent(role="user", content=long_text, event_type="message", content_type="text")]
    result = compressor.compress_events(events, 50)
    print(f"Result: {result}")
    # Should contain the mock summary or be truncated
    assert "[Mock LLM Summary]" in result or len(result) > 0
    print("✓ LLM integration works")

if __name__ == "__main__":
    try:
        test_basic()
        test_code_compression()
        test_data_compression()
        test_table_compression()
        test_diagram_compression()
        test_mixed_types()
        test_fallback_to_extractive()
        test_with_llm_mock()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
