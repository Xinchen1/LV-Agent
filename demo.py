#!/usr/bin/env python3
"""
Demo of content-type-aware compression in ContextEngine.
"""
from agent_project.context_engine import ContextCompressor, WorkingMemoryEvent

def demo():
    compressor = ContextCompressor(llm_client=None)
    
    print("=== Content-Type-Aware Compression Demo ===\n")
    
    # Example 1: Code compression
    print("1. Code Compression:")
    code_event = WorkingMemoryEvent(
        role="assistant",
        content="""def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",
        event_type="message",
        content_type="code"
    )
    result = compressor.compress_events([code_event], target_tokens=30)
    print(f"Original: {len(code_event.content)} chars")
    print(f"Compressed: {result}")
    print()
    
    # Example 2: Data compression
    print("2. Data Compression:")
    data_event = WorkingMemoryEvent(
        role="assistant",
        content="""Revenue: $5,432,100.50
Expenses: $3,210,000.75
Net Profit: $2,222,099.75""",
        event_type="message",
        content_type="data"
    )
    result = compressor.compress_events([data_event], target_tokens=30)
    print(f"Original: {len(data_event.content)} chars")
    print(f"Compressed: {result}")
    print()
    
    # Example 3: Table compression
    print("3. Table Compression:")
    table_event = WorkingMemoryEvent(
        role="assistant",
        content="""Product\tQ1\tQ2\tQ3\tQ4
Widget A\t100\t120\t130\t150
Widget B\t80\t90\t100\t110
Widget C\t60\t70\t80\t90""",
        event_type="message",
        content_type="table"
    )
    result = compressor.compress_events([table_event], target_tokens=30)
    print(f"Original: {len(table_event.content)} chars")
    print(f"Compressed: {result}")
    print()
    
    # Example 4: Mixed types
    print("4. Mixed Types Compression:")
    mixed_events = [
        WorkingMemoryEvent(role="user", content="Please analyze the sales data and trend.", event_type="message", content_type="text"),
        WorkingMemoryEvent(role="assistant", content="""Sales trending upward: Q1: $100k, Q2: $120k, Q3: $140k, Q4: $160k""", event_type="message", content_type="data"),
        WorkingMemoryEvent(role="assistant", content="""def forecast(trend):
    return trend * 1.2""", event_type="message", content_type="code"),
    ]
    result = compressor.compress_events(mixed_events, target_tokens=50)
    print(f"Number of events: {len(mixed_events)}")
    print(f"Compressed:\n{result}")
    print()

if __name__ == "__main__":
    demo()
