# tools/file_ops.py:1015-1026 - 并行调用
def _parallel_call(self, payloads, timeout=120.0):
    """使用线程池并行执行多个 Rust 调用"""
    futures = {self._batch_executor.submit(self._call_rust_fast, p, timeout): i 
               for i, p in enumerate(payloads)}
    # ... as_completed 收集结果