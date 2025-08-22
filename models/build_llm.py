from vllm import LLM


# Create an LLM.
def build_llm(model_name: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True)
    return llm
