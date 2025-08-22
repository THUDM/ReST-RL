from vllm import LLM, SamplingParams
from models.api import get_reply


def sample_with_vllm(llm: LLM, prompts: list[str], temperature: float = 0.7, max_tokens: int = 1024, n: int = 1, stop: list[str] = None) -> tuple[list[list[str]], int, int]:
    # Create a sampling params object
    sampling_params = SamplingParams(n=n, temperature=temperature, stop=stop, max_tokens=max_tokens)

    # Generate texts from the prompt. The output is a list of RequestOutput objects
    outputs = llm.generate(prompts, sampling_params)
    # Process the outputs
    all_generated_texts = []
    current_prompt_tokens = 0
    current_completion_tokens = 0
    for output in outputs:
        generated_texts = [output.outputs[i].text for i in range(n)]
        all_generated_texts.append(generated_texts)
        completion_token_usages = [len(output.outputs[i].token_ids) for i in range(n)]
        prompt_token_usages = [len(output.prompt_token_ids) for i in range(n)]
        current_completion_tokens += sum(completion_token_usages)
        current_prompt_tokens += sum(prompt_token_usages)
    return all_generated_texts, current_prompt_tokens, current_completion_tokens


def sample_with_api(prompt: str, base: str, temperature: float = 0.7, max_tokens: int = 1024, n: int = 1, stop: list[str] = None) -> tuple[list[str], int, int]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    outputs, current_prompt_tokens, current_completion_tokens = get_reply(messages, base=base, temperature=temperature,
                                                                          max_tokens=max_tokens, n=n, stop=stop)
    return outputs, current_prompt_tokens, current_completion_tokens


def sample(prompts: list[str], base: str = None, llm: LLM = None, temperature: float = 0.7, max_tokens: int = 1024, n: int = 1, stop: list[str] = None) -> tuple[list[list[str]], int, int]:
    """
    Sampling method
    Priority: api > llm
    Stop strings will not be kept
    """
    if base is not None:
        all_samples = []
        all_prompt_tokens = 0
        all_completion_tokens = 0
        for prompt in prompts:
            samples, prompt_tokens, completion_tokens = sample_with_api(prompt, base, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
            all_samples.append(samples)
            all_prompt_tokens += prompt_tokens
            all_completion_tokens += completion_tokens
        return all_samples, all_prompt_tokens, all_completion_tokens

    elif llm is not None:
        return sample_with_vllm(llm, prompts, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

    else:
        raise ValueError("No backend provided\n")
