from models.sample import sample
from prompts.base import Formatter
from verifiers.base import Verifier
from vllm import LLM


class Reasoning_Task:
    def __init__(self, q: str, backend: str = None, llm: LLM = None, use_api: bool = False, formatter: Formatter = None,
                 verifier: Verifier = None, initial_state: str = ""):
        self.q = q
        self.initial_state = initial_state
        assert backend is not None, "You should provide a valid backend\n"
        self.backend = backend
        self.llm = llm if not use_api else None
        self.base = backend if use_api else None
        if use_api:
            self.backend_type = 'api'
            print(f'Using API backend: {backend}\n')
        else:
            self.backend_type = 'vllm'
            print(f'Using VLLM backend: {backend}\n')
            assert llm is not None, "You should provide a valid LLM instance\n"
        self.formatter = formatter
        self.verifier = verifier
        self.token_usage = {'prompt': 0, 'completion': 0}

    def show_token_usage(self):
        print(f'Token usage: prompt={self.token_usage["prompt"]}, completion={self.token_usage["completion"]}\n')

    def get_token_usage(self):
        return self.token_usage

    def get_completion(self, prompt: str, temperature=0.7, max_tokens=1024, n=1, stop=None) -> list[str]:
        # for single prompt sampling
        tol = 10
        replies = []
        prompt_tokens, completion_tokens = 0, 0
        while tol > 0:
            try:
                replies, prompt_tokens, completion_tokens = sample([prompt], self.base, self.llm, temperature=temperature,
                                                                   max_tokens=max_tokens, n=n, stop=stop)
                break
            except Exception as e:
                print(f'Exception occurred: [{e}]! Retrying...\n')
                tol -= 1
        if tol <= 0:
            raise RuntimeError("Failed to get completion!\n")

        self.token_usage['prompt'] += prompt_tokens
        self.token_usage['completion'] += completion_tokens
        return replies[0]

    def get_completions(self, prompts: list[str], temperature=0.7, max_tokens=1024, n=1, stop=None) -> list[list[str]]:
        # for multiple prompts sampling
        tol = 10
        replies = []
        prompt_tokens, completion_tokens = 0, 0
        while tol > 0:
            try:
                replies, prompt_tokens, completion_tokens = sample(prompts, self.base, self.llm,
                                                                   temperature=temperature,
                                                                   max_tokens=max_tokens, n=n, stop=stop)
                break
            except Exception as e:
                print(f'Exception occurred: [{e}]! Retrying...\n')
                tol -= 1
        if tol <= 0:
            raise RuntimeError("Failed to get completion!\n")

        self.token_usage['prompt'] += prompt_tokens
        self.token_usage['completion'] += completion_tokens
        return replies

    def verify(self, completions: list[str]) -> list[float]:
        return self.verifier.verify(completions, self.initial_state)

    def format_prompt(self, cur_state: str) -> str:
        return self.formatter.format(self.q, cur_state)
