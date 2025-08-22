from tasks.base import *
from rms.reward_models import BaseRM


class Common_Task(Reasoning_Task):
    def __init__(self, q: str, backend: str = None, llm: LLM = None, use_api: bool = False, formatter: Formatter = None,
                 verifier: Verifier = None, initial_state: str = "", phase: str = 'train', num_sample: int = 5, temperature: float = 0.7,
                 max_tokens: int = 1024, stop: list[str] = None, rm: BaseRM = None, do_format: bool = True):
        super().__init__(q, backend, llm, use_api, formatter, verifier, initial_state)
        assert phase in ['train', 'test'], "Argument 'phase' must be 'train' or 'test'\n"
        self.phase = phase
        self.num_sample = num_sample
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.do_format = do_format
        if self.phase == 'train':
            self.rm = None
            self.rm_type = 'none'
        else:
            if rm is not None:
                assert hasattr(rm, 'eval'), 'Reward model must have eval method\n'
                assert hasattr(rm, 'eval_batch'), 'Reward model must have eval_batch method\n'
                self.rm = rm
                self.rm_type = rm.rm_type
                print(f"Using {self.rm_type} for evaluation\n")
            else:
                self.rm = None
                self.rm_type = 'none'

    def get_reward(self, completions: list[str]) -> list[float]:
        if self.phase == 'train':
            return self.verify(completions)
        else:
            assert self.rm is not None, "Reward model must be implemented before evaluation\n"
            return self.rm.eval_batch(self.q, completions)

    def run(self) -> dict:
        if self.do_format:
            prompt = self.format_prompt(self.initial_state)
        else:
            prompt = self.q
        samples = self.get_completion(prompt, self.temperature, self.max_tokens, self.num_sample, self.stop)
        formatted_samples = [self.formatter.format_sample(self.initial_state, s) for s in samples]
        formatted_completions = [self.initial_state + s.rstrip() for s in formatted_samples]
        if self.phase == 'train':
            rewards = self.get_reward(formatted_completions)
            result = {'formatted_prompt': prompt, "completions": [{'sample': sample, 'formatted_sample': formatted_sample, 'formatted_completion': formatted_completion, 'reward': reward} for sample, formatted_sample, formatted_completion, reward in zip(samples, formatted_samples, formatted_completions, rewards)]}
        else:
            if self.rm is not None:
                rewards = self.get_reward(formatted_completions)
                result = {'formatted_prompt': prompt, "completions": [{'sample': sample, 'formatted_sample': formatted_sample, 'formatted_completion': formatted_completion, 'reward': reward} for sample, formatted_sample, formatted_completion, reward in zip(samples, formatted_samples, formatted_completions, rewards)]}
            else:
                result = {'formatted_prompt': prompt, "completions": [{'sample': sample, 'formatted_sample': formatted_sample, 'formatted_completion': formatted_completion} for sample, formatted_sample, formatted_completion in zip(samples, formatted_samples, formatted_completions)]}
        return result
