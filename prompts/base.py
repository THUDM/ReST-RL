from abc import ABC


class Formatter(ABC):
    def __init__(self, domain: str = None, tokenizer=None):
        self.domain = domain
        self.tokenizer = tokenizer

    def format(self, q: str, state: str) -> str:
        raise NotImplementedError("The method 'format' must be implemented for a formatter\n")

    def format_for_think(self, q: str, state: str) -> str:
        raise NotImplementedError("The method 'format_for_think' must be implemented for a formatter when thought tokens are used\n")

    def format_for_act(self, q: str, state: str, thought: str) -> str:
        raise NotImplementedError("The method 'format_for_act' must be implemented for a formatter when thought tokens are used\n")

    def format_thought(self, thought: str) -> str:
        raise NotImplementedError("The method 'format_thought' must be implemented for a formatter\n")

    def format_sample(self, state: str, sample: str) -> str:
        raise NotImplementedError("The method 'format_sample' must be implemented for a formatter\n")
