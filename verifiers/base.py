from abc import ABC


class Verifier(ABC):
    """
    Used for verifying a completion for a specific task
    """

    def __init__(self, domain: str = None, reference=None, low_b: float = 0):
        self.domain = domain
        self.reference = reference
        self.low_b = low_b

    def verify(self, completions: list[str], initial_state: str = "") -> list[float]:
        raise NotImplementedError("The method 'verify' must be implemented for a verifier\n")
