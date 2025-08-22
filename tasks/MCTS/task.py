from algorithms.MCTS.base import DecisionNode
from algorithms.MCTS.mcts import MCTS
from rms.reward_models import BaseRM
from tasks.base import *


class MCTS_Task(Reasoning_Task):
    def __init__(self, q: str, backend: str = None, llm: LLM = None, use_api: bool = False, formatter: Formatter = None,
                 verifier: Verifier = None, initial_state: str = "", phase: str = 'train', time_limit: float = None,
                 iteration_limit: int = None, num_sample: int = 5,
                 num_decision: int = 3, exploration_constant: float = 0.2, eps: float = 0.1,
                 temperature: float = 0.7, max_tokens: int = 1024, stop: list[str] = None, rm: BaseRM = None,
                 use_thought: bool = False, stop_think: list[str] = None, max_thought_tokens: int = 128):
        super().__init__(q, backend, llm, use_api, formatter, verifier, initial_state)
        self.algorithm = 'mcts'
        assert phase in ['train', 'test'], "Argument 'phase' must be 'train' or 'test'\n"
        self.phase = phase
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.num_sample = num_sample
        self.num_decision = num_decision
        self.exploration_constant = exploration_constant
        self.eps = eps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.use_thought = use_thought
        self.stop_think = stop_think
        self.max_thought_tokens = max_thought_tokens
        if self.phase == 'train':
            self.rm = None
            self.rm_type = 'none'
        else:
            assert rm is not None, 'For reasoning test, reward model must be implemented\n'
            assert hasattr(rm, 'eval'), 'Reward model must have eval method\n'
            assert hasattr(rm, 'eval_batch'), 'Reward model must have eval_batch method\n'
            self.rm = rm
            self.rm_type = rm.rm_type
            print(f"Using {self.rm_type} for evaluation\n")
        self.should_do_rollout = False if self.rm_type == 'prm' else True

    def do_sample(self, node: DecisionNode, num_sample: int) -> tuple[list[str], list[str]]:
        cur_state = node.state
        if self.use_thought:
            think_prompt = self.format_think_prompt(cur_state)
            thoughts_ = self.get_completion(think_prompt, self.temperature, self.max_thought_tokens, num_sample, self.stop_think)
            thoughts = [self.formatter.format_thought(thought) for thought in thoughts_]
            act_prompts = [self.format_act_prompt(cur_state, thought) for thought in thoughts]
            samples_ = self.get_completions(act_prompts, self.temperature, self.max_tokens, 1, self.stop)
            samples = [s[0] for s in samples_]
            return [self.formatter.format_sample(cur_state, s) for s in samples], thoughts
        else:
            prompt = self.format_prompt(cur_state)
            samples = self.get_completion(prompt, self.temperature, self.max_tokens, num_sample, self.stop)
            return [self.formatter.format_sample(cur_state, s) for s in samples], []

    def evaluate(self, node: DecisionNode) -> DecisionNode:
        if self.should_do_rollout:
            print("Skipping evaluation for nodes during train phase or when using an orm\n")
            return node
        else:
            # for new nodes only
            print("Using prm for reward estimation during test phase\n")
            cur_state = node.state
            print(f"<State>\n\n{cur_state}\n")
            reward = self.rm.eval(self.q, cur_state)
            print(f"<Reward>\n{reward}\n")
            node.V = reward
            node.numVisits += 1
            node.sumReward += reward
            return node

    def get_reward(self, completions: list[str]) -> list[float]:
        if self.phase == 'train':
            return self.verify(completions)
        else:
            # for orm or prm based evaluation during test
            return self.rm.eval_batch(self.q, completions)

    def do_rollout(self, node: DecisionNode) -> tuple[DecisionNode, int, float]:
        if self.should_do_rollout:
            if node.isTerminal:
                print(f"<State>\n\n{node.state}\n")
                reward = self.get_reward([node.state])[0]
                print(f"<Reward>\n{reward}\n")
                return node, 1, reward
            else:
                completions = []
                for item in node.completions:
                    completion = next(iter(item))
                    completions.append(completion)
                rewards = self.get_reward(completions)

                for idx in range(len(node.completions)):
                    item = node.completions[idx]
                    completion = next(iter(item))
                    print(f"<Completion>\n\n{completion}\n")
                    reward = rewards[idx]
                    print(f"<Reward>\n{reward}\n")
                    item[completion] = reward

                n_visit = len(rewards)
                sum_reward = sum(rewards)
                return node, n_visit, sum_reward
        else:
            # this will be an intermediate node
            print("Estimating completion rewards with prm for verification and selection during test phase\n")
            completions = []
            for item in node.completions:
                completion = next(iter(item))
                completions.append(completion)
            rewards = self.get_reward(completions)

            for idx in range(len(node.completions)):
                item = node.completions[idx]
                completion = next(iter(item))
                print(f"<Completion>\n\n{completion}\n")
                reward = rewards[idx]
                print(f"<Reward>\n{reward}\n")
                item[completion] = reward

            print("Using child node values for rollout during test phase\n")
            n_visit = 0
            sum_reward = 0
            for child in node.children.values():
                n_visit += child.numVisits
                sum_reward += child.sumReward
            return node, n_visit, sum_reward

    def format_think_prompt(self, cur_state: str) -> str:
        return self.formatter.format_for_think(self.q, cur_state)

    def format_act_prompt(self, cur_state: str, thought: str) -> str:
        return self.formatter.format_for_act(self.q, cur_state, thought)

    def run(self) -> DecisionNode:
        mcts = MCTS(self, time_limit=self.time_limit, iteration_limit=self.iteration_limit, num_sample=self.num_sample,
                    num_decision=self.num_decision, exploration_constant=self.exploration_constant, eps=self.eps)
        root = mcts.search()
        return root
