import time
import math
import random
from algorithms.MCTS.base import DecisionNode


class MCTS:
    def __init__(self, mcts_task, time_limit=None, iteration_limit=None, num_sample=5, num_decision=3, exploration_constant=0.2, eps=0.1):
        """
        :param mcts_task: The task object to complete
        :param time_limit: Setting time limit(s)
        :param iteration_limit: Setting maximum iteration, prior to time limit
        :param num_sample: Number of samples generated for rollout and expansion
        :param num_decision: Number of samples selected for tree expansion
        :param exploration_constant: Used for UCT
        :param eps: Used for UCT
        """
        self.mcts_task = mcts_task
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.num_sample = num_sample
        self.num_decision = num_decision
        self.exploration_constant = exploration_constant
        self.eps = eps
        self.low = -10000
        if iteration_limit is not None:
            self.limit_type = 'iteration'
        else:
            assert time_limit is not None, 'Argument time_limit or iteration_limit must be set'
            self.limit_type = 'time'

    def search(self):
        root = DecisionNode(state=self.mcts_task.initial_state)

        if self.limit_type == 'time':
            timeLimit = time.time() + self.time_limit
            time_start = time.time()
            print(f'Starting MCTS search with time limit: {self.time_limit}s\n')
            print('-' * 70, 'Search Task', '-' * 70)
            while time.time() < timeLimit:
                print(f'<Starting new iteration with used time: {time.time() - time_start}s>\n')
                flag, node, root = self.executeRound(root)
        else:
            print(f'Starting MCTS search with iteration limit: {self.iteration_limit} iters\n')
            print('-' * 70, 'Search Task', '-' * 70)
            for i in range(self.iteration_limit):
                print(f'<Starting new iteration with completed iterations: {i}>\n')
                flag, node, root = self.executeRound(root)

        print('-' * 70, 'End of Search Task', '-' * 70)
        return root

    def executeRound(self, root: DecisionNode):
        """
        execute a selection-expansion-simulation-backpropagation round
        """
        print('-' * 50, 'Iteration', '-' * 50)
        print('*' * 30, 'Selection', '*' * 30)
        flag, node = self.selectNode(root)

        print('*' * 30, 'Expansion', '*' * 30)
        if flag:
            print('Skipping expansion phase...\n')
        else:
            node = self.expand(node)

        print('*' * 30, 'Simulation', '*' * 30)
        if flag:
            if node.has_been_visited():
                print(f"This terminal node has been visited {node.numVisits} times. No simulation or evaluation is needed\n")
                new_n_visits = 1
                new_sum_reward = node.V
            else:
                print('No simulation needed for terminal nodes. Evaluating node value instead\n')
                node, new_n_visits, new_sum_reward = self.rollout(node)
        else:
            node, new_n_visits, new_sum_reward = self.rollout(node)

        print('*' * 30, 'Backpropagation', '*' * 30)
        self.back_propagate(node, new_n_visits, new_sum_reward)

        print('-' * 50, 'End of Iteration', '-' * 50)
        return flag, node, root

    def selectNode(self, node: DecisionNode):
        while node.isFullyExpanded:
            node = self.getBestChild(node)

        print(f"<Current State>\n\n{node.state}\n")
        if node.isTerminal:
            print("Notice: This is a terminal state\n")
            return True, node
        else:
            return False, node

    def expand(self, node: DecisionNode):
        samples, thoughts = self.mcts_task.do_sample(node, self.num_sample)
        actions = []
        for i in range(len(samples)):
            sample = samples[i]
            sample_ = sample.rstrip()
            if self.mcts_task.use_thought:
                print(f"<Thought {i + 1}>\n\n{thoughts[i]}\n")
            print(f"<Sample {i + 1}>\n\n{sample_}\n")
            if sample_.count('\n') < 1:
                # terminal child
                action = sample_ + '\n'
                is_terminal = True
                node.completions.append({node.state + action: self.low})
            else:
                # normal leaf child
                action = sample_.split('\n')[0] + '\n'
                is_terminal = False
                node.completions.append({node.state + sample_ + '\n': self.low})

            if action not in actions:
                actions.append(action)
                if self.mcts_task.use_thought:
                    new_node = DecisionNode(parent=node, state=node.state + action, is_terminal=is_terminal,
                                            thought=thoughts[i])
                else:
                    new_node = DecisionNode(parent=node, state=node.state + action, is_terminal=is_terminal)

                print("Evaluating the new node...\n")
                new_node = self.mcts_task.evaluate(new_node)
                node.children[action] = new_node
            else:
                if not self.mcts_task.should_do_rollout:
                    node.children[action].numVisits += 1
                    node.children[action].sumReward += node.children[action].V

        if len(node.children.values()) > self.num_decision:
            # set some nodes invisible
            sorted_actions = sorted(node.children.keys(), key=lambda x: node.children[x].V, reverse=True)
            for i in range(self.num_decision, len(sorted_actions)):
                node.children[sorted_actions[i]].set_invisible()

        node.isFullyExpanded = True
        return node

    def rollout(self, node: DecisionNode):
        node, n_visits, sum_value = self.mcts_task.do_rollout(node)
        return node, n_visits, sum_value

    @staticmethod
    def back_propagate(node: DecisionNode, new_n_visits: int, new_sum_reward: float):
        while node is not None:
            node.sumReward += new_sum_reward
            node.numVisits += new_n_visits
            node.V = node.sumReward / node.numVisits
            node = node.parent

    def getBestChild(self, node: DecisionNode):
        bestNodes = []
        bestValue = self.low
        for child in node.children.values():
            if not child.visible:
                continue
            nodeValue = self.UCT(child)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def UCT(self, node: DecisionNode):
        return node.V + self.exploration_constant * math.sqrt((1 + math.log(node.parent.numVisits)) / (self.eps + node.numVisits))
