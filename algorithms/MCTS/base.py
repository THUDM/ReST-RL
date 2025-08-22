import copy
import random


class DecisionNode:
    """
    Decision node class, labelled by a state
    """

    def __init__(self, parent=None, state: str = "", is_terminal: bool = False, thought: str = ""):
        self.parent = parent
        self.state = state
        self.isTerminal = is_terminal
        self.thought = thought  # the thought used to access this state
        self.numVisits = 0  # int
        self.V = 0.0  # node value: float
        self.sumReward = 0.0
        self.children = {}  # dict[str:DecisionNode] [action->child]
        self.isFullyExpanded = False  # expanded
        self.visible = True  # whether the node is selected or not
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1

        # we use this for saving completions generated from it
        self.completions = []  # list of map {completion: str -> reward: float}, items may not be unique

    def is_fully_expanded(self):
        return self.isFullyExpanded

    def set_invisible(self):
        self.visible = False

    def has_been_visited(self):
        return self.numVisits > 0
