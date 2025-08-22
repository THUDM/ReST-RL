from algorithms.MCTS.base import DecisionNode


def get_all_completions(root: DecisionNode) -> list[dict]:
    all_completions = root.completions
    if root.isFullyExpanded:
        for child in root.children.values():
            all_completions.extend(get_all_completions(child))
    return all_completions


def get_process_rewards(root: DecisionNode) -> list[dict[str, list]]:
    all_rewards = [{root.state: [root.V, root.isTerminal, root.isFullyExpanded, root.visible]}]
    if root.isFullyExpanded:
        for child in root.children.values():
            all_rewards.extend(get_process_rewards(child))
    return all_rewards


def record_tree(root: DecisionNode) -> list[dict]:
    """
    record tree for further usage
    :param root: search tree root
    :return: list of tree nodes recorded in preorder traversal
    """
    all_nodes = []
    node_object = {'state': root.state, 'is_terminal': root.isTerminal, 'thought': root.thought,
                   'num_visits': root.numVisits, 'value': root.V, 'sum_reward': root.sumReward,
                   'is_fully_expanded': root.isFullyExpanded, 'visible': root.visible, 'depth': root.depth,
                   'completions': root.completions}
    all_nodes.append(node_object)

    if root.isFullyExpanded:
        for child in root.children.values():
            all_nodes.extend(record_tree(child))
    return all_nodes


def rebuild_tree(tree_nodes: list[dict]) -> DecisionNode | None:
    """
    reconstruct the search tree using the list of nodes
    :param tree_nodes: the list of tree nodes recorded in preorder traversal
    :return: tree root
    """
    # check input validity
    if len(tree_nodes) < 1:
        return None

    # build root node
    root_info = tree_nodes[0]
    root = DecisionNode(state=root_info['state'], is_terminal=root_info['is_terminal'], thought=root_info['thought'])
    root.isFullyExpanded = root_info['is_fully_expanded']
    root.numVisits = root_info['num_visits']
    root.V = root_info['value']
    root.sumReward = root_info['sum_reward']
    root.completions = root_info['completions']
    root.visible = root_info['visible']

    # build children
    cur_node = root
    cur_idx = 1
    while cur_idx < len(tree_nodes):
        cur_info = tree_nodes[cur_idx]
        cur_depth = cur_info['depth']
        assert cur_node.depth + 2 > cur_depth, "Failed to build tree, please check the provided node list\n"
        while cur_node.depth + 1 > cur_depth:
            cur_node = cur_node.parent

        node = DecisionNode(parent=cur_node, state=cur_info['state'], is_terminal=cur_info['is_terminal'], thought=cur_info['thought'])
        node.isFullyExpanded = cur_info['is_fully_expanded']
        node.numVisits = cur_info['num_visits']
        node.V = cur_info['value']
        node.sumReward = cur_info['sum_reward']
        node.completions = cur_info['completions']
        node.visible = cur_info['visible']
        assert node.state.startswith(cur_node.state), "Failed to build tree, please check the provided node list\n"
        action = node.state[len(cur_node.state):]
        cur_node.children[action] = node
        cur_idx += 1
        cur_node = node

    return root
