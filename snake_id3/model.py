import pickle
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import pytorch

"""Implement your model, training code and other utilities here. Please note, you can generate multiple
pickled data files and merge them into a single data list."""


def game_state_to_data_sample(state):
    head_pos = state['snake_body'][-1]
    dir = state['snake_direction']
    snake_len = len(state['snake_body'])
    barrier_up = int(head_pos[1] / 30)
    barrier_down = int((270 - head_pos[1]) / 30)
    barrier_left = int(head_pos[0] / 30)
    barrier_right = int((270 - head_pos[0]) / 30)
    food_pos = state['food']
    food_down = food_up = food_left = food_right = 0
    if head_pos[1] > food_pos[1]:
        food_up = 1
    if head_pos[1] < food_pos[1]:
        food_down = 1
    if head_pos[0] > food_pos[0]:
        food_left = 1
    if head_pos[0] < food_pos[0]:
        food_right = 1
    data = np.array([dir.value, snake_len, barrier_up, barrier_down,
                    barrier_left, barrier_right, food_up,
                    food_down, food_left, food_right])
    # data = np.array([food_up, food_down, food_left, food_right])
    return data


def get_data(file_name):
    data = np.empty((0, 11), int)
    with open(file_name, 'rb') as f:
        data_file = pickle.load(f)
    for game_state in data_file['data'][1:]:
        x, y = game_state
        x = game_state_to_data_sample(x)
        y = np.array([y.value])
        arr = np.concatenate(([x, y]))
        data = np.vstack((data, arr))
    return data


def get_all_data():
    data = np.empty((0, 11), int)
    x = get_data(f"data/2022-12-03_16:00:02.pickle")
    data = np.vstack((data, x))
    x = get_data(f"data/2022-12-03_16:05:01.pickle")
    data = np.vstack((data, x))
    x = get_data(f"data/2022-12-03_16:10:19.pickle")
    data = np.vstack((data, x))
    x = get_data(f"data/2022-12-03_17:04:14.pickle")
    data = np.vstack((data, x))
    x = get_data(f"data/2022-12-03_17:09:06.pickle")
    data = np.vstack((data, x))
    x = get_data(f"data/2022-12-03_17:15:42.pickle")
    data = np.vstack((data, x))
    x = get_data(f"data/2022-12-03_19:06:32.pickle")
    data = np.vstack((data, x))
    return data


def get_most_often(sets):
    up = 0
    right = 0
    down = 0
    left = 0
    for i in range(len(sets)):
        if sets[i][-1] == 0:
            up += 1
        if sets[i][-1] == 1:
            right += 1
        if sets[i][-1] == 2:
            down += 1
        if sets[i][-1] == 3:
            left += 1
    dirs = [up, right, down, left]
    most_occurances = max(dirs)
    if most_occurances:
        often_item = dirs.index(most_occurances)
    return often_item


class Node():
    def __init__(self) -> None:
        self.value = None
        self.children = []
        self.attribute = None
        self.prediction = None


def get_entropy(sets):
    up = 0
    right = 0
    down = 0
    left = 0
    for i in range(len(sets)):
        if sets[i][-1] == 0:
            up += 1
        if sets[i][-1] == 1:
            right += 1
        if sets[i][-1] == 2:
            down += 1
        if sets[i][-1] == 3:
            left += 1
    dirs = [up, right, down, right]
    entropy = 0.0
    for dir in dirs:
        x = dir / len(sets)
        if x != 0:
            entropy += -x * np.log(x)
    return entropy


def best_arg_inf_gain(attributes, sets):
    entropy = get_entropy(sets)
    best_arg = None
    best_gain = 0.0
    for attr in attributes:
        attr_entropy = 0.0
        vals_lst = []
        for i in range(len(sets)):
            if sets[i][attr] not in vals_lst:
                vals_lst.append(sets[i][attr])
        for val in vals_lst:
            val_data = sets[(sets[:, attr] == val), :]
            x = len(val_data) / len(sets)
            attr_entropy += x * get_entropy(val_data)
        gain = entropy - attr_entropy
        if gain > best_gain:
            best_arg = attr
            best_gain = gain
    return best_arg


def decision_tree_id3(attributes, sets, max_depth=100, val=0):
    root = Node()
    root.value = val

    res = sets[0][-1]
    for row in sets:
        if res != row[-1]:
            break
    if res == row[-1]:
        root.prediction = res
        return root

    best_attr = best_arg_inf_gain(attributes, sets)
    item = get_most_often(sets)
    root.prediction = item

    if len(attributes) == 0 or best_attr is None:
        return root

    if max_depth == 0:
        return root

    max_depth -= 1

    root.attribute = best_attr
    attributes.remove(best_attr)

    vals_lst = []
    for i in range(len(sets)):
        if sets[i][best_attr] not in vals_lst:
            vals_lst.append(sets[i][best_attr])
    for value in sorted(vals_lst):
        new_data = sets[(sets[:, best_attr] == value), :]
        new_attrs = copy.deepcopy(attributes)
        new_tree = decision_tree_id3(new_attrs, new_data, max_depth, value)
        root.children.append(new_tree)
    return root


def use_tree(root: Node, state):
    for child in root.children:
        if child.value == state[root.attribute]:
            if len(child.children) == 0:
                return child.prediction
            else:
                return use_tree(child, state)
    return root.prediction


def test_accuracy(train_data, test_data, max_depth=100):
    # dataset = get_all_data()
    # train_data, test_data = train_test_split(dataset, test_size=0.2, train_size=0.8)
    attrs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    model = decision_tree_id3(attrs, train_data, max_depth)
    Y_train = train_data[:, -1]
    train_output = []
    for row in train_data:
        train_output.append(use_tree(model, row[:-1]))
    train_acc = accuracy_score(Y_train, np.array(train_output))
    Y_test = test_data[:, -1]
    test_output = []
    for row in test_data:
        out = use_tree(model, row[:-1])
        test_output.append(out)
    test_acc = accuracy_score(Y_test, np.array(test_output))
    print("Depth: ", max_depth)
    print("Train: ", train_acc)
    print("Test: ", test_acc)


# attrs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# data = get_all_data()
# root = decision_tree_id3(attrs, data)

# dataset = get_all_data()
# train_data, test_data = train_test_split(dataset, test_size=0.2, train_size=0.8)
# test_accuracy(train_data, test_data, 1)
# test_accuracy(train_data, test_data, 2)
# test_accuracy(train_data, test_data, 3)
# test_accuracy(train_data, test_data, 4)
# test_accuracy(train_data, test_data)
