import csv
import numpy as np


def get_data(path):
    file = open(path)
    csvreader = csv.reader(file)
    rows = []
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)
    file.close()
    data = np.array(rows)
    data = np.delete(data, [0, 3, 8, 9, 10, 11], 1)
    data = data[:, [1, 2, 3, 4, 5, 0]]
    for i in range(len(data[:, 2])):
        status = ''
        age = data[i][2]
        if age != '':
            age = float(age)
            if age < 18:
                status = 'child'
            elif age >= 18 and age < 60:
                status = 'adult'
            elif age >=60:
                status = 'elderly'
        else:
            status = 'adult'
        data[i][2] = status
    return data


def get_classes(y_data):
    classes = []
    for i in y_data:
        if i not in classes:
            classes.append(i)
    return classes


def classes_probs(data):
    y_data = data[:, -1]
    classes = get_classes(y_data)
    probs = {}
    for c in classes:
        counter = 0
        for j in y_data:
            if j == c:
                counter += 1
        probs[c] = counter / len(y_data)
    return probs


def conditional_probs(data):
    y_data = data[:, -1]
    classes = get_classes(y_data)
    cond_probs = {}
    for c in classes:
        cond_probs[c] = {}
        for feature in range(data.shape[1] - 1):
            unique_vals = []
            feature_val = data[data[:, -1] == c][:, feature]
            for val in feature_val:
                if val not in unique_vals:
                    unique_vals.append(val)
            cond_probs[c][feature] = {}
            for value in unique_vals:
                counter = 0
                for i in feature_val:
                    if (i == value):
                        counter += 1
                cond_probs[c][feature][value] = counter / len(feature_val)
    return cond_probs


def predict(X, class_probs, cond_probs):
    max_prob = 0
    best_class = None
    for c in class_probs.keys():
        prob = class_probs[c]
        for feature in range(len(X)):
            if X[feature] in cond_probs[c][feature]:
                prob *= cond_probs[c][feature][X[feature]]
        prob /= get_probability_certain_args(train, X)
        if prob > max_prob:
            max_prob = prob
            best_class = c
    return best_class


def get_probability_certain_args(train, row):
    indexes = []
    for i in range(len(row)):
        if row[i] != '':
            indexes.append(i)
    counter = 0
    for x in train:
        isCertain = True
        for ind in indexes:
            if x[ind] != row[ind]:
                isCertain = False
        if isCertain:
            counter += 1
    return counter / train.shape[0]



def get_probability_survive(train, cond_probs, data):
    valid_data = {}
    for i in range(len(data)):
        if data[i] != '':
            valid_data[i] = data[i]
    survive_prob = class_probs['1']
    for feature in valid_data.keys():
        if valid_data[feature] in cond_probs['1'][feature]:
            survive_prob *= cond_probs['1'][feature][valid_data[feature]]
    survive_prob /= get_probability_certain_args(train, data)
    return survive_prob


train = get_data('train.csv')


class_probs = classes_probs(train)
cond_probs = conditional_probs(train)


# test = get_data('train.csv')
# y = test[:, -1]
# preds = []
# for row in test:
#     pred = predict(row[0:4], class_probs, cond_probs)
#     preds.append(pred)
# counter = 0
# for i in range(len(y)):
#     if preds[i] == y[i]:
#         counter += 1
# accuracy = counter / len(y)
# print("Accuracy: ", accuracy)

woman = ['', 'female', 'adult', '', '']
man = ['', 'male', 'adult', '', '']
girl = ['', 'female', 'child', '', '']
boy = ['', 'male', 'child', '', '']
prob = get_probability_survive(train, cond_probs, woman)
print('Probability woman: ', prob)
prob = get_probability_survive(train, cond_probs, man)
print('Probability man: ', prob)
prob = get_probability_survive(train, cond_probs, boy)
print('Probability boy: ', prob)
prob = get_probability_survive(train, cond_probs, girl)
print('Probability girl: ', prob)
