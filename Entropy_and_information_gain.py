import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_entropy(data):
    target = data.iloc[:, -1]
    values = target.unique()
    entropy = 0
    for value in values:
        fraction = target.value_counts()[value] / len(target)
        entropy += -fraction * np.log2(fraction)
    return entropy

def calculate_information_gain(data, feature):
    entropy_total = calculate_entropy(data)
    values = data[feature].unique()
    entropy_feature = 0
    for value in values:
        subset = data[data[feature] == value]
        entropy = calculate_entropy(subset)
        fraction = len(subset) / len(data)
        entropy_feature += fraction * entropy
    information_gain = entropy_total - entropy_feature
    # print(f'Entropy of {count} is',entropy_total)
    # print(information_gain,feature)
    print(f'entropy = {entropy_total}, information_gain = {information_gain}, attribute =  {feature}')
    return information_gain

def select_best_feature(data):
    features = data.columns[:-1]
    information_gains = []
    # count=1
    for feature in features:
        information_gain = calculate_information_gain(data, feature)
        information_gains.append(information_gain)
    best_feature_index = np.argmax(information_gains)
    best_feature = features[best_feature_index]
    print(f'\nBest feature is {best_feature}\n')
    return best_feature

def build_decision_tree(data):
    target = data.iloc[:, -1]
    if len(target.unique()) == 1:
        return target.unique()[0]
    elif len(data.columns) == 1:
        return target.value_counts().idxmax()
    else:
        best_feature = select_best_feature(data)
        values = data[best_feature].unique()
        for value in values:
            subset = data[data[best_feature] == value].drop(best_feature, axis=1)
            subtree = build_decision_tree(subset)

data = pd.read_csv('/content/sample_data/eggs.csv')
data2 = pd.read_csv('/content/sample_data/newdataset.csv')
print("First Dataset\n")
decision_tree = build_decision_tree(data)
print("\nSecond dataset\n")
decision_tree2 = build_decision_tree(data2)


# output
# First Dataset

# entropy = 0.9182958340544896, information_gain = 0.109170338675599, attribute =  type
# entropy = 0.9182958340544896, information_gain = 0.4591479170272448, attribute =  feather
# entropy = 0.9182958340544896, information_gain = 0.31668908831502096, attribute =  fur
# entropy = 0.9182958340544896, information_gain = 0.044110417748401076, attribute =  swims

# Best feature is feather

# entropy = 0.9182958340544896, information_gain = 0.9182958340544896, attribute =  type
# entropy = 0.9182958340544896, information_gain = 0.2516291673878229, attribute =  fur
# entropy = 0.9182958340544896, information_gain = 0.2516291673878229, attribute =  swims

# Best feature is type


# Second dataset

# entropy = 0.9402859586706311, information_gain = 0.24674981977443933, attribute =  outlook
# entropy = 0.9402859586706311, information_gain = 0.02922256565895487, attribute =  temp
# entropy = 0.9402859586706311, information_gain = 0.09027634939276519, attribute =  hum
# entropy = 0.9402859586706311, information_gain = 0.04812703040826949, attribute =  wind

# Best feature is outlook

# entropy = 0.9709505944546686, information_gain = 0.5709505944546686, attribute =  temp
# entropy = 0.9709505944546686, information_gain = 0.9709505944546686, attribute =  hum
# entropy = 0.9709505944546686, information_gain = 0.01997309402197489, attribute =  wind

# Best feature is hum

# entropy = 0.9709505944546686, information_gain = 0.01997309402197489, attribute =  temp
# entropy = 0.9709505944546686, information_gain = 0.01997309402197489, attribute =  hum
# entropy = 0.9709505944546686, information_gain = 0.9709505944546686, attribute =  wind

# Best feature is wind
