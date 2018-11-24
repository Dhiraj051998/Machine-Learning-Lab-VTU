"""
Build an Artificial Neural Network by implementing the Back propagation
algorithm and test the same using appropriate data sets.
"""
from collections import Counter
from pprint import pprint
import pandas as pd
import math


# -p*log2*p
def entropy(probability):
    return sum([-prob * math.log(prob, 2) for prob in probability])


def entropy_of_list(a_list):
    cnt = Counter(x for x in a_list)
    num_instances = len(a_list)  # = 14
    probability = [x / num_instances for x in cnt.values()]
    return entropy(probability)  # Call Entropy


def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    df_split = df.groupby(split_attribute_name)
    nobs = len(df.index)
    df_agg_ent = df_split.agg({target_attribute_name: [entropy_of_list, lambda x: len(x) / nobs]})[
        target_attribute_name]
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    new_entropy = sum(df_agg_ent['Entropy'] * df_agg_ent['PropObservations'])
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy


def id3(df, target_attribute_name, attribute_names, default_class=None):
    cnt = Counter(x for x in df[target_attribute_name])  # class of YES /NO
    if len(cnt) == 1:
        return next(iter(cnt))
    elif df.empty or (not attribute_names):
        return default_class  # Return None for Empty Data Set
    else:
        default_class = max(cnt.keys())  # No of YES and NO Class
        gainz = [information_gain(df, attr, target_attribute_name) for
                 attr in attribute_names]
        index_of_max = gainz.index(max(gainz))  # Index of Best Attribute
        best_attr = attribute_names[index_of_max]

        tree = {best_attr: {}}
        remaining_attribute_names = [i for i in attribute_names if i !=
                                     best_attr]

        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target_attribute_name,
                          remaining_attribute_names, default_class)
            tree[best_attr][attr_val] = subtree
        return tree


def classify(instance, tree, default=None):
    attribute = next(iter(tree))  # Outlook/Humidity/Wind
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):  # this is a tree, delve deeper.
            return classify(instance, result)
        else:
            return result  # this is a label
    else:
        return default


dataset = pd.read_csv('data.csv')

total_entropy = entropy_of_list(dataset['PlayTennis'])
attribute_names = list(dataset.columns)
attribute_names.remove('PlayTennis')  # Remove the class attribute
tree = id3(dataset, 'PlayTennis', attribute_names)
pprint(tree)
attribute = next(iter(tree))
dataset['predicted'] = dataset.apply(classify, axis=1,
                                     args=(tree, 'No'))
print('\n Accuracy is:\n' + str(sum(dataset['PlayTennis'] == dataset['predicted']) / (1.0 * len(dataset.index))))
dataset[['PlayTennis', 'predicted']]
print(dataset)
training_data = dataset.iloc[1:-4]  # all but last four instances
test_data = dataset.iloc[-4:]  # just the last four
train_tree = id3(training_data, 'PlayTennis', attribute_names)
test_data['predicted2'] = test_data.apply(
    classify,  # <---- test_data source
    axis=1,
    args=(train_tree, 'Yes'))  # <---- train_data tree
print('\n\n Accuracy is : ' + str(
    sum(test_data['PlayTennis'] == test_data['predicted2']) /
    (1.0 * len(test_data.index))))
