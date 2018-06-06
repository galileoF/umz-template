import pandas as pd
import graphviz
from sklearn import tree
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X, train_Y)


def show_plot():
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=train.columns[:-1],
                                    class_names=[str(x)
                                                 for x in [1, 2, 3, 4, 5, 6, 7]],
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.view()

# MAX LEAF NODES 6

clf = tree.DecisionTreeClassifier(max_leaf_nodes=6)
clf = clf.fit(train_X, train_Y)


print('MAX LEAF NODES 6')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

# CRITERION 'entropy', MIN SAMPLES SPLIT 3

clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=3)
clf = clf.fit(train_X, train_Y)


print('CRITERION entropy, MIN SAMPLES SPLIT 3')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))


# CRITERION 'entropy', MIN SAMPLES SPLIT 4

clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=4)
clf = clf.fit(train_X, train_Y)


print('CRITERION entropy, MIN SAMPLES SPLIT 4')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

# MIN WEIGHT FRACTION LEAF 0.1

clf = tree.DecisionTreeClassifier(min_weight_fraction_leaf=0.1)
clf = clf.fit(train_X, train_Y)


print('MIN WEIGHT FRACTION LEAF 0.1')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

# MIN WEIGHT FRACTION LEAF 0.3
clf = tree.DecisionTreeClassifier(min_weight_fraction_leaf=0.3)
clf = clf.fit(train_X, train_Y)


print('MIN WEIGHT FRACTION LEAF 0.3')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

# PRESORT True, MAX DEPTH 4

clf = tree.DecisionTreeClassifier(presort=True, max_depth=4)
clf = clf.fit(train_X, train_Y)
#show_plot()

print('PRESORT True, MAX DEPTH 4')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))
