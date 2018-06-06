import pandas as pd
import graphviz
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']


#def show_plot():
 #   dot_data = tree.export_graphviz(clf, out_file=None,
  #                                  feature_names=train.columns[:-1],
   #                                 class_names=[str(x)
    #                                             for x in [1, 2, 3, 4, 5, 6, 7]],
     #                               filled=True, rounded=True)
    #graph = graphviz.Source(dot_data)
    #graph.view()

# n_neighbors=5 (default)

clf = KNeighborsClassifier()
clf = clf.fit(train_X, train_Y)


print('n_neighbors=5')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

# n_neighbors=6

clf = KNeighborsClassifier(n_neighbors=6)
clf = clf.fit(train_X, train_Y)


print('n_neighbors=6')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))


# n_neighbors=8

clf = KNeighborsClassifier(n_neighbors=8)
clf = clf.fit(train_X, train_Y)


print('n_neighbors=8')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

# n_neighbors=10

clf = KNeighborsClassifier(n_neighbors=10)
clf = clf.fit(train_X, train_Y)


print('n_neighbors=10')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

# n_neighbors=2
clf = KNeighborsClassifier(n_neighbors=2)
clf = clf.fit(train_X, train_Y)


print('n_neighbors=2')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

# n_neighbors=4
clf = KNeighborsClassifier(n_neighbors=4)
clf = clf.fit(train_X, train_Y)


print('n_neighbors=4')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

# n_neighbors=100
clf = KNeighborsClassifier(n_neighbors=100)
clf = clf.fit(train_X, train_Y)


print('n_neighbors=100')
print('TRAIN SET')
#print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
#print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))