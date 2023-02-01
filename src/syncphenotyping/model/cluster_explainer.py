from sklearn import tree


def fit_cluster(x, c, max_depth=5):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(x, c)

    return clf


def explain(clf, feature_names):
    exp = tree.export_text(clf, feature_names=feature_names)
    return exp
