import privjail as pj
from privjail import pandas as ppd
import numpy as np
import pandas as pd

np.random.seed(0)

def calc_gain(df, split_attr, target_attr):
    s = 0
    for category, df_child in df.groupby(split_attr):
        s += df_child[target_attr].value_counts(sort=False).max()
    return s

def noisy_count(df, epsilon):
    return max(0, pj.laplace_mechanism(df.shape[0], epsilon=epsilon))

def best_split(df, attributes, target_attr, epsilon):
    gains = {attr: calc_gain(df, attr, target_attr) for attr in attributes}
    return pj.exponential_mechanism(gains, epsilon)

def build_decision_tree(df, attributes, target_attr, max_depth, epsilon):
    t = max([len(df.schema[attr]["categories"]) for attr in attributes])
    n_classes = len(df.schema[target_attr]["categories"])
    n_rows = noisy_count(df, epsilon)

    if len(attributes) == 0 or max_depth == 0 or n_rows / (t * n_classes) < (2 ** 0.5) / epsilon:
        class_counts = {c: noisy_count(df_c, epsilon) for c, df_c in df.groupby(target_attr)}
        return max(class_counts, key=class_counts.get)

    best_attr = best_split(df, attributes, target_attr, epsilon)

    child_nodes = []
    for category, df_child in df.groupby(best_attr):
        child_node = build_decision_tree(df_child, [a for a in attributes if a != best_attr], target_attr, max_depth - 1, epsilon)
        child_nodes.append(dict(category=category, child=child_node))

    return dict(attr=best_attr, children=child_nodes)

def make_bins(ser, vmin, vmax, n_bins):
    delta = (vmax - vmin) / n_bins
    bins = [vmin + i * delta for i in range(n_bins + 1)]
    labels = [vmin + i * delta / 2 for i in range(n_bins)]

    if isinstance(ser, ppd.PrivSeries):
        return ppd.cut(ser, bins=bins, labels=labels, right=False, include_lowest=True)
    else:
        return pd.cut(ser, bins=bins, labels=labels, right=False, include_lowest=True)

def train(max_depth=5, n_bins=20, epsilon=1.0):
    df_train = ppd.read_csv("data/adult_train.csv", "schema/adult.json")
    df_train = df_train.dropna()

    original_schema = df_train.schema.copy()

    for attr, attrprop in df_train.schema.items():
        if attrprop["type"] == "int64":
            [vmin, vmax] = attrprop["range"]
            df_train[attr] = make_bins(df_train[attr], vmin, vmax, n_bins)

    target_attr = "income"
    attributes = [attr for attr in df_train.columns if attr != target_attr]

    eps = epsilon / (2 * (max_depth + 1))
    dtree = build_decision_tree(df_train, attributes, target_attr, max_depth, eps)

    print("Decision tree constructed.")

    return dict(n_bins=n_bins, schema=original_schema, tree=dtree)

def classify(dtree, row):
    if type(dtree) is str:
        return dtree

    for child_node in dtree["children"]:
        if child_node["category"] == row[dtree["attr"]]:
            return classify(child_node["child"], row)

    raise Exception

def test(dtree):
    df_test = pd.read_csv("data/adult_test.csv")
    df_test = df_test.replace("?", np.nan).dropna()

    n_bins = dtree["n_bins"]

    for attr, attrprop in dtree["schema"].items():
        if attrprop["type"] == "int64":
            [vmin, vmax] = attrprop["range"]
            df_test[attr] = make_bins(df_test[attr], vmin, vmax, n_bins)

    correct_count = 0
    for i, row in df_test.iterrows():
        ans = row["income"]
        result = classify(dtree["tree"], row.drop("income"))
        correct_count += (ans == result)

    print(f"Accuracy: {correct_count / len(df_test)} ({correct_count} / {len(df_test)})")

def tree_stats(dtree, depth=1):
    if type(dtree) is str:
        return (1, 1, depth)

    node_count = 1
    leaf_count = 0
    max_depth = depth

    for child_node in dtree["children"]:
        nc, lc, d = tree_stats(child_node["child"], depth + 1)
        node_count += nc
        leaf_count += lc
        max_depth = max(d, max_depth)

    return node_count, leaf_count, max_depth

dtree = train()

print(pj.current_privacy_budget())

# import pprint
# pprint.pprint(dtree)

node_count, leaf_count, depth = tree_stats(dtree["tree"])
print(f"node count: {node_count}, leaf count: {leaf_count}, depth: {depth}")

test(dtree)
