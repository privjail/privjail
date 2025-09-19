# Copyright 2025 TOYOTA MOTOR CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
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

def noisy_count(df, eps):
    return max(0, pj.laplace_mechanism(df.shape[0], eps=eps))

def best_split(df, attributes, target_attr, eps):
    gains = [calc_gain(df, attr, target_attr) for attr in attributes]
    return attributes[pj.argmax(gains, eps=eps)]

def build_decision_tree(df, attributes, target_attr, max_depth, eps):
    t = max([len(df.domains[attr].categories) for attr in attributes])
    n_classes = len(df.domains[target_attr].categories)
    n_rows = noisy_count(df, eps)

    if len(attributes) == 0 or max_depth == 0 or n_rows / (t * n_classes) < (2 ** 0.5) / eps:
        class_counts = {c: noisy_count(df_c, eps) for c, df_c in df.groupby(target_attr)}
        return max(class_counts, key=class_counts.get)

    best_attr = best_split(df, attributes, target_attr, eps)

    child_nodes = []
    for category, df_child in df.groupby(best_attr):
        child_node = build_decision_tree(df_child, [a for a in attributes if a != best_attr], target_attr, max_depth - 1, eps)
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

def train(max_depth=5, n_bins=20, eps=1.0):
    df_train = ppd.read_csv("data/adult_train.csv", "schema/adult.json")
    df_train = df_train.dropna()

    original_domains = df_train.domains.copy()

    for attr, domain in df_train.domains.items():
        if domain.dtype == "int64":
            vmin, vmax = domain.range
            df_train[attr] = make_bins(df_train[attr], vmin, vmax, n_bins)

    target_attr = "income"
    attributes = [attr for attr in df_train.columns if attr != target_attr]

    eps_each = eps / (2 * (max_depth + 1))
    dtree = build_decision_tree(df_train, attributes, target_attr, max_depth, eps_each)

    print("Decision tree constructed.")

    return dict(n_bins=n_bins, domains=original_domains, tree=dtree)

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

    for attr, domain in dtree["domains"].items():
        if domain.dtype == "int64":
            vmin, vmax = domain.range
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

def main():
    dtree = train()

    print(pj.budgets_spent())

    # import pprint
    # pprint.pprint(dtree)

    node_count, leaf_count, depth = tree_stats(dtree["tree"])
    print(f"node count: {node_count}, leaf count: {leaf_count}, depth: {depth}")

    test(dtree)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        mode = "local"
    else:
        mode = sys.argv[1]

    if mode in ("server", "client"):
        if len(sys.argv) < 4:
            raise ValueError(f"Usage: {sys.argv[0]} {mode} <host> <port>")

        host = sys.argv[2]
        port = int(sys.argv[3])

    if mode == "local":
        main()

    elif mode == "server":
        # print(pj.proto_file_content())
        pj.serve(port)

    elif mode == "client":
        pj.connect(host, port)
        main()

    else:
        raise ValueError(f"Usage: {sys.argv[0]} [local|server|client] [host] [port]")
