import privjail as pj
from privjail import pandas as ppd
import numpy as np
import pandas as pd

np.random.seed(0)

def calc_gain(df, attributes, split_attr, target_attr):
    s = 0
    for category, df_child in df.groupby(split_attr, keys=attributes[split_attr]["categories"]):
        s += df_child[target_attr].value_counts(sort=False, values=attributes[target_attr]["categories"]).max()
    return s

def noisy_count(df, epsilon):
    return max(0, pj.laplace_mechanism(df.shape[0], epsilon=epsilon))

def best_split(df, attributes, target_attr, epsilon):
    gains = {attr: calc_gain(df, attributes, attr, target_attr) for attr in attributes.keys() if attr != target_attr}
    return pj.exponential_mechanism(gains, epsilon)

def build_decision_tree(df, attributes, target_attr, max_depth, epsilon):
    t = max([len(attrprop["categories"]) for _, attrprop in attributes.items()])
    n_rows = noisy_count(df, epsilon)

    if max_depth == 0 or len(attributes) == 1 or n_rows / t / 2 < 2 ** (1/2) / epsilon:
        best_count = -1
        best_cls = None

        for cls, df_c in df.groupby(target_attr, keys=attributes[target_attr]["categories"]):
            count = noisy_count(df_c, epsilon)
            if best_count < count:
                best_count = count
                best_cls = cls

        return best_cls

    best_attr = best_split(df, attributes, target_attr, epsilon)

    attributes_child = attributes.copy()
    attributes_child.pop(best_attr)

    child_nodes = []
    for category, df_child in df.groupby(best_attr, keys=attributes[best_attr]["categories"]):
        child_node = build_decision_tree(df_child, attributes_child, target_attr, max_depth - 1, epsilon)
        child_nodes.append(dict(category=category, child=child_node))

    return dict(attr=best_attr, children=child_nodes)

def make_bins(ser, vmin, vmax, n_bins):
    delta = (vmax - vmin) / n_bins
    bins = [vmin + i * delta for i in range(n_bins + 1)]
    labels = [vmin + i * delta / 2 for i in range(n_bins)]

    if isinstance(ser, ppd.PrivSeries):
        return ppd.cut(ser, bins=bins, labels=labels, right=False, include_lowest=True), labels
    else:
        return pd.cut(ser, bins=bins, labels=labels, right=False, include_lowest=True), labels

def train(max_depth=5, n_bins=20, epsilon=1.0):
    attribute_props = {
        "age"             : dict(attrtype="continuous", domain=[17, 91]),
        "workclass"       : dict(attrtype="categorical", categories=['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc',
                                                                     'Self-emp-not-inc', 'State-gov', 'Without-pay']),
        "fnlwgt"          : dict(attrtype="continuous", domain=[13492, 1490401]),
        "education"       : dict(attrtype="categorical", categories=['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th',
                                                                     'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad',
                                                                     'Masters', 'Preschool', 'Prof-school', 'Some-college']),
        "educational-num" : dict(attrtype="continuous", domain=[1, 17]),
        "marital-status"  : dict(attrtype="categorical", categories=['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
                                                                     'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']),
        "occupation"      : dict(attrtype="categorical", categories=['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                                                                     'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct',
                                                                     'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv',
                                                                     'Sales', 'Tech-support', 'Transport-moving']),
        "relationship"    : dict(attrtype="categorical", categories=['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried',
                                                                     'Wife']),
        "race"            : dict(attrtype="categorical", categories=['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']),
        "gender"          : dict(attrtype="categorical", categories=['Female', 'Male']),
        "capital-gain"    : dict(attrtype="continuous", domain=[0, 100000]),
        "capital-loss"    : dict(attrtype="continuous", domain=[0, 4357]),
        "hours-per-week"  : dict(attrtype="continuous", domain=[1, 100]),
        "native-country"  : dict(attrtype="categorical", categories=['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
                                                                     'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece',
                                                                     'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong',
                                                                     'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan',
                                                                     'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru',
                                                                     'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South',
                                                                     'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam',
                                                                     'Yugoslavia']),
        "income"          : dict(attrtype="categorical", categories=['<=50K', '>50K']),
    }

    df_train = ppd.read_csv("data/adult_train.csv")
    df_train = df_train.replace("?", np.nan).dropna()

    for attr, attrprop in attribute_props.items():
        if attrprop["attrtype"] == "continuous":
            [vmin, vmax] = attrprop["domain"]
            df_train[attr], labels = make_bins(df_train[attr], vmin, vmax, n_bins)
            attrprop["categories"] = labels

    eps = epsilon / 2 / (max_depth + 1)
    dtree = build_decision_tree(df_train, attribute_props, "income", max_depth, eps)

    print("Decision tree constructed.")

    return dict(n_bins=n_bins, attributes=attribute_props, tree=dtree)

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

    for attr, attrprop in dtree["attributes"].items():
        if attrprop["attrtype"] == "continuous":
            [vmin, vmax] = attrprop["domain"]
            df_test[attr], _ = make_bins(df_test[attr], vmin, vmax, n_bins)

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
