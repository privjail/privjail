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
import argparse
import privjail as pj
import privjail.pandas as ppd

def main(args):
    df = ppd.read_csv("data/adult_train.csv", "schema/adult.json").dropna()

    n = 1000
    eps = args.eps

    print()
    print("Income stats of younger age groups:")
    print(df.sort_values("age").head(n)["income"].value_counts(sort=False).reveal(eps=eps))

    print()
    print("Income stats of older age groups:")
    print(df.sort_values("age").tail(n)["income"].value_counts(sort=False).reveal(eps=eps))

    print()
    print("Average age by income:")
    for income, df_ in df.groupby("income"):
        mean_age = df_["age"].mean(eps=eps)
        print(f"{income}: {mean_age}")

    print()
    print("Average age and hours-per-week by income:")
    print(df.groupby("income")[["age", "hours-per-week"]].mean(eps=eps))

    print()
    print("Crosstab between gender and income")
    print(ppd.crosstab(df["gender"], df["income"]).reveal(eps=eps))

    print()
    print("Sum of capital-gain and capital-loss of PhDs aged under 40")
    print(df[(df["education"] == "Doctorate") & (df["age"] < 40)][["capital-gain", "capital-loss"]].sum().reveal(eps=eps))

    print()
    print("Consumed Privacy Budget:")
    print(pj.budgets_spent())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--eps", type=float, default=1.0)
    args = parser.parse_args()

    if args.remote:
        pj.connect(args.host, args.port)

    main(args)
