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

def main():
    df = ppd.read_csv("data/adult_train.csv", "schema/adult.json").dropna()

    n = 1000

    print()
    print("Income stats of younger age groups:")
    print(df.sort_values("age").head(n)["income"].value_counts(sort=False).reveal(eps=1.0))

    print()
    print("Income stats of older age groups:")
    print(df.sort_values("age").tail(n)["income"].value_counts(sort=False).reveal(eps=1.0))

    print()
    print("Average age by income:")
    for income, df_ in df.groupby("income"):
        mean_age = df_["age"].mean(eps=1.0)
        print(f"{income}: {mean_age}")

    print()
    print("Average age and hours-per-week by income:")
    print(df.groupby("income")[["age", "hours-per-week"]].mean(eps=1.0))

    print()
    print("Crosstab between gender and income")
    print(ppd.crosstab(df["gender"], df["income"]).reveal(eps=1.0))

    print()
    print("Sum of capital-gain and capital-loss of PhDs aged under 40")
    print(df[(df["education"] == "Doctorate") & (df["age"] < 40)][["capital-gain", "capital-loss"]].sum().reveal(eps=1.0))

    print()
    print("Consumed Privacy Budget:")
    print(pj.consumed_privacy_budget())

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
