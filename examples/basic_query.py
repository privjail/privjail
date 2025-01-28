import sys
import privjail as pj
from privjail import pandas as ppd

def main():
    df = ppd.read_csv("data/adult_train.csv", "schema/adult.json").dropna()

    n = 1000

    print()
    print("Income stats of younger age groups:")
    print(pj.laplace_mechanism(df.sort_values("age").head(n)["income"].value_counts(sort=False), eps=1.0))

    print()
    print("Income stats of older age groups:")
    print(pj.laplace_mechanism(df.sort_values("age").tail(n)["income"].value_counts(sort=False), eps=1.0))

    print()
    print("Average age by income:")
    for income, df_ in df.groupby("income"):
        mean_age = df_["age"].mean(eps=1.0)
        print(f"{income}: {mean_age}")

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
