import pandas as pd
from util.util import str_to_jnp_array


def main():
    df = pd.read_csv("logs/0/data.csv")
    print(df["p"].apply(str_to_jnp_array).apply(lambda arr: arr[:3]))


if __name__ == "__main__":
    main()