import os
import pandas as pd

import argparse



def parse_args():
    
    parser = argparse.ArgumentParser(description="Results directory.")

    parser.add_argument("-d", "--directory",
                        help="Results directory.", 
                        required=False,
                        default="dataset", 
                        type=str)

    return parser.parse_args()


def main():

  args = parse_args()

  df = pd.read_csv(args.directory)
  
  df["f1 score"] = 2 * ((df["recall"] * df["precision"]) / (df["recall"] + df["precision"]))

  df = df.round(2)

  print(df.describe())
  df.plot()

  df = df.groupby("threshold").agg("mean")
  
  df.plot()
  

  df.describe()

  #download to that directory






if __name__ == "__main__":
    main()
