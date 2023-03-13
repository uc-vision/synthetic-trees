import os
import pandas as pd


import argparse

import matplotlib.pyplot as plt

from util.math import calculate_AuC

    
def parse_args():
    
    parser = argparse.ArgumentParser(description="Results directory.")

    parser.add_argument("-p", "--path",
                        help="Results path.", 
                        required=True,
                        default="dataset", 
                        type=str)

    return parser.parse_args()


def main():

  args = parse_args()

  df = pd.read_csv(args.path)
  
  df["f1"] = (2 * ((df["recall"] * df["precision"]) / (df["recall"] + df["precision"]))).fillna(0)

  df = df.round(2)

  df = df.groupby("threshold").agg("mean", numeric_only=True)
  
  print(f"F1 AUC: {calculate_AuC(df['f1'])}")
  print(f"Recall AUC: {calculate_AuC(df['recall'])}")
  print(f"Precision AUC: {calculate_AuC(df['precision'])}")

  #download to that directory
  ax = df.plot()
  ax.set_ylim(0, 100)
  ax.set_xlim(0, 1)

  plt.show()

if __name__ == "__main__":
    main()