import pandas as pd

from typing import Dict


def save_results(results_dict: Dict, save_path: str ="results.csv"):
  
  rows = []
  
  for tree_name, results in results_dict.items():

    species = tree_name.split("_")[0]
    
    for threshold, recall, precision in zip(results["thresholds"], results["recall"], results["precision"]):
    
      rows.append([tree_name, species, threshold, recall, precision])
  
  df = pd.DataFrame(rows, columns=["tree_name", "species", "threshold", "recall", "precision"])
  df.to_csv(f"{save_path}", index=False)
