# Globally balanced multi-task splits

A tool to create well-balanced multi-task splits without data leakage between different tasks for QSAR modelling.

This package is based on the work of Giovanni Tricarico presented in [Construction of balanced, chemically dissimilar training, validation and test sets for machine learning on molecular datasets](https://chemrxiv.org/engage/chemrxiv/article-details/6253d85d88636ca19c0de92d). 

# Installation

```
pip install git+ssh:git@github.com:sohviluukkonen/gbmt-splits.git 
```

# Getting started
## API

Load or create pivoted dataset (each row corresponds to a unique molecules and each task has it's own column)
```
import pandas as pd
dataset = pd.read_csv('data.csv')
```

Splitting data with a random initial clustering of molecules
```
from gbmtsplits import RandomGloballyBalancedSplit as rgbs
splitter = rgbs()
data_rgbs = splitter(data, min_distance=True)
```

or molecular dissimialrity-based clustering
```
from gbmtsplits import DissimilarityDrivenGloballyBalancedSplit as dgbs
splitter = dgbs()
data_dgbs = splitter(data, min_distance=True)
```
as `min_distance=True`, the minimum Tanimoto distance between a molecule in a subset and all the molecules in the other subsets.

The chemical (dis)similarity of the subsets and the balance of subsets per task can visualized either for a single dataset/split:
```
from gbmt.plot import PlottingSingleDataset
plotter = PlottingSingleDataset(data_rgbs)
plotter.plot_all()
```
<p float="left">
  <img src="https://user-images.githubusercontent.com/25030163/219752130-01e16b4a-e056-4136-9b88-44a6e5e26385.png" width="220"> 
  <img src="https://user-images.githubusercontent.com/25030163/219752240-90bb5df9-e8db-4c3a-9a47-0d3e1f130acb.png" width="220"> 
  <img src="https://user-images.githubusercontent.com/25030163/220315575-f9c12fb6-0ff9-4cf7-aa93-f218be36cd97.png" width="150"> 
  <img src="https://user-images.githubusercontent.com/25030163/220315711-6ddb8470-c966-47a6-85ba-cae036f59b1f.png" width="150"> 
  <img src="https://user-images.githubusercontent.com/25030163/219752297-7b952b1e-ad4e-485f-b786-f8a6087e084c.png" width="200"> 
</p>

or to compare multiple datasets/splits:
```
from gbmt.plot import PlottingCompareDatasets

data_rgbs['Dataset'] = 'RGBS'
data_dgbs['Dataset'] = 'DGBS'
data_both = pd.concat([data_rgbs, data_dgbs], ignore_index=True])

plotter = PlottingCompareDatasets(data_both, compare_col='Dataset')
plotter.plot_all()
```
<p float="left">
  <img src="https://user-images.githubusercontent.com/25030163/219757615-d64b869d-dc19-4506-a3da-b23658009677.png" width="500"> 
  <img src="https://user-images.githubusercontent.com/25030163/219757669-f298712e-8409-41bc-8722-5df3e829f1de.png" width="500"> 
  <img src="https://user-images.githubusercontent.com/25030163/219757718-51dd6b3a-b5cd-40fb-b525-427762833258.png" width="300"> 
  <img src="https://user-images.githubusercontent.com/25030163/220318196-2016ba5a-b641-414f-b11a-afff9be2edfe.png" width="300"> 
  <img src="https://user-images.githubusercontent.com/25030163/219757781-e41b68f0-f1f8-4eac-9d13-88e2b42010e4.png" width="300"> 
</p>
