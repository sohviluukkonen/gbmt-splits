# Globally balanced multi-task splits

A tool to create well-balanced multi-task splits without data leakage between different tasks for QSAR modelling.

This package is based on the work of Giovanni Tricarico presented in [Construction of balanced, chemically dissimilar training, validation and test sets for machine learning on molecular datasets](https://chemrxiv.org/engage/chemrxiv/article-details/6253d85d88636ca19c0de92d). 

## Installation
```
pip install git+https://git@github.com/sohviluukkonen/gbmt-splits.git@main
```

# Getting started
GBMT splits is organised in five classes:
1. **Clustering** - `ClusterMethods` to make initial clusters of molecules
2. **Splitters** - four GBMT splitters consistent with sklearn's splitters
   1. `GBMTSplit` - for a single split (with any number of output subsets
   2. `GBMTRepeatedSplit` - for repeated splits
   3. `GBMTKFold` - for a k-fold split
   4. `GBMTRepeatedKFold` - for repated k-fold splits
3. **Data** - wrapper to process to make splits and analyse them from dataset
4. **Plotting** - functions to visualize the splits
5. **CLI** - command line interface to make splits

## CLI
The split can be easily created from the command line with

```
gbmtsplits -i <dataset.csv> -c <random/dissimilarity_maxmin/dissimilarity_leader/scaffold> 
```
with <datasets.csv> an pivoted dataset where each row corresponds to a unique molecules and each task has it's own column. For more options use `-h/--help`.

## API

### Raw splitters

### Splitting a dataset

### Visualize the split

The chemical (dis)similarity of the subsets and the balance of subsets per task can visualized either for a single dataset/split:
```
from gbmtsplits.plot import PlottingSingleDataset
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

