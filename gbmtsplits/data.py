"""
Module for splitting data in a dataframe with pre- and post-processing.

The module contains the following classes:

Author: Sohvi Luukkonen
"""
import numpy as np
import pandas as pd

from typing import Callable

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from .logs import logger
from .splitters import GBMTSplit, GBMTRepeatedSplit, GBMTKFold, GBMTRepeatedKFold

def split_dataset(
        data : pd.DataFrame = None,
        data_path : str = None,
        splitter : Callable = GBMTSplit(test_size=0.2),
        smiles_col : str = 'SMILES',
        task_cols : list = None,
        ignore_cols : list = None,
        output_path : str = None,
        compute_stats : bool = True,
        preassigned_smiles : list = None,
        dissimilarity_fp_calculator : Callable = GetMorganGenerator(radius=2, fpSize=1024),
        keep_stratified : bool = False,
        ) -> pd.DataFrame:
        
        # Get data
        if data is None and data_path is None:
            raise ValueError('Both data and data_path cannot be defined.')
        elif data is not None and data_path is not None:
            raise ValueError('Neither data nor data_path is defined.')
        elif data_path:
            data = pd.read_csv(data_path)

        # Header : 80 characters : text padded with '=' on both sides
        # TODO : give lots of info about the split

        # Get task columns
        if not task_cols:
            task_cols = [col for col in data.columns if col != smiles_col]
        if ignore_cols:
            task_cols = [col for col in task_cols if col not in ignore_cols]

        # Splitter arguments
        smiles_list = data[smiles_col].tolist()
        y = data[task_cols].to_numpy()
        X = np.zeros((y.shape[0], 1)) # Dummy X required by sklearn splitters

        # Split data
        split = splitter.split(
            X, 
            y, 
            smiles_list, 
            task_cols,
            preassigned_smiles=preassigned_smiles
        )

        # Append split indices to dataframe
        if isinstance(splitter, GBMTSplit):
            for i, subset in enumerate(next(split)):
                data.loc[subset, "Split"]  = i
            # Check that all indices are assigned
            assert data['Split'].isna().sum() == 0

        elif isinstance(splitter, GBMTRepeatedSplit):
            for i, subsets in enumerate(split):
                for j, subset in enumerate(subsets):
                    data.loc[subset, f"Split_{i}"]  = j
            # Check that all indices are assigned
            assert data[f"Split_{i}"].isna().sum() == 0

        elif isinstance(splitter, GBMTKFold):
            for i, (_, test) in enumerate(split):
                data.loc[test, "Folds"] = i
            # Check that all indices are assigned
            assert data["Folds"].isna().sum() == 0

        elif isinstance(splitter, GBMTRepeatedKFold):
            n_folds = splitter.n_splits
            for i, (_, test) in enumerate(split):
                repeat = i // n_folds
                fold = i % n_folds
                data.loc[test, f"Folds_{repeat}"] = fold
            # Check that all indices are assigned
            assert data[f"Folds_{repeat}"].isna().sum() == 0

        # Save data
        if output_path:
            data.to_csv(output_path, index=False)

        # Compute statistics
        if compute_stats:
            if splitter.stratify: # Recreate startification to compute stats
                stratified_y, stratified_task_cols = splitter._stratify(y, task_cols)
                df_stratified_y = pd.DataFrame(stratified_y, columns=stratified_task_cols)
                data = pd.concat([data, df_stratified_y], axis=1)
            compute_balance_stats(data, splitter.sizes, task_cols, stratified_task_cols)
            compute_dissimilarity_stats(data, smiles_col, dissimilarity_fp_calculator)
            logger.info('-' * 80)
            if not keep_stratified:
                data = data.drop(columns=stratified_task_cols)

        return data



def compute_balance_stats(data : pd.DataFrame, sizes : list[float], task_cols : list[str], startified_task_cols : list[str] = None):
    """
    Compute split balance statistics for a dataframe.
    """

    split_cols = [col for col in data.columns if col.startswith('Split') or col.startswith('Folds')]
    if not split_cols:
        raise ValueError('No split columns found in dataframe.')
    
    # Compute balance stats for each split
    for split_col in split_cols:

        # Header : 80 characters : text padded with '=' on both sides
        text = f" Balance statistics for {split_col} "
        header = '-' * 50
        header = header[:len(text)] + text + header[len(text):]
        logger.info(header)

        padding_witdth = None

        # If stratified, compute balance stats for each stratified task
        if startified_task_cols:
            padding_witdth = max([len(task) for task in startified_task_cols]) + 2
            for task in startified_task_cols:
                counts = data[[task, split_col]].groupby(split_col).count()
                n = counts[task].sum()
                txt = ''
                for subset in sorted(data[split_col].unique()):
                    n_subset = counts.loc[subset, task]
                    txt += f"{int(subset)}: {n_subset/n:.3f} [{n_subset}]\t"
                logger.info(f"{task:<{padding_witdth}}\t {txt}")
            logger.info('\n')
        
        # Compute balance stats for all original tasks
        for task in task_cols:
            padding_witdth = padding_witdth if padding_witdth else max([len(task) for task in task_cols]) + 2
            counts = data[[task, split_col]].groupby(split_col).count()
            n = counts[task].sum()
            txt = ''
            for subset in sorted(data[split_col].unique()):
                n_subset = counts.loc[subset, task]
                txt += f"{int(subset)}: {n_subset/n:.3f} [{n_subset}]\t"
            logger.info(f"{task:<{padding_witdth}}\t {txt}")
        logger.info('\n')

        # Compute balance stats for all tasks
        balance_score = 0
        txt = f"{'Overall':<{padding_witdth}}\t"
        n = data.shape[0]
        for i, subset in enumerate(sorted(data[split_col].unique())):
            n_subset = data[data[split_col] == subset].shape[0]
            txt += f"{int(subset)}: {n_subset/n:.3f} [{n_subset}]\t"
            balance_score += np.abs(n_subset/n - sizes[i])
        
        logger.info(f"{txt}")
        
        
        logger.info(f"Balance score: {balance_score:.3f}")      

def compute_dissimilarity_stats(
        data : pd.DataFrame, 
        smiles_col : str, 
        dissimilarity_fp_calculator : Callable = GetMorganGenerator(2, 1024)):

    """
    Compute chemical dissimilarity between subsets of a split.
    """

    split_cols = [col for col in data.columns if col.startswith('Split') or col.startswith('Folds')]
    if not split_cols:
        raise ValueError('No split columns found in dataframe.')
    
    # Compute molecular fingerprints
    mols = [Chem.MolFromSmiles(smiles) for smiles in data[smiles_col].tolist()]
    fps = [dissimilarity_fp_calculator.GetFingerprint(mol) for mol in mols]

    # Compute dissimilarity for each split
    for split_col in split_cols:

        # Header : 80 characters : text padded with '=' on both sides
        text = f" Dissimilarity statistics for {split_col} "
        header = '-' * 50
        header = header[:len(text)] + text + header[len(text):]
        logger.info(header)

        # Compute dissimilarity for each subset
        medians = []
        for subset in sorted(data[split_col].unique()):
            min_interset_distances = []
            subset_fps = [fps[i] for i in data[data[split_col] == subset].index]
            other_fps = [fps[i] for i in data[data[split_col] != subset].index]
            for fp in subset_fps:
                sim = DataStructs.BulkTanimotoSimilarity(fp, other_fps)
                min_interset_distances.append(1 - np.max(sim))
            mean, std, median = np.mean(min_interset_distances), np.std(min_interset_distances), np.median(min_interset_distances)
            medians.append(median)
            logger.debug(mean, std, median)
            logger.info(f"Subset {int(subset)}: {mean:.3f} +/- {std:.3f} (median: {median:.3f})")
        logger.info(f"Chemical dissimilarity score: {np.min(medians):.3f}")