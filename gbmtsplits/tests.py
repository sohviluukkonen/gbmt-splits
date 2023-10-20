# Unit test for gbmt-splits
import os
import pandas as pd
import numpy as np
from unittest import TestCase
from parameterized import parameterized

from .clustering import (
    RandomClustering, 
    MaxMinClustering, 
    LeaderPickerClustering, 
    MurckoScaffoldClustering
)
from .splitters import GBMTSplit, GBMTRepeatedSplit, GBMTKFold, GBMTRepeatedKFold
from .data import split_dataset

preassigned_smiles = {
    'Brc1cccc(Nc2nc3c(N4CCCC4)ncnc3s2)c1' : 0,
    'C#CCn1c(=O)c2c(nc3n2CCCN3C2CCC2)n(C)c1=O' : 1,
}

# logging.basicConfig(level=logging.INFO)

class TestSplits(TestCase):

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.csv')
    seed = 2022
    time_limit = 10

    df = pd.read_csv(test_data_path)
    smiles_list = df['SMILES'].tolist()
    y = df.drop(columns=['SMILES']).to_numpy()
    X = np.zeros((y.shape[0], 1))

    @parameterized.expand([
        ([0.8, 0.2], None, None, True), 
        (None, 0.2, preassigned_smiles, False), 
    ])  
    
    def test_GBMTSplit(self, sizes, test_size, preassigned_smiles, stratify):
        splitter = GBMTSplit(test_size=test_size, sizes=sizes, stratify=stratify)
        split = splitter.split(
            self.X, 
            self.y, 
            self.smiles_list, 
            preassigned_smiles=preassigned_smiles
            )

        for i, s in enumerate(split):
            assert len(s) == 2
            if preassigned_smiles:
                for asg_smiles, asg_subset in preassigned_smiles.items():
                    indices = s[asg_subset]
                    assert asg_smiles in [self.smiles_list[i] for i in indices]

    @parameterized.expand([
        (2, None),
        (5, preassigned_smiles), 
    ])

    def test_GBMTRepeatedSplit(self, n_repeats, preassigned_smiles):
        splitter = GBMTRepeatedSplit(n_repeats=n_repeats)
        split = splitter.split(self.X, self.y, self.smiles_list, preassigned_smiles=preassigned_smiles)

        count = 0
        for i, s in enumerate(split):
            assert len(s) == 2
            count += 1
            if preassigned_smiles:
                for asg_smiles, asg_subset in preassigned_smiles.items():
                    indices = s[asg_subset]
                    assert asg_smiles in [self.smiles_list[i] for i in indices]
        assert count == n_repeats
    
    @parameterized.expand([
        RandomClustering(),
        MaxMinClustering(),
        LeaderPickerClustering(),
        MurckoScaffoldClustering(),
        'predefined_clusters'
    ])

    def test_GBMTKFold(self, clustering_method=None):
        if clustering_method == 'predefined_clusters':
            # Create predifined clusters
            mol_idx = np.arange(len(self.smiles_list))
            clusters = np.array_split(mol_idx, 20)
            clustering_method = {i : [] for i in range(20)}
            for i, cluster in enumerate(clusters):
                for idx in cluster:
                    clustering_method[i].append(self.smiles_list[idx])
            
        splitter = GBMTKFold(n_splits=5, clustering_method=clustering_method)
        split = splitter.split(self.X, self.y, self.smiles_list)

        count = 0
        for i, s in enumerate(split):
            assert len(s) == 2
            count += 1
        assert count == 5

    def test_GBMTRepeatedKFold(self,):
        splitter = GBMTRepeatedKFold(n_splits=5, n_repeats=3)
        split = splitter.split(self.X, self.y, self.smiles_list)

        count = 0
        for i, s in enumerate(split):
            assert len(s) == 2
            count += 1
        assert count == 5 * 3

class TestDatasetSplit(TestCase):
    
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.csv')
    seed = 2022
    time_limit = 10

    df = pd.read_csv(test_data_path)
    smiles_list = df['SMILES'].tolist()
    y = df.drop(columns=['SMILES']).to_numpy()
    X = np.zeros((y.shape[0], 1))

    @parameterized.expand([
        (GBMTSplit(test_size=0.2), None, preassigned_smiles, True, ),
        (GBMTRepeatedSplit(test_size=0.2), 'Binary', None, False,),
        (GBMTKFold(n_splits=3), 'String', None, True),
        (GBMTRepeatedKFold(n_repeats=2, n_splits=3), 'Contineous', None, False),
    ])

    def test_split_dataset(self, splitter, ignore_cols, preassigned_smiles, keep_stratified):

        data = split_dataset(
            data=self.df.copy(),
            splitter=splitter,
            smiles_col='SMILES',
            ignore_cols=ignore_cols,
            preassigned_smiles=preassigned_smiles,
        )

        if isinstance(splitter, GBMTSplit):
            assert 'Split' in data.columns
            assert len(data['Split'].unique()) == 2

        elif isinstance(splitter, GBMTRepeatedSplit):
            assert 'Split_0' in data.columns
            assert 'Split_1' in data.columns
            assert len(data['Split_0'].unique()) == 2
            assert len(data['Split_1'].unique()) == 2

        elif isinstance(splitter, GBMTKFold):
            assert 'Folds' in data.columns
            assert len(data['Folds'].unique()) == 3

        elif isinstance(splitter, GBMTRepeatedKFold):
            # Assert that two columns starting with 'Folds' exist
            assert 'Folds_0' in data.columns
            assert 'Folds_1' in data.columns
            assert len(data['Folds_0'].unique()) == 3
            assert len(data['Folds_1'].unique()) == 3

        # Check preassigned smiles
        if preassigned_smiles:
            for asg_smiles, asg_subset in preassigned_smiles.items():
                indices = data[data['Split'] == asg_subset].index
                assert asg_smiles in [self.smiles_list[i] for i in indices]

        # Check numbers of columns, if keep_stratified is False
        if not keep_stratified:
            n_original_cols = len(self.df.columns)
            n_splits = splitter.n_repeats if ( isinstance(splitter, GBMTRepeatedSplit) 
                                               or isinstance(splitter, GBMTRepeatedKFold) ) else 1
            assert len(data.columns) == n_original_cols + n_splits, f"Expected {n_original_cols + n_newcols} columns, got {len(data.columns)} instead."