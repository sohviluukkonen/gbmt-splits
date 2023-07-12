import tqdm

import numpy as np
import pandas as pd

from pulp import *
from typing import List, Dict, Callable
from abc import ABC, abstractmethod

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from .clustering import MaxMinClustering, LeaderPickerClustering, MurckoScaffoldClustering, RandomClustering

class GloballyBalancedSplit:

    """
    Globally balanced splits.

    Attributes
    ----------
    sizes : List[int], optional
        List of sizes of the splits.
    clusters : dict, optional
        Dictionary of clusters, where keys are cluster indices and values are indices of molecules.
    clustering_method : Callable, optional
        Clustering method.
    n_splits : int, optional
        Number of splits.
    equal_weight_perc_compounds_as_tasks : bool, optional
        Whether to weight the tasks equally or not.
    relative_gap : float, optional  
        Relative gap for the linear programming problem.
    time_limit_seconds : int, optional
        Time limit for the linear programming problem.
    n_jobs : int, optional
        Number of jobs.
    min_distance : bool, optional
        Whether to calculate the minimum interset distance between the splits.
    """

    def __init__(
            self,
            sizes : List[int] = [0.8, 0.1, 0.1],
            clusters : dict = None,
            clustering_method : Callable | None = MaxMinClustering(),
            n_splits : int = 1,
            equal_weight_perc_compounds_as_tasks : bool = True,
            relative_gap : float = 0.1,
            time_limit_seconds : int = 60,
            n_jobs : int = 1,
            min_distance : bool = True,    
            ) -> None:     
        
        if clusters is None and clustering_method is None:
            raise ValueError('Either clusters or clustering_method must be provided.')
        elif clusters is not None and clustering_method is not None:
            raise ValueError('Only one of clusters or clustering_method must be provided.')
        
        self.sizes = sizes
        self.clusters = clusters
        self.clustering_method = clustering_method
        self.n_splits = n_splits
        self.equal_weight_perc_compounds_as_tasks = equal_weight_perc_compounds_as_tasks
        self.relative_gap = relative_gap
        self.time_limit_seconds = time_limit_seconds
        self.n_jobs = n_jobs
        self.min_distance = min_distance

    def __call__(
            self,
            data : pd.DataFrame,
            smiles_column : str = 'SMILES',
            targets : List[str] = None,
            ignore_columns : List[str] = None,
            ) -> pd.DataFrame:
        
        """
        Globally balanced splits.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with SMILES strings and targets.
        smiles_column : str, optional
            Name of the column with SMILES strings.
        targets : List[str], optional
            List of target columns.
        ignore_columns : List[str], optional
            List of columns to ignore.
        
        Returns
        -------
        splits : pd.DataFrame
            Dataframe with the splits.
        """

        # Multiple splits doesn't work with all clustering methods
        if self.n_splits > 1 :
            if self.clusters is not None:
                raise ValueError('n_splits > 1 does not work with precomputed clusters.')
            elif not (isinstance(self.clustering_method, MaxMinClustering)
                                    or isinstance(self.clustering_method, RandomClustering)):
                raise ValueError(f'n_splits > 1 does not work with type \
                                  {type(self.clustering_method).__name__}.')

        self.smiles_column = smiles_column
        self.ignore_columns = ignore_columns
        
        # Save the original targets and create the targets for balancing
        self._set_original_targets(data, targets)
        self._set_targets_for_balancing(data)

        smiles_list = data[smiles_column].tolist()

        for split in range(self.n_splits):

            if self.n_splits == 1:
                split_name = 'Split'
                mintd_name = 'minInterSetTd'
            else:
                split_name = f'Split_{split}'
                mintd_name = f'minInterSetTd_{split}'

            # Cluster molecules
            if self.clusters :
                clusters = self.clusters
            else:
                clusters = self.clustering_method(smiles_list)
            # Compute the number of datapoints per task for each cluster
            tasks_per_cluster = self._compute_tasks_per_cluster(data, self.targets_for_balancing, clusters)
            
            # Merge the clusters with a linear programming method to create the subsets
            merged_clusters_mapping = self._merge_clusters_with_balancing_mapping(
                tasks_per_cluster, 
                self.sizes, 
                self.equal_weight_perc_compounds_as_tasks, 
                self.relative_gap,
                self.time_limit_seconds,
                self.n_jobs)  
            for i, idx in clusters.items(): 
                data.loc[idx, split_name] = merged_clusters_mapping[i]-1
        
            # Compute the minimum interset distance between the splits
            if self.min_distance:
                if hasattr(self.clustering_method, 'fp_calculator'):
                    fp_calculator = self.clustering_method.fp_calculator
                    self._compute_min_interset_Tanimoto_distances(data, split_name, mintd_name, fp_calculator)
                else:
                    self._compute_min_interset_Tanimoto_distances(data, split_name, mintd_name)

            # Print balance and chemical bias metrics
            self._print_metrics(data)

            # Update clustering seed
            if self.n_splits > 1:
                self.clustering_method.seed += 1


        # Drop the targets for balancing
        cols2drop = [col for col in self.targets_for_balancing if col not in self.original_targets]
        data.drop(cols2drop, axis=1, inplace=True)

        return data


    def _compute_tasks_per_cluster(self, 
                               df : pd.DataFrame,
                               targets : List[str],
                               clusters : Dict[int, List[int]]) -> Dict[int, List[str]]:
        
        """
        Compute the number of datapoints per task for each cluster.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the data
        targets : List[str]
            List of targets
        clusters : Dict[int, List[int]]
            Dictionary of clusters and list of indices of molecules in the cluster

        Returns
        -------
        np.array of shape (len(targets)+1, len(clusters))
            Array with each columns correspoding to a cluster and each row to a target
            plus the 1st row for the number of molecules per cluster
        """

        target_vs_clusters = np.zeros((len(targets)+1, len(clusters)))
        target_vs_clusters[0,:] = [ len(cluster) for cluster in clusters.values() ]

        for i, target in enumerate(targets):
            for j, indices_per_cluster in clusters.items():
                data_per_cluster = df.iloc[indices_per_cluster]
                target_vs_clusters[i+1,j] = data_per_cluster[target].dropna().shape[0]

        return target_vs_clusters  
    
    # Bash command to delete conda environment
    # conda env remove --name myenv

        
    def _set_original_targets(self, data : pd.DataFrame, targets : List[str] | None) -> None:
        """
        Set the original targets.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with SMILES strings and targets.
        """

        if targets is None:
            columns = data.columns.tolist()
            # Drop the SMILES column.
            columns.remove(self.smiles_column)
            # Drop the ignore columns.
            if self.ignore_columns is not None:
                columns = [column for column in columns if column not in self.ignore_columns]
            # Drop columns starting with 'Split' and 'MinIntersetDistance'.
            columns = [column for column in columns if not column.startswith('Split')]
            columns = [column for column in columns if not column.startswith('MinIntersetDistance')]
            # Set original targets.
            self.original_targets = columns
        else:
            self.original_targets = targets

    def _set_targets_for_balancing(self, data : pd.DataFrame) -> None:

        """
        Set the targets for balancing. If all values for a target are integers, 
        the target is considered a classification target, and a separate column is
        created for each class. If the target is not a classification target,
        the target is considered a regression target, and the target is used as is.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with SMILES strings and targets.
        """

        self.targets_for_balancing = []
        for target in self.original_targets:
            if all(isinstance(x, int) for x in data[target].dropna()):
                for cls in data[target].dropna().unique():
                    data[target + '_' + str(cls)] = (data[target] == cls).map({True: 1, False: np.nan})
                    self.targets_for_balancing.append(f'{target}_{cls}')
            else:
                self.targets_for_balancing.append(target)

    def _merge_clusters_with_balancing_mapping(
            self, 
            tasks_vs_clusters_array : np.array,
            sizes : List[float] = [0.9, 0.1, 0.1], 
            equal_weight_perc_compounds_as_tasks : bool = False,
            relative_gap : float = 0,
            time_limit_seconds : int = 60*60,
            max_N_threads : int = 1) -> List[List[int]]:
            """
            Linear programming function needed to balance the data while merging clusters.

            Paper: Tricarico et al., Construction of balanced, chemically dissimilar training, validation 
            and test sets for machine learning on molecular datasets, 2022, 
            DOI: https://doi.org/10.26434/chemrxiv-2022-m8l33-v2

            Parameters
            ----------
            tasks_vs_clusters_array : 2D np.array
                - the cross-tabulation of the number of data points per cluster, per task.
                - columns represent unique clusters.
                - rows represent tasks, except the first row, which represents the number of records (or compounds).
                - Optionally, instead of the number of data points, the provided array may contain the *percentages*
                    of data points _for the task across all clusters_ (i.e. each *row*, NOT column, may sum to 1).
                IMPORTANT: make sure the array has 2 dimensions, even if only balancing the number of data records,
                    so there is only 1 row. This can be achieved by setting ndmin = 2 in the np.array function.
            sizes : list
                - list of the desired final sizes (will be normalised to fractions internally).
            equal_weight_perc_compounds_as_tasks : bool
                - if True, matching the % records will have the same weight as matching the % data of individual tasks.
                - if False, matching the % records will have a weight X times larger than the X tasks.
            relative_gap : float
                - the relative gap between the absolute optimal objective and the current one at which the solver
                stops and returns a solution. Can be very useful for cases where the exact solution requires
                far too long to be found to be of any practical use.
                - set to 0 to obtain the absolute optimal solution (if reached within the time_limit_seconds)
            time_limit_seconds : int
                - the time limit in seconds for the solver (by default set to 1 hour)
                - after this time, whatever solution is available is returned
            max_N_threads : int
                - the maximal number of threads to be used by the solver.
                - it is advisable to set this number as high as allowed by the available resources.
            
            Returns
            ------
            List (of length equal to the number of columns of tasks_vs_clusters_array) of final cluster identifiers
                (integers, numbered from 1 to len(sizes)), mapping each unique initial cluster to its final cluster.
            Example: if sizes == [20, 10, 70], the output will be a list like [3, 3, 1, 2, 1, 3...], where
                '1' represents the final cluster of relative size 20, '2' the one of relative size 10, and '3' the 
                one of relative size 70.
            """

            # Calculate the fractions from sizes

            fractional_sizes = sizes / np.sum(sizes)

            S = len(sizes)

            # Normalise the data matrix
            tasks_vs_clusters_array = tasks_vs_clusters_array / tasks_vs_clusters_array.sum(axis = 1, keepdims = True)

            # Find the number of tasks + compounds (M) and the number of initial clusters (N)
            M, N = tasks_vs_clusters_array.shape
            if (S > N):
                errormessage = 'The requested number of new clusters to make ('+ str(S) + ') cannot be larger than the initial number of clusters (' + str(N) + '). Please review.'
                raise ValueError(errormessage)

            # Given matrix A (M x N) of fraction of data per cluster, assign each cluster to one of S final ML subsets,
            # so that the fraction of data per ML subset is closest to the corresponding fraction_size.
            # The weights on each ML subset (WML, S x 1) are calculated from fractional_sizes harmonic-mean-like.
            # The weights on each task (WT, M x 1) are calculated as requested by the user.
            # In the end: argmin SUM(ABS((A.X-T).WML).WT)
            # where X is the (N x S) binary solution matrix
            # where T is the (M x S) matrix of target fraction sizes (repeat of fractional_sizes)
            # constraint: assign one cluster to one and only one final ML subset
            # i.e. each row of X must sum to 1

            A = np.copy(tasks_vs_clusters_array)

            # Create WT = obj_weights
            if ((M > 1) & (equal_weight_perc_compounds_as_tasks == False)):
                obj_weights = np.array([M-1] + [1] * (M-1))
            else:
                obj_weights = np.array([1] * M)

            obj_weights = obj_weights / np.sum(obj_weights)

            # Create WML
            sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)

            # Create the pulp model
            prob = LpProblem("Data_balancing", LpMinimize)

            # Create the pulp variables
            # x_names represent clusters, ML_subsets, and are binary variables
            x_names = ['x_'+str(i) for i in range(N * S)]
            x = [LpVariable(x_names[i], lowBound = 0, upBound = 1, cat = 'Integer') for i in range(N * S)]
            # X_names represent tasks, ML_subsets, and are continuous positive variables
            X_names = ['X_'+str(i) for i in range(M * S)]
            X = [LpVariable(X_names[i], lowBound = 0, cat = 'Continuous') for i in range(M * S)]

            # Add the objective to the model

            obj = []
            coeff = []
            for m in range(S):
                for t in range(M):
                    obj.append(X[m*M+t])
                    coeff.append(sk_harmonic[m] * obj_weights[t])

            prob += LpAffineExpression([(obj[i],coeff[i]) for i in range(len(obj)) ])

            # Add the constraints to the model

            # Constraints forcing each cluster to be in one and only one ML_subset
            for c in range(N):
                prob += LpAffineExpression([(x[c+m*N],+1) for m in range(S)]) == 1

            # Constraints related to the ABS values handling, part 1 and 2
            for m in range(S):
                for t in range(M):
                    cs = [c for c in range(N) if A[t,c] != 0]
                    prob += LpAffineExpression([(x[c+m*N],A[t,c]) for c in cs]) - X[t] <= fractional_sizes[m]
                    prob += LpAffineExpression([(x[c+m*N],A[t,c]) for c in cs]) + X[t] >= fractional_sizes[m]

            # Solve the model
            prob.solve(PULP_CBC_CMD(gapRel = relative_gap, timeLimit = time_limit_seconds, threads = max_N_threads, msg=False))
            #solver.tmpDir = "/zfsdata/data/erik/erik-rp1/pQSAR/scaffoldsplit_trial/tmp"
            #prob.solve(solver)

            # Extract the solution

            list_binary_solution = [value(x[i]) for i in range(N * S)]
            list_initial_cluster_indices = [(list(range(N)) * S)[i] for i,l in enumerate(list_binary_solution) if l == 1]
            list_final_ML_subsets = [(list((1 + np.repeat(range(S), N)).astype('int64')))[i] for i,l in enumerate(list_binary_solution) if l == 1]
            mapping = [x for _, x in sorted(zip(list_initial_cluster_indices, list_final_ML_subsets))]

            return mapping

    def _compute_min_interset_Tanimoto_distances(
            self, 
            df, 
            split_col = 'Split',
            mTd_col = 'minInterSetTd',
            fp_calculator = GetMorganGenerator(radius=3, fpSize=2048)
            ):
        """
        Compute the minimum Tanimoto distance per compound to the compounds in the other subsets.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the data with a column 'Split' containing the subset number
        split_col : str, optional
            Name of the column containing the subset number, by default 'Split'
        mTd_col : str, optional
            Name of the column to store the minimum Tanimoto distance in, by default 'minInterSetTd'
        fp_calculator : rdkit.Chem.rdMolDescriptors.GetMorganGenerator, optional
            Fingerprint calculator, by default GetMorganGenerator(radius=3, nBits=2048)
        
        Returns
        -------
        pd.DataFrame
            Dataframe containing the data with a column 'MinInterSetTd' containing the minimum Tanimoto distance

        """

        # Compute fingerprints
        mols = [ Chem.MolFromSmiles(smi) for smi in df[self.smiles_column].tolist() ]
        fps = [ fp_calculator.GetFingerprint(mol) for mol in mols ]

        min_distances = np.zeros(len(fps))
        # Iterate over subsets and compute minimum Tanimoto distance per compound to the compounds in the other subsets
        for j in df[split_col].unique():
            ref_idx = df[df[split_col] == j].index.tolist()
            other_fps = [ fp for i, fp in enumerate(fps) if i not in ref_idx ]
            for i in ref_idx :
                sims = DataStructs.BulkTanimotoSimilarity(fps[i], other_fps)
                min_distances[i] = 1. - max(sims)

        df[mTd_col] = min_distances

        # Print average and std  of minimum distances per subset
        txt = f'Average and std of minimum Tanimoto distance per subset ({split_col}):'
        for subset in sorted(df[split_col].unique()):
            dist = df[df[split_col] == subset][mTd_col].to_numpy()
            txt += f' {subset}: {np.mean(dist):.2f} ({np.std(dist):.2f})'
        print(txt)

        return df

    def _print_metrics(self, data):
        pass
