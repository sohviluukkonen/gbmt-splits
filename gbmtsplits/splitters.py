import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from pulp import *
from typing import Callable

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from .clustering import ClusteringMethod, MaxMinClustering, RandomClustering
from .logs import logger

class GBMTSplitter():

    """ Globally Balanced Multi-Task Splitter in sklearn style"""

    def __init__(
            self,
            sizes : list[int] = [0.8, 0.1, 0.1],
            clustering_method : ClusteringMethod | None = MaxMinClustering(),
            n_repeats : int = 1,
            equal_weight_perc_compounds_as_tasks : bool = True,
            absolute_gap : float = 1e-3,
            time_limit_seconds : int = None,
            n_jobs : int = 1,
            min_distance : bool = True,  
            stratify : bool = True,
            stratify_reg_nbins : int = 5,  
    ):
        self.sizes = sizes
        self.clustering_method = clustering_method
        self.n_repeats = n_repeats
        self.equal_weight_perc_compounds_as_tasks = equal_weight_perc_compounds_as_tasks
        self.absolute_gap = absolute_gap
        self.time_limit_seconds = time_limit_seconds
        self.n_jobs = n_jobs
        self.min_distance = min_distance
        self.stratify = stratify
        self.stratify_reg_nbins = stratify_reg_nbins    
    
    @abstractmethod
    def split(
        self,
        X : np.array, 
        y : np.array, 
        smiles_list : list[str] | None = None,
        task_names : list[str] | None = None,
        preassigned_smiles : dict[str, int] | None = None ) -> list:
        """
        Split the data into subsets
        
        Parameters
        ----------
        X : np.array
            The input data
        y : np.array
            The target values
        smiles_list : list[str] | None
            The list of SMILES strings
        task_names : list[str] | None
            The list of task names
        preassigned_smiles : dict[str, int] | None
            The dictionary of preassigned smiles. The keys are the smiles and the values are the subset/fold indices.
        """
        
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"
        if smiles_list is not None:
            assert X.shape[0] == len(smiles_list), "X and smiles_list must have the same number of rows"
        if task_names is not None:
            assert y.shape[1] == len(task_names), "y and task_names must have the same number of columns"

        if not smiles_list:
            if preassigned_smiles:
                raise ValueError("smiles_list must be provided when preassigned_smiles is provided")    
            if not isinstance(self.clustering_method, RandomClustering):
                raise ValueError("smiles_list must be provided when clustering_method is not RandomClustering")
            else:
                smiles_list = [f"smiles_{i}" for i in range(X.shape[0])]    

    
        if self.n_repeats > 1 and not (isinstance(self.clustering_method, MaxMinClustering)
                                    or isinstance(self.clustering_method, RandomClustering)):
                raise ValueError("n_repeat > 1 is only supported for MaxMinClustering and RandomClustering")
        
        if self.stratify:
             # TODO: implement stratification
             pass
        
        # One hot encode the target value matrix
        y_encoded = self._one_hot_encode(y)
            
        splits = []
        for i in range(self.n_repeats):
            logger.info(f"Splitting data, repeat {i+1}/{self.n_repeats}")
            split = self._split(X, y_encoded, smiles_list, task_names, preassigned_smiles)
            splits.append(split)

        return splits
            # yield self._split(X, y, smiles_list, task_names, preassigned_smiles)


    def _stratify(self, y : np.array, task_names : list[str] | None = None) -> np.array:
        """
        Stratify the data

        Parameters
        ----------
        y : np.array
            The target values
        task_names : list[str] | None
            The list of task names
        
        Returns
        -------
        np.array
            The stratified target values
        """

        raise NotImplementedError
        
        def is_convertible_to_float(x):
            try:
                float(x)
                return True
            except:
                return False
        
        if task_names is None:
            task_names = [f"task_{i}" for i in range(y.shape[1])]

        stratified_y = np.empty((y.shape[0],))
        stratified_task_names = []
        for i, task_name in enumerate(task_names):
            task_values = y[:, i]
            all_values_numerical = np.all([is_convertible_to_float(task_value) for task_value in task_values])
            any_values_numerical = np.any([is_convertible_to_float(task_value) for task_value in task_values])
            # If values both numerical and non-numerical, raise error
            if not all_values_numerical and any_values_numerical:
                raise ValueError(f"Task {task_name} has both numerical and non-numerical values, which is not supported for stratification")
            # If only non-numerical values, use one-hot encoding
            elif not any_values_numerical:
                unique_task_values = np.unique(task_values)
                task_values_one_hot = np.empty((task_values.shape[0], len(unique_task_values)))
                for j, unique_task_value in enumerate(unique_task_values):
                    stratified_task_names.append(f"{task_name}_{unique_task_value}")
                    task_values_one_hot[:, j] = task_values == unique_task_value



        

        
        raise NotImplementedError
    
    def _split(
              self,
              X : np.array,
              y : np.array,
              smiles_list : list[str] | None = None,
              task_names : list[str] | None = None,
              preassigned_smiles : dict[str, int] | None = None ) -> list:
        """
        Split the data into subsets
        """            

        # Cluster the data
        clusters = self.clustering_method.__call__(smiles_list)

        # Preassign clusters to subset/folds based on preassigned smiles
        if preassigned_smiles:
            preassigned_clusters = self._get_preassigned_clusters(preassigned_smiles, clusters)
        else:
            preassigned_clusters = None

        # Compute the number of datapoints per task for each cluster
        task_vs_clusters = self._get_task_vs_clusters(y, clusters)

        # Set time limit for linear programming
        if self.time_limit_seconds is None:
            self.time_limit_seconds = self._get_default_time_limit_seconds(y.shape[0], y.shape[1])

        # Merge the clusters with a linear programming method to create the subsets
        merged_clusters_mapping = self._merge_clusters_with_balancing_mapping(
            task_vs_clusters, 
            self.sizes, 
            self.equal_weight_perc_compounds_as_tasks, 
            self.absolute_gap,
            self.time_limit_seconds, 
            preassigned_clusters)  


        return merged_clusters_mapping




    def _get_preassigned_clusters(
            self, 
            smiles_list : list[str],
            preassigned_smiles : dict[str, int], 
            clusters : dict) -> dict:
        """
        Preassign clusters to subset/folds based on preassigned smiles

        Parameters
        ----------
        smiles_list : list[str]
            The list of SMILES strings
        preassigned_smiles : dict[str, int]
            The dictionary of preassigned smiles. The keys are the smiles and the values are the subset/fold indices.
        clusters : dict
            The dictionary of clusters. The keys are cluster indices and values are indices of SMILES strings.

        Returns
        -------
        dict
            The dictionary of preassigned clusters. The keys are the cluster indices and the values are the subset/fold indices.
        """

        raise NotImplementedError

        preassigned_clusters = {}
        for smiles, subset_index in preassigned_smiles.items():
            for cluster_index, cluster in clusters.items():
                if smiles in cluster:
                    preassigned_clusters[smiles] = subset_index
                    break
            else:
                raise ValueError(f"SMILES {smiles} not found in any cluster")

    def _one_hot_encode(self, y : np.array) -> np.array:
        """
        One hot encode the target values. All np.nans are replaced with 0s and all other values are replaced with 1s.
        
        Parameters
        ----------
        y : np.array
            The target values 
        
        Returns
        -------
        np.array
            The one hot encoded target values.
        """
        y_ = np.zeros(y.shape)
        # Loop through the elements and set NaNs to 0 and non-NaN floats to 1
        for i, value in enumerate(y):
            if isinstance(value, float) and np.isnan(value):
                y_[i] = 0
            elif isinstance(value, str):
                try:
                    if not np.isnan(float(value)):
                        y_[i] = 1
                except ValueError:
                    pass
        return y_

    def _get_task_vs_clusters(
            self,
            y : np.array,
            clusters : dict ) -> dict:
        
        """
        Compute the number of datapoints per task for each cluster.

        Parameters
        ----------
        y : np.array
            The target values
        clusters : dict
            The dictionary of clusters. The keys are cluster indices and values are indices of SMILES strings.

        Returns
        -------
        np.array of shape (len(tasks)+1, len(clusters))
            Array with each columns correspoding to a cluster and each row to a task
            plus the 1st row for the number of molecules per cluster    
        """

        ntasks = y.shape[1]
        task_vs_clusters = np.zeros((ntasks+1, len(clusters)))

        # 1st row is the number of molecules per cluster
        task_vs_clusters[0, :] = [len(cluster) for cluster in clusters.values()]

        # Compute the number of datapoints per task for each cluster
        for i in range(ntasks):
            for j, cluster in clusters.items():
                # print(y[cluster, i])
                # Number non-NaN values
                task_vs_clusters[i+1, j] = np.count_nonzero(y[cluster, i])
        
        return task_vs_clusters

    def _get_default_time_limit_seconds(self, nmols : int, ntasks : int) -> int:
        """
        Compute the default time limit for linear programming based on 
        number of datapoints and number of tasks.
        
        Parameters
        ----------
        nmols : int
            Number of datapoints
        ntasks : int
            Number of tasks
        
        Returns
        -------
        int
            The default time limit in seconds
        """
        tmol = nmols ** (1/3)
        ttarget = np.sqrt(ntasks)
        tmin = 10
        tmax = 60 * 60
        tlim = int(min(tmax, max(tmin, tmol * ttarget)))
        logger.info(f'Time limit for LP: {tlim}s')
        return tlim



    def _merge_clusters_with_balancing_mapping(
            self, 
            tasks_vs_clusters_array : np.array,
            sizes : list[float] = [0.9, 0.1, 0.1], 
            equal_weight_perc_compounds_as_tasks : bool = False,
            absolute_gap : float = 1e-3,
            time_limit_seconds : int = 60*60,
            max_N_threads : int = 1,
            preassigned_clusters : dict[int, int] | None = None) -> list[list[int]]:
            """
            Linear programming function needed to balance the self.df while merging clusters.

            Paper: Tricarico et al., Construction of balanced, chemically dissimilar training, validation 
            and test sets for machine learning on molecular self.dfsets, 2022, 
            DOI: https://doi.org/10.26434/chemrxiv-2022-m8l33-v2

            Parameters
            ----------
            tasks_vs_clusters_array : 2D np.array
                - the cross-tabulation of the number of self.df points per cluster, per task.
                - columns represent unique clusters.
                - rows represent tasks, except the first row, which represents the number of records (or compounds).
                - Optionally, instead of the number of self.df points, the provided array may contain the *percentages*
                    of self.df points _for the task across all clusters_ (i.e. each *row*, NOT column, may sum to 1).
                IMPORTANT: make sure the array has 2 dimensions, even if only balancing the number of self.df records,
                    so there is only 1 row. This can be achieved by setting ndmin = 2 in the np.array function.
            sizes : list
                - list of the desired final sizes (will be normalised to fractions internally).
            equal_weight_perc_compounds_as_tasks : bool
                - if True, matching the % records will have the same weight as matching the % self.df of individual tasks.
                - if False, matching the % records will have a weight X times larger than the X tasks.
            absolute_gap : float
                - the absolute gap between the absolute optimal objective and the current one at which the solver
                stops and returns a solution. Can be very useful for cases where the exact solution requires
                far too long to be found to be of any practical use.
            time_limit_seconds : int
                - the time limit in seconds for the solver (by default set to 1 hour)
                - after this time, whatever solution is available is returned
            max_N_threads : int
                - the maximal number of threads to be used by the solver.
                - it is advisable to set this number as high as allowed by the available resources.
            preassigned_clusters : dict
                - a dictionary of the form {cluster_index: ML_subset_index} to force the clusters to be assigned
                    to the ML subsets as specified by the user.
            
            Returns
            ------
            list (of length equal to the number of columns of tasks_vs_clusters_array) of final cluster identifiers
                (integers, numbered from 1 to len(sizes)), mapping each unique initial cluster to its final cluster.
            Example: if sizes == [20, 10, 70], the output will be a list like [3, 3, 1, 2, 1, 3...], where
                '1' represents the final cluster of relative size 20, '2' the one of relative size 10, and '3' the 
                one of relative size 70.
            """

            # Calculate the fractions from sizes

            fractional_sizes = sizes / np.sum(sizes)

            S = len(sizes)

            # Normalise the self.df matrix
            tasks_vs_clusters_array = tasks_vs_clusters_array / tasks_vs_clusters_array.sum(axis = 1, keepdims = True)

            # Find the number of tasks + compounds (M) and the number of initial clusters (N)
            M, N = tasks_vs_clusters_array.shape
            if (S > N):
                errormessage = 'The requested number of new clusters to make ('+ str(S) + ') cannot be larger than the initial number of clusters (' + str(N) + '). Please review.'
                raise ValueError(errormessage)

            # Given matrix A (M x N) of fraction of self.df per cluster, assign each cluster to one of S final ML subsets,
            # so that the fraction of self.df per ML subset is closest to the corresponding fraction_size.
            # The weights on each ML subset (WML, S x 1) are calculated from fractional_sizes harmonic-mean-like.
            # The weights on each task (WT, M x 1) are calculated as requested by the user.
            # In the end: argmin SUM(ABS((A.X-T).WML).WT)
            # where X is the (N x S) binary solution matrix
            # where T is the (M x S) matrix of task fraction sizes (repeat of fractional_sizes)
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

            # Round all values to have only 3 decimals > reduce computational time
            A = np.round(A, 3)
            fractional_sizes = np.round(fractional_sizes, 3)
            obj_weights = np.round(obj_weights, 3)
            sk_harmonic = np.round(sk_harmonic, 3)           

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

            # If preassigned_clusters is pro[int, int]vided, add the constraints to the model to force the clusters
            # to be assigned to the ML subset preassigned_clusters[t]
            if preassigned_clusters:
                for c, subset in preassigned_clusters.items():
                    # prob += LpAffineExpression(x[c+(subset)*N]) == 1
                    prob += x[c+(subset)*N] == 1

            # Constraints related to the ABS values handling, part 1 and 2
            for m in range(S):
                for t in range(M):
                    cs = [c for c in range(N) if A[t,c] != 0]
                    prob += LpAffineExpression([(x[c+m*N],A[t,c]) for c in cs]) - X[t] <= fractional_sizes[m]
                    prob += LpAffineExpression([(x[c+m*N],A[t,c]) for c in cs]) + X[t] >= fractional_sizes[m]

            # Solve the model
            prob.solve(PULP_CBC_CMD(gapAbs = absolute_gap, timeLimit = time_limit_seconds, threads = max_N_threads, msg=False))

            # Extract the solution
            list_binary_solution = [value(x[i]) for i in range(N * S)]
            list_initial_cluster_indices = [(list(range(N)) * S)[i] for i,l in enumerate(list_binary_solution) if l == 1]
            list_final_ML_subsets = [(list((1 + np.repeat(range(S), N)).astype('int64')))[i] for i,l in enumerate(list_binary_solution) if l == 1]
            mapping = [x for _, x in sorted(zip(list_initial_cluster_indices, list_final_ML_subsets))]

            return mapping 

    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator"""
        return len(self.sizes)
    

if __name__ == '__main__':
    
    import os
    print(os.getcwd())
    df = pd.read_csv('C:\\Users\\admin\\RESEARCH\\Leiden\\gbmt-splits\\gbmtsplits\\test_data.csv')
    smiles_list = df['SMILES'].tolist()
    y = df.drop(columns=['SMILES']).to_numpy()
    X = np.zeros((y.shape[0], 1))

    splitter = GBMTSplitter()
    split = splitter.split(X, y, smiles_list)

    






