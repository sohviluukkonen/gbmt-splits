"""
Module for splitting data into subsets for globally balanced multi-task learning.

Authors: Sohvi Luukkonen & Giovanni Tricarico
"""

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from pulp import *

from .clustering import ClusteringMethod, MaxMinClustering, RandomClustering
from .logs import logger

class GBMTBase(ABC):
    """
    Base class for GBMT splitters
    """
    def __init__(
        self,
        sizes : list[int] = [0.8, 0.1, 0.1],
        clustering_method : ClusteringMethod | dict = MaxMinClustering(),
        equal_weight_perc_compounds_as_tasks : bool = True,
        absolute_gap : float = 1e-3,
        time_limit_seconds : int = None,
        n_jobs : int = 1,
        stratify : bool = True,
        stratify_reg_nbins : int = 5,  
    ):
        self.sizes = sizes
        self.clustering_method = clustering_method
        self.equal_weight_perc_compounds_as_tasks = equal_weight_perc_compounds_as_tasks
        self.absolute_gap = absolute_gap
        self.time_limit_seconds = time_limit_seconds
        self.n_jobs = n_jobs
        self.stratify = stratify
        self.stratify_reg_nbins = stratify_reg_nbins


    @abstractmethod
    def split(X : np.array, y : np.array, *args, **kwargs):
        pass

    @abstractmethod
    def get_n_splits(X : np.array, y : np.array = None, *args, **kwargs):
        pass

    def get_meta_routing():
        # For consistency with sklearn style (to be implemented?)
        logger.warning("get_meta_routing is not implemented")
        pass

    def _split(
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

        Returns
        -------
        tuple of lists of ints
            Tuple of molecule indices for each subset. The tuple length is equal to the number of subsets.
        """       

        # Stratify the data
        if self.stratify:
            y, task_names = self._stratify(y, task_names)

        # One hot encode the target value matrix
        y = self._one_hot_encode(y)     

        # Cluster the data
        if isinstance(self.clustering_method, dict):
            clusters = self._get_predefined_clusters(smiles_list)
        else:
            clusters = self.clustering_method.__call__(smiles_list)

        # Preassign clusters to subset/folds based on preassigned smiles
        if preassigned_smiles:
            preassigned_clusters = self._get_preassigned_clusters(smiles_list, preassigned_smiles, clusters)
        else:
            preassigned_clusters = None

        # Compute the number of datapoints per task for each cluster
        task_vs_clusters = self._get_data_summary_per_cluster(y, clusters)

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
            self.n_jobs,
            preassigned_clusters)  
        
        # Tuple of molecule indices for each subset
        split_tuple = ()
        for subset in range(1,len(self.sizes)+1):
            cluster_indices = [i for i, x in enumerate(merged_clusters_mapping) if x == subset]
            smiles_indices = [x for i, cluster in clusters.items() if i in cluster_indices for x in cluster]
            split_tuple += (smiles_indices,)
            
        return split_tuple
    
    def _get_predefined_clusters(self, smiles_list : list):
        """
        Get predefined clusters, check that the SMILES strings in the clustering dictionary
        match the SMILES strings in the SMILES list, and transform SMILES strings to indices.

        Parameters
        ----------
        smiles_list : list[str]
            The list of SMILES strings

        Returns
        -------
        dict
            The dictionary of clusters. The keys are cluster indices and values are indices of SMILES strings.
        """

        smiles_from_clusters = [smiles for cluster in self.clustering_method.values() for smiles in cluster]
        if set(smiles_list) != set(smiles_from_clusters):
            raise ValueError("The SMILES strings in the clustering dictionary must match the SMILES strings in the SMILES list")
        # Clusters : cluster index -> list of smiles indices
        clusters = {i : [j for j, smiles in enumerate(smiles_list) if smiles in smiles_from_clusters] for i in self.clustering_method.keys()}

        return clusters
    
    def _check_input_consistency(
            self, 
            X : np.array, 
            y : np.array, 
            smiles_list : list[str] | None = None, 
            task_names : list[str] | None = None,
            preassigned_smiles : dict[str, int] | None = None):

        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"
        if smiles_list is not None:
            assert X.shape[0] == len(smiles_list), "X and smiles_list must have the same number of rows"
        if task_names is not None:
            assert y.shape[1] == len(task_names), "y and task_names must have the same number of columns"

        if not smiles_list:
            if preassigned_smiles:
                raise ValueError("smiles_list must be provided when preassigned_smiles is provided")
            if not isinstance(self.clustering_method, RandomClustering) or \
                not isinstance(self.clustering_method, dict):
                raise ValueError("smiles_list must be provided when clustering_method is not RandomClustering")
            else:
                smiles_list = [f"smiles_{i}" for i in range(X.shape[0])]   
    
    def _stratify(self, y : np.array, task_names : list[str] | None = None) -> np.array:
        """
        Stratify the data in each task. If the values are floats, the data is stratified
        into bins, and each bin is one-hot encoded to new columns. If values are integers
        or strings, the data is stratified into one-hot encoded to new columns. 
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
        def is_numeric(x):
            try:
                float(x)
                return True
            except:
                return False
        
        if task_names is None:
            task_names = [f"task_{i}" for i in range(y.shape[1])]
        
        df = pd.DataFrame(y, columns=task_names)
        stratified_task_names = []
        for task_name in task_names:
            task_values = df[task_name].dropna().unique()
            
            # Check if values are numerical
            task_values_numerical = np.array([is_numeric(task_value) for task_value in task_values])
            all_values_numerical = np.all(task_values_numerical)
            any_values_numerical = np.any(task_values_numerical)
            
            # If values both numerical and non-numerical, raise error
            if not all_values_numerical and any_values_numerical:
                raise ValueError(f"Task {task_name} has both numerical and non-numerical values, which is not supported for stratification")
            
            # If only non-numerical values or only integers, stratify into one-hot encoded columns
            elif not any_values_numerical or all( value % 1 == 0 for value in task_values):
                for value in task_values:
                    stratified_task_names.append(f"{task_name}_{value}")
                    df[f"{task_name}_{value}"] = df[task_name].apply(lambda x: 1 if x == value else np.nan)
            # If values are floats, stratify into bins
            else:
                sorted_task_values = np.sort(task_values)
                bins = np.array_split(sorted_task_values, self.stratify_reg_nbins)
                for i, bin in enumerate(bins):
                    key = f"{task_name}_{bin[0]:.2f}_{bin[-1]:.2f}"
                    stratified_task_names.append(key)
                    df[key] = df[task_name].apply(lambda x: 1 if x in bin else np.nan)
            df = df.drop(columns=[task_name])

        stratified_y = df.to_numpy()
        logger.info(f"The initial {len(task_names)} tasks are stratified into {len(stratified_task_names)} tasks for balancing")
        
        return stratified_y, stratified_task_names

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
            The dictionary of preassigned smiles. The keys are the smiles and
            the values are the subset/fold indices.
        clusters : dict
            The dictionary of clusters. The keys are cluster indices and 
            the values are indices of SMILES strings.

        Returns
        -------
        dict
            The dictionary of preassigned clusters. The keys are the cluster indices and
            the values are the subset/fold indices.
        """

        preassigned_clusters = {}
        for smiles, subset in preassigned_smiles.items():
            if smiles not in smiles_list:
                raise ValueError(f"Preassigned SMILES string {smiles} not found in smiles_list")
            else:
                smiles_idx = smiles_list.index(smiles)
            for cluster_idx, cluster in clusters.items():
                if smiles_idx in cluster:
                    preassigned_clusters[cluster_idx] = subset
                    logger.info(f"Preassigned cluster {cluster_idx} (containing {smiles}) to subset {subset}")

        return preassigned_clusters

    def _one_hot_encode(self, y : np.array) -> np.array:
        """
        One hot encode the target values. All non-NaN values are encoded as 1,
        and all NaN values are encoded as NaN.
        
        Parameters
        ----------
        y : np.array
            The target values 
        
        Returns
        -------
        np.array
            The one hot encoded target values.
        """
        y_ = np.empty(y.shape)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if not isinstance(y[i, j], float) or not np.isnan(y[i, j]):
                    y_[i, j] = 1
                else:
                    y_[i, j] = y[i, j]
        return y_

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

    def _get_data_summary_per_cluster(
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
                task_vs_clusters[i+1, j] = np.count_nonzero(y[cluster, i])
        
        return task_vs_clusters

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

            # If preassigned_clusters is provided, add the constraints to the model to force the clusters
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
    
class GBMTSplit(GBMTBase):
    """
    Globally Balanced Multi-Task Splitter for single split
    """

    def __init__(
            self, 
            test_size : float | None = None,
            sizes: list[int] | None = None,
            clustering_method: ClusteringMethod | dict = MaxMinClustering(),
            equal_weight_perc_compounds_as_tasks: bool = True,
            absolute_gap: float = 0.001,
            time_limit_seconds: int = None,
            n_jobs: int = 1,
            stratify: bool = True,
            stratify_reg_nbins: int = 5):
        
        super().__init__(sizes, clustering_method,  equal_weight_perc_compounds_as_tasks, absolute_gap, time_limit_seconds, n_jobs, stratify, stratify_reg_nbins)

        if test_size is None and sizes is None:
            raise ValueError("Either test_size or sizes must be provided")
        elif test_size is not None and sizes is not None:
            raise ValueError("Either test_size or sizes must be provided, but not both")
        elif test_size is not None:
            self.sizes = [1-test_size, test_size]
        elif np.sum(sizes) != 1:
            raise ValueError("The sum of subset sizes must be equal to 1")
    
    def get_n_splits(self, X: np.array, y: np.array  = None, *args, **kwargs):
        return 1 
    
    def split(
            self,
            X : np.array,
            y : np.array,
            smiles_list : list[str] | None = None,
            task_names : list[str] | None = None,
            preassigned_smiles : dict[str, int] | None = None,
            *args, **kwargs):

        """
        Generate list of indices to split data into subsets.
        
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
            The dictionary of preassigned smiles. The keys are the smiles and 
            the values are the subset indices.

        Yields
        ------
        tuple of lists of ints
            Tuple of molecule indices for each subset. The tuple length is equal to 
            the number of subsets.
        """

        self._check_input_consistency(X, y, smiles_list, task_names, preassigned_smiles)

        logger.info(f"Splitting data into {len(self.sizes)} subsets of sizes {self.sizes}")
        split = self._split(X, y, smiles_list, task_names, preassigned_smiles)
        yield split
    
class GBMTRepeatedSplit(GBMTBase):
    """
    Globally Balanced Multi-Task Splitter for repeated splits
    """
    def __init__(
            self,
            test_size : float | None = 0.2, 
            sizes: list[int] = None,
            n_repeats : int = 5,
            clustering_method: ClusteringMethod | dict = MaxMinClustering(),
            equal_weight_perc_compounds_as_tasks: bool = True,
            absolute_gap: float = 0.001,
            time_limit_seconds: int = None,
            n_jobs: int = 1,
            stratify: bool = True,
            stratify_reg_nbins: int = 5):
        super().__init__(sizes, clustering_method, equal_weight_perc_compounds_as_tasks, absolute_gap, time_limit_seconds, n_jobs, stratify, stratify_reg_nbins)

        if test_size is None and sizes is None:
            raise ValueError("Either test_size or sizes must be provided")
        elif test_size is not None and sizes is not None:
            raise ValueError("Either test_size or sizes must be provided, but not both")
        elif test_size is not None:
            self.sizes = [1-test_size, test_size]
        elif np.sum(sizes) != 1:
            raise ValueError("The sum of subset sizes must be equal to 1")

        self.n_repeats = n_repeats

        if self.n_repeats < 1:
            raise ValueError("n_repeats must be greater than 0")
        
        if not (isinstance(self.clustering_method, MaxMinClustering) or 
                    not isinstance(self.clustering_method, RandomClustering)):
                raise ValueError("GBMTRepeatedSplit only supports MaxMinClustering and RandomClustering")
        
    def get_n_splits(self, X: np.array, y: np.array  = None, *args, **kwargs):
        return self.n_repeats
    
    def split(
            self,
            X : np.array,
            y : np.array,
            smiles_list : list[str] | None = None,
            task_names : list[str] | None = None,
            preassigned_smiles : dict[str, int] | None = None,
            *args, **kwargs):
        """
        Repeats of GBMTSplit.split method with different clusterings at each repeat.
        
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
            The dictionary of preassigned smiles. The keys are the smiles and
            the values are the subset indices.

        Yields
        ------
        tuple of lists of ints
            Tuple of molecule indices for each subset. The tuple length is equal to 
            the number of subsets.
        """

        self._check_input_consistency(X, y, smiles_list, task_names, preassigned_smiles)

        for i in range(self.n_repeats):
            logger.info(f"Splitting data, repeat {i+1}/{self.n_repeats}")
            split = self._split(X, y, smiles_list, task_names, preassigned_smiles)
            self.clustering_method.seed += 1
            yield split

class GBMTKFold(GBMTBase):
    """
    Globally Balanced Multi-Task Splitter for k-fold cross-validation
    
    """

    def __init__(
            self, 
            n_splits : int = 5,
            clustering_method: ClusteringMethod | dict = MaxMinClustering(),
            equal_weight_perc_compounds_as_tasks: bool = True,
            absolute_gap: float = 0.001,
            time_limit_seconds: int = None,
            n_jobs: int = 1,
            stratify: bool = True,
            stratify_reg_nbins: int = 5):
        super().__init__(None, clustering_method, equal_weight_perc_compounds_as_tasks, absolute_gap, time_limit_seconds, n_jobs, stratify, stratify_reg_nbins)

        self.n_splits = n_splits
        if self.n_splits < 2:
            raise ValueError("n_splits must be greater than 1")
        
        self.sizes = [1/n_splits for _ in range(n_splits)]

    def get_n_splits(self, X: np.array, y: np.array  = None, *args, **kwargs):
        return self.n_splits
    
    def split(
            self,
            X : np.array,
            y : np.array,
            smiles_list : list[str] | None = None,
            task_names : list[str] | None = None,
            *args, **kwargs):
        """
        Generate list of indices to split data into subsets for k-fold cross-validation.

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

        Yields
        ------
        tuple of lists of ints
            Tuple of molecule indices for each subset. The tuple length is equal to 
            the number of subsets.
        """
        self._check_input_consistency(X, y, smiles_list, task_names)

        logger.info(f"Creating {self.n_splits}-fold cross-validation splits")
        split = self._split(X, y, smiles_list, task_names)
        for i in range(self.n_splits):
            test_indices = split[i]
            train_indices = [x for j in range(self.n_splits) if j != i for x in split[j]]
            yield train_indices, test_indices

class GBMTRepeatedKFold(GBMTBase):
    """
    Globally Balanced Multi-Task Splitter for repeated cross-validation splits.
    """

    def __init__(
            self, 
            n_splits : int = 5,
            n_repeats : int = 5,
            clustering_method: ClusteringMethod | dict = MaxMinClustering(),
            equal_weight_perc_compounds_as_tasks: bool = True,
            absolute_gap: float = 0.001,
            time_limit_seconds: int = None,
            n_jobs: int = 1,
            stratify: bool = True, 
            stratify_reg_nbins: int = 5):
        super().__init__(None, clustering_method, equal_weight_perc_compounds_as_tasks, absolute_gap, time_limit_seconds, n_jobs, stratify, stratify_reg_nbins)


        self.n_splits = n_splits
        if self.n_splits < 2:
            raise ValueError("n_splits must be greater than 1")
        self.sizes = [1/n_splits for _ in range(n_splits)]

        self.n_repeats = n_repeats
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be greater than 0")
        
        if not (isinstance(self.clustering_method, MaxMinClustering) or 
                    not isinstance(self.clustering_method, RandomClustering)):
                raise ValueError("GBMTRepeatedKfold only supports MaxMinClustering and RandomClustering")
        
    def get_n_splits(self, X: np.array, y: np.array  = None, *args, **kwargs):
        return self.n_repeats * self.n_splits
    
    def split(
            self,
            X : np.array,
            y : np.array,
            smiles_list : list[str] | None = None,
            task_names : list[str] | None = None,
            *args, **kwargs):
        """
        Repeats of GBMTKFold.split method with different clusterings at each repeat.

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

        Yields
        ------
        tuple of lists of ints
            Tuple of molecule indices for each subset. The tuple length is equal to 
            the number of subsets.
        """
        self._check_input_consistency(X, y, smiles_list, task_names)

        for i in range(self.n_repeats):
            logger.info(f"Creating {self.n_splits}-fold cross-validation splits, repeat {i+1}/{self.n_repeats}")
            split = self._split(X, y, smiles_list, task_names)
            self.clustering_method.seed += 1
            for j in range(self.n_splits):
                test_indices = split[j]
                train_indices = [x for k in range(self.n_splits) if k != j for x in split[k]]
                yield train_indices, test_indices