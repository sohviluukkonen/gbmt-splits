
import numpy as np
import pandas as pd

from pulp import *
from typing import Callable

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from .clustering import MaxMinClustering, LeaderPickerClustering, MurckoScaffoldClustering, RandomClustering
from .logs import logger

class GloballyBalancedSplit:

    """
    Globally balanced splits.

    Attributes
    ----------
    sizes : list[int], optional
        list of sizes of the splits.
    clusters : dict, optional
        dictionary of clusters, where keys are cluster indices and values are indices of molecules.
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
            sizes : list[int] = [0.8, 0.1, 0.1],
            clusters : dict = None,
            clustering_method : Callable | None = MaxMinClustering(),
            n_splits : int = 1,
            equal_weight_perc_compounds_as_tasks : bool = True,
            relative_gap : float = 0.1,
            time_limit_seconds : int = None,
            n_jobs : int = 1,
            min_distance : bool = True,  
            stratify : bool = True,
            stratify_reg_nbins : int = 5,  
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
        self.stratify = stratify
        self.stratify_reg_nbins = stratify_reg_nbins

    def __call__(
            self,
            data : pd.DataFrame,
            smiles_column : str = 'SMILES',
            tasks : list[str] = None,
            ignore_columns : list[str] = None,
            preassigned_smiles : dict[str, int] = None,
            ) -> pd.DataFrame:
        
        """
        Globally balanced splits.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with SMILES strings and tasks.
        smiles_column : str, optional
            Name of the column with SMILES strings.
        tasks : list[str], optional
            list of task columns.
        ignore_columns : list[str], optional
            list of columns to ignore.
        
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

        self.df = data.copy()
        self.smiles_column = smiles_column
        self.ignore_columns = ignore_columns
        self.preassigned_smiles = preassigned_smiles
        
        # Save the original tasks and create the tasks for balancing
        self._set_original_tasks(tasks)
        self._set_tasks_for_balancing()


        smiles_list = self.df[smiles_column].tolist()

        for split in range(self.n_splits):

            if self.n_splits == 1:
                split_name = 'Split'
                mintd_name = 'minInterSetTd'
            else:
                split_name = f'Split_{split}'
                mintd_name = f'minInterSetTd_{split}'

            txt = f' {split_name} '
            n = int((80 - len(txt)) / 2)
            logger.info('=' * n + txt + '=' * n)
            logger.info(f'Clustering method: {self.clustering_method.get_name() if self.clustering_method else "precomputed clusters"}')
            logger.info(f'Original tasks: {self.original_tasks}')
            logger.info(f'Tasks for balancing: {self.tasks_for_balancing}')
            logger.info(f'Subset sizes: {self.sizes}')

            # Cluster molecules
            if self.clusters :
                clusters = self.clusters
            else:
                clusters = self.clustering_method(smiles_list, )
            logger.info(f'Number of initial clusters: {len(clusters)}')
            
            # Get clusters indices of pre-assigned molecules
            preassigned_clusters = self._get_preassigned_clusters(clusters) if preassigned_smiles else None
            # Compute the number of self.dfpoints per task for each cluster
            tasks_per_cluster = self._compute_tasks_per_cluster(self.tasks_for_balancing, clusters)
                     
            # Merge the clusters with a linear programming method to create the subsets
            merged_clusters_mapping = self._merge_clusters_with_balancing_mapping(
                tasks_per_cluster, 
                self.sizes, 
                self.equal_weight_perc_compounds_as_tasks, 
                self.relative_gap,
                self.time_limit_seconds if self.time_limit_seconds else self.get_default_time_limit_seconds(len(smiles_list), len(self.tasks_for_balancing)),
                self.n_jobs,
                preassigned_clusters)  

            for i, idx in clusters.items(): 
                self.df.loc[idx, split_name] = merged_clusters_mapping[i]-1

            # Print balance
            self._compute_task_balace(split_name)
        
            # Compute the minimum interset distance between the splits
            if self.min_distance:
                if hasattr(self.clustering_method, 'fp_calculator'):
                    fp_calculator = self.clustering_method.fp_calculator
                    self._compute_min_interset_Tanimoto_distances(split_name, mintd_name, fp_calculator)
                else:
                    self._compute_min_interset_Tanimoto_distances(split_name, mintd_name)

            # # Print balance and chemical bias metrics
            # self._print_metrics(self.df)

            # Update clustering seed
            if self.n_splits > 1:
                self.clustering_method.seed += 1


        # Drop the tasks for balancing
        cols2drop = [col for col in self.tasks_for_balancing if col not in self.original_tasks]
        self.df.drop(cols2drop, axis=1, inplace=True)


        logger.info('=' * 80)

        return self.df
    
    def get_default_time_limit_seconds(self, nmols : int, ntasks : int) -> int:
        """
        Default time limit in seconds for the optimization problem.

        tlim = nmol^(1/3) * sqrt(ntask) and tlim >= 10s and tlim <= 1h

        Parameters
        ----------
        nmols : int
            Number of molecules.
        ntasks : int
            Number of tasks.
        
        Returns
        -------
        int
            Time limit in seconds.
        """
        
        tmol = nmols ** (1/3)
        ttarget = np.sqrt(ntasks)
        tmin = 10
        tmax = 60 * 60
        tlim = min(tmax, max(tmin, tmol * ttarget)) 
        logger.info(f'Time limit: {int(tlim)}s')
        return tlim


    def _compute_tasks_per_cluster(
            self, 
            tasks : list[str],
            clusters : dict[int, list[int]]
        ) -> dict[int, list[str]]:
        
        """
        Compute the number of datapoints per task for each cluster.

        Parameters
        ----------
        tasks : list[str]
            list of tasks
        clusters : dict[int, list[int]]
            dictionary of clusters and list of indices of molecules in the cluster

        Returns
        -------
        np.array of shape (len(tasks)+1, len(clusters))
            Array with each columns correspoding to a cluster and each row to a task
            plus the 1st row for the number of molecules per cluster
        """

        task_vs_clusters = np.zeros((len(tasks)+1, len(clusters)))
        task_vs_clusters[0,:] = [ len(cluster) for cluster in clusters.values() ]

        for i, task in enumerate(tasks):
            for j, indices_per_cluster in clusters.items():
                self.df_per_cluster = self.df.iloc[indices_per_cluster]
                task_vs_clusters[i+1,j] = self.df_per_cluster[task].dropna().shape[0]

        return task_vs_clusters  
    
    # Bash command to delete conda environment
    # conda env remove --name myenv

        
    def _set_original_tasks(self, tasks : list[str] | None) -> None:
        """
        Set the original tasks.
        
        Parameters
        ----------
        tasks : list[str] | None
            list of task columns.
        """

        if tasks is None:
            columns = self.df.columns.tolist()
            # Drop the SMILES column.
            columns.remove(self.smiles_column)
            # Drop the ignore columns.
            if self.ignore_columns is not None:
                columns = [column for column in columns if column not in self.ignore_columns]
            # Drop columns starting with 'Split' and 'MinIntersetDistance'.
            columns = [column for column in columns if not column.startswith('Split')]
            columns = [column for column in columns if not column.startswith('MinIntersetDistance')]
            # Set original tasks.
            self.original_tasks = columns
        else:
            self.original_tasks = tasks

    def _set_tasks_for_balancing(self) -> None:

        """
        Set the tasks for balancing. 

        If stratify is True, the tasks are used to create a column for each class (in case of classification tasks) and
        bin the data (in case of regression tasks). If stratify is False, the tasks are used as is.
        """

        def is_convertible(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        self.tasks_for_balancing = []
        if self.stratify:
            for task in self.original_tasks:
                values = self.df[task].dropna().unique()
                values_is_numerical = [is_convertible(value) for value in values]
                
                # Check if contains non-convertible strings
                if not (all(values_is_numerical)): # Some non numerical values

                    if any(values_is_numerical): # But some values are numerical
                        raise ValueError(f'Column {task} contains both numerical and strings values.')
                    else:
                        # Create a task per string
                        for string in values:
                            key = f'{task}_{string}'
                            self.df[key] = (self.df[task] == string).map({True: 1, False: np.nan})
                            self.tasks_for_balancing.append(key)
                        logger.info(f'String classification {task} stratified into {len(self.df[task].dropna().unique())} tasks.')
                # Classification: task per class
                elif all( x % 1 == 0 for x in values):
                    for cls in values:
                        key = f'{task}_{int(cls)}'
                        self.df[key] = (self.df[task] == cls).map({True: 1, False: np.nan})
                        self.tasks_for_balancing.append(key)
                    logger.info(f'Classification task {task} stratified into {len(self.df[task].dropna().unique())} tasks.')
                # Regression: bin data and use as tasks
                else:
                    sorted_values = np.sort(values)
                    bins = np.array_split(sorted_values, self.stratify_reg_nbins)
                    for i, bin in enumerate(bins):
                        key = f'{task}_{bin[0]:.2f}_{bin[-1]:.2f}'
                        self.df[key] = self.df[task].apply(lambda x: x if x in bin else np.nan)
                        self.tasks_for_balancing.append(key)
                    logger.info(f'Regression task {task} stratified into {self.stratify_reg_nbins} tasks.')
        else:
            self.tasks_for_balancing = self.original_tasks

    def _get_preassigned_clusters(
            self, 
            clusters : dict[int, list[int]]) -> dict[int, int]:

        """
        Get the pre-assigned clusters.

        Parameters
        ----------
        clusters : dict[int, list[int]]
            dictionary of clusters and list of indices of molecules in the clusters.
        
        Returns
        -------
        dict[int, int]
            dictionary with cluster indices as keys and subset indices as values.
        """

        preassigned_clusters = {}
        for smi, subset in self.preassigned_smiles.items():
            imol = self.df[self.df[self.smiles_column] == smi].index[0]
            for idx, cluster in clusters.items():
                if imol in cluster:
                    if (idx in preassigned_clusters.keys()) and (subset != preassigned_clusters[idx]):
                        raise ValueError(f'Pre-assigned cluster {idx} is assigned to multiple subsets.')
                    preassigned_clusters[idx] = subset
                    logger.info(f'Cluster {idx} contaning {smi} is preassigned to subset {subset}.')
                    break

        preassigned_clusters
        
        return preassigned_clusters

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

    def _compute_min_interset_Tanimoto_distances(
            self, 
            split_col = 'Split',
            mTd_col = 'minInterSetTd',
            fp_calculator = GetMorganGenerator(radius=3, fpSize=2048)
            ):
        """
        Compute the minimum Tanimoto distance per compound to the compounds in the other subsets.
        
        Parameters
        ----------
        split_col : str, optional
            Name of the column containing the subset number, by default 'Split'
        mTd_col : str, optional
            Name of the column to store the minimum Tanimoto distance in, by default 'minInterSetTd'
        fp_calculator : rdkit.Chem.rdMolDescriptors.GetMorganGenerator, optional
            Fingerprint calculator, by default GetMorganGenerator(radius=3, nBits=2048)
        """

        # Print header
        txt = f' Min. inter-set Tanimoto distance '
        n = int((80 - len(txt)) / 2)
        logger.info('-' * n + txt + '-' * n) 

        # Compute fingerprints
        mols = [ Chem.MolFromSmiles(smi) for smi in self.df[self.smiles_column].tolist() ]
        fps = [ fp_calculator.GetFingerprint(mol) for mol in mols ]

        min_distances = np.zeros(len(fps))
        # Iterate over subsets and compute minimum Tanimoto distance per compound 
        # to the compounds in the other subsets
        for j in self.df[split_col].unique():
            ref_idx = self.df[self.df[split_col] == j].index.tolist()
            other_fps = [ fp for i, fp in enumerate(fps) if i not in ref_idx ]
            for i in ref_idx :
                sims = DataStructs.BulkTanimotoSimilarity(fps[i], other_fps)
                min_distances[i] = 1. - max(sims)

        self.df[mTd_col] = min_distances

        # Print average and std  of minimum distances per subset
        for subset in sorted(self.df[split_col].unique()):
            dist = self.df[self.df[split_col] == subset][mTd_col] #.to_numpy()
            logger.info(f'Subset {int(subset)}: {dist.mean():.2f} +/- {dist.std():.2f} | {dist.median():.2f}')

        # Chemical dissimilarity score
        cd_score = self.df.groupby(split_col)[mTd_col].median().min()
        logger.info(f'Chemical dissimilarity score: {cd_score:.2f}')
    
    def _compute_task_balace(self, split_col : str = 'Split'):

        """
        Compute the task balance per subset.
        
        Parameters
        ----------
        split_col : str, optional
            Name of the column containing the subset number, by default 'Split'
        """

        # Header
        txt = f' {split_col} balance '
        n = int((80 - len(txt)) / 2)
        logger.info('-' * n + txt + '-' * n) 


        # 1. Print out for each balancing task the fraction and number of self.df points per subset
        # 2. Print out for each original task the fraction and number of self.df points per subset
        # 3. Print the fraction and number of self.df point per subset for all tasks combined

        # Get name of longets task
        longest_task = max(self.tasks_for_balancing, key=len)   

        # 1. Print out for each balancing task the fraction and number of self.df points per subset
        if self.stratify:
            for task in self.tasks_for_balancing:
                txt = f'{task} balance:'
                txt += ' ' * (len(longest_task) - len(task)) + '\t'
                counts = self.df[[task, split_col]].groupby(split_col).count()
                n = counts[task].sum()
                for subset in sorted(self.df[split_col].unique()):
                    n_subset = counts.loc[subset, task]
                    txt += f' {int(subset)}: {n_subset/n:.2f} [{n_subset}]\t'
                logger.info(txt)
            logger.info('')

        # 2. Print out for each original task the fraction and number of self.df points per subset
        for task in self.original_tasks:
            txt = f'{task} balance:'
            txt += ' ' * (len(longest_task) - len(task)) + '\t'
            counts = self.df[[task, split_col]].groupby(split_col).count()
            n = counts[task].sum()
            for subset in sorted(self.df[split_col].unique()):
                n_subset = counts.loc[subset, task]
                txt += f' {int(subset)}: {n_subset/n:.2f} [{n_subset}]\t'
            logger.info(txt)
        logger.info('')

        # 3. Print the fraction and number of self.df point per subset for all tasks combined
        txt = f'Overall balance:' + ' ' * (len(longest_task) - len(task)) + '\t'
        n = self.df.shape[0]
        balance_score = 0
        subsets = sorted(self.df[split_col].unique())
        for i, subset in enumerate(subsets):
            n_subset = self.df[self.df[split_col] == subset].shape[0]
            txt += f' {int(subset)}: {n_subset/n:.2f} [{n_subset}]\t'
            balance_score += np.abs(n_subset/n - self.sizes[i])
        logger.info(txt)
        logger.info(f'Balance score: {balance_score/len(self.sizes):.4f}')