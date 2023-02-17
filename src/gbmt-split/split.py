import tqdm
import argparse

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.SimDivFilters import rdSimDivPickers

from typing import List, Dict
from timeit import default_timer as timer

from pulp import *


class GloballyBalancedSplit(ABC):

    """ 
    Class to create well-balanced splits for multi-task learning with data-leakage between the tasks.

    Molecules are initially clustered (randomly or based on a similarity measure) and then the clusters 
    are combined with a linear programming method to ensure that the number of datapoints per task is 
    balanced in each split and no molecule is present in more than one split.
    
    The method ensures that the number of datapoints per task is balanced in each split and no molecule is present in more than one split.
    In addition, the method can be used to ensure that the compounds in each split are chemically dissimilar.

    Paper: Tricarico et al., Construction of balanced, chemically dissimilar training, validation
    and test sets for machine learning on molecular datasets, 2022, doi: https://doi.org/10.26434/chemrxiv-2022-m8l33-v2
    """

    def __call__(self, 
            data : pd.DataFrame,
            smiles_column : str = 'SMILES',
            targets : List[str] = None,
            ignore_columns : List[str] = None,
            sizes : List[int] = [0.8, 0.1, 0.1],
            min_distance : bool = False,
            equal_weight_perc_compounds_as_tasks = False,
            relative_gap = 0,
            time_limit_seconds = 60,
            n_jobs = 1        
            ) -> pd.DataFrame:
        
        """
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with the data
        smiles_column : str, optional
            Name of the column with the SMILES, by default 'SMILES'
        targets : List[str], optional
            List of targets, by default None. If None, all columns except the smiles_column and the ignore_columns are used.
        ignore_columns : List[str], optional
            List of columns to ignore, by default None
        sizes : List[int], optional
            List of sizes for the splits, by default [0.8, 0.1, 0.1]
        min_distance : bool, optional
            If True, the minimum inter-subset Tanimoto distance is computed, by default False
        equal_weight_perc_compounds_as_tasks : bool, optional
            If True, the percentage of compounds per task is used as weight for the balancing, by default False
        relative_gap : float, optional
            Relative gap for the linear programming, by default 0
        time_limit_seconds : int, optional
            Time limit for the linear programming, by default 60
        n_jobs : int, optional
            Number of jobs for the linear programming, by default 1

        Returns
        -------
        pd.DataFrame
            DataFrame with the data and a column 'Subset' with the split assignment (and optionally 'MinInterSetTd' the minimum inter-subset Tanimoto distance).
        """

        # Get the list of SMILES
        smiles_list = data[smiles_column].tolist()
        
        # Get the list of targets
        if targets is None:
            targets = [c for c in data.columns if c != smiles_column and c not in ignore_columns]


        # Check for each target if all values are integers (i.e. classification) and create a seperate target for each class
        original_targets = targets
        targets = []
        for target in original_targets:
            if all(x.is_integer() for x in data[target].dropna()):
                print(f'{target} seems to be a classification labels. A separate target is created for each class for the balancing.')
                for c in data[target].dropna().unique():
                    data[target + '_' + str(c)] = (data[target] == c).map({True: 1, False: np.nan})
                    targets.append(target + '_' + str(c))
            else:
                targets.append(target)
        if set(original_targets) != set(targets):
            print('New targets for the balancing are', targets)


        # Cluster the molecules
        clusters = self.clustering(smiles_list)    
        # Compute the number of datapoints per task for each cluster
        tasks_per_cluster = self.computeTasksPerCluster(data, targets, clusters)
        # Merge the clusters with a linear programming method
        merged_clusters_mapping = self.mergeClustersWithBalancingMapping(tasks_per_cluster, sizes, equal_weight_perc_compounds_as_tasks, 
                                                                         relative_gap, time_limit_seconds, n_jobs)
        for i, idx in clusters.items(): data.loc[idx, 'Subset'] = merged_clusters_mapping[i]

        self.printBalanceMetrics(data, targets)
            
        # Compute the minimum inter-subset Tanimoto distance
        if min_distance: self.computeInterSubsetMinimumTanimotoDistance(data, smiles_column)
                
        return data        

    
    @abstractmethod
    def clustering(self, smiles_list : List[str], **kwargs) -> Dict[int, List[int]]:
        """ 
        Cluster the molecules.
        """
        pass

    def getMorganFingerprints(self, smiles_list : List[str], **kwargs) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
        """
        Compute the Morgan fingerprints for the molecules.
        """
        return [rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), **kwargs) for s in smiles_list]
    
    def computeTasksPerCluster(self, 
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
    
    def mergeClustersWithBalancingMapping(self, tasks_vs_clusters_array : np.array,
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
    
    def printBalanceMetrics(self, data : pd.DataFrame, targets : List[str]):
        """ 
        Print the balance metrics for the given subsets and targets.
        """
            
        txt = 'Overall balance:'
        for subset in sorted(data['Subset'].unique()):
            n = len(data[data['Subset'] == subset])
            frac = n/ len(data)
            txt += f' {subset}: {n} ({frac:05.2%})'
        print(txt)
        
        for target in targets:
            txt = f'{target} balance:'
            df = data.dropna(subset=[target])
            for subset in sorted(data['Subset'].unique()):
                n = len(df[df['Subset'] == subset])
                frac = n / len(df)
                txt += f' {subset}: {n} ({frac:05.2%})'
            print(txt)

    def computeInterSubsetMinimumTanimotoDistance(self, df, smiles_column='SMILES'):
        """
        Compute the minimum Tanimoto distance per compound to the compounds in the other subsets.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing the data with a column 'Subset' containing the split
        smiles_column : str, optional
            Name of the column containing the SMILES, by default 'SMILES'
        
        Returns
        -------
        pd.DataFrame
            Dataframe containing the data with a column 'MinInterSetTd' containing the minimum Tanimoto distance

        """
        n = len(df)

        # Compute Morgan Fingerprints
        fps = self.getMorganFingerprints(df[smiles_column].tolist(), radius=3, nBits=2048)

        min_distances = np.zeros(n)
        # Iterate over subsets and compute minimum Tanimoto distance per compound to the compounds in the other subsets
        for j in df.Subset.unique():
            ref_idx = df[df.Subset == j].index.tolist()
            other_fps = [ fp for i, fp in enumerate(fps) if i not in ref_idx ]
            for i in tqdm.tqdm(ref_idx, total=len(ref_idx), desc=f'Computing minimum Tanimoto distance for subset {j}') :
                sims = DataStructs.BulkTanimotoSimilarity(fps[i], other_fps)
                min_distances[i] = 1. - max(sims)

        df['MinInterSetTd'] = min_distances

        # Print average and std  of minimum distances per subset
        txt = 'Average and std of minimum Tanimoto distance per subset:'
        for subset in sorted(df['Subset'].unique()):
            dist = df[df['Subset'] == subset]['MinInterSetTd'].to_numpy()
            txt += f' {subset}: {np.mean(dist):.2f} ({np.std(dist):.2f})'
        print(txt)

        return df

class RandomGloballyBalancedSplit(GloballyBalancedSplit):
    """ 
    Randomly split the data into clusters and assign each cluster to a subset by linear programming 
    to ensure balanced subsets for each target.
    """
     
    def __init__(self, n_clusters : int = None, seed : int = 42):
        super().__init__()
        """ 
        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters. If None, the number of clusters is equal to the number of molecules divided by 100, by default None
        seed : int, optional
            Random seed, by default 42
        """
        self.n_clusters = n_clusters
        self.seed = seed

        np.random.seed(self.seed)

    def clustering(self, smiles_list : List[str]) -> Dict[int, List[int]]:
        """
        Create random clusters of molecules.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings
        n_clusters : int, optional
            Number of clusters. If None, the number of clusters is equal to the number of molecules divided by 100, by default None
        seed : int, optional
            Random seed, by default 42
        
        Returns
        -------
        Dict[List[int]]
            Dictionary of clusters and list of indices of molecules in the cluster
        """

        if self.n_clusters is None:
            self.n_clusters = len(smiles_list) // 100
        clusters = { i: [] for i in range(self.n_clusters) }

        cluster_thresholds = np.linspace(0, 1, self.n_clusters + 1)
        random_values = np.random.rand(len(smiles_list))

        for i, threshold in enumerate(cluster_thresholds[:-1]):
            indices = np.where((random_values >= threshold) & (random_values < cluster_thresholds[i+1]))[0]
            clusters[i] = indices.tolist()

        print('Molecules were randomly clustered into {} clusters.'.format(len(clusters)))

        return clusters
        
class DissimilarityDrivenGloballyBalancedSplit(GloballyBalancedSplit):
    """ 
    Splits the data into clusters with Sayle's clustering algorithm and assign each cluster to a subset by linear programming 
    to ensure balanced subsets for each target and dissimilarity between subsets.
    """
     
    def __init__(self, similarity_threshold : float = 0.736):
        super().__init__()
        """
        Parameters
        ----------
        similarity_threshold : float, optional
            Minimum distance between cluster centers, by default 0.736
        """
        self.similarity_threshold = similarity_threshold
    
    def clustering(self, smiles_list) -> Dict[int, List[int]]:
        """
        Clustering of molecules based on fingerprint similarity with selection of cluster centers
        based on the Sayle algorithm.

        Source: https://github.com/rdkit/UGM_2019/raw/master/Presentations/Sayle_Clustering.pdf
        
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings
        fingerprint_calculator : rdkit.Chem.rdMolDescriptors.MolFingerprint, optional
            Fingerprint calculator, by default MorganFingerprint (radius=3, nBits=2048)
        threshold : float, optional
            Minimum distance between cluster centers, by default 0.736

        Returns
        -------
        Dict[List[int]]
            Dictionary of clusters and list of indices of molecules in the cluster
        """

        fps = self.getMorganFingerprints(smiles_list, radius=3, nBits=2048)

        # Get the cluster centers
        print("Pick cluster centers with Sayle's algorithm...")
        lead_picker = rdSimDivPickers.LeaderPicker()
        centroids_indices = lead_picker.LazyBitVectorPick(fps, len(fps), self.similarity_threshold)
        clusters = { i: [centroid_idx] for i, centroid_idx in enumerate(centroids_indices) }

        # Calculate the Tanimoto similarities between the cluster centers 
        # and the other points
        print('Calculating Tanimoto similarities between cluster centers and other points...')
        sims = np.zeros((len(centroids_indices),len(fps)))
        for i, centroid_idx in enumerate(centroids_indices):
            sims[i,:] = DataStructs.BulkTanimotoSimilarity(fps[centroid_idx],fps)
            # sims[i,i] = 0

        # Assign the points to clusters
        print('Assigning points to clusters...')
        best_cluster = np.argmax(sims,axis=0)
        for i, idx in enumerate(best_cluster):
            if i not in centroids_indices:
                clusters[idx].append(i)

        print('Molecules were clustered based on dissimilarity into {} clusters.'.format(len(clusters)))

        return clusters
    
def CommandLineInterface():

    # Parse command line arguments ###############################
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input file with the data in a pivoted csv/tsv format. A column with the SMILES must be provided and each target must be in a separate column.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file with the data with additional columns the assigned subset (and Minimum interset Tanimoto distance).')
    parser.add_argument('-sc','--smiles_column', type=str, default='SMILES',
                        help='Name of the column with the SMILES')
    parser.add_argument('-tc','--target_columns', type=str, nargs='+', default=None,
                        help="Name of the columns with the targets. If not provided, all columns except the SMILES and --ignore_columns' columns will be used")
    parser.add_argument('-ic','--ignore_columns', type=str, nargs='+', default=None,
                        help='Name of the columns to ignore')
    parser.add_argument('-c','--clustering', type=str, default='dissimilarity',
                        help='Clustering algorithm to use. Options: random, dissimilarity')
    parser.add_argument('-nc','--n_clusters', type=int, default=None,
                        help='Number of clusters to use. Only used for random clustering. If None, the number of clusters is equal to the number of molecules divided by 100')
    parser.add_argument('-rs','--random_seed', type=int, default=42,
                        help='Seed for the random clustering')
    parser.add_argument('-ct','--cluster_threshold', type=float, default=0.7,
                        help='Minimum distance between cluster centers. Only used for dissimilarity clustering.')
    parser.add_argument('-s', '--sizes', type=float, nargs='+', default=[0.8, 0.1, 0.1],
                        help='Sizes of the subsets')
    parser.add_argument('-t','--time_limit', type=int, default=60,
                        help='Time limit for linear combination of clusters in seconds')
    parser.add_argument('-mtd', '--min_Tanimoto_distance', action='store_true',
                        help='Compute the minimum Tanimoto distance between a molecule in a subset and the molecules in the other subsets')
    
    # Start the timer
    start_time = timer()
    
    args = parser.parse_args()
    
    # Read input data from csv/tsv file ##########################
    if '.csv' in args.input:
        df = pd.read_csv(args.input)
    elif '.tsv' in args.input:
        df = pd.read_csv(args.input, sep='\t')
    else:
        raise ValueError('Input file must be a csv or tsv file')
    
    if args.target_columns is None and args.ignore_columns is None:
        args.target_columns = [col for col in df.columns if col != args.smiles_column]
    elif args.target_columns is None and args.ignore_columns is not None:
        args.target_columns = [col for col in df.columns if col != args.smiles_column and col not in args.ignore_columns]

    # Setup splitter #############################################
    if args.clustering == 'random':
        splitter = RandomGloballyBalancedSplit(n_clusters=args.n_clusters, seed=args.random_seed)
    elif args.clustering == 'dissimilarity':
        splitter = DissimilarityDrivenGloballyBalancedSplit(similarity_threshold=args.cluster_threshold)
    
    # Split data #################################################
    df = splitter(df, time_limit_seconds=args.time_limit, min_distance=args.min_Tanimoto_distance, sizes=args.sizes, targets=args.target_columns, smiles_column=args.smiles_column)

    # Write output ###############################################
    
    if not args.output:
        args.output = args.input.split('.')[0] 
    
    # Add clustering method to output file name
    args.output += '_RGBS.' if args.clustering == 'random' else '_DGBS.'
    # Use same extension  and compression as input file
    args.output += '.'.join(args.input.split('.')[1:])

    if '.csv' in args.output:
        df.to_csv(args.output, index=False)
    elif '.tsv' in args.output:
        df.to_csv(args.output, sep='\t', index=False)
    
    # Print elapsed time #########################################
    elapsed_time = timer() - start_time
    print('Elapsed time: {:.2f} seconds'.format(elapsed_time))
        
if __name__ == '__main__':

    CommandLineInterface()


