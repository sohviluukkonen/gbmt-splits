import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from typing import List

def compute_target_balance(df : pd.DataFrame, targets : List[str], subset_col : str ='Subset'):
    """
    Compute balance of subsets for each target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a column per target and a column with the subset number.
    targets : List[str]
        List of target columns.
    subset_col : str, optional
        Name of the column with the subset number, by default 'Subset'
    
    Returns
    -------
    pd.DataFrame of shape (len(targets), len(subsets)+1)
        Dataframe with a row per target and columns containing the fraction of compounds in each subset.
    """

    subsets = sorted(df[subset_col].unique())
    df_balance = pd.DataFrame(columns=['Task']+subsets, index=range(len(targets))) 

    for i, target in enumerate(targets):
        df_target = df[[target, subset_col]].dropna()
        df_balance.loc[i, 'Task'] = target
        ntarget = len(df_target)
        for subset in subsets:
            df_balance.loc[i, subset] = len(df_target[df_target[subset_col] == subset]) / ntarget

    return df_balance

class PlottingSingleDataset():

    """Class to plot split/subset properties for a single dataset."""

    def __init__(self, data : pd.DataFrame, smiles_col : str = 'SMILES', split : str = None, dataset : str = None, targets : List[int] = None, colors : List[str] = None):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with each row containing a compound and a column per target and a column with the subset number.
        smiles_col : str, optional
            Name of the column with the SMILES, by default 'SMILES'
        split : str, optional
            Name of the split, by default None
        dataset : str, optional
            Name of the dataset, by default None
        targets : List[int], optional
            List of target columns, by default None
        colors : List[str], optional
            List of colors for the subsets, by default None
        """

        self.data = data
        self.smiles_col = smiles_col
        self.split = split
        self.dataset = dataset
        self.subset_col = 'Subset'
        self.targets = targets if targets is not None else [ col for col in data.columns if col not in [smiles_col, 'Subset', 'Dataset', 'Split', 'MinInterSetTd'] ]
        self.colors = colors if colors is not None else sns.color_palette('colorblind')

        self.title_suffix = ''
        if self.dataset : self.title_suffix += f' - ({self.dataset})'
        if self.split : self.title_suffix += f' - ({self.split})'

    def plot_pca(self, figsize : tuple = (4, 4), fname : str = None):
        """
        Plots PCA of the dataset with a different color for each subset.
        
        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure, by default (4, 4)
        fname : str, optional
            Name of the file to save the figure, by default None
        
        
        Returns
        -------
        fig, ax
            Figure and axis objects.
        """
        fps = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 3, nBits=2048) for s in self.data[self.smiles_col]])
        pca = PCA(n_components=2)
        x = pca.fit_transform(fps)
        
        fig, ax = plt.subplots(1, figsize=figsize)
        for i, subset in enumerate(sorted(self.data[self.subset_col].unique())):
            ax.scatter(x[self.data[self.subset_col] == subset, 0], x[self.data[self.subset_col] == subset, 1], label=subset, alpha=0.5, color=self.colors[i], marker='.')
        
        ax.legend(frameon=False)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'PCA {self.title_suffix}')

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches='tight')

        return fig, ax
    
    def plot_tsne(self, figsize : tuple = (4, 4), fname : str = None):
        """
        Plots t-SNE of the dataset with a different color for each subset.
        
        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure, by default (4, 4)
        fname : str, optional
            Name of the file to save the figure, by default None
        
        Returns
        -------
        fig, ax
            Figure and axis objects.
        """
        fps = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 3, nBits=2048) for s in self.data[self.smiles_col]])
        tsne = TSNE(n_components=2)
        x = tsne.fit_transform(fps)
        
        fig, ax = plt.subplots(1, figsize=figsize)
        for i, subset in enumerate(sorted(self.data[self.subset_col].unique())):
            ax.scatter(x[self.data[self.subset_col] == subset, 0], x[self.data[self.subset_col] == subset, 1], label=subset, alpha=0.5, color=self.colors[i], marker='.')
        
        ax.legend(frameon=False)
        ax.set_xlabel('t-SNE1')
        ax.set_ylabel('t-SNE2')
        ax.set_title(f't-SNE {self.title_suffix}')

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches='tight')

        return fig, ax
    
    def plot_target_balance_bars(self, figsize : tuple = None, ideal_sizes : List[float] = None, fname : str = None, **kwargs):
        """
        Plots the fraction of compounds per subset for each target as stacked bars.
        
        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure, by default None
        ideal_sizes : List[float], optional
            List of ideal fractions for each subset, by default None
        fname : str, optional
            Name of the file to save the figure, by default None
        
        Returns
        -------
        fig, ax
            Figure and axis objects.
        """
        df_balance = compute_target_balance(self.data, self.targets, subset_col=self.subset_col)

        if figsize is None: figsize = ( len(self.targets)/4, 4)

        fig, ax = plt.subplots(1, figsize=figsize)
        df_balance.plot.bar(x='Task', ax=ax, stacked=True, color=self.colors, width=0.8)

        if ideal_sizes is not None:
            cum = 0
            for i, size in enumerate(ideal_sizes[:-1]):
                cum += size	    
                ax.axhline(cum, color='k', linestyle='--')

        ax.set_title(f'Target balance {self.title_suffix}')
        ax.set_ylabel('Fraction of compounds')
        ax.set_xlabel('Target')
        ax.legend(frameon=False, bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1)

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_target_balance_distributions(self, ideal_sizes : List[float], figsize : tuple = None, plot_function : callable = sns.boxplot, fname : str = None, **kwargs):
        """
        Plots the distribution of the fraction of compounds per subset for each target.
        
        Parameters
        ----------
        ideal_sizes : List[float]
            List of ideal fractions for each subset
        figsize : tuple, optional
            Size of the figure, by default None
        plot_function : callable, optional
            Function to use for plotting, by default sns.boxplot
        fname : str, optional
            Name of the file to save the figure, by default None
        
        Returns
        -------
        fig, ax
            Figure and axis objects.
        """

        subsets = sorted(self.data[self.subset_col].unique())
        if len(ideal_sizes) != len(subsets):
            raise ValueError('The number of ideal sizes must be equal to the number of subsets.')
        
        df_balance = compute_target_balance(self.data, self.targets, subset_col=self.subset_col)
        
        df_balance_diff = pd.DataFrame(columns=['Subset', 'Task', 'dFraction'])
        for task in df_balance['Task'].unique():
            for subset, size in zip(subsets, ideal_sizes):
                df_balance_diff = pd.concat([df_balance_diff,
                                             pd.DataFrame({
                                                    'Subset': [subset],
                                                    'Task': [task],
                                                    'dFraction': [df_balance[df_balance['Task'] == task][subset].values[0] - size]
                                                    })
                                                ],
                                                ignore_index=True
                                            )
                
        if figsize is None: figsize = ( len(self.targets)/4, 4)

        fig, ax = plt.subplots(1, figsize=figsize)
        plot_function(x='Subset', y='dFraction', data=df_balance_diff, ax=ax, palette=self.colors)

        ax.axhline(0, color='k', linestyle='--')

        ax.set_title(f'Target balance {self.title_suffix}')
        ax.set_ylabel('dFraction (observed - ideal)')
        ax.set_xlabel('Subset')
        ax.set_ylim(-0.05, 0.05)

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches='tight')
        
        return fig, ax

    def plot_min_tanimoto_distance_distributions(self, figsize : tuple = None, plot_function : callable = sns.boxplot, fname : str = None, **kwargs):
        """
        Plots the distribution of the minimum Tanimoto distance between compounds in different subsets.

        
        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure, by default None
        plot_function : callable, optional
            Function to use for plotting, by default sns.boxplot
        fname : str, optional
            Name of the file to save the figure, by default None
        
        Returns
        -------
        fig, ax
            Figure and axis objects.
        """

        if figsize is None: figsize = (self.data[self.subset_col].nunique(), 4)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        plot_function(x=self.subset_col, y='MinInterSetTd', data=self.data, ax=ax, palette=self.colors)

        ax.set_title(f'Min. interset Tanimoto dist. {self.title_suffix}')
        ax.set_ylabel('Min. interset Tanimoto dist.')
        ax.set_xlabel('Subset')
        ax.set_ylim(0, 1)

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches='tight')

        return fig, ax

    def plot_all(self, ideal_subset_sizes : List[float] = None):
        """Plot all plots."""
        self.plot_pca()
        self.plot_tsne()
        self.plot_target_balance_bars(ideal_sizes=ideal_subset_sizes)
        self.plot_target_balance_distributions(ideal_subset_sizes)
        self.plot_min_tanimoto_distance_distributions()

class PlottingCompareDatasets():
    """
    Class for plotting properties of splits/subsets of multple datasets.
    """

    def __init__(self, data : pd.DataFrame, compare_col : str, dataset_names : List[str] = None, smiles_col : str = 'SMILES', targets : List[int] = None, colors : List[str] = None):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing the data.
        compare_col : str
            Column containing the name of the different datasets.
        dataset_names : List[str], optional
            List of names of the datasets. If None, all unique values in compare_col are used, by default None
        smiles_col : str, optional
            Name of the column containing the SMILES, by default 'SMILES'
        targets : List[int], optional
            List of targets. If None, all columns except smiles_col, compare_col, 'Subset', 'Dataset', 'Split', 'MinInterSetTd' are used, by default None
        colors : List[str], optional
            List of colors, by default None
        """
        self.data = data
        self.compare_col = compare_col
        self.dataset_names = dataset_names if dataset_names is not None else sorted(data[compare_col].unique())
        self.smiles_col = smiles_col
        self.subset_col = 'Subset'
        self.targets = targets if targets is not None else [ col for col in data.columns if col not in [smiles_col, compare_col, 'Subset', 'Dataset', 'Split', 'MinInterSetTd'] ]
        self.colors = colors if colors is not None else sns.color_palette('colorblind')

    def plot_pca(self, figsize : tuple = None, fname : str = None, **kwargs):
        """
        Plots a PCA of each dataset in subfigures with a different color for each subset.
        
        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure, by default (4, 4)
        fname : str, optional
            Name of the file to save the figure, by default None
        
        Returns
        -------
        fig, ax
            Figure and axis objects.
        """
        
        fps = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 3, nBits=2048) for s in self.data[self.smiles_col]])
        
        # Fit PCA on unique compounds
        pca = PCA(n_components=2)
        unique_idx = self.data[self.smiles_col].drop_duplicates().index
        pca.fit(fps[unique_idx])

        # Transform all compounds
        x = pca.fit_transform(fps)

        if figsize is None: figsize = (len(self.dataset_names)*4, 4)
        
        fig, ax = plt.subplots(1, len(self.dataset_names), figsize=figsize)
        for i, dataset in enumerate(self.dataset_names):
            for j, subset in enumerate(sorted(self.data.Subset.unique())):
                ax[i].scatter(x[(self.data[self.compare_col] == dataset) & (self.data.Subset == subset), 0], 
                              x[(self.data[self.compare_col] == dataset) & (self.data.Subset == subset), 1], 
                              label=subset, alpha=0.5, color=self.colors[j], marker='.')
                ax[i].set_xlabel('PC1')
                ax[i].set_title(dataset)  
                ax[i].legend(frameon=False)
                
        ax[0].set_ylabel('PC2')  

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches='tight')

        return fig, ax
    
    def plot_tsne(self, figsize : tuple = None, fname : str = None, **kwargs):
        """
        Plots a T-SNE of each dataset in subfigures with a different color for each subset.
        
        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure, by default (4, 4)
        fname : str, optional
            Name of the file to save the figure, by default None
        
        Returns
        -------
        fig, ax
            Figure and axis objects.
        """

        fps = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 3, nBits=2048) for s in self.data[self.smiles_col]])

        # Fit t-SNE on unique compounds
        tsne = TSNE(n_components=2)
        unique_idx = self.data[self.smiles_col].drop_duplicates().index
        tsne.fit(fps[unique_idx])

        # Transform all compounds
        x = tsne.fit_transform(fps)

        if figsize is None: figsize = (len(self.dataset_names)*4, 4)

        fig, ax = plt.subplots(1, len(self.dataset_names), figsize=figsize)
        for i, dataset in enumerate(self.dataset_names):
            for j, subset in enumerate(sorted(self.data.Subset.unique())):
                ax[i].scatter(x[(self.data[self.compare_col] == dataset) & (self.data.Subset == subset), 0], 
                              x[(self.data[self.compare_col] == dataset) & (self.data.Subset == subset), 1], 
                              label=subset, alpha=0.5, color=self.colors[j], marker='.')
                ax[i].set_xlabel('t-SNE 1')
                ax[i].set_title(dataset)  
                ax[i].legend(frameon=False)

        ax[0].set_ylabel('t-SNE 2')

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches='tight')

        return fig, ax
    
    def plot_target_balance_bars(self, figsize : tuple = None, ideal_sizes : List[float] = None, drop_task_names : bool = False, fname : str = None, **kwargs):
        """
        Plots the fraction of compound per subset for each target of each dataset in subfigures as stacked bars.
        
        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure, by default None
        ideal_sizes : List[float], optional
            List of ideal sizes for each subset, by default None
        fname : str, optional
            Name of the file to save the figure, by default None
        
        Returns
        -------
        fig, ax
            Figure and axis objects.
        """

        if figsize is None: figsize = (len(self.dataset_names)*4, 4)

        fig, ax = plt.subplots(1, len(self.dataset_names), figsize=figsize, sharey=True, sharex=True)
        for i, dataset in enumerate(self.dataset_names):

            df_balance = compute_target_balance(self.data[self.data[self.compare_col] == dataset], self.targets, self.subset_col)
            df_balance.plot.bar(x='Task', ax=ax[i], stacked=True, color=self.colors, width=0.8, legend=False)

            if ideal_sizes is not None:
                cum = 0
                for size in ideal_sizes[:-1]:
                    cum += size	    
                    ax[i].axhline(cum, color='k', linestyle='--')
            
            ax[i].set_title(dataset)
            if drop_task_names:
                ax[i].set_xticklabels([])

        ax[0].set_ylabel('Fraction of compounds')
        ax[0].set_ylim(0, 1)

        # Add legend only to the last plot
        handles, labels = ax[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(self.data.Subset.unique()), frameon=False)

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches='tight')

        return fig, ax 
    
    def plot_target_balance_distributions(self, ideal_sizes : List[float], figsize : tuple = None, plot_function : callable = sns.boxplot, fname : str = None, **kwargs):

        """ 
        Plots the distribution of the fraction of compounds per subset for each target of each dataset.
        
        Parameters
        ----------
        ideal_sizes : List[float]
            List of ideal sizes for each subset.
        figsize : tuple, optional
            Size of the figure, by default None
        plot_function : callable, optional
            Function to use for plotting, by default sns.boxplot
        fname : str, optional
            Name of the file to save the figure, by default None
        
        Returns
        -------
        fig, ax
            Figure and axis objects.
        """

        subsets = sorted(self.data[self.subset_col].unique())
        if len(ideal_sizes) != len(subsets):
            raise ValueError('The number of ideal sizes must be equal to the number of subsets.')

        if figsize is None: figsize = (self.data[self.subset_col].nunique() * len(self.dataset_names), 4)
        
        df_balance_diff = pd.DataFrame(columns=[self.compare_col, 'Task', 'Subset', 'dFraction'])
        for dataset in self.dataset_names:
            df_balance = compute_target_balance(self.data[self.data[self.compare_col] == dataset], self.targets, self.subset_col)
            for task in df_balance['Task'].unique():
                for subset, size in zip(subsets, ideal_sizes):
                    df_balance_diff = pd.concat([df_balance_diff,
                                                pd.DataFrame({
                                                        self.compare_col: [dataset],
                                                        'Subset': [subset],
                                                        'Task': [task],
                                                        'dFraction': [df_balance[df_balance['Task'] == task][subset].values[0] - size]
                                                        })
                                                    ],
                                                    ignore_index=True
                                                )
                    
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_function(x=self.subset_col, y='dFraction', hue=self.compare_col, hue_order=self.dataset_names, data=df_balance_diff, ax=ax, palette=self.colors, **kwargs)

        ax.axhline(0, color='k', linestyle='--')
        ax.set_ylabel('dFraction (observed - ideal)')
        ax.set_xlabel('Subset')
        ax.set_ylim(-0.05, 0.05)


        handles, labels = ax.get_legend_handles_labels()
        ax.legend([],[], frameon=False)
        fig.legend(handles, labels, loc='upper center', ncol=len(self.data.Subset.unique()), frameon=False)

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches='tight')


    
    def plot_min_tanimoto_distance_distributions(self, figsize : tuple = None, plot_function : callable = sns.boxplot, fname : str = None, **kwargs):
        """
        Plots the distribution of the minimum Tanimoto distance between compounds in different subsets.
        
        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure, by default None
        plot_function : callable, optional
            Function to use for plotting, by default sns.boxplot
        fname : str, optional
            Name of the file to save the figure, by default None
        
        Returns
        -------
        fig, ax
            Figure and axis objects.
        """

        if figsize is None: figsize = (self.data[self.subset_col].nunique() * len(self.dataset_names), 4)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        plot_function(x=self.subset_col, y='MinInterSetTd', hue=self.compare_col, hue_order=self.dataset_names, data=self.data, ax=ax, palette=self.colors, **kwargs)

        ax.set_ylabel('Min. interset Tanimoto dist.')
        ax.set_xlabel('Subset')
        ax.set_ylim(0, 1)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend([],[], frameon=False)
        fig.legend(handles, labels, loc='upper center', ncol=len(self.data.Subset.unique()), frameon=False)

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches='tight')

        return fig, ax

    def plot_all(self, ideal_subset_sizes : List[float] = None):
        """Plot all plots."""
        self.plot_pca()
        self.plot_tsne()
        self.plot_target_balance_bars(ideal_sizes=ideal_subset_sizes)
        self.plot_target_balance_distributions(ideal_sizes=ideal_subset_sizes)
        self.plot_min_tanimoto_distance_distributions()