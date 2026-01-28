"""
This script contains a set of prewritten plotting functions to create all of
the figures in the Reclassifying Lethal Heat paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
import seaborn as sn
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from apollo import thermo as th


def wet_bulb_plot(path: str, df: pd.DataFrame, target: str, temp_var: str,
                  humid_var: str, colour1: str, colour2 : str, ratio: int=2,
                  boundaries: list=[20,25,30,35], spectrum: str='flare',
                  dpi: int=144):
    '''
    Generates a scatter plot of lethal and nonlethal heatwaves in terms of
    temperature and humidity with specified wet bulb temperature thresholds.
    
    Parameters
    ----------
    path : str
        Filepath for where the figure will be saved.
    df : pd.DataFrame
        Dataframe of heatwaves along with meteorological variables of concern.
    target : str
        Column name for the lethal/nonlethal heatwave labelling.
    temp_var : str
        Column name for the temperature variable of concern (e.g. maximum).
    humid_var : str
        Column name for the humidity variable (e.g. mean).
    colour1 : str
        Colour to be applied to lethal heatwave scatter points.
    colour2 : str
        Colour to be applied to nonlethal heatwave scatter points.
    ratio : int, optional
        The relative number of nonlethal events to plot. The default is 2.
    boundaries : list, optional
        The wet bulb temperature thresholds to be plotted. The default is
        [20,25,30,35].
    spectrum : str, optional
        The colour spectrum from which the threshold colours are taken. The
        default is 'flare'.
    dpi : int, optional
        The resolution of the saved figure file. The default is 144.

    Returns
    -------
    Saves the output graph to the pathway specified.
    '''
    positive_subset = df[df[target] == 1]
    negative_subset = df[df[target] <= 0].sample(len(positive_subset)*ratio,
                                                 random_state=32)
    wbt_frame = pd.DataFrame(np.linspace(0, 99.9, 1000), columns=['Humidity'])
    colours = []
    cmap = sn.color_palette(spectrum, as_cmap=True)
    c_points = np.linspace(0,1,len(boundaries))
    for point in c_points:
        colours.append(cmap(point))
    colours.reverse()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(positive_subset[temp_var], positive_subset[humid_var],
               marker='o', s=48, c=colour1, label='Lethal')
    ax.scatter(negative_subset[temp_var], negative_subset[humid_var],
               marker='x', s=48, c=colour2, label='Nonlethal')
    for i in range(len(boundaries)):
        b = boundaries[i]
        b_col = str(b) + '°C'
        label = str(b) +'°C WBT'
        wbt_frame[b_col] = wbt_frame['Humidity'].apply(lambda h: th.threshold_wb(h, b))
        ax.plot(wbt_frame[b_col], wbt_frame['Humidity'], lw=4, c=colours[i], label=label)    
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 100])
    ax.set_xlabel('Maximum Temperature (°C)')
    ax.set_ylabel('Relative Humidity (%)')
    ax.legend(loc="lower left")
    plt.savefig(path, dpi=dpi, format='eps')
    plt.show()


def correlation_matrix(path: str, df: pd.DataFrame, features: list, 
                       labels: list, dpi: float=144):
    '''
    Generates a correlation matrix for all input variables with numerical
    values and a colour scale.
    
    Parameters
    ----------
    path : str
        Filepath for where the figure will be saved.
    df : pd.DataFrame
        Dataframe of heatwaves along with input variables.
    features : list
        The input variables for the model.
    labels : list
        Corresponding plotting labels for each of the input variables.
    dpi : float, optional
        The resolution of the saved figure file. The default is 144.

    Returns
    -------
    Saves the output graph to the pathway specified.
    '''
    corr_df = df[features]
    corr_df.columns = labels
    CM = corr_df.corr()
    CM = CM.round(decimals=2)
    mask = np.triu(np.ones_like(CM.corr()), k=1)
    fig, ax = plt.subplots(figsize=(16, 12))
    sn.heatmap(CM, vmin= -0.6, vmax=1.01, annot=True, cmap='flare', mask=mask)
    ax.set_xticklabels(labels, rotation=-45, ha='left')
    plt.savefig(path, dpi=dpi, bbox_inches='tight', format='eps')
    plt.show()


def heat_bubble(path: str, df: pd.DataFrame, target: str, lons: list, lats: list,
                colours: list, scaling: list, num: int, dpi=144):
    '''
    Generates a bubble plot of the total number of heatwaves of a given category
    by location.

    Parameters
    ----------
    path : str
        Filepath for where the figure will be saved.
    df : pd.DataFrame
        Dataframe of heatwave predictions or labels along with latitudinal and
        longitudinal coordinates.
    target : str
        The target category of heatwaves (e.g. Lethal Observations).
    lons : list
        Maximum and minimum longitudinal coordinate pair, to give area extent.
    lats : list
        Maximum and minimum latitudinal coordinate pair, to give area extent.
    colours : list
        Pair of colours for the bubble fill and edge.
    scaling : list
        Pair of numbers for scaling the size of the bubbles in the plot and
        legend.  See matplotlib documentation for more information.
    num : int
        The number of bubble sizes to be included in the legend.
    dpi : TYPE, optional
        The resolution of the saved figure file. The default is 144.

    Returns
    -------
    Saves the output graph to the pathway specified.
    '''
    df = df[df['Longitude'] > lons[0]]
    df = df[df['Longitude'] < lons[1]]
    proj = ccrs.PlateCarree()    
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection':proj})
    f = lambda x: (x**scaling[0])*scaling[1]
    g = lambda y: (y/scaling[1])**(1/scaling[0])
    sc = ax.scatter(df['Longitude'], df['Latitude'], s=f(df[target]),
                    c=colours[0], alpha=0.5, edgecolors=colours[1], lw=2.5)
    ax.set_extent([lons[0], lons[1], lats[0], lats[1]], proj)  
    ax.coastlines(resolution='50m', alpha=0.5)
    gl = ax.gridlines(crs=proj, draw_labels=True, )
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlocator = mtk.MaxNLocator(5)
    gl.ylocator = mtk.MaxNLocator(5)
    gl.top_labels = False
    gl.right_labels = False
    ax.legend(*sc.legend_elements("sizes", num=num, func=g), labelspacing=2)
    plt.savefig(path, dpi=dpi, format='eps')
    plt.show()


def sensitivity_bar(path:str, df: pd.DataFrame, features: list, labels: list,
                    xlabel: str, limits: list, metric: str='F1 Score',
                    spectrum: str='flare', dpi: float=144):
    '''
    Generates bar plots of the sensitivity of the model to each feature,
    according to some sensitivity analysis method.

    Parameters
    ----------
    path : str
        Filepath for where the figure will be saved.
    df : pd.DataFrame
        Data frame containing the input variables and metrics of concern.
    features : list
        The input variables for the model.
    labels : list
        Corresponding plotting labels for each of the input variables.
    xlabel : str
        Name of the x axis.
    limits : list
        Pair of values of the maximum and minimum metric for graph extent.
    metric : str, optional
        Metric of concern. The default is 'F1 Score'.
    spectrum : str, optional
        The colour spectrum from which the bar colours are taken. The default
        is 'flare'.
    dpi : float, optional
        The resolution of the saved figure file. The default is 144.

    Returns
    -------
    Saves the output graph to the pathway specified.
    '''
    colours = []
    cmap = sn.color_palette(spectrum, as_cmap=True)
    c_points = np.linspace(0,1,len(features))
    for point in c_points:
        colours.append(cmap(point))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features, df[metric], color=colours, edgecolor="black", lw=0.75)
    ax.invert_yaxis() 
    ax.axvline(x=0, color='black')
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_xlim(limits)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', format='eps')
    plt.show()


def grid_tuning_plot(path: str, df: pd.DataFrame, hyperparameter: str,
                     metrics: list=['Precision','Recall','F1 Score'],
                     l_styles: list=['-', '--', '-.'], spectrum: str='flare',
                     dpi: int=144):
    '''
    Generates a plot of the evolution of precision, recall, and F1 score over
    varying a given hyperparameter.
    
    Parameters
    ----------
    path : str
        Filepath for where the figure will be saved.
    df : pd.DataFrame
        Data frame containing the input variables and metrics of concern..
    hyperparameter : str
        Name of the hyperparameter being tuned.
    metrics : list, optional
        The three metrics to be plotted. The default is
        ['Precision','Recall','F1 Score'].
    l_styles : list, optional
        List of linestyles for accesibility. The default is ['-', '--', '-.'].
    spectrum : str, optional
        The colour spectrum from which the line colours are taken. The default
        is 'flare'.
    dpi : int, optional
        The resolution of the saved figure file. The default is 144.

    Returns
    -------
    Saves the output graph to the pathway specified.
    '''
    cmap = sn.color_palette(spectrum, as_cmap=True)
    colours = [cmap(0.9), cmap(0.1), cmap(0.5)]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(df[hyperparameter], df[metrics[0]], lw=4, c=colours[0],
            ls=l_styles[0], label=metrics[0])
    ax.plot(df[hyperparameter], df[metrics[1]], lw=4, c=colours[1], 
            ls=l_styles[1], label=metrics[1])
    ax.plot(df[hyperparameter], df[metrics[2]], lw=4, c=colours[2], 
            ls=l_styles[2], label=metrics[2])
    ax.set_xlabel(hyperparameter)
    ax.set_ylabel('Validation Set Prediction Score')
    ax.set_xlim([0, df[hyperparameter].max()])
    ax.set_ylim([0, 1.0])
    ax.axhline(y=1, c='black', lw=0.5)
    ax.xaxis.set_major_locator(mtk.MaxNLocator(5))
    # ax.yaxis.set_major_locator(mtk.MaxNLocator(6))
    ax.legend(loc='lower right')
    plt.savefig(path, dpi=dpi, bbox_inches='tight', format='eps')
    plt.show()


def regional_date_stack(path: str, date_dict: dict, plot_ratio: int,
                        spectrum: str='flare', dpi: int=144):
    '''
    Generates a plot of where the date split occurs between the training/validation
    and test sets using a bar chart for a subset of the regions investigated.
    Parameters
    ----------
    path : str
        Filepath for where the figure will be saved.
    plot_ratio : int
        Inverse of the proportion of regions to be displayed out of the total.
    date_dict : dict
        Dictionary of region keys with date value pairs for when the
        train/test split occurs.
    spectrum : str, optional
        The colour spectrum from which the bar colours are taken. The default
        is 'flare'.
    dpi : int, optional
        The resolution of the saved figure file. The default is 144.

    Returns
    -------
    Saves the output graph to the pathway specified.
    '''
    subset = dict(list(date_dict.items())[0::plot_ratio])
    plot_df = pd.DataFrame.from_dict([subset]).transpose().reset_index()
    plot_df.columns = ['City','Date']
    plot_df['End'] = '2013-12-01'
    plot_df['Date'] = pd.to_datetime(plot_df['Date'])
    plot_df['End'] = pd.to_datetime(plot_df['End'])
    cmap = sn.color_palette(spectrum, as_cmap=True)
    colours = [cmap(0.8), cmap(0.15)]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(plot_df['City'], plot_df['End'], color=colours[0], 
             edgecolor="black", lw=0.75, label='Test Set')
    ax.barh(plot_df['City'], plot_df['Date'], color=colours[1], 
             edgecolor="black", lw=0.75, label='Training & Validation Set')
    ax.set_xlabel('Date')
    ax.set_xlim([pd.Timestamp(1990, 1, 1), pd.Timestamp(2013, 12, 31)])
    ax.legend(loc='upper left', framealpha=1)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', format='eps')
    plt.show()


def feature_dists(path: str, df1: pd.DataFrame, df2: pd.DataFrame, names: list,
                  features: list, labels: list, bins=16, spectrum: str='flare',
                  dpi: int=144):
    '''
    Generates a grid of histogram plots of feature distributions between an
    original dataset and a resampled dataset.

    Parameters
    ----------
    path : str
        Filepath for where the figure will be saved.
    df1 : pd.DataFrame
        The original pandas dataframe containing all features.
    df2 : pd.DataFrame
        The resampled pandas dataframe containing all features.
    names : list
        The names of the original and resampled datasets for the plot legend.
    features : list
        The input variables for the model.
    labels : list
        Corresponding plotting labels for each of the input variables.
    bins : TYPE, optional
        Number of histogram bins to separate the data into. The default is 20.
    spectrum : str, optional
        The colour spectrum from which the bar colours are taken. The default
        is 'flare'.
    dpi : int, optional
        The resolution of the saved figure file. The default is 144.

    Returns
    -------
    Saves the output graph to the pathway specified.
    '''
    fig, axs = plt.subplots(5, 3, figsize=(14, 21))
    cmap = sn.color_palette(spectrum, as_cmap=True)
    colours = [cmap(0.9), cmap(0.025)]
    for i, ax in enumerate(axs.flatten()):
        x1 = df1[features[i]].values
        x2 = df2[features[i]].values
        ax.hist([x1, x2], bins, weights=[np.ones_like(x1)/len(x1),
                                       np.ones_like(x2)/len(x2)],
                color=colours, edgecolor='grey', rwidth=0.8)
        ax.set_xlabel(labels[i])
    plt.tight_layout()
    fig.text(-0.0125, 0.5, 'Density', va='center', ha='center', rotation='vertical')
    plt.legend(labels=names, loc="lower center", bbox_to_anchor=(-0.9, -0.7))
    plt.savefig(path, dpi=dpi, bbox_inches='tight', format='eps')
    plt.show()
