"""
This script trains and tests the random forest model for classifying heatwaves
as either lethal or nonlethal, using a precompiled database.  Also pre and post
processes the results and input data to create some of the plots for the
Reclassifying Lethal Heat paper 
"""

import config
import pandas as pd
import visualisations as vi
from apollo import mechanics as ma
from classifier import LethalHeatClassifier as cl
from joblib import dump


'''
Load configurations and setup the classifier for the random forest model.
'''
ma.textstyle()
heatwave_data = config.lethal_heat_data
target = config.target
features = list(config.feature_dict.keys())
params = config.ForestConfig()
config.results_filepath

classifier = cl(heatwave_data, features, target, params.class_weights,
                params.n_trees, params.tree_depth, params.date_column,
                [params.train_split, params.val_split], params.random_seed,
                params.n_neighbors, params.platt_method, downsample=False,
                synthetic=True, validate=False)


'''
Generate a plot of lethal and nonlethal heatwave events in terms of maximum
temperature and mean relative humidity along with specific wet bulb temperature
thresholds equivalent to Figure 1 in the paper.
'''
vi.wet_bulb_plot(config.fig1_path, classifier.data.data, target[0],
                 features[0], features[1], params.colour_1A, params.colour_2A)


'''
Generate correlation heatmap for all input features, equivalent to Figure 5 in
the paper.
'''
vi.correlation_matrix(config.fig5_path, classifier.data.data, features,
                      config.feature_dict.values())


'''
Train the random forest, output model performance statistics.
'''
classifier.model = classifier.train(processors=3)
classifier.model_output(threshold=0.6)
dump(classifier.model, config.model_filepath)
dump(classifier.data.norm_cache, config.model_dictpath)


'''
Collapse the results into true and false positives and negatives and generate
the lethal/nonlethal heatwave bubble count plots, equivalent to Figures 2 & 3. 
'''
df = classifier.data.test_data
df['Observed +'] = (df[target[0]] > 0).astype(int)
df['Observed -'] = (df[target[0]] == 0).astype(int)
df['Predicted +'] = (df['Predicted'] > 0).astype(int)
df['Predicted -'] = (df['Predicted'] == 0).astype(int)
df['True +'] = (df[target[0]] > 0) & (df['Predicted'] > 0).astype(int)
df['True -'] = (df[target[0]] == 0) & (df['Predicted'] == 0).astype(int)
df['False +'] = (df[target[0]] == 0) & (df['Predicted'] > 0).astype(int)
df['False -'] = (df[target[0]] > 0) & (df['Predicted'] == 0).astype(int)
retain = ['Latitude','Longitude','Observed +','Observed -',
          'Predicted +','Predicted -','True +','True -','False +','False -']
df = df[retain].groupby(retain[0:2], as_index=False).sum()
cset = [[params.colour_1A, params.colour_1B],
        [params.colour_2A, params.colour_2B]]

vi.heat_bubble(config.fig2a_path, df, 'Observed +', [-20, 40], [30, 70],
               cset[0], [5,4], 4)
vi.heat_bubble(config.fig2b_path, df, 'Observed -', [-20, 40], [30, 70],
               cset[1], [3.2,0.0005], 4)
vi.heat_bubble(config.fig2c_path, df, 'Predicted +', [-20, 40], [30, 70],
               cset[0], [2,10], 5)
vi.heat_bubble(config.fig2d_path, df, 'Predicted -', [-20, 40], [30, 70],
               cset[1], [3.2,0.0005], 4)
vi.heat_bubble(config.fig2e_path, df, 'False -', [-20, 40], [30, 70],
               cset[0], [5,4], 5)
vi.heat_bubble(config.fig2f_path, df, 'False +', [-20, 40], [30, 70],
               cset[1], [2,10], 5)

vi.heat_bubble(config.fig3a_path, df, 'Observed +', [60, 180], [0, 80],
               cset[0], [5,4], 4)
vi.heat_bubble(config.fig3b_path, df, 'Observed -', [60, 180], [0, 80],
               cset[1], [3.2,0.0005], 4)
vi.heat_bubble(config.fig3c_path, df, 'Predicted +', [60, 180], [0, 80],
               cset[0], [5,4], 5)
vi.heat_bubble(config.fig3d_path, df, 'Predicted -', [60, 180], [0, 80],
               cset[1], [3.2,0.0005], 4)
vi.heat_bubble(config.fig3e_path, df, 'False -', [60, 180], [0, 80],
               cset[0], [5,4], 5)
vi.heat_bubble(config.fig3f_path, df, 'False +', [60, 180], [0, 80],
               cset[1], [2,10], 5)