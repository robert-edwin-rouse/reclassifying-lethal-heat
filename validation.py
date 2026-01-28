"""
This script generates the random forest hyperparameter grid search results for
the Reclassifying Lethal Heat paper based on the validation dataset.
"""

import config
from classifier import LethalHeatClassifier as cl
import pandas as pd
import numpy as np
import visualisations as vi
from apollo import mechanics as ma


'''
Load configurations and setup the classifier for the random forest model.
'''
ma.textstyle()
heatwave_data = config.lethal_heat_data
target = config.target
features = list(config.feature_dict.keys())
labels = config.feature_dict.values()
params = config.ForestConfig()


'''
Loop over different ratios of nonlethal to lethal heatwaves and return
model performance.
'''
downsampled_ratios = {}
n=100
for i in range(n):
    classifier = cl(heatwave_data, features, target, params.class_weights,
                    params.n_trees, params.tree_depth, params.date_column,
                    [params.train_split, params.val_split], params.random_seed,
                    params.n_neighbors, params.platt_method, downsample=True,
                    downsample_frac=((1+i)/n), validate=True)
    classifier.model = classifier.train(processors=3)
    accuracy, f1score, precision, recall = classifier.model_output(threshold=0.5)
    downsampled_ratios[i] = [i, accuracy, precision, recall, f1score]
downsampling_results = pd.DataFrame.from_dict(downsampled_ratios, orient='index',
                                      columns=['Nonlethal:Lethal Event Ratio',
                                               'Accuracy', 'Precision',
                                               'Recall', 'F1 Score'])
downsampling_results.to_csv('data/downsampling.csv')
vi.grid_tuning_plot(config.figa1a_path, downsampling_results,
                    'Nonlethal:Lethal Event Ratio')


'''
Generate feature distributions for different sampling strategies, SMOTE and 
downsampling the nonlethal events, using the validation set.
'''
base_case = cl(heatwave_data, features, target, params.class_weights,
                params.n_trees, params.tree_depth, params.date_column,
                [params.train_split, params.val_split], params.random_seed,
                params.n_neighbors, params.platt_method, downsample=False,
                validate=True, normalise=False)
synthetic = cl(heatwave_data, features, target, params.class_weights,
                params.n_trees, params.tree_depth, params.date_column,
                [params.train_split, params.val_split], params.random_seed,
                params.n_neighbors, params.platt_method, synthetic=True,
                validate=True, normalise=False)
equal_lnl = cl(heatwave_data, features, target, params.class_weights,
                params.n_trees, params.tree_depth, params.date_column,
                [params.train_split, params.val_split], params.random_seed,
                params.n_neighbors, params.platt_method, downsample=True,
                downsample_frac=0.0071, validate=True, normalise=False)
base_case_array = np.hstack((base_case.x_res,
                             base_case.y_res.reshape(len(base_case.y_res), 1)))
synthetic_array = np.hstack((synthetic.x_res,
                             synthetic.y_res.reshape(len(synthetic.y_res), 1)))
equal_lnl_array = np.hstack((equal_lnl.x_res,
                             equal_lnl.y_res.reshape(len(equal_lnl.y_res), 1)))
base_case_df = pd.DataFrame(base_case_array, columns=features+target)
synthetic_df = pd.DataFrame(synthetic_array, columns=features+target)
equal_lnl_df = pd.DataFrame(equal_lnl_array, columns=features+target)
vi.feature_dists(config.figa2_path, base_case_df, synthetic_df,
                 ['Original dataset','Synthetic dataset'], features, list(labels))
vi.feature_dists(config.figa3_path, base_case_df, equal_lnl_df,
                 ['Original dataset','Downsampled dataset'], features, list(labels))


'''
Loop over tree depth and tree count and return model performance.
'''
forestgrid = {}
classifier = cl(heatwave_data, features, target, params.class_weights,
                params.n_trees, params.tree_depth, params.date_column,
                [params.train_split, params.val_split], params.random_seed,
                params.n_neighbors, params.platt_method, synthetic=True,
                validate=True)
max_tree_depth = 50
max_tree_count = 200
depth_stride = 2
tree_stride = 2
index = 1
for i in range(int(max_tree_depth/depth_stride)):
    for j in range(int(max_tree_count/tree_stride)):
        print('Iteration: ' + str(index))
        classifier.tree_depth = (i+1)*depth_stride
        classifier.n_trees = (j+1)*tree_stride
        print('Tree Depth: ' + str(classifier.tree_depth) + ' | Forest Size: ' \
              + str(classifier.n_trees))
        classifier.model = classifier.train(processors=3)
        accuracy, f1score, precision, recall = classifier.model_output(threshold=0.5)
        forestgrid[index] = [classifier.tree_depth, classifier.n_trees,
                             accuracy, precision, recall, f1score]
        index += 1


'''
Export the results from this hyperparameter tuning study and generate line
charts of hyperparameter affect on performance for given subsets for all model
validation runs, equivalent to Appendix Figure X in the paper.
'''
grid_results = pd.DataFrame.from_dict(forestgrid, orient='index',
                                      columns=['Maximum Tree Depth',
                                               'Number of Trees',
                                               'Accuracy', 'Precision',
                                               'Recall', 'F1 Score'])
grid_results.to_csv('data/forestgridsearch.csv')
grid_subset = grid_results[grid_results['Maximum Tree Depth']==32]
vi.grid_tuning_plot(config.figa2b_path, grid_subset, 'Number of Trees')
grid_subset = grid_results[grid_results['Number of Trees']==128]
vi.grid_tuning_plot(config.figa2a_path, grid_subset, 'Maximum Tree Depth')