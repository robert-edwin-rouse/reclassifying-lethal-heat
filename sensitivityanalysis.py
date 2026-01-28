"""
This script generates the ablation study results for the Reclassifying Lethal
Heat paper whilst also conducting the feature permutation study.
"""

import config
from classifier import LethalHeatClassifier as cl
import pandas as pd
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
metric_columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score']


'''
Feature permutation study applied to evaluation metrics to assess the impact
to each of them and identify priority variables.
'''
permutation_results = pd.DataFrame(index=features)
classifier = cl(heatwave_data, features, target, params.class_weights,
                params.n_trees, params.tree_depth, params.date_column,
                [params.train_split, params.val_split], params.random_seed,
                params.n_neighbors, params.platt_method, downsample=False,
                synthetic=True)
classifier.model = classifier.train(processors=3)
baseline = classifier.model_output(threshold=0.6)
ranking_metrics = ['accuracy', 'precision','recall','f1']
for r in ranking_metrics:
    results = classifier.feature_permutation(metric=r).to_frame()
    permutation_results = pd.concat([permutation_results, results], axis=1)
    

'''
Export the results from this feature engineering study.
'''
permutation_results.columns = metric_columns
permutation_results.to_csv(config.permutation_filepath)


'''
Ablation study that loops over the feature set and removes variables one at a
time with replacement to assess impact to evaluation metrics.
'''
ablation_dict = {}
for f in range(len(features)):
    inputs = features[:f] + features[f+1:]
    classifier = cl(heatwave_data, inputs, target, params.class_weights,
                    params.n_trees, params.tree_depth, params.date_column,
                    [params.train_split, params.val_split], params.random_seed,
                    params.n_neighbors, params.platt_method, downsample=False,
                    synthetic=True)
    classifier.model = classifier.train(processors=3)
    ablation_dict[features[f]] = list(classifier.model_output(threshold=0.6))    


'''
Process and export the results from this feature engineering study.
'''
ablation_results = pd.DataFrame.from_dict(ablation_dict, orient='index',
                                 columns=metric_columns)
for i in range(len(metric_columns)):
    ablation_results[metric_columns[i]] = baseline[i] - \
                                            ablation_results[metric_columns[i]]
ablation_results.to_csv(config.ablation_filepath)


'''
Generate bar charts of feature permutation importance and ablation impact,
equivalent to Figure 4 in the paper.
'''
vi.sensitivity_bar(config.fig4a_path, permutation_results, features, labels,
                'Mean F1 Decrease', [-0.08, 0.2], metric='F1 Score',)
vi.sensitivity_bar(config.fig4b_path, ablation_results, features, labels,
                'F1 Decrease', [-0.01, 0.2], metric='F1 Score',)