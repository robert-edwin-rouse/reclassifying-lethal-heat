"""
This script generates the greedy feature engineering results for the Reclassifying
Lethal Heat paper along with the two comparative models that use wet bulb
temperature alone and the adaptive variable set less the wet bulb temperature.
"""

import config
import pandas as pd
from classifier import LethalHeatClassifier as cl
import visualisations as vi


heatwave_data = config.lethal_heat_data
target = config.target
features = list(config.feature_dict.keys())
params = config.ForestConfig()


'''
Greedy agorithm that adds the next best feature, in terms of F1 score, to the
list of model inputs.
'''
featurebuilder = {}
while len(features)>0:
    best_f1score = -1
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    for f in features:
        inputs = list(featurebuilder.keys()) + [f]
        classifier = cl(heatwave_data, inputs, target, params.class_weights,
                        params.n_trees, params.tree_depth, params.date_column,
                        [params.train_split, params.val_split], params.random_seed,
                        params.n_neighbors, params.platt_method,
                        downsample=False, synthetic=True)
        classifier.model = classifier.train(processors=3)
        accuracy, precision, recall, f1score = classifier.model_output(threshold=0.6)
        if f1score >= best_f1score:
            bestkey = f
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_f1score = f1score
    featurebuilder[bestkey] = [best_accuracy, best_precision,
                               best_recall, best_f1score]
    features.remove(bestkey)


'''
Generate results for the model without wet bulb temperature variables and 
using wet bulb temperature variables on their own.
'''
features = list(config.feature_dict.keys())

classifier = cl(heatwave_data, features[2:], target, params.class_weights,
                params.n_trees, params.tree_depth, params.date_column,
                [params.train_split, params.val_split], params.random_seed,
                params.n_neighbors, params.platt_method,
                downsample=False, synthetic=True)
classifier.model = classifier.train(processors=3)
featurebuilder['Without Wet Bulb'] = list(classifier.model_output(threshold=0.6))

classifier = cl(heatwave_data, features[:2], target, params.class_weights,
                params.n_trees, params.tree_depth, params.date_column,
                [params.train_split, params.val_split], params.random_seed,
                params.n_neighbors, params.platt_method,
                downsample=False, synthetic=True)
classifier.model = classifier.train(processors=3)
classifier.model_output(threshold=0.6)
featurebuilder['Wet Bulb Only'] = list(classifier.model_output(threshold=0.6))


'''
Collapse the results into true and false positives and negatives and generate
the lethal/nonlethal heatwave bubble count plots, equivalent to Figures 6 & 7. 
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

vi.heat_bubble(config.fig6a_path, df, 'Observed +', [-20, 40], [30, 70], cset[0], [5,4], 4)
vi.heat_bubble(config.fig6b_path, df, 'False -', [-20, 40], [30, 70], cset[0], [5,4], 5)
vi.heat_bubble(config.fig6c_path, df, 'Observed -', [-20, 40], [30, 70], cset[1], [3.2,0.0005], 4)
vi.heat_bubble(config.fig6d_path, df, 'False +', [-20, 40], [30, 70], cset[1], [2,5], 5)

vi.heat_bubble(config.fig7a_path, df, 'Observed +', [60, 180], [0, 80], cset[0], [5,4], 4)
vi.heat_bubble(config.fig7b_path, df, 'False -', [60, 180], [0, 80], cset[0], [5,4], 5)
vi.heat_bubble(config.fig7c_path, df, 'Observed -', [60, 180], [0, 80], cset[1], [3.2,0.0005], 4)
vi.heat_bubble(config.fig7d_path, df, 'False +', [60, 180], [0, 80], cset[1], [2,5], 5)


'''
Export the results from this feature engineering study.
'''
results = pd.DataFrame.from_dict(featurebuilder, orient='index',
                                 columns=['Accuracy', 'Precision',
                                          'Recall', 'F1 Score'])
results.to_csv(config.feature_builder_filepath)