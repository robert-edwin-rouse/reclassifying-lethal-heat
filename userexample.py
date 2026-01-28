"""
Example script on how to load and use the trained random forest model to
perform classification on new datasets.
"""

import config
import pandas as pd
import numpy as np
from apollo import mechanics as ma
from joblib import load


'''
Load configurations and setup the classifier for the random forest model.
'''
target = config.target
features = list(config.feature_dict.keys())
params = config.ForestConfig()
model = load(config.model_filepath)
norm_cache = load(config.model_dictpath)
df = pd.read_csv('data/new_data.csv')


'''
Normalise and extract relevant features from the new dataset.
'''
for f in features:
    df[f] = ma.normalise(df, f, norm_cache, write_cache=False)
xspace = ma.featurelocator(df, features)
array = df.to_numpy()
x_array = array[:,xspace].reshape(len(array), len(xspace)).astype(float)


'''
Make predictions of lethal/nonlethal heatwaves using the model and append
them to the dataframe for analysis.
'''
y_prob = model.predict_proba(x_array)
y_pred = np.array([np.argmax(prob) for prob in y_prob])
df['Consensus'] = y_pred
df['Probability'] = y_prob[:,1]
df['Predicted'] = (df['Probability']>=config.platt_threshold).astype(int)