"""
This script contains the classes for the dataset handling and model architecture
upon which all of the experiments in the Reclassifying Lethal Heat paper are
based.
"""

import numpy as np
import pandas as pd
import sklearn.ensemble as sle
import sklearn.metrics as slm
from sklearn.inspection import permutation_importance as p_i
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN
from apollo import mechanics as ma
from pathlib import Path
import warnings


def temporal_split(data: pd.DataFrame, label_column: str, label: int,
                   date_column: str, split_point: float):
    '''
    Function to filter, sort, and split data from a pandas dataframe about a
    pivot, typically expected to be a date, to help mitigate temporal leakage
    from a training set into a test set.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset to be split.
    label_column : str
        Name of the column to create a sorted subset of the data.
    label : int
        Label with which to subset the data.
    date_column : str
        Name of the date, or other split point, column.
    split_point : float
        Proportion of the data to be split.

    Returns
    -------
    cache : pd.DataFrame
        A time filtered and sorted dataframe.
    '''
    cache = data[data[label_column] == label].sort_values(by=[date_column])
    cache = cache[:int(len(cache)*(split_point))]
    return cache


class LethalHeatData:
    '''
    This class creates is used to create a preprocessed dataframe of lethal
    and nonlethal heatwaves in preparation for use with a machine learning
    classification algorithm.
    '''
    def __init__(self, heat_file: Path, features: list, target: list,
                 label_column: str, date_column: str, set_split: list):
        '''
        Creates a lethal heatwave dataframe to be used by the classification
        algorithm.
        
        Parameters
        ----------
        heat_file : Path
            Filepath to the heatwave dataset, stored as a csv file.
        features : list
            List of input features to be used.
        target : list
            List of length 1 of the target variable being mapped to.
        label_column : str
            Name of the target variable column if different from target.
        date_column : str
            Name of the date column in the csv file.
        set_split : list
            Desired splits for the dataset that takes two values: the first
            is the proportion of data in the training set, the second is
            the proportion of data in the validation set.  The remaining
            data will form the test set.

        Returns
        -------
        None.
        '''
        self.data = pd.read_csv(heat_file).fillna(value=0)
        self.features = features
        self.target = target[0]
        self.xspace = ma.featurelocator(self.data, features)
        self.yspace = ma.featurelocator(self.data, target)
        self.label_column = self.target
        self.date_column = date_column
        self.set_split = set_split
    
    def convert_datetime(self, column: str):
        '''
        Converts a pandas column to a standard datetime format.
        Parameters
        ----------
        column : str
            Name of the column to be converted.

        Returns
        -------
        Overwrites the original column with the converted one in the dataframe.
        '''
        self.data[column] = pd.to_datetime(self.data[column],
                                           format='%Y-%m-%d').dt.date
    
    def data_splitter(self, label_column: str, date_column: str,
                      set_split: list, random_state: int=42):
        '''
        Creates a regional chronological split in the data to minimise temporal
        leakage.
        
        Parameters
        ----------
        label_column : str
            Name of the target variable column in the csv file.
        date_column : str
            Name of the date column in the csv file.
        set_split : list
            Desired splits for the dataset that takes two values: the first
            is the proportion of data in the training set, the second is
            the proportion of data in the validation set.  The remaining
            data will form the test set.
        random_state : int, optional
            Random seed for reproducibility of the split into the training
            and validation data sets. The default is 42.

        Returns
        -------
        train_data : pd.DataFrame
            Dataframe of the training data.
        val_data : pd.DataFrame
            Dataframe of the validation data.
        test_data : pd.DataFrame
            Dataframe of the test data.
        date_splits : dict
            Dictionary of regions along with their respective date about which
            the data is split into the training and test sets.
        '''
        self.convert_datetime(date_column)
        data_subset = pd.DataFrame(columns=self.data.columns)
        city_list = self.data['City'].unique().tolist()
        train_split, val_split = set_split[0], set_split[1]
        date_splits = {}
        for i in range(len(city_list)):
            cache = self.data[self.data['City'] == city_list[i]]
            cache_positive = temporal_split(cache, self.label_column, 1,
                                       self.date_column, (train_split+val_split))
            cache_negative = temporal_split(cache, self.label_column, 0, 
                                       self.date_column, (train_split+val_split))
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                data_subset = pd.concat([data_subset, cache_positive, cache_negative])
            date_splits[city_list[i]] = cache_negative['EndDate'].max()
        train_data = data_subset.sample(frac=train_split, random_state=random_state)
        val_data = data_subset[~data_subset.isin(train_data)].dropna()
        test_data = self.data[~self.data.isin(data_subset)].dropna()
        return train_data, val_data, test_data, date_splits
    
    def process(self, normalise: bool=True):
        '''
        Preprocesses the data and splits it into training, validation, and test
        sets accordingly.  Normalises the data if normalisation is required.
        
        Parameters
        ----------
        normalise : bool, optional
            Whether or not to normalise the input features of the dataset using
            the training data only. The default is True.

        Returns
        -------
        None.
        '''
        self.data[self.target] = np.where(self.data[self.target] > 0, 1, 0)
        self.train_data, self.val_data, self.test_data, self.date_splits = \
            self.data_splitter(self.label_column, self.date_column, self.set_split)
        if normalise==True:
            norm_cache = {}
            for f in self.features:
                self.train_data[f] = ma.normalise(self.train_data, f,
                                                  norm_cache, write_cache=True)
                self.norm_cache = norm_cache
                self.val_data[f] = ma.normalise(self.val_data, f,
                                                norm_cache, write_cache=False)
                self.test_data[f] = ma.normalise(self.test_data, f,
                                                 norm_cache, write_cache=False)

    def downsample(self, subset: pd.DataFrame, downsample_frac: float=1):
        '''
        Downsamples a dataset based on reducing the number of majority,
        nonlethal heatwave events.

        Parameters
        ----------
        subset : pd.DataFrame
            The dataset to be downsampled.
        downsample_frac : float, optional
            The reduction factor to be applied when downsampling. The default
            is 1.

        Returns
        -------
        pd.DataFrame
            The downsampled dataset.
        int
            The number of positive, lethal events in the dataset.
        int
            The number of negative, nonlethal events in the dataset.
        '''
        positives = subset[subset[self.target] == 1]
        negatives = subset[subset[self.target] == 0].sample(frac=downsample_frac)
        downsampled_set = pd.concat([positives, negatives])
        return downsampled_set, len(positives), len(negatives)

    def to_arrays(self, subset: pd.DataFrame):
        '''
        Generates input and output data arrays from a pandas dataframe for use
        with a classification algorithm.
        
        Parameters
        ----------
        subset : pd.DataFrame
            Pandas dataframe to be split into input and output data arrays.

        Returns
        -------
        x_array : np.array
            Array of model input values.
        y_array : np.array
            Array of model output values.
        '''
        array = subset.to_numpy()
        x_array = array[:,self.xspace].reshape(len(array),
                                               len(self.xspace)).astype(float)
        y_array = array[:,self.yspace].reshape(len(array),
                                               ).astype(float)
        return x_array, y_array


class LethalHeatClassifier:
    '''
    Class to specify, train, and evaluate a random forest architecture using
    probabilistic Platt scaling.  Uses the LethalHeatData class to subset and
    transform the input-output data pairs according to a desired temporal data
    split.  Also takes advantage of functions to determine whether or not the
    model is being used for validation or testing, and to either downsample or
    upsample the data using synthetic data generation.  For further information 
    on standard functions taken from the scikit learn library, please refer to
    https://scikit-learn.org/stable/user_guide.html
    '''
    def __init__(self, heat_file: Path, features: list, target: list,
                 class_weights: dict, n_trees: int, tree_depth: int,
                 date_column: str, set_split: list, random_state: int,
                 n_neighbors: int, platt_method: str, downsample: bool=False,
                 downsample_frac:float=1, synthetic: bool=False,
                 validate: bool=False, normalise: bool=True):
        '''
        Initialises the random forest classifier and prepares all data for
        training.
        
        Parameters
        ----------
        heat_file : Path
            Filepath to the heatwave dataset, stored as a csv file.
        features : list
            List of input features to be used.
        target : list
            List of length 1 of the target variable being mapped to.
        class_weights : dict
            Dictionary of weights to apply to the lethal and nonlethal
            classes.
        n_trees : int
            Number of decision trees to use in the random forest.
        tree_depth : int
            Maximum tree depth that the decision trees can grow to.
        date_column : str
            Name of the date column in the csv file.
        set_split : list
            Desired splits for the dataset that takes two values: the first
            is the proportion of data in the training set, the second is
            the proportion of data in the validation set.  The remaining
            data will form the test set.
        random_state : int
            Random state for reproducibility.
        n_neighbors : int
            Number of existing examples to use to generate each new synthetic
            data point.
        platt_method : str
            Platt scaling method to be used for the classifier calibration.
        downsample : bool, optional
            Whether or not to downsample the majority, nonlethal class. The
            default is False.
        downsample_frac : float, optional
            The reduction factor to be applied when downsampling. The default
            is 1.
        synthetic : bool, optional
            Whether or not to upsample the minority, lethal class using
            synthetic minority oversampling. The default is False.
        validate : bool, optional
            Whether or not the model is being used on the test set or the is
            being tuned using the validation set. The default is False.

        Returns
        -------
        None.
        '''
        self.data = LethalHeatData(heat_file, features, target, target[0],
                                   date_column, set_split)
        self.data.convert_datetime(date_column)
        self.features = features
        self.target = target
        self.xspace = ma.featurelocator(self.data.data, self.features)
        self.yspace = ma.featurelocator(self.data.data, self.target)
        self.class_weights = class_weights
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.platt_method = platt_method
        self.downsample = downsample
        self.downsample_frac = downsample_frac
        self.synthetic = synthetic
        self.validate = validate
        self.normalise = normalise
        self.model = sle.RandomForestClassifier(n_estimators=n_trees,
                                                max_depth=tree_depth,
                                                class_weight=class_weights)
        self.data.process(self.normalise)
        if self.downsample == False:
            self.downsample_frac = 1
        if self.validate == False:
            self.data.train_data = pd.concat([self.data.train_data,
                                              self.data.val_data])            
            self.data.train_data, self.lethal_count, self.non_lethal_count = \
                self.data.downsample(self.data.train_data, self.downsample_frac)
            self.x_train, self.y_train = self.data.to_arrays(self.data.train_data)
            self.x_eval, self.y_eval = self.data.to_arrays(self.data.test_data)
        else:
            self.data.train_data, self.lethal_count, self.non_lethal_count = \
                self.data.downsample(self.data.train_data, self.downsample_frac)
            self.x_train, self.y_train = self.data.to_arrays(self.data.train_data)
            self.x_eval, self.y_eval = self.data.to_arrays(self.data.val_data)
        print('You have ' + str(self.lethal_count) + ' lethal heatwaves and ' + \
              str(self.non_lethal_count) + ' nonlethal heatwaves in your ' + \
                  'training set for a lethal proportion of ' +
                  str(100*self.lethal_count/len(self.x_train))[:5] + '%')
        self.x_res, self.y_res = self.x_train, self.y_train
        if self.synthetic == True:
            sm = ADASYN(random_state=self.random_state)#, n_neighbors=self.n_neighbors)
            self.x_res, self.y_res = sm.fit_resample(self.x_train, self.y_train)
            synthetic_count = len(self.x_res) - len(self.x_train)
            synthetic_ratio = (synthetic_count+self.lethal_count)/len(self.x_res)
            print(str(synthetic_count) + ' synthetic training data points ' + \
                  'have been generated for a new lethal proportion in your' + \
                  ' training set of ' + str(100*synthetic_ratio)[:5] + '%')
    
    def train(self, processors: int):
        '''
        Trains the random forest model.  Refer to the scikit learn
        documentation for further information.
        
        Parameters
        ----------
        processors : int
            Number of processors that can be used for training.

        Returns
        -------
        platt_model : CalibratedClassifierCV
            A Platt scaled random forest model that outputs probabilistic
            predictions of lethal/nonlethal class labels.
        '''
        platt_model = CalibratedClassifierCV(self.model, method=self.platt_method,
                                             n_jobs=processors)
        platt_model.fit(self.x_res, self.y_res)
        return platt_model
    
    def predict(self, subset: pd.DataFrame,
                trained_model: sle.RandomForestClassifier, threshold: float):
        '''
        Creates predictions of class labels using a probability threshold for
        Platt scaling of either lethal or nonlethal.  Also generates standard
        consensus predictions.  Values are added to the subset dataframe.
        Refer to the scikit learn documentation for further information.
        
        
        Parameters
        ----------
        subset : pd.DataFrame
            Subset of a dataframe upon which to make the prediction, such as
            the test subset from the whole dataset.
        trained_model : sle.RandomForestClassifier
            The trained random forest model.
        threshold : float
            Probability threshold between 0 and 1.

        Returns
        -------
        None.
        '''
        y_prob = trained_model.predict_proba(self.x_eval)
        y_pred = np.array([np.argmax(prob) for prob in y_prob])
        subset['Consensus'] = y_pred
        subset['Probability'] = y_prob[:,1]
        subset['Predicted'] = (subset['Probability']>=threshold).astype(int)

    def predict_all(self, trained_model: sle.RandomForestClassifier,
                    threshold: float):
        '''
        Generates predictions for all datapoints, including the training set,
        according to a given probability threshold.  Values are added to the
        subset dataframe.  Refer to the scikit learn documentation for further
        information.  
        
        Parameters
        ----------
        trained_model : sle.RandomForestClassifier
            The trained random forest model.
        threshold : float
            Probability threshold between 0 and 1.

        Returns
        -------
        None.
        '''
        x_array, _ = self.data.to_arrays(self.data.data)
        y_prob = trained_model.predict_proba(x_array)
        y_pred = np.array([np.argmax(prob) for prob in y_prob])
        self.data.data['Consensus'] = y_pred
        self.data.data['Probability'] = y_prob[:,1]
        self.data.data['Predicted'] = (self.data.data['Probability']>=threshold).astype(int)
    
    def evaluate(self, subset: pd.DataFrame):
        '''
        Evaluates the performance of the model on a given subset of the data
        using standard classification metrics.
        
        Parameters
        ----------
        subset : pd.DataFrame
            Subset of the data upon which to evaluate the performance of the
            classifier.

        Returns
        -------
        accuracy : float
            Accuracy.
        precision : float
            Precision.
        recall : float
            Recall.
        f1score : float
            F1 score.
        '''
        accuracy = slm.accuracy_score(subset[self.target].astype(int), 
                                      subset['Predicted'])
        CM = slm.confusion_matrix(subset[self.target].astype(int), 
                                  subset['Predicted'])
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1score = slm.f1_score(subset[self.target].astype(int), 
                               subset['Predicted'])
        return accuracy, precision, recall, f1score
    
    def print_metrics(self, subset: pd.DataFrame):
        '''
        Prints the output statistics used for classification to the console.
        
        Parameters
        ----------
        subset : pd.DataFrame
            Subset of a dataframe upon which to make the assessment, such as
            the test subset from the whole dataset.

        Returns
        -------
        None.
        '''
        accuracy, precision, recall, f1score = self.evaluate(subset)
        print('Test Accuracy: ' + str(accuracy))
        print('Test Precision: ' + str(precision))
        print('Test Recall: ' + str(recall))
        print('Test F1 score: ' + str(f1score))
        print('- - - - - - - - - - - - - - - - - - - -')

    def model_output(self, threshold:float):
        '''
        Uses the class prediction and evaluation methods to create predictions
        of lethal/nonlethal labels for heatwaves and generate output
        statistics, specifically accuracy, precision, recall, and f1 score.
        Refer to the scikit learn documentation for further information.
        
        Parameters
        ----------
        threshold : float
            Probability threshold between 0 and 1.

        Returns
        -------
        acc : float
            Accuracy.
        prec : float
            Precision.
        rec : float
            Recall.
        f1 : float
            F1 score.

        '''
        if self.validate == True:
            self.predict(self.data.val_data, self.model, threshold)
            acc, prec, rec, f1 = self.evaluate(self.data.val_data)
            self.print_metrics(self.data.val_data)
        else:
            self.predict(self.data.test_data, self.model, threshold)
            acc, prec, rec, f1 = self.evaluate(self.data.test_data)
            self.print_metrics(self.data.test_data)
        return acc, prec, rec, f1
    
    def feature_permutation(self, metric: str='f1', repeats: int=8):
        '''
        Performs feature permutation for the random forest classifier using
        a given metric, such as f1 score.  Refer to the scikit learn
        documentation for further information.

        Parameters
        ----------
        metric : str, optional
            The metric upon which impact is assessed. The default is 'f1'.
        repeats : int, optional
            The number of repeats. The default is 8.

        Returns
        -------
        pd.Series
            Series of feature permutation importances that uses the features
            as an index.
        '''
        self.ranking = p_i(self.model, self.x_eval, self.y_eval, scoring=metric,
                           n_repeats=repeats, random_state=self.random_state)
        return pd.Series(self.ranking.importances_mean, index=self.features)