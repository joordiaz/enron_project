#helper furnctions for final_project
import sys
from pandas import *
import math
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def add_email_exist_feature(d):

	for key, value in d.iteritems():

		if value['from_messages'] == 'NaN' and \
			value['from_poi_to_this_person'] == 'NaN' and \
			value['from_this_person_to_poi'] == 'NaN' and \
			value['shared_receipt_with_poi'] == 'NaN' and \
			value['to_messages'] == 'NaN' :

			d[key]['email_exist'] = 0
		else:
			d[key]['email_exist'] = 1

	return d

def add_ratio_email_to_poi(d):
	for key, value in d.iteritems():
    
	    from_message =  float(value['from_messages'])
	    to_poi = float(value['from_this_person_to_poi'])
	    if isnull(to_poi) or isnull(from_message):
	        d[key]['ratio_emails_to_poi'] = 'NaN'
	    else:
	        d[key]['ratio_emails_to_poi'] = to_poi/from_message
	
	return d

def add_ratio_email_from_poi(d):
	for key, value in d.iteritems():
    
	    to_messages =  float(value['to_messages'])
	    from_poi = float(value['from_poi_to_this_person'])
	    if isnull(from_poi) or isnull(to_messages):
	        d[key]['ratio_emails_from_poi'] = 'NaN'
	    else:
	        d[key]['ratio_emails_from_poi'] = from_poi/to_messages
	
	return d

def add_product_ratios_poi(d):
	for key, value in d.iteritems():
    
	    from_poi =  float(value['ratio_emails_from_poi'])
	    to_poi = float(value['ratio_emails_to_poi'])
	    if isnull(from_poi) or isnull(to_poi):
	        d[key]['product_ratios_poi'] = 'NaN'
	    else:
	        d[key]['product_ratios_poi'] = from_poi*to_poi
	
	return d

def add_total_compensation(d):
	for key, value in d.iteritems():
    
	    total_payments =  float(value['total_payments'])
	    total_stock_value = float(value['total_stock_value'])
	    
	    if isnull(total_payments):
	    	total_payments = 0.0

	    if isnull(total_stock_value):
	        total_stock_value = 0.0
	    
	    d[key]['total_compensation'] = total_payments + total_stock_value
	
	return d

def impute_email_data(d):

	for key, value in d.iteritems():

		if value['from_messages'] == 'NaN':
			d[key]['from_messages'] = 608

		if value['from_poi_to_this_person'] == 'NaN':
			d[key]['from_poi_to_this_person'] = 65

		if value['from_this_person_to_poi'] == 'NaN':
			d[key]['from_this_person_to_poi'] = 41

		if value['shared_receipt_with_poi'] == 'NaN':
			d[key]['shared_receipt_with_poi'] = 1176

		if value['to_messages'] == 'NaN':
			d[key]['to_messages'] = 2073


	return d

def remove_obs_with_no_email(d):

	for key, value in d.copy().iteritems():

		if value['email_exist'] == 0:
			d.pop(key)

	print len(d.keys())
	return d

class hybrid_model:
    """A split model for classification"""
    def __init__(self, clf0, clf1, clf0_feature_idx, clf1_feature_idx):
        self.clf0 = clf0
        self.clf1 = clf1
        self.clf0_feature_idx = clf0_feature_idx
        self.clf1_feature_idx = clf1_feature_idx
    def fit(self, X, y):
    	#print 'X.shape' , X[0].shape
    	#print 'y.shape' , len(y)
    	X0 = []
    	X1 = []
    	y0 = []
    	y1 = []

    	for obs, label in zip(X,y):
    		if obs[0] == 1:
    			X1.append(obs[self.clf1_feature_idx])
    			y1.append(label)
    		X0.append(obs[self.clf0_feature_idx])
    		y0.append(label)


        self.clf0.fit(X0,y0)
        self.clf1.fit(X1,y1)

    def predict(self, X):
    	pred = []
    	for obs in X:
    		if obs[0] == 1:
    			pred.append(self.clf1.predict(obs[self.clf1_feature_idx]))
    		else:
    			pred.append(self.clf0.predict(obs[self.clf0_feature_idx]))

        return pred


def transform_to_dataframe(d_dict):
    ''' Transforms data from the data_dict format into a data_frame format '''
    df_dict = {}
    features = d_dict[d_dict.keys()[0]].keys()
    #for each feature create a Series
    for feature in features:
        #Initialize a Series with a dictionary 'name': feature_value
        series_dict = {}
        for key,value in d_dict.iteritems():
            series_dict[key] = value[feature]
        # Create a series
        f_series = Series(series_dict)
        # Add to a dictionary 'feature_name': Series
        df_dict[feature] = f_series
        
    #create a DataFrame with the dictionary
    df = DataFrame(df_dict)
    return df

#Transform numerical values in dataframe to float
def transform_values_to_float(df):
    non_num_features = ['poi', 'email_address']
    for feature in df.columns:
        if feature not in non_num_features:
            df[feature]  = df[feature].map(lambda x: float(x))
    return df

def get_email_comp_dict(d):
	ec_d = {}
	for key,value in d.iteritems():
		if value['email_exist'] == 1:
			ec_d[key] = value

	return ec_d

from sklearn.grid_search import GridSearchCV

# Do grid test with cross validation given the fatures, model, data, and tunning parameters
def grid_test_model(clf_features, model, data_dict, param_grid):
	# select features
	clf_feature_list = ['poi'] + clf_features
	# form features and lables
	data = featureFormat(data_dict, clf_feature_list,sort_keys = True)
	labels, features = targetFeatureSplit(data)
	# create cross-validation object
	cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
	# set up grid parameter search
	grid = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring = 'f1')
	# train the model grid
	grid.fit(features, labels)
	print("The best parameters are %s with a score of %0.3f"
	      % (grid.best_params_, grid.best_score_))
	return grid



