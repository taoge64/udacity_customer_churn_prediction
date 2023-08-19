import os
import logging
import churn_library as cls
import pandas as pd
logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	# isolation test dataset
	test_data = {
        'CLIENTNUM': [768805383, 818770008],
        'Attrition_Flag': ["Existing Customer", "Existing Customer"],
        'Customer_Age': [45, 49],
        'Gender': ['M', 'F'],
        'Dependent_count': [3, 5],
        'Education_Level': ['High School', 'Graduate'],
        'Marital_Status': ['Married', 'Single'],
        'Income_Category': ['$60K - $80K', 'Less than $40K'],
        'Card_Category': ['Blue', 'Blue'],
        'Months_on_book': [39, 44],
        'Total_Relationship_Count': [5, 6],
        'Months_Inactive_12_mon': [1, 1],
        'Contacts_Count_12_mon': [3, 2],
        'Credit_Limit': [12691.0, 8256.0],
        'Total_Revolving_Bal': [777, 864],
        'Avg_Open_To_Buy': [11914.0, 7392.0],
        'Total_Amt_Chng_Q4_Q1': [1.335, 1.541],
        'Total_Trans_Amt': [1144, 1291],
        'Total_Trans_Ct': [42, 33],
        'Total_Ct_Chng_Q4_Q1': [1.625, 3.714],
        'Avg_Utilization_Ratio': [0.061, 0.105]
    }
	df = pd.DataFrame(test_data)
	try:
		perform_eda(df)
		logging.info("Testing eda: SUCCESS")
	except RuntimeError as err:
		logging.error("Testing eda: The function doesn't run successful")
		raise err

	try:
		assert list(df['Churn']) == [0, 0], "Churn transformation failed"
	except AssertionError as err:
		logging.error(f"Testing eda: Churn transformation failed")
		raise err

	try:
		paths = [
			"./image/eda/churn_hist.png",
			"./image/eda/customer_age.png",
			"./image/eda/marital_status.png",
			"./image/eda/total_trans_density.png",
			"./image/eda/feature_importance.png"
		]
		for path in paths:
			assert os.path.exists(path)
		for path in paths:
			if os.path.exists(path):
				# remove testing files after finished
				os.remove(path)
	except AssertionError as err:
		logging.error(f"Testing eda: Expected file at {path} does not exist")
		raise err

def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	test_import(cls.import_data)
	test_eda(cls.perform_eda)









