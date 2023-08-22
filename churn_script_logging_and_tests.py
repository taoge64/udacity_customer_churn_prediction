'''
unit test section for churn_library
'''
import os
import logging
import pytest
import churn_library as cls
logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def pytest_configure():
    '''
    define pytest configure
    :return: None
    '''
    pytest.df = None
    pytest.encoder_df = None
    pytest.x_train = None
    pytest.x_test = None
    pytest.y_train = None
    pytest.y_test = None

def test_import(import_data):
    '''

    :param import_data: function of import_data
    :return: None
    '''
    try:
        import_df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        pytest.df = import_df
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert import_df.shape[0] > 0
        assert import_df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    # isolation test dataset
    eda_df = pytest.df
    try:
        perform_eda(eda_df)
        logging.info("Testing eda: SUCCESS")
    except RuntimeError as err:
        logging.error("Testing eda: The function doesn't run successful")
        raise err

    try:
        paths = [
            "./images/eda/churn_hist.png",
            "./images/eda/customer_age.png",
            "./images/eda/marital_status.png",
            "./images/eda/total_trans_density.png",
            "./images/eda/feature_importance.png"
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
    test perform encoder_helper function
    '''
    try:
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        encoder_df = pytest.df
        encoder_df = encoder_helper(encoder_df,cat_columns,['Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn'])
        pytest.encoder_df= encoder_df
    except RuntimeError as err:
        logging.error("Testing Encoder Helper: Encoder Failed")
        raise err



def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                     'Total_Relationship_Count', 'Months_Inactive_12_mon',
                     'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                     'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                     'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                     'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                     'Income_Category_Churn', 'Card_Category_Churn']
        # use only gender as an example to test encoder
        encoder_df = pytest.encoder_df
        x_train, x_test, y_train, y_test = perform_feature_engineering(encoder_df,keep_cols)
        pytest.x_train = x_train
        pytest.x_test = x_test
        pytest.y_train = y_train
        pytest.y_test = y_test
    except RuntimeError as err:
        logging.error("Testing Feature Engineering : Function Failed")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        train_models(pytest.x_train,pytest.x_test,pytest.y_train,pytest.y_test)
    except RuntimeError as err:
        logging.error(f"Testing Train Models : Function Failed")
        raise err





if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
