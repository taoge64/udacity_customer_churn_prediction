'''
libraries for perform churn prediction jobs
Author: Tao Liu
Creation Date: 08/21/2023
'''
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
# library doc string
# import libraries
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    original_df = pd.read_csv(pth)
    original_df['Churn'] = original_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return original_df


def perform_eda(eda_df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    eda_df['Churn'].hist()
    plt.savefig("./images/eda/churn_hist.png")
    plt.close()
    plt.figure(figsize=(20,10))
    eda_df['Customer_Age'].hist()
    plt.savefig("./images/eda/customer_age.png")
    plt.close()
    plt.figure(figsize=(20,10))
    eda_df.Marital_Status.value_counts(
        'normalize').plot(kind='bar')
    plt.savefig("./images/eda/marital_status.png")
    plt.close()
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct'
    # add a smooth curve obtained using a kernel density estimate
    plt.figure(figsize=(20,10))
    sns.histplot(eda_df['Total_Trans_Ct'],
                 stat='density',
                 kde=True)
    plt.savefig("./images/eda/total_trans_density.png")
    plt.close()
    plt.figure(figsize=(20,10))
    sns.heatmap(eda_df.corr(),
                annot=False,
                cmap='Dark2_r',
                linewidths=2
                )
    plt.savefig("./images/eda/feature_importance.png")


def encoder_helper(encoder_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            df: pandas dataframe with new columns for
    '''
    for index in range(len(category_lst)):
        mean_lst = []
        mean_groups = encoder_df.groupby(category_lst[index]).mean()['Churn']
        for val in encoder_df[category_lst[index]]:
            mean_lst.append(mean_groups.loc[val])
        encoder_df[response[index]] = mean_lst
    return encoder_df


def perform_feature_engineering(feature_engineering_df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y_data = feature_engineering_df['Churn']
    x_data = feature_engineering_df[response]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores

    reports = {
        'Random Forest Test': classification_report(y_test, y_test_preds_rf),
        'Random Forest Training': classification_report(y_train, y_train_preds_rf),
        'Logistic Regression Test': classification_report(y_test, y_test_preds_lr),
        'Logistic Regression Training': classification_report(y_train, y_train_preds_lr)
    }

    # Plot each classification report
    for title, report in reports.items():
        plt.figure(figsize=(10, 3))
        plt.text(0.01,
                 0.5,
                 report,
                 {'fontsize': 12},
                 fontproperties='monospace')  # Adjust fontsize if needed
        plt.axis('off')
        plt.tight_layout()
        # Adjust these parameters to fit the text perfectly
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.1)
        plt.title(title)
        plt.savefig(f'images/results/{title}.png', dpi=300)
        plt.close()

def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth, bbox_inches='tight')


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    print("now finish grid search")
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    # scores
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    # plots
    plt.figure(figsize=(15, 8))
    _ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=_ax,
        alpha=0.8)
    rfc_disp.plot(ax=_ax, alpha=0.8)
    plt.savefig("./images/results/roc.png")
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    feature_importance_plot(
        cv_rfc,
        x_train,
        "images/results/feature_importance.png")


#     print('now start shap plot')
#     explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
#     shap_values = explainer.shap_values(x_test)
#     shap.summary_plot(shap_values, x_test, plot_type="bar")
#     plt.savefig("images/results/shap.png")
if __name__ == "__main__":
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    bank_df = import_data("./data/bank_data.csv")
    perform_eda(bank_df)
    encoder_churn_df = encoder_helper(bank_df, cat_columns, [
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn'
    ])
    churn_x_train, churn_x_test, churn_y_train, churn_y_test = perform_feature_engineering(
        encoder_churn_df, keep_cols)
    print("now start training models")
    train_models(churn_x_train, churn_x_test, churn_y_train, churn_y_test)