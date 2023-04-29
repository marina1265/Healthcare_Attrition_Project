# API Documention

This project is to perform data analytics, as well as build a better Machine Learning classification model that can predict employee attrition in the healthcare industry.  It mainly includes data preprocessing, visualization of each attribute in the dataset, the calculation of the correlation coefficient matrix, building the classification models, calculation of the accuracy rates, and prediction of the attrition outcome. 

### read_csv

```python
def read_csv(filename):
    """
    :param filename: Name of the read csv file
    :return: The data table in the dataframe format
    """
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = None
    return df
```

We use this function to read csv file. Input the filename of a parameter, if the file does not exist, return None, otherwise return dataframe format dataframe. 


### data_deal

```python
def data_deal(df):
    """
    :param df: dataframe
		:return: some plots of the data processing
    """
    # Display data info
    print("==" * 10, 'data info', "==" * 10)
    print(df.info())
    print("==" * 10, 'data head', "==" * 10)
    print(df.head())
    # Check missing values
    print("==" * 10, 'data missing value', "==" * 10)
    print(df.isnull().sum())
    # Drop all categorical variables and numeric variables that only have one value
    # Display all useful numeric variables
    numCol = df.select_dtypes(include=np.number)
    numCol = numCol.drop(['EmployeeID', 'EmployeeCount', 'StandardHours'], axis=1)
    print("==" * 10, 'data describe', "==" * 10)
    print(numCol.describe().round(3))
    # Display all categorical variables
    cagCol = df.select_dtypes(exclude=np.number)
    print("==" * 10, 'categorical variables', "==" * 10)
    print(cagCol.head())
    print("==" * 10, 'categorical variables describe', "==" * 10)
    print(cagCol.describe())
    num_count = numCol.drop(['HourlyRate', 'DailyRate', 'MonthlyIncome', 'MonthlyRate'], axis=1)
    return num_count
```

This function takes a single argument --df, dataframe, and returns numeric dataframe data. This function displays the basic information of the data. The first five rows (data.head() method) as well as the missing value information of the data are displayed. Show all the numerical data of the data, and show the basic data description of the numerical data, such as the maximum value, minimum value, median value, upper quartile, etc. Display all categorical variables, and show the descriptive statistics of categorical variables, including the value of the maximum frequency, etc. Finally, the categorical variables are transformed to the numerical dataframe format.



### data_visualization

```python

def data_visualization(num_count, df):
    """
    :param num_count: dataframe data in numeric format
		:param df: dataframe, including numeric and categorical data
		:return: some tables of the data processing
    """
    plt.figure(figsize=(30, 60))
    for index, column in enumerate(num_count):
        plt.subplot(5, 4, index + 1)
        sns.countplot(data=num_count, x=column)
        plt.xticks(rotation=90)
    plt.tight_layout(pad=1.0)
   	    ...
   	plt.figure(figsize=(25, 25))
    sns.heatmap(num_count.corr(), cmap='RdPu', linewidths=1, annot=True, fmt='.3f')
    plt.show()
    return df
```

This function is a data visualization function that takes two parameters, num_count: numeric dataframe data; df: For the numerical data and categorical variables, we did the data processing, such as replacing the label, replacing yes with 1 and No with 0, etc. Therefore, the last return of the function is the processed data.



### object_to_numerical

```python
def object_to_numerical(df1):
    """
    :param df: dataframe 
    :return: 1. feature engineering: convert object to numerical
             2. map_dict: the mapping relation of the transformation from categorical variables to numerical variables in the form of dictionary
    """
    var_mod = []
    for c in df1.columns:
        if df1[c].dtype=='object':
            var_mod.append(c)
    le = LabelEncoder()
    map_dict = {}
    for i in var_mod:
        
        df1[i] = le.fit_transform(df1[i])
        res = {}
        for cl in le.classes_:
            res.update({cl:le.transform([cl])[0]})
        map_dict[i]=res
    # After processing the data, it can be seen that all the data is numerical
    return df1, map_dict
```

Convert a categorical variable function to numeric data by using LabelEncoder. It returns the mapping relation of the transformation from categorical variables to numerical variables in the form of dictionary



### feature_engineer

```python
def feature_engineer(X, y):
    """
    :param X: independent variable in dataframe format
    :return: 1.Normalized characteristic (independent variable) data 
             2.The method of normalization
             3.Independent variable left after feature screening
    """
    
    # feature selection
    def rmse_cv(model):
        rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 6))
        return(rmse)

    # Call LassoCV with cross validation. Default cv=10
    model_lasso = LassoCV(alphas = [0.1,1,0.001,0.0001, 0.0005]).fit(X, y)

    #    alpha, the optimal regularization parameter chosen by the model
    print(model_lasso.alpha_)

    #The parameter value or weight parameter of each feature column is 0,
    # which means that the feature has been eliminated by the model
    print(model_lasso.coef_)

    #Finally, several feature vectors are selected and several feature vectors are eliminated
    coef = pd.Series(model_lasso.coef_, index = X.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

    #Output the average residual in the case of the optimal regularized parameter selected, since it is 10% off, so look at the average
    print(rmse_cv(model_lasso).mean())
    #Draw the importance of the characteristic variables, and select the first three important and the last three unimportant examples
    imp_coef = pd.concat([coef.sort_values().head(3),
                         coef.sort_values().tail(3)])

    matplotlib.rcParams['figure.figsize'] = (12,5)
    imp_coef.plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show()
    
    rest_features=coef[coef!=0].index
    print('Lasso select features:',list(rest_features))
    
    # Data normalization
    scaler=MinMaxScaler()
    scaler_X=scaler.fit(X[rest_features])
    X=pd.DataFrame(scaler_X.transform(X[rest_features]), columns=X[rest_features].columns)
    return X, scaler_X, list(rest_features)
```

This is a feature engineering function of the data for the independent variables. First, it goes through the feature screening of LassoCV, and then carries on the feature scaling of the independent variables left by the feature screening. The scaling mode is the 0-1 normalization of MinMax. We input the independent variable in the dataframe format, then the function returns the normalized independent variables.

### model_predict ###

```python
def model_predict(model, x_train, y_train, x_test):
    """
    :param model: model, such as decision tree, random forest, etc
    :param x_train: independent variable of the training model
    :param y_train: label data of the training model :param x_test: independent variable data for the model test
    :return: 1.The test result of the model against the test data
             2. The trained model
    """
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred, model
```

This function is used for model training and prediction. Its inputs have four parameters, which are instantiated models, including decision tree,random forest, Bayesian model, etc. x_train: separated independent variable training data; Y_train: Separate dependent variable training data; After the training of x_train and y_train, the model has certain cognitive ability. Finally, the x_test data of the input model is predicted, namely the independent variable of the test. This function returns the predicted results of the test data and the trained model, so that the trained data can be easily predicted on a single item.



#### model_evaluate

```python
def model_evaluate(y_true, y_pred):
    """
    : paramy_true: indicates the actual test  data
	: paramy_pred: Label for the test data predicted by a model
	:return: Test accuracy of the model on the test data, decimal, the maximum is 1, the minimum is 0
    """
    acc = accuracy_score(y_true, y_pred)
    mat = confusion_matrix(y_true, y_pred)
    mat = pd.DataFrame(mat, columns=['No', 'Yes'])
    mat.index = mat.columns
    plt.figure(figsize=(10, 5), dpi=100)
    sns.heatmap(mat, fmt='d', annot=True)
    plt.show()
    return acc
```

This evaluation function is used to visualize the confusion matrix and return the accuracy rate of each model.