# README
## Motivation
We are interested in exploring the relationships between attrition outcomes and different employee attributes in the healthcare industry. We will provide a detailed data analysis on those employee attributes and build classification models to predict the attrition outcomes in healthcare for any given employee data. It will be helpful for the government to examine the attrition in healthcare. As a result, the HR department can adjust the current organizational structure/policy to reduce future attrition. 


## Description
The project aims to predict employee turnover in the health care industry. Data sets can be found and downloaded from Kaggle( [watson_healthcare_modified.csv](https://www.kaggle.com/datasets/jpmiller/employee-attrition-for-healthcare)). 

The dataset consists of 1677 rows and 35 columns,  the attributes including one dependent variable named Attrition, and independent variables such as the employee background information (age, education level, education field, distance from home…etc) and job-related variables (department, job level, job satisfaction, percent salary hike, year at the company, years with current manager…etc). 

Raw data is complex, and we explore, analyze, and process it through functionalization. After processing, we conducted data cleaning, data partitioning training and testing, and verified the accuracy of the classification on each model.

In the data exploration stage, we understand the basic information of the data, such as the dimension of the data, the type of the data field and so on. During the analysis phase, we drew some conclusions from the visualized images. In the data processing stage, we process the data into numerical fields, and finally build, train and evaluate the models, and visualize and compare the accuracy indicators of each model

## Getting Started 

The project operating environment is python3.X version, and the overall project requires the following third-party libraries:

- matplotlib
- numpy
- scikit-learn==1.1.1
- seaborn

If your compiler is jupyter, open the INF1340 Final Project.ipynb directly with the jupyter notebook and run it cell by cell until the end. If your compiler is pycharm or IDLE, Run INF1340 Final Project.py directly, or if you are running from a terminal, type the command line:

```shell
python INF1340 Final Project.py
```



## Output

This code carries out a series of data analysis and data processing. After running the code, there will be a lot of images and tables output, the main ones are:

1. Display all useful numeric variables 

![image-20221206202512247](archive/image-20221206202512247.png)

2. HourlyRate、DailyRate、MonthlyRate、MonthlyIncome box plot

![image-20221206202652789](archive/image-20221206202652789.png)

3. The Percentage of Employee Attrition per Gender

![](archive/image-20221206202814838.png)

4. Percent Salary Hike

![image-20221206202946222](archive/image-20221206202946222.png)

5. Display a heatmap to show the correlation between features

![image-20221206203008839](archive/image-20221206203008839.png)

The above output is belonging to feature engineering. We used the decision tree model, random forest model, Bayesian model, SVM and logistic regression model to do the predictive analysis, and finally carried out the single item prediction.

1.Feature Selection:

Feature screening for LassoCV's ten fold cross validation, Lasso regression is regularization method to minimize overfitting in machine learning models and is widely used in industry. In many projects, especially feature selection will use it. Lasso adds L1 regularization to simple linear regression, which can shrink the coefficient of insignificant variables to 0, which is a very important feature of Lasso regression model. Therefore, feature selection is realized based on this.

Through the Lasso model feature selection with 10 fold cross-validation, we drop 4 variables and select 29 factors that are related to attrition. Among them, the 29 variables are screened respectively.

```
Lasso picked 29 variables and eliminated the other 4 variables
Lasso select features: ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'RelationshipSatisfaction', 'Shift', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
```

The feature of importance ranking of LassoCV screening is visualized as：

![image-20221208154349347](archive/image-20221208154349347.png)

OverTime is considered the most important feature.

2.Application of model

(1) Decision Tree Classifier Model

```
tree accuracy: 0.8711217183770883
```

![image-20221206203134985](archive/image-20221206203134985.png)

(2) Random Forest Classifier Model

```
rfc accuracy: 0.9116945107398569
```

![image-20221206203309583](archive/image-20221206203309583.png)

(3) Logistic Regression Model

```
logistic accuracy: 0.9045346062052506
```

![image-20221206203253863](archive/image-20221206203253863.png)

(4) Bayes Model

```
bayes accuracy: 0.7804295942720764
```

![image-20221206203331357](archive/image-20221206203331357.png)

(5) SVM Model

![image-20221206203343248](archive/image-20221206203343248.png)

```
svm accuracy: 0.9093078758949881
```

So the best model is random forest model with 91% accuracy rate, the worst is Bayesian model which has 78% accuracy rate.





3.Predict Single Item:

Only the values of 29 variables can be used as inputs for the feature screening and the optimal random forest model can be used for sample prediction. This can predict the input data after the normalization. The mapping relationship of classification variables is:

```json
{'BusinessTravel': {'Non-Travel': 0,
  'Travel_Frequently': 1,
  'Travel_Rarely': 2},
 'Department': {'Cardiology': 0, 'Maternity': 1, 'Neurology': 2},
 'EducationField': {'Human Resources': 0,
  'Life Sciences': 1,
  'Marketing': 2,
  'Medical': 3,
  'Other': 4,
  'Technical Degree': 5},
 'JobRole': {'Admin': 0,
  'Administrative': 1,
  'Nurse': 2,
  'Other': 3,
  'Therapist': 4},
 'MaritalStatus': {'Divorced': 0, 'Married': 1, 'Single': 2},
 'Over18': {'Y': 0},
 'OverTime': {'No': 0, 'Yes': 1}}
```

Prediction of samples [35, 2, 809, 1, 16, 3, 3, 1, 1, 1, 84, 4, 1, 2, 2, 1, 2426, 16479, 0, 0, 0, 13, 3, 1, 6, 5, 3, 5, 4, 0, 3], The input characteristic variables are successively represented as:

```
['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'RelationshipSatisfaction', 'Shift', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
```

```python
# predict single item
import warnings
warnings.filterwarnings("ignore")
print("Please enter the following characteristic data:")
print(features)
test_array = [35, 2, 809, 1, 16, 3, 3, 1, 1, 84, 4, 1, 2, 2, 1, 2426, 16479, 0, 0, 13, 3, 1, 6, 5, 3, 5, 4, 0, 3]
scaler_sample = scaler_method.transform(np.array(test_array).reshape(1,-1))
predict_label = rfc_model.predict(scaler_sample)
print("predict label:", predict_label[0])
```

The final prediction is 0

```
predict label: 0
```