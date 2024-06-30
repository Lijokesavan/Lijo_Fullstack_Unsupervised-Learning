# 1. Importing Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

#from google.colab import drive
#drive.mount('/content/drive')
#df = pd.read_csv('/content/drive/MyDrive/heart_data.csv')

df = pd.read_csv('heart_data.csv')
df.head()

# 3. Check the shape of the data (number of rows and columns). Check the general information about the dataframe using the .info() method.
df.shape
df.info()

# 4. Check the statistical summary of the dataset and write your inferences.
df.describe().T
df.describe(include='O')

# 5. Check the percentage of missing values in each column of the data frame. Drop the missing values if there are any.
df.isnull().sum()/len(df)*100

# 6. Check if there are any duplicate rows. If any drop them and check the shape of the dataframe after dropping duplicates.
df.duplicated().sum()
df = df.drop_duplicates()
df.shape

# 7. Check the distribution of the target variable (i.e. 'HeartDisease') and write your observations.
df['HeartDisease'].value_counts().plot(kind='pie',autopct='%1.0f%%')
plt.show()

# 8. Visualize the distribution of the target column 'heart disease' with respect to various categorical features and write your observations.
categorical_features = df.select_dtypes(include=[np.object])
categorical_features.columns

plt.figure(figsize = (30,25))
for i, feature in enumerate(categorical_features):
    plt.subplot(6, 3, i+1)
    sns.countplot(x=feature, hue='HeartDisease', data=df)
plt.show()

# 9. Check the unique categories in the column 'Diabetic'. Replace 'Yes (during pregnancy)' as 'Yes' and 'No, borderline diabetes' as 'No'.
df['Diabetic'].unique()
df['Diabetic'] = df['Diabetic'].replace({'Yes (during pregnancy)':'Yes','No, borderline diabetes':'No'})
df['Diabetic'].value_counts()

# 10. For the target column 'HeartDiease', Replace 'No' as 0 and 'Yes' as 1. 
df['HeartDisease'] = df['HeartDisease'].replace({'Yes':1, 'No':0})
df['HeartDisease'].value_counts()

# 11. Label Encode the columns "AgeCategory", "Race", and "GenHealth". Encode the rest of the columns using dummy encoding approach.
object_type_variables = ['AgeCategory', 'Race', 'GenHealth']

le = LabelEncoder()
for col in object_type_variables:
    df[col] = le.fit_transform(df[col])

df = pd.get_dummies(df, drop_first=True)
df.head(2)

# 12. Store the target column (i.e.'HeartDisease') in the y variable and the rest of the columns in the X variable.
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# 13. Split the dataset into two parts (i.e. 70% train and 30% test) and print the shape of the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# 14. Standardize the numerical columns using Standard Scalar approach for both train and test data.
ss = StandardScaler()

X_train.iloc[:, :7] = ss.fit_transform(X_train.iloc[:, :7])
X_test.iloc[:, :7] = ss.transform(X_test.iloc[:, :7])

X_train.head(2)
X_test.head(2)

# 15. Write a function.
def fit_n_print(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    return accuracy

# 16. Use the function and train various classifiers.
lr = LogisticRegression()
nb = GaussianNB()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
adb = AdaBoostClassifier()
gb = GradientBoostingClassifier()

estimators = [('rf', rf), ('knn', knn), ('gb', gb), ('adb', adb)]
sc = StackingClassifier(estimators=estimators, final_estimator=rf)

result = pd.DataFrame(columns = ['Accuracy'])

for model, model_name in zip([lr, nb, knn, dt, rf, adb, gb, sc], 
                             ['Logistic Regression','Naive Bayes','KNN','Decision tree', 
                              'Random Forest', 'Ada Boost', 'Gradient Boost', 'Stacking']):
    
    result.loc[model_name] = fit_n_print(model, X_train, X_test, y_train, y_test)

    result