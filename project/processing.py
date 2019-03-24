import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

import random
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# For Keras usage
from keras import optimizers
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# To achieve same accuracies everytime
np.random.seed(42)

# The columns to be used as input
# TODO: decide on which ones to use
# got these recommended from https://www.kaggle.com/uds5501/conflicting-result-classifications
# x_columns = ['CHROM', 'REF', 'ALT', 'CLNVC', 'MC', 'ORIGIN', 'Allele', 'Consequence', 'IMPACT', 'Feature_type', 'BIOTYPE', 'STRAND']
onehot_columns = ['CHROM']
numeric_columns = ['POS', 'AF_ESP', 'AF_EXAC', 'AF_TGP', 'STRAND', 'CADD_PHRED']
x_columns = onehot_columns + numeric_columns

def load_data():
    enc = OneHotEncoder()
    df = pd.read_csv("clinvar_conflicting.csv", dtype={0: object, 38: str, 40: object})
    
    # Ensuring that our final dataset has equal distribution of 1's and 0's
    g = df.groupby('CLASS')
    df_balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    df = df_balanced
    
    # Sampling the dataset - too large to train on laptop with full dataset
    # df_balanced = df_balanced.sample(3000)
    
    df = df[x_columns + ['CLASS']].dropna()
    # df = df_balanced[x_columns + ['CLASS']].dropna()
    # df_x = df[x_columns]
    df_onehot = df[onehot_columns]
    df_numeric = df[numeric_columns]
    y = df['CLASS']
    # Naive for now: just one-hot encode every attribute
    # OneHotEncoder had problems with datatype even after updating it.
    X_onehot = pd.get_dummies(df_onehot,drop_first=True)
    # X_onehot = enc.fit_transform(df_onehot)
    X = pd.concat([X_onehot, df_numeric], axis=1)
    print("x shape", X.shape)
    return X,y

# Main Code begins here:

X,y = load_data()
print("Loaded data, final shape: ", X.shape)
print("Target data, final shape: ", y.shape)

train_X, test_X, train_y, test_y = train_test_split(X, y)
# ros = RandomOverSampler(random_state = 42)
ros = RandomUnderSampler(random_state = 42)
# ros = SMOTE(random_state = 42)
# train_X,train_y = ros.fit_resample(train_X,train_y)
rus = RandomUnderSampler(random_state=42)
# test_X,test_y = rus.fit_resample(test_X,test_y)
print("Training data size: ", train_X.shape)
print("Test data size: ", test_X.shape)

# Normalize using StandardScaler
scaler=StandardScaler()
train_X=scaler.fit_transform(train_X)
test_X=scaler.transform(test_X)

#NOTE: If you are trying to add some of your models - your input will be train_X and then test it on test_X.

# Models to try
try_models = [LogisticRegression(), 
          LogisticRegressionCV(), 
          KNeighborsClassifier(n_neighbors=20),
          DecisionTreeClassifier(), 
          RandomForestClassifier(n_estimators=500), 
          MLPClassifier(hidden_layer_sizes=(300,),verbose=True,max_iter=500,alpha=0.00001),
          SVC()
         ]
# Gather metrics here
accuracy_by_model={}

# Train then evaluate each model
i = 0
for model in try_models:
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    score = accuracy_score(test_y, pred_y)
    # Fill metrics dictionary
    model_name = model.__class__.__name__ + str(i)
    i += 1
    accuracy_by_model[model_name]=score  
    print(model_name)
    print(classification_report(test_y, pred_y))
    
# Draw accuracy by model chart
acc_df = pd.DataFrame(list(accuracy_by_model.items()), columns=['Model', 'Accuracy']).sort_values('Accuracy', ascending=False).reset_index(drop=True)
acc_df.index=acc_df.index+1
sns.barplot(data=acc_df,y='Model',x='Accuracy')
plt.xlim(0,1)
plt.title('Accuracy of models')
plt.xticks(rotation=45)
# plt.show()

# Print table
print(acc_df)




