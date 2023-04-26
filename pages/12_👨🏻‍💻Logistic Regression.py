import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import gc; gc.enable()


#Display images and titles
st.image("4.png", output_format="auto")
st.title("American Express Fraud Detection")

data = st.session_state['data']

_ = gc.collect()

# #features = data.columns.to_list()
xg_features = ["S_2","B_30", "B_38",'D_114','D_116','D_117','D_120','D_126','D_63','D_64','D_66','D_68']
le_encoder = LabelEncoder()
for categorical_feature in xg_features:
    data[categorical_feature] = le_encoder.fit_transform(data[categorical_feature])


st.title("Dataset")
st.write(data.head())

smote = SMOTE(random_state=47)

X = data.drop(["target","customer_ID"], axis = 1)
y = data['target']

X, y = smote.fit_resample(X, y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=111)

start = time.time()
st.markdown("#### ⏰ Model Building in Progress..." )
model = LogisticRegression()
model.fit(X_train,y_train)
end = time.time()
st.markdown(f"### 🦾 Model Built and Fitted Succesfully in {end - start} seconds")

test_predicted = model.predict(X_test)
logloss = round(log_loss(y_test, model.predict_proba(X_test)),2)
f1score_t = round(f1_score(y_test, test_predicted, average='micro'),2)
st.markdown(f"#### 🎯 The F1 score of XGBoost Classifier Model for Test Data: {f1score_t}")
st.markdown(f"#### 🎯 The Log Loss of XGBoost Classifier Model for Test Data: {logloss}")


#PLotting Confusion Matrix
st.title("Confusion Matrix of Logistic Regression Classifier Model: \n\n")
confusion_matrix = confusion_matrix(y_test, test_predicted)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
fig,ax = plt.subplots(1,1)
sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='magma',ax=ax)
st.pyplot(fig)


#save the model
#model.save_model('LogisticRegression.json')

st.session_state['model'] = model
