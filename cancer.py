#importing dataset

import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\SRAVAN\Downloads\Cancer_Data.csv")

# Convert 'diagnosis' column: M to 1, B to 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

dataframe=df.head()

#cleaning dataset

df.info()
df.duplicated().sum()
X=df.iloc[:,2:31]
y=df.iloc[:,1]
print(y)

#train test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)   #training

#predction
y_predict=(knn.predict(X_test))

#accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)

#streamlit

import streamlit as st

st.title("BREAST CANCER DATASET")
st.header("Predict if a tumor is Malignant (cancer) or Benign (no cancer).")
if st.checkbox("show dataset"):
    st.dataframe(df.head())
st.write(f"Model Accuracy: **{accuracy:.2f}**")
st.markdown("### *Enter the details*")

#for horizontal sl;ider
st.markdown("""
    <style>
    .scroll-container {
        display: flex;
        overflow-x: auto;
        padding: 15px;
        border: 1px solid #ddd;
        background-color: #f9f9f9;
        white-space: nowrap;
    }
    .scroll-item {
        display: inline-block;
        margin-right: 20px;
        width: 250px;
    }
    </style>
""", unsafe_allow_html=True)


#input the details

st.markdown('<div class="scroll-container">', unsafe_allow_html=True)

input_data = []
for col in X.columns:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    default_val = float(X[col].mean())
    slider_html = f"""
        <div class="scroll-item">
            <p><b>{col}</b></p>
        </div>
    """
    st.markdown(slider_html, unsafe_allow_html=True)
    value = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=default_val, key=col)
    input_data.append(value)

st.markdown('</div>', unsafe_allow_html=True)
# Predict Button
if st.button("Predict"):
    input_array = np.array([input_data])
    prediction = knn.predict(input_array)[0]

    if prediction == 1:
        st.error("The tumor is **Malignant (Cancer)**")
    else:
        st.success("The tumor is **Benign (No Cancer)**")




