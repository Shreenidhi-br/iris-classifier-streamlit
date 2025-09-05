import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🌸 Iris Flower Classifier")
st.write("A Machine Learning Web App to classify Iris flowers into Setosa, Versicolor, or Virginica.")

# Input fields
sl = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sw = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, step=0.1)
pl = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
pw = st.number_input("Petal Width (cm)", min_value=0.1, max_value=3.0, step=0.1)

# Prediction button
if st.button("Predict"):
    sample = np.array([[sl, sw, pl, pw]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    st.success(f"🌼 Predicted Species: {iris.target_names[prediction][0]}")
