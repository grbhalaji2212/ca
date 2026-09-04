import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


iris = load_iris()


model = RandomForestClassifier()
model.fit(iris.data, iris.target)


st.title(" Iris Flower Classifier")


sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")


if st.button("Predict"):

    data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(data)

    st.success(
        "Predicted Species: " +
        iris.target_names[prediction[0]]
    )
