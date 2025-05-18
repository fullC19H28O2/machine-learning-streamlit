
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="ML Eğitimi ve Optimizasyon", layout="wide")
st.sidebar.title("🔍 Menü")
page = st.sidebar.radio("Sayfa Seç", ["Model Eğitimi", "Hiperparametre Optimizasyonu"])

uploaded_file = st.file_uploader("Veri seti yükle (.csv)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Yüklenen Veri Seti")
    st.dataframe(df)

    target_column = st.selectbox("🎯 Hedef sütunu seçin", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        if page == "Model Eğitimi":
            st.title("📘 Model Eğitimi")
            model_name = st.selectbox("Model seçin", [
                "Lojistik Regresyon",
                "Karar Ağacı",
                "Rastgele Orman",
                "Destek Vektör Makineleri",
                "K-En Yakın Komşu"
            ])

            if st.button("Modeli Eğit"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                if model_name == "Lojistik Regresyon":
                    model = LogisticRegression(max_iter=1000)

                elif model_name == "Karar Ağacı":
                    model = DecisionTreeClassifier()

                elif model_name == "Rastgele Orman":
                    model = RandomForestClassifier()

                elif model_name == "Destek Vektör Makineleri":
                    model = SVC(probability=True)

                elif model_name == "K-En Yakın Komşu":
                    model = KNeighborsClassifier()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.write("Sınıflandırma Metrikleri")
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.write(f"Precision: {precision_score(y_test, y_pred, average='macro'):.2f}")
                st.write(f"Recall: {recall_score(y_test, y_pred, average='macro'):.2f}")
                st.write(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.2f}")
                st.write(f"MCC: {matthews_corrcoef(y_test, y_pred):.2f}")

                st.write("Sınıflandırma Matrisi")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                st.write("Sınıflandırma Raporu")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

        elif page == "Hiperparametre Optimizasyonu":
            st.title("🔧 Hiperparametre Optimizasyonu")
            model_choice = st.selectbox("ML Modeli Seçin", ["Lojistik Regresyon", "Rastgele Orman", "Destek Vektör Makineleri"])
            optimize = st.button("Optimize Et")

            if optimize:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if model_choice == "Lojistik Regresyon":
                    param_grid = {"C": [0.1, 1, 10]}
                    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

                elif model_choice == "Rastgele Orman":
                    param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
                    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

                elif model_choice == "Destek Vektör Makineleri":
                    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)

                st.subheader("📌 En İyi Parametreler")
                st.json(grid.best_params_)

                st.subheader("📊 Sınıflandırma Matrisi")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                st.subheader("📄 Sınıflandırma Raporu")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
