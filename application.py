import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

st.title("Makine Öğrenmesi Sınıflandırma Paneli")

uploaded_file = st.file_uploader("Veri seti yükle (.csv)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Yüklenen Veri Seti")
    st.dataframe(df)

    target_column = st.selectbox("Hedef sütunu seçin", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

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
            y_prob = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba") and len(model.classes_) == 2
                else None
            )

            # Metirkler
            st.write("Sınıflandırma Metrikleri")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            mcc = matthews_corrcoef(y_test, y_pred)

            st.write(f"Accuracy: {acc:.2f}")
            st.write(f"Precision: {prec:.2f}")
            st.write(f"Recall: {rec:.2f}")
            st.write(f"F1 Score: {f1:.2f}")
            st.write(f"MCC (Matthews Corr Coef): {mcc:.2f}")

            # Confusion matrix
            st.write("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Tahmin")
            ax.set_ylabel("Gerçek")
            st.pyplot(fig)

            # ROC-AUC
            if y_prob is not None:
                st.write("ROC-AUC Grafiği")
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title("ROC Curve")
                ax2.legend(loc="lower right")
                st.pyplot(fig2)

            # Başarı grafiği
            st.write("Başarı Bar Grafiği")
            metrics = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "MCC": mcc
            }
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())

            fig3, ax3 = plt.subplots()
            sns.barplot(x=metric_names, y=metric_values, ax=ax3)
            ax3.set_ylim(0, 1)
            ax3.set_title("Model Performans Metrikleri")
            st.pyplot(fig3)

            # Modeli kaydet
            joblib.dump(model, "egitilmis_model.pkl")
            st.success("Model başarıyla 'egitilmis_model.pkl' dosyasına kaydedildi.")

        # Harici test verisi
        st.subheader("Harici Test Verisi ile Tahmin")
        test_file = st.file_uploader("Test verisi yükle (.csv)", type="csv", key="test")

        if test_file is not None:
            test_df = pd.read_csv(test_file)
            st.write("Yüklenen test verisi:")
            st.dataframe(test_df)

            try:
                model = joblib.load("egitilmis_model.pkl")
                test_predictions = model.predict(test_df)
                st.write("Tahmin Sonuçları:")
                st.dataframe(pd.DataFrame({"Tahmin": test_predictions}))
            except Exception as e:
                st.error(f"Model yüklenemedi veya tahmin başarısız: {e}")