# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# # Загрузка данных
# data = pd.read_csv("data/predictive_maintenance.csv")
# # Удаление ненужных столбцов
# data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF',
# 'RNF'])
# # Преобразование категориальной переменной Type в числовую
# data['Type'] = LabelEncoder().fit_transform(data['Type'])

# # Проверка на пропущенные значения
# # print(data.isnull().sum())

# from sklearn.preprocessing import StandardScaler
# # Масштабирование числовых признаков
# scaler = StandardScaler()
# numerical_features = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
# data[numerical_features] = scaler.fit_transform(data[numerical_features])
# # Вывод первых строк данных после масштабирования
# # print(data.head())

# from sklearn.model_selection import train_test_split
# # Признаки (X) и целевая переменная (y)
# X = data.drop(columns=['Machine failure'])
# y = data['Machine failure']
# # Разделение данных
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# from sklearn.linear_model import LogisticRegression
# # Создание и обучение модели
# log_reg = LogisticRegression()
# log_reg.fit(X_train, y_train)

# from sklearn.ensemble import RandomForestClassifier
# # Создание и обучение модели
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# from xgboost import XGBClassifier
# # Создание и обучение модели
# xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
# xgb.fit(X_train, y_train)

# from sklearn.svm import SVC
# # Создание и обучение модели
# svm = SVC(kernel='linear', random_state=42, probability=True) #probability=True для ROC-AUC
# svm.fit(X_train, y_train)

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
# import matplotlib.pyplot as plt
# # Функция для оценки модели
# def evaluate_model(model, X_test, y_test):
#     # Предсказания
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)[:, 1] # Вероятности для ROC-AUC
#     # Метрики
#     accuracy = accuracy_score(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     class_report = classification_report(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred_proba)
#     # Вывод результатов
#     print("Accuracy:", accuracy)
#     print("Confusion Matrix:\n", conf_matrix)
#     print("Classification Report:\n", class_report)
#     print("ROC-AUC:", roc_auc)
#     # Построение ROC-кривой
#     fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#     plt.plot(fpr, tpr, label=f"{model.__class__.__name__} (AUC = {roc_auc:.2f})")

# # Оценка Logistic Regression
# print("Logistic Regression:")
# evaluate_model(log_reg, X_test, y_test)
# # Оценка Random Forest
# print("Random Forest:")
# evaluate_model(rf, X_test, y_test)
# # Оценка XGBoost
# print("XGBoost:")
# evaluate_model(xgb, X_test, y_test)
# # Оценка SVM
# print("SVM:")
# evaluate_model(svm, X_test, y_test)
# # Визуализация ROC-кривых
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC-кривые')
# plt.legend()
# plt.show()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder

def analysis_and_model_page():
    st.title("Анализ данных и модель")
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Удаление ненужных столбцов
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        # Преобразование категориальной переменной Type в числовую
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Проверка на пропущенные значения
        # print(data.isnull().sum())

        from sklearn.preprocessing import StandardScaler
        # Масштабирование числовых признаков
        scaler = StandardScaler()
        numerical_features = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        # Вывод первых строк данных после масштабирования
        # print(data.head())

        # Признаки (X) и целевая переменная (y)
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        # Оценка модели (дописать)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        
        st.header("Результаты обучения модели")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        st.subheader("Classification Report")
        st.text(classification_rep)

        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для предсказания:")
            productID = st.selectbox("productID", ["L", "M", "H"])
            air_temp = st.number_input("air temperature [K]")
            process_temp = st.number_input("process temperature [K]")
            rotational_speed = st.number_input("rotational speed [rpm]")
            torque = st.number_input("torque [Nm]")
            tool_wear = st.number_input("tool wear [min]")
            submit_button = st.form_submit_button("Предсказать")

        if submit_button:
                    # Преобразование введенных данных (дописать)
            input_dict = {
                "Type": [productID],  
                "Air temperature": [air_temp],  
                "Process temperature": [process_temp],  
                "Rotational speed": [rotational_speed],  
                "Torque": [torque],  
                "Tool wear": [tool_wear]  
            }
            
            input_df = pd.DataFrame(input_dict)
            
            type_mapping = {"L": 0, "M": 1, "H": 2}
            input_df["Type"] = input_df["Type"].map(type_mapping)
            
            input_df[numerical_features] = scaler.transform(input_df[numerical_features])

            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)[:, 1]
            st.write(f"Предсказание: {prediction[0]}")
            st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")

analysis_and_model_page()

