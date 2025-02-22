import streamlit as st
# Настройка навигации
pages = {
    "Анализ и модель": st.Page("analysis_and_model.py", title="Анализ и модель"),
    "Презентация": st.Page("presentation.py", title="Презентация"),
}
# Отображение навигации
current_page = st.navigation([pages["Анализ и модель"], pages["Презентация"]], position="sidebar", expanded=True)
current_page.run()
# import streamlit as st

# def analysis_and_model_page():
#     st.title("Анализ и модель")
#     # Ваш основной контент

# def presentation_page():
#     st.title("Презентация")
#     # Ваш код презентации

# # Настройка страниц
# pages = {
#     "Анализ и модель": analysis_and_model_page,
#     "Презентация": presentation_page,
# }

# # Показ навигации в сайдбаре
# st.sidebar.title("Навигация")
# selected_page = st.sidebar.radio("Выберите страницу", list(pages.keys()))

# # Отображение выбранной страницы
# pages[selected_page]()