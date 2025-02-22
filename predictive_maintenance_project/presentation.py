# import streamlit as st
# import reveal_slides as rs

# def presentation_page():
#     st.title("Презентация проекта")
#     # Содержание презентации в формате Markdown
#     presentation_markdown = """
#     # Прогнозирование отказов оборудования
#     ---
#     ## Введение
#     - Описание задачи и датасета.
#     - Цель: предсказать отказ оборудования (Target = 1) или его отсутствие
#     (Target = 0).
#     ---
#     ## Этапы работы
#     1. Загрузка данных.
#     2. Предобработка данных.
#     3. Обучение модели.
#     4. Оценка модели.
#     5. Визуализация результатов.
#     ---
#     ## Streamlit-приложение
#     - Основная страница: анализ данных и предсказания.
#     - Страница с презентацией: описание проекта.
#     ---
#     ## Заключение
#     - Итоги и возможные улучшения.
#     """
#     # Настройки презентации
#     with st.sidebar:
#         st.header("Настройки презентации")
#         theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
#         height = st.number_input("Высота слайдов", value=500)
#         transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
#         plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])
#         # Отображение презентации
#         rs.slides(
#             presentation_markdown,
#             height=height,
#             theme=theme,
#             config={
#                 "transition": transition,
#                 "plugins": plugins,
#             },
#             markdown_props={"data-separator-vertical": "^--$"},
#         )

# presentation_page()
import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.set_page_config(layout="wide")  # Разворачиваем на всю ширину
    st.title("Презентация проекта")

    # Markdown для презентации
    presentation_markdown = """
    # Прогнозирование отказов оборудования
    ---

    ## 1. Введение
    - Проблема: отказы промышленного оборудования могут привести к сбоям и финансовым потерям.
    - Цель проекта: предсказать возможные отказы оборудования заранее.
    - Используемые технологии: **Python, Streamlit, scikit-learn, Pandas**.
    ---

    ## 2. Датасет и предобработка данных
    - Источник данных: [добавить источник, если есть].
    - Основные признаки:
    - **Температура** воздуха и процесса.
    - **Скорость вращения**, крутящий момент.
    - **Износ инструмента** и другие.
    - Данные обработаны:
    - Удалены ненужные столбцы.
    - Кодирован категориальный признак **Type**.
    - Применено **масштабирование** числовых признаков.
    ---

    ## 3. Обучение модели
    1. Разделение данных на **обучающую (80%) и тестовую (20%) выборки**.
    2. Используем **логистическую регрессию**.
    3. Оцениваем качество модели по **Accuracy, Confusion Matrix, F1-score**.

    ---

    ## 4. Оценка модели
    - Точность (Accuracy): **XX.XX%** (указать точное значение).
    - Confusion Matrix:
    - **True Positives (TP)**: предсказали отказ, и он реально произошёл.
    - **False Positives (FP)**: предсказали отказ, но он не произошёл.
    - Precision, Recall, F1-score для оценки качества модели.
    ---

    ## 5. Streamlit-приложение
    - **Основная страница**:
    - Анализ данных (графики, корреляции).
    - Обучение модели.
    - Интерфейс предсказаний.
    - **Страница с презентацией**:
    - Описание проекта.
    - Выводы.
    ---

    ## 6. Выводы и улучшения
    - **Результаты**: модель даёт точность **XX%**, но её можно улучшить.
    - **Возможные улучшения**:
    - Добавление других алгоритмов (Random Forest, SVM).
    - Устранение дисбаланса классов (если данные несбалансированы).
    - Инженерия новых признаков.
    """

    st.sidebar.header("Настройки презентации")
    theme = st.sidebar.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
    height = st.sidebar.slider("Высота слайдов", min_value=300, max_value=1000, value=600)
    transition = st.sidebar.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
    plugins = st.sidebar.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

    st.markdown("<style>.block-container { max-width: 100% !important; }</style>", unsafe_allow_html=True)
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )

presentation_page()

