# Проект: Бинарная классификация для предиктивного обслуживания оборудования
## Описание проекта
Цель проекта — разработать модель машинного обучения, которая
предсказывает, произойдет ли отказ оборудования (Target = 1) или нет
(Target = 0). Результаты работы оформлены в виде Streamlit-приложения
## Датасет
Используется датасет **"AI4I 2020 Predictive Maintenance Dataset"**,
содержащий 10 000 записей с 14 признаками. Подробное описание датасе
можно найти в [документации](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+data)
## Установка и запуск
1. Клонируйте репозиторий:
`git clone <ссылка на репозиторий>`
2. Установите зависимости:
`pip install -r requirements.txt`
3. Запустите приложение:
`streamlit run app.py`
## Структура репозитория
- `app.py`: Основной файл приложения.
- `analysis_and_model.py`: Страница с анализом данных и моделью.
- `presentation.py`: Страница с презентацией проекта.
- `requirements.txt`: Файл с зависимостями.
- `data/`: Папка с данными.
- `video/`: Папка с видео-демонстрацией.
- `README.md`: Описание проекта.
## Видео-демонстрация
[Ссылка на видео-демонстрацию](video/demo.mp4) или видео ниже:
[Видео-демонстрация](https://nifton-gh.github.io/predictive_maintenance_project/)

