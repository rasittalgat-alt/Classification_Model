# Классификация наименований ТРУ


## 1. Идея

По колонке `DESCRIPTION` (текст наименования ТРУ) предсказываю категорию  
(например, «Продукты питания и напитки», «Обувь и средства индивидуальной защиты (СИЗ)» и т.п.).

Подход:
- предобработка текста (lowercase, нормализация пробелов);
- TF-IDF по символьным n-граммам (3–5);
- линейный классификатор `SGDClassifier (loss="log_loss")`.

Обучение выполняется на вручную размеченном сэмпле `esf_label_sample_5000.csv` (21 категория).

---

## 2. Установка

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

```

## 3. Подготовка данных (локально)

Исходный большой файл esf_fulll_2025....csv находится локально и не выкладывается в GitHub.
Путь к нему указывается в константе SRC_PATH в make_sample.py.


```bash
# 1) Делаем облегчённый сэмпл (200k строк) из большого CSV
python make_sample.py
# создаётся файл esf_sample_200k.csv (колонка DESCRIPTION)

# 2) Формируем выборку для ручной разметки категорий (5k строк)
python make_label_sample.py
# создаётся файл esf_label_sample_5000.csv с колонками DESCRIPTION и CATEGORY

```
## 4. Обучение модели

```bash
python train_model.py

```
Наш скрипт:

- читает esf_label_sample_5000.csv;
- обучает TF-IDF + SGDClassifier (loss="log_loss");
- выводит метрики (accuracy, F1);
- сохраняет модель в models/tfidf_sgd_model.pkl (файл игнорируется в git).

## 5. Инференс

```bash
python infer_model.py

```

Наш скрипт:

- выводит примеры текст → предсказанная категория в консоль;
- применяет модель к esf_sample_200k.csv;
- сохраняет результат в esf_sample_200k_predicted.csv с колонкой PREDICTED_CATEGORY.

  ## 6. Категории

Всего используется 21 категория («Продукты питания и напитки», «Одежда и текстиль», «Обувь и средства индивидуальной защиты (СИЗ)»,
«Лекарства», «Прочие услуги» и др.). Полный список можно посмотреть в колонке CATEGORY файла esf_label_sample_5000.csv.











