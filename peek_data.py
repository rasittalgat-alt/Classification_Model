import pandas as pd
from pathlib import Path

# полный путь к CSV
csv_path = Path(r"C:\Users\talgat.rashit\Desktop\AML Тестовое задание\esf_fulll_202511211949.csv")

print("Файл существует?", csv_path.exists())
print("Путь к файлу:", csv_path, "\n")

# пробуем разные разделители: ; и ,
for sep in [";", ","]:
    print("=" * 40)
    print(f"Пробуем читать с разделителем '{sep}'")
    try:
        df = pd.read_csv(csv_path, sep=sep, nrows=10)
        print("Колонки:", list(df.columns))
        print("\nПервые строки:")
        print(df.head())
        break  # если успешно прочитали — выходим из цикла
    except Exception as e:
        print(f"Ошибка для разделителя '{sep}': {e}")
