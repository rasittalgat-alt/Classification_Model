import pandas as pd
from pathlib import Path

# исходный большой файл. Указываем путь к файлу
SRC_PATH = Path(r"C:\Users\talgat.rashit\Desktop\AML Тестовое задание\esf_fulll_202511211949.csv")

# выходной облегченный файл
DST_PATH = Path("esf_sample_200k.csv")

N_ROWS = 200_000  # строк хотим в сэмпле
CHUNK_SIZE = 50_000  # размер кусочка при чтении

def main():
    print("Источник:", SRC_PATH)
    print("Облегченный файл будет сохранен как:", DST_PATH)

    if not SRC_PATH.exists():
        print("❌ Исходный файл не найден!")
        return

    total_kept = 0
    chunks = []

    print(f"Читаем файл по {CHUNK_SIZE} строк...")

    # Важно: engine='python' + on_bad_lines='skip' — пропускаем проблемные строки
    reader = pd.read_csv(
        SRC_PATH,
        sep=";",                # как мы уже выяснили
        engine="python",
        on_bad_lines="skip",    # пропускать строки с неправильным числом колонок
        chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader, start=1):
        print(f"Обрабатываем чанк {i}, размер {len(chunk)}")

        # Если уже набрали достаточно строк - останавливаемся
        rows_left = N_ROWS - total_kept
        if rows_left <= 0:
            break

        if len(chunk) > rows_left:
            chunk = chunk.iloc[:rows_left]

        chunks.append(chunk)
        total_kept += len(chunk)
        print(f"Набрано строк: {total_kept}")

        if total_kept >= N_ROWS:
            break

    if not chunks:
        print("❌ Не удалось собрать ни одного чанка.")
        return

    df = pd.concat(chunks, ignore_index=True)
    print("Итоговая форма датафрейма:", df.shape)
    print("Колонки:", list(df.columns))

    print("Сохраняем выборку...")
    df.to_csv(DST_PATH, index=False, encoding="utf-8")

    print("✅ Готово! Файл сохранен как:", DST_PATH.resolve())

if __name__ == "__main__":
    main()
