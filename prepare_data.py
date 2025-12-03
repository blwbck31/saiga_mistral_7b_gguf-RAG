import os
import json
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- НАСТРОЙКИ ---
PDF_DIRECTORY = "pdfs"  # Папка, в которую вы положите свои PDF-файлы
OUTPUT_FILE = "processed_chunks.json"
CHUNK_SIZE = 1500  # Длина чанка в символах (примерно 200-250 слов)
CHUNK_OVERLAP = 300   # Перекрытие между чанками в символах (20%)

def extract_text_from_pdfs(pdf_dir):
    """Извлекает текст из всех PDF-файлов в указанной директории."""
    full_text = ""
    print(f"Начинаю извлечение текста из PDF в папке '{pdf_dir}'...")
    
    if not os.path.isdir(pdf_dir):
        print(f"Ошибка: Директория '{pdf_dir}' не найдена.")
        return ""
        
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"В папке '{pdf_dir}' не найдено PDF-файлов.")
        return ""

    for filename in pdf_files:
        filepath = os.path.join(pdf_dir, filename)
        try:
            doc = fitz.open(filepath)
            print(f"  - Обрабатываю файл: {filename} ({len(doc)} страниц)")
            for page_num, page in enumerate(doc):
                # Добавляем текст страницы и разделитель, чтобы сохранить структуру
                text = page.get_text("text")
                if text:
                    # Простое удаление разрывов строк внутри предложений
                    clean_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
                    full_text += clean_text + "\n"
            doc.close()
        except Exception as e:
            print(f"    Не удалось обработать файл {filename}: {e}")

    print("Извлечение текста завершено.")
    return full_text

def main():
    """Основная функция для подготовки данных."""
    # 1. Извлечение текста
    raw_text = extract_text_from_pdfs(PDF_DIRECTORY)
    
    if not raw_text:
        print("Текст не был извлечен. Завершение работы.")
        return

    # 2. Разбиение на чанки
    print("Разбиение текста на чанки...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""], # Приоритет разделителей
        length_function=len
    )
    
    chunks = text_splitter.split_text(raw_text)
    
    # 3. Сохранение чанков в JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
        
    print(f"Данные обработаны и разделены на {len(chunks)} чанков.")
    print(f"Результат сохранен в файл: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
