from flask import Flask, request, jsonify, render_template
import json
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- НАСТРОЙКИ ---
MODEL_PATH = "saiga-mistral-7b.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNKS_FILE = "processed_chunks.json"
FAISS_INDEX_PATH = "faiss_index"

app = Flask(__name__)
qa_chain = None

def setup_rag_pipeline():
    global qa_chain
    print("Загрузка эмбеддинг-модели...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(FAISS_INDEX_PATH):
        print("Загрузка существующего индекса FAISS...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Создание нового индекса FAISS...")
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"Индекс сохранен в {FAISS_INDEX_PATH}")

    print("Загрузка LLM...")
    llm = LlamaCpp(
        model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=0, n_batch=512, verbose=False, temperature=0.3
    )

    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    
    prompt_template = """
    Используй следующий контекст, чтобы ответить на вопрос. Отвечай кратко и только по-русски на основе предоставленной информации.
    Если ты не знаешь ответа или информация отсутствует в контексте, скажи "Я не нашел информации по этому вопросу в документах".

    Контекст: {context}

    Вопрос: {question}

    Ответ:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT}
    )
    print("RAG-пайплайн готов к работе!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_chain:
        return jsonify({"error": "Система еще не инициализирована."}), 503
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "Вопрос не может быть пустым."}), 400
    try:
        print(f"Получен вопрос: {question}")
        result = qa_chain({"query": question})
        answer = result.get('result', "Не удалось получить ответ.")
        print(f"Сгенерирован ответ: {answer}")
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Ошибка при обработке запроса: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера."}), 500

if __name__ == '__main__':
    setup_rag_pipeline()
    app.run(host='0.0.0.0', port=5000, debug=False)
