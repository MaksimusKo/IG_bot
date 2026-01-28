# -*- coding: utf-8 -*-
"""
integrated_rag_instagram.py

Интеграция RAG (FAISS + OpenAI) с Instagram API через FastAPI.
Читает PDF-файлы, создает векторный индекс (FAISS), затем отвечает на вопросы через Instagram.
"""

import os
import glob
import faiss
import numpy as np
from openai import OpenAI
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
import requests
from pyngrok import ngrok
import uvicorn
import json

# ======================================================================
# 1) Если не установлены, установите нужные библиотеки:
#    pip install faiss-cpu sentence-transformers pypdf2 openai fastapi uvicorn pyngrok requests
#
# 2) Задайте свой ключ OpenAI и другие необходимые токены:
OPENAI_API_KEY = ''
VERIFY_TOKEN = ""
APP_USER_IG_ID = ""
NGROK_AUTH_TOKEN = ""
ACCESS_TOKEN = ""
GRAPH_API_URL = "https://graph.instagram.com/v21.0"
PDF_FOLDER_PATH = r"D:\projects\work1\pdfs"  # Укажите путь к вашим PDF-файлам
# ======================================================================

# Инициализация OpenAI клиента
client = OpenAI(api_key=OPENAI_API_KEY)

# Глобальные переменные для хранения индекса и документов
index = None
documents = None
embedding_model = None

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает весь текст из PDF-файла.
    """
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""
    return text

def load_and_chunk_pdfs(folder_path: str):
    """
    Ищет все PDF в папке, извлекает из каждого текст,
    разбивает на чанки, возвращает список чанков (documents).
    """
    pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))
    docs = []

    for pdf_file in pdf_files:
        pdf_text = extract_text_from_pdf(pdf_file)
        # Разбиваем текст на абзацы или блоки по двойным переводам строки
        chunks = pdf_text.split('\n\n')
        # Очищаем от пустых строк
        chunks = [c.strip() for c in chunks if c.strip()]
        docs.extend(chunks)

    return docs

def build_faiss_index(docs, model):
    """
    По списку документов создаёт FAISS-индекс и возвращает:
    - сам индекс FAISS (index)
    - список документов (docs)
    - матрицу эмбеддингов (document_embeddings)
    """
    # Получаем эмбеддинги
    doc_embeddings = model.encode(docs, convert_to_tensor=False)
    doc_embeddings = np.array(doc_embeddings, dtype='float32')  # faiss ожидает float32

    # Создаем FAISS-индекс (L2)
    idx = faiss.IndexFlatL2(doc_embeddings.shape[1])
    idx.add(doc_embeddings)

    return idx, doc_embeddings

def get_relevant_docs(question: str, idx, docs, model, k=5):
    """
    Ищет k наиболее релевантных чанков для вопроса. Возвращает список строк (чанков).
    """
    question_embedding = model.encode([question], convert_to_tensor=False)
    question_embedding = np.array(question_embedding, dtype='float32')
    distances, indices = idx.search(question_embedding, k)
    relevant_docs = [docs[i] for i in indices[0]]
    return relevant_docs

def generate_answer(question: str, idx, docs, model) -> str:
    """
    Семантический поиск по FAISS -> формируем контекст -> отправляем в OpenAI -> получаем ответ.
    """
    # 1) Найти релевантные тексты (чанки)
    relevant_docs = get_relevant_docs(question, idx, docs, model)
    context = "\n".join(relevant_docs)

    # 2) Формируем промпт для ChatCompletion
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"

    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — виртуальный помощник компании KORSHOP. "
                    "Опираешься на данные из векторных баз и ведёшь себя как менеджер: дружелюбно и с пониманием. "
                    "Отвечаешь только на вопросы, связанные с деятельностью KORSHOP; всё, что выходит за рамки компании, "
                    "не комментируешь. Отвечай кратко и по существу."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=200,
        top_p=1.0,
        frequency_penalty=0.2,
        presence_penalty=0.2
    )

    answer = response.choices[0].message.content.strip()
    return answer

def init_system():
    """
    1. Загружаем PDF из папки.
    2. Разбиваем их на чанки (documents).
    3. Строим векторный индекс (FAISS).
    4. Глобально сохраняем index, documents, embedding_model.
    """
    global index, documents, embedding_model

    print("Загружаем PDF из папки:", PDF_FOLDER_PATH)
    documents = load_and_chunk_pdfs(PDF_FOLDER_PATH)
    print(f"Найдено {len(documents)} чанков текста.")

    print("Загружаем модель 'all-MiniLM-L6-v2'...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Строим FAISS-индекс...")
    index, doc_embs = build_faiss_index(documents, embedding_model)
    print("Индекс готов.\n")

def send_message(recipient_id: str, message_text: str):
    """
    Отправляет сообщение через Instagram API.
    """
    url = f"{GRAPH_API_URL}/{APP_USER_IG_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text},
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print(f"Сообщение отправлено пользователю {recipient_id}: {message_text}")
    else:
        print(f"Не удалось отправить сообщение: {response.json()}")

# Инициализация FastAPI
app = FastAPI()

@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Обработка GET-запросов для проверки Webhook.
    """
    hub_mode = request.query_params.get("hub.mode")
    hub_challenge = request.query_params.get("hub.challenge")
    hub_verify_token = request.query_params.get("hub.verify_token")

    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("Webhook успешно подтвержден!")
        return PlainTextResponse(hub_challenge, status_code=200)
    else:
        print("Проверка Webhook не удалась!")
        return PlainTextResponse("Verification failed", status_code=403)

@app.post("/webhook")
async def instagram_webhook(request: Request):
    """
    Обработка входящих сообщений через Webhook.
    """
    try:
        # Получение JSON-данных из тела запроса
        data = await request.json()
        print("Webhook получен:", data)

        # Проверяем, что уведомление от Instagram
        if data.get("object") == "instagram":
            for entry in data.get("entry", []):
                app_user_id = entry.get("id")
                if app_user_id != APP_USER_IG_ID:
                    print(f"Неожиданный IG User ID: {app_user_id}")
                    continue

                for event in entry.get("messaging", []):
                    sender_id = event["sender"]["id"]  # ID отправителя
                    message = event.get("message", {})  # Сообщение

                    # Проверка на эхо-сообщение
                    if message.get("is_echo"):
                        print("Получено эхо-сообщение. Игнорируем.")
                        continue

                    message_text = message.get("text")  # Текст сообщения

                    if message_text:
                        print(f"Получен текст: {message_text}")
                        # Генерация ответа с использованием RAG
                        answer = generate_answer(message_text, index, documents, embedding_model)
                        # Отправка ответа пользователю
                        send_message(sender_id, answer)

        return JSONResponse({"status": "ok"})
    except Exception as e:
        print(f"Ошибка при обработке Webhook: {e}")
        return JSONResponse({"status": "error", "message": str(e)})

def setup_ngrok():
    """
    Настраивает туннель ngrok на порт 8000.
    """
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    tunnel = ngrok.connect(8000)
    print(f"ngrok туннель установлен: {tunnel.public_url}")
    return tunnel.public_url

def main():
    # 1) Инициализируем систему (PDF → чанки → FAISS)
    init_system()

    # 2) Настраиваем ngrok для публичного доступа к FastAPI
    public_url = setup_ngrok()
    print(f"Публичный URL для Webhook: {public_url}")

    # 3) Запуск FastAPI сервера
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

