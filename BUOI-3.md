# Buổi 3 — RAG thật: chunk + embed + Chroma, `/chat` retrieve trước khi trả lời (khoảng 45 phút)

Bước nhỏ hôm nay trong briefing sáng 19/11. Chỉ làm bước này thôi,
không đụng tới frontend hay deploy — mấy cái đó để buổi sau.

## Điều kiện trước

- Buổi 2 đã chạy được: `/chat` trả lời thật bằng OpenAI (xem
  `BUOI-2.md`). Nếu chưa thì làm buổi 2 trước.
- Đã có key OpenAI trong `.env`.

> Lưu ý (25/11): t đang dùng key **Gemini free** thay cho OpenAI.
> Ở bước 3 dưới đây, thay OpenAI embeddings bằng model
> `gemini-embedding-001` qua Gemini API, gọi bằng `urllib` giống cách
> `main.py` đang gọi Gemini (`GEMINI_API_KEY` trong `.env`). Không cần
> cài thêm thư viện nào cho embeddings.

## 1. Cài thêm 2 thư viện

Thêm vào `requirements.txt` rồi cài:

```
chromadb
langchain-text-splitters
```

```
pip install -r requirements.txt
```

## 2. Chuẩn bị 1 file tài liệu mẫu

Tạo file `docs/gioi-thieu.md` trong thư mục project, viết khoảng
10–15 dòng giới thiệu về một chủ đề t thích (ví dụ: RAG là gì,
tự viết bằng tiếng Việt). Đây là "kiến thức" chatbot sẽ tra cứu.

## 3. Viết `ingest.py` — nạp tài liệu vào Chroma

```python
"""Buoi 3 — chunk + embed tai lieu vao Chroma."""
import os

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

client = OpenAI()
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("docs")

with open("docs/gioi-thieu.md", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

embeddings = [
    client.embeddings.create(model="text-embedding-3-small", input=c).data[0].embedding
    for c in chunks
]
collection.add(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings,
)
print(f"Da nap {len(chunks)} chunk vao Chroma.")
```

Chạy: `python ingest.py`. Đúng thì in ra số chunk đã nạp.

## 4. Sửa `/chat` trong `main.py` — retrieve trước, rồi mới trả lời

Thêm đoạn này vào đầu hàm `chat`, sau khi lấy `question`:

```python
    query_emb = client.embeddings.create(
        model="text-embedding-3-small", input=question
    ).data[0].embedding
    results = collection.query(
        query_embeddings=[query_emb], n_results=3
    )
    context = "\n\n".join(results["documents"][0])
    citations = [f"docs/gioi-thieu.md (chunk {i})" for i in range(len(results["documents"][0]))]
```

Và đổi system prompt thành:

```python
        {
            "role": "system",
            "content": (
                "Tra loi ngan gon, tieng Viet, chi dua tren tai lieu duoi day. "
                "Neu tai lieu khong co thong tin thi noi thang la khong biet.\n\n"
                f"{context}"
            ),
        },
```

Trả về `{"answer": ..., "citations": citations}` thay vì `citations: []`.

Nhớ import và khởi tạo `collection` ở đầu `main.py` giống như
trong `ingest.py` (cùng `path="./chroma_db"` và tên collection `"docs"`).

## Test lại

```
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG la gi?"}'
```

Đúng thì `answer` nhắc tới nội dung trong file `gioi-thieu.md`
của t, và `citations` không còn `[]` nữa. Thử hỏi một câu ngoài
tài liệu — chatbot phải nói không biết, đó mới là RAG hoạt động
đúng (không bịa).

## Xong buổi này thì t có gì

- Chatbot RAG thật: trả lời dựa trên tài liệu của t, có trích dẫn.
- Đây đúng là thứ job Upwork chatbot MVP ($1.200, đăng 05/11)
  và job Senior AI/ML RAG ($15–30/h, đăng 09/11) đang hỏi —
  xong buổi này t đã có 1 demo RAG để khoe.

## Nếu kẹt

Nhắn m đúng chỗ kẹt kèm đoạn lỗi copy ra, đừng mò một mình
quá 15 phút ở một lỗi. Nhớ đừng commit file `.env`, key,
hay thư mục `chroma_db/` lên GitHub.
