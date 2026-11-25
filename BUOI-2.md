# Buổi 2 — Nối `/chat` với LLM thật (khoảng 30 phút)

Bước nhỏ hôm nay trong briefing sáng 18/11. Chỉ làm bước này thôi,
không đụng tới chunking hay vector DB — mấy cái đó để buổi sau.

`main.py` hiện tại hỗ trợ 2 đường, chỉ cần 1 key trong `.env`:
OpenAI `gpt-4o-mini` (đường cũ, cần nạp tiền) hoặc Gemini
`gemini-3.5-flash-lite` (free tier, không cần thẻ). Có cả hai
key thì OpenAI được ưu tiên. Đường Gemini gọi thẳng Gemini API
bằng `urllib` có sẵn trong Python, không cần cài thêm thư viện.

## Điều kiện trước

- Buổi 1 đã chạy được: `uvicorn main:app --reload` rồi mở
  `/health` thấy `{"status": "ok"}`. Nếu chưa thì làm buổi 1 trước,
  5 phút thôi (xem `BUOI-1.md`).
- Đã có 1 trong 2 key, dán vào `.env` (copy từ `.env.example`):
  ```
  OPENAI_API_KEY=<redacted>
  ```
  hoặc key Gemini free (lấy ở aistudio.google.com -> Get API key):
  ```
  GEMINI_API_KEY=<redacted>
  ```

## Đã kiểm chứng với key Gemini thật (24/11/2026)

- `GET /v1beta/models` trả về danh sách model, key hoạt động.
- `gemini-3.5-flash-lite` trả lời thật bằng tiếng Việt cho câu
  "RAG la gi?" — đây là model mặc định trong `main.py`.
- `gemini-2.5-flash-lite` đã bị Google gỡ cho user mới (404,
  bảo đổi sang `gemini-3.5-flash-lite`), đừng dùng.
- `gemini-3.8-flash` lúc test bị 503 quá tải tạm thời, đừng
  dùng làm mặc định lúc này.

## Thay `main.py` bằng bản dưới đây

Giữ nguyên shape trả về (`answer` + `citations`) để frontend sau
này không phải đổi. `citations` vẫn là `[]` ở buổi này vì chưa
có retrieval — RAG thật tới sau khi thêm chunking + vector DB.

```python
"""Buoi 2 — /chat goi OpenAI that, giu nguyen response shape."""
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI

load_dotenv()

app = FastAPI(title="rag-chatbot-starter")

client = OpenAI() if os.getenv("OPENAI_API_KEY") else None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: dict):
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question trong rong")
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="Chua co OPENAI_API_KEY trong .env",
        )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Tra loi ngan gon, tieng Viet, dung nhu cau hoi.",
            },
            {"role": "user", "content": question},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "citations": [],
    }
```

Không cần cài thêm gì: `requirements.txt` đã có sẵn `openai`
và `python-dotenv` từ buổi 1.

## Test lại

Chạy server như buổi 1, rồi:

```
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG la gi?"}'
```

Đúng thì trả về `{"answer": "...câu trả lời thật...", "citations": []}`.
Nếu báo `Chua co OPENAI_API_KEY hoac GEMINI_API_KEY` thì kiểm tra lại file `.env`
đang nằm cùng thư mục với `main.py` chưa, và key có thừa khoảng
trắng hay dấu ngoặc không.

## Xong buổi này thì t có gì

- Chatbot trả lời thật bằng OpenAI hoặc Gemini, không còn mock nữa.
- Shape `answer` + `citations` giữ nguyên, sẵn sàng gắn
  retrieval vào `citations` ở buổi RAG sau.

## Nếu kẹt

Nhắn m đúng chỗ kẹt kèm đoạn lỗi copy ra, đừng mò một mình
quá 15 phút ở một lỗi. Nhớ đừng commit file `.env` hay key
lên GitHub.
