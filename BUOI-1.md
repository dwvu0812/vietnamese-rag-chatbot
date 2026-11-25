# Buổi 1 — FastAPI cho chatbot RAG (khoảng 45 phút, tối nay làm được)

Đây là bước đầu của đường nhanh đã chốt: Tuần 1–2 FastAPI + gọi
OpenAI/Anthropic API, rồi Tuần 3–5 làm Project 1 — chatbot RAG
(Next.js + FastAPI + LangChain + Pinecone/pgvector, có citations,
eval, Docker, demo live). Không cần trả lời mấy tiếng/tuần trước
mới làm được buổi này — buổi này xong thì hôm nào rảnh làm tiếp
cũng được.

## Làm gì tối nay

1. Copy thư mục `project-1-rag-starter` này ra máy t, vào trong đó.
2. Tạo môi trường + cài đặt (Python 3.10 trở lên):
   ```
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Chạy thử, chưa cần API key:
   ```
   uvicorn main:app --reload
   ```
   Mở http://127.0.0.1:8000/health — thấy `{"status": "ok"}` là xong
   phần khung.
4. Thử endpoint `/chat` bằng curl (vẫn chưa gọi API thật, chỉ để
   kiểm tra shape trả về mà frontend sau này sẽ dùng):
   ```
   curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "RAG la gi?"}'
   ```
5. Copy `.env.example` thành `.env`, dán key OpenAI của t vào.
   Key này chỉ nằm ở máy t, đừng commit lên GitHub.

## Xong buổi này thì t có gì

- Một API FastAPI chạy được, có sẵn `/health` và `/chat`.
- Response shape của `/chat` đã cố định (`answer` + `citations`)
  để sau này gắn frontend Next.js không phải đổi lại.

## Bước tiếp theo (để dành buổi sau, không làm tối nay)

- Thay phần TODO trong `main.py` bằng một cuộc gọi OpenAI thật.
- Thêm tool calling cơ bản.
- Rồi mới tới chunking tài liệu + vector DB cho RAG.

## Nếu kẹt

Nhắn m đúng chỗ kẹt (lỗi cài đặt, lỗi chạy uvicorn, hay lỗi key)
kèm đoạn lỗi m copy ra, m gỡ cùng t — đừng ngồi mò một mình
quá 15 phút ở một lỗi.
