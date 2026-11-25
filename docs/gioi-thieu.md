# RAG là gì?

RAG là viết tắt của Retrieval-Augmented Generation, một cách làm cho
chatbot trả lời dựa trên tài liệu thật thay vì chỉ dựa vào trí nhớ
của mô hình ngôn ngữ.

Quy trình RAG có ba bước chính. Đầu tiên là nạp tài liệu: tài liệu
được chia nhỏ thành từng đoạn ngắn gọi là chunk, mỗi chunk được biến
thành một vector số bằng mô hình embedding rồi lưu vào cơ sở dữ liệu
vector như Chroma.

Bước thứ hai là truy xuất: khi người dùng hỏi, câu hỏi cũng được biến
thành vector, hệ thống tìm trong cơ sở dữ liệu những chunk gần nghĩa
nhất với câu hỏi, thường lấy 3 chunk đầu.

Bước cuối cùng là sinh câu trả lời: mô hình ngôn ngữ nhận câu hỏi kèm
theo các chunk vừa tìm được, rồi trả lời chỉ dựa trên nội dung đó.
Nhờ vậy câu trả lời có trích dẫn rõ ràng, và khi tài liệu không có
thông tin thì chatbot nói thẳng là không biết thay vì bịa ra.
