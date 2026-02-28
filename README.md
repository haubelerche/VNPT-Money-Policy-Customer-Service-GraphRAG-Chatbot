# VNPT Money GraphRAG Chatbot

> Chatbot hỗ trợ khách hàng thông minh cho dịch vụ VNPT Money, xây dựng trên kiến trúc **GraphRAG** với Neo4j Knowledge Graph.
> **Thời gian thực hiện:** 15/12/2025 – 28/02/2026 | **Phiên bản:** 3.2 | Dự án thực tập cá nhân

---

## Giới thiệu

Khác với RAG truyền thống (chunk-based), hệ thống khai thác **đồ thị tri thức (Knowledge Graph)** trên Neo4j để truy xuất thông tin có cấu trúc và ngữ cảnh phong phú hơn. Chatbot trả lời câu hỏi về chính sách, điều khoản và 22 loại dịch vụ của VNPT Money hoàn toàn dựa trên knowledge base đã kiểm duyệt — không hallucinate.

---

## Điểm nổi bật

### Kiến trúc & Thuật toán 

| # | Thuật toán / Tính năng | Mô tả ngắn |
|---|------------------------|------------|
| 1 | **Hybrid Intent Parser** | Rule-based (confidence ≥ 0.6) + LLM fallback (GPT-4o-mini) |
| 2 | **Constraint-based Retrieval** | Vector search giới hạn theo graph — tránh false positives |
| 3 | **Multi-Signal Ranking (RRF)** | Kết hợp vector · keyword · graph · intent với trọng số tùy chỉnh |
| 4 | **Confidence Scoring** | Đa tín hiệu: 0.35×RRF + 0.30×Intent + 0.20×Gap + 0.15×Slot |
| 5 | **Adaptive Decision Engine** | Phân loại DIRECT / CLARIFY / ESCALATE dựa trên certainty score |
| 6 | **Fast Path (≥0.90 sim)** | Bỏ qua LLM synthesis → giảm latency từ ~15s xuống **~0.5s** |
| 7 | **Adaptive Context Filtering** | Điều chỉnh ngưỡng lọc context linh hoạt theo similarity |
| 8 | **Multi-Intent Handling** | Tự phát hiện và trả lời đầy đủ câu hỏi nhiều ý |
| 9 | **Vietnamese NLP Preprocessing** | 60+ quy tắc viết tắt, 150+ quy tắc khôi phục dấu tiếng Việt |
| 10 | **Cross-Check Fallback** | Tìm lại toàn bộ KB khi constrained search dưới ngưỡng 0.88 |
| 11 | **Embedding Caching** | Redis cache vectors giảm OpenAI API calls lặp lại |

### Kết quả đánh giá RAGAS (27/02/2026 · 20 mẫu · Full Pipeline)

| Chỉ số | Điểm | Ngưỡng | Kết quả |
|--------|------|--------|---------|
| **Faithfulness** | **98.75%** | ≥ 80% | PASS |
| **Answer Relevancy** | **97.00%** | ≥ 75% |  PASS |
| **Context Recall** | **97.83%** | ≥ 70% |  PASS |
| **Context Precision** | **90.00%** | ≥ 70% |  PASS |
| **Answer Similarity** | **87.75%** | ≥ 70% |  PASS |

---

## Tech Stack

| Thành phần | Công nghệ |
|------------|-----------|
| UI | Chainlit |
| Backend | Python 3.11 |
| Graph DB | Neo4j 5.x + Vector Index |
| Cache | Redis 7.x |
| LLM | OpenAI GPT-4o-mini |
| Embedding | OpenAI text-embedding-3-large (3072 dims) |
| Monitoring | Prometheus + Grafana |
| Container | Docker Compose |

---

## Cài đặt và Chạy

### Yêu cầu

- Python 3.10+ (khuyến nghị 3.11)
- Docker Desktop
- OpenAI API Key

### Bước 1 — Tạo môi trường

```bash
conda create -n vnpt-chatbot python=3.11 -y
conda activate vnpt-chatbot
pip install -r requirements.txt
```

### Bước 2 — Cấu hình biến môi trường

Tạo file `.env` ở thư mục gốc:

```env
OPENAI_API_KEY=sk-...

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=

REDIS_URL=redis://localhost:6379/0
```

> Neo4j Docker Compose chạy với `NEO4J_AUTH=none` — để `NEO4J_PASSWORD` trống.

### Bước 3 — Khởi động services

```bash
docker-compose up -d
```

Khởi chạy 4 services: Neo4j (`7474`/`7687`), Redis (`6379`), Prometheus (`9090`), Grafana (`3000`).  
Đợi khoảng 90 giây để Neo4j sẵn sàng, kiểm tra tại `http://localhost:7474`.

### Bước 4 — Nạp Knowledge Base

```bash
python src/ingest_data_v3.py
```

Script tạo graph nodes/relationships, tạo vector embeddings và Neo4j vector index.

### Bước 5 — Chạy Chatbot

```bash
chainlit run src/app.py -w
```

Ứng dụng chạy tại **http://localhost:8000**

---

## Kiểm tra nhanh

| Hành động | Kết quả mong đợi |
|-----------|-----------------|
| Gửi: *"Hướng dẫn chuyển tiền Mobile Money"* | Câu trả lời chi tiết từ knowledge base |
| Gửi: *"abc xyz 123"* | Escalation → gợi ý số tổng đài |
| `http://localhost:7474` → `MATCH (n) RETURN count(n)` | Trả về số node đã nạp |
| `http://localhost:3000` | Grafana dashboard (admin / admin123) |

---

## Other Commands

```bash
# Xem logs
docker-compose logs -f neo4j

# Dừng tất cả services
docker-compose down

# Chạy đánh giá RAGAS
python src/ragas_evaluation.py

# Metrics server (standalone)
uvicorn src/metrics_server:app --host 0.0.0.0 --port 8001
```

---

*Tài liệu kỹ thuật chi tiết: xem [DOCUMENT.md](DOCUMENT.md)*
