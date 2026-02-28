# VNPT Money Policy Customer Service GraphRAG Chatbot
# Dự án thực tập cá nhân
> **Thời gian thực hiện**: 15/12/2025 - 28/02/2026 | **Phiên bản**: 3.2

---
<img width="8192" height="7768" alt="Neo4j Vector Retrieval Flow-2026-02-05-095816" src="https://github.com/user-attachments/assets/0406d53e-095e-4119-8a09-def62545794e" />

<img width="4009" height="8192" alt="flowchart" src="https://github.com/user-attachments/assets/a976afad-7dd2-4b85-910f-06434d96018f" />

https://github.com/user-attachments/assets/243ca33e-4ef8-4d73-bd56-2e1aa2f3ee28

## MỤC LỤC

1. [Giới thiệu và Mục đích](#1-giới-thiệu-và-mục-đích)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Các thuật toán](#3-các-thuật-toán)
   - 3.1 Phân tích ý định trong câu 
   - 3.2 Truy hồi có ràng buộc
   - 3.3 Thuật toán xếp hạng kết quả tính toán đa tín hiệu (Multi-Signal Ranking - RRF)
   - 3.4 Thuật toán tính độ chắc chắn
   - 3.5 Thuật toán ra quyết định
   - 3.6 LLM tạo sinh xử lý câu hỏi đa ý
   - 3.7 Tối ưu tốc độ trả lời câu đơn giản
   - 3.8 Adaptive Context Filtering (V2)
   - 3.9 Tiền xử lý dữ liệu tiếng Việt
   - 3.10 Xử lý các biến thể của một câu hỏi
   - 3.11 Embedding Caching 
 
4. [Luồng xử lý (Pipeline Flow)](#4-luồng-xử-lý-pipeline-flow)
5. [Chi tiết từng Module](#5-chi-tiết-từng-module)
6. [Monitoring & Metrics](#6-monitoring--metrics)
7. [Đánh giá RAGAS cho GraphRAG](#7-đánh-giá-ragas-cho-graphrag)
   - 7.1 Tổng quan về RAGAS
   - 7.2 Chuẩn bị tập dữ liệu đánh giá
   - 7.3 Các chỉ số đánh giá cốt lõi
   - 7.4 Kiến trúc module đánh giá
   - 7.5 Quy trình thực thi đánh giá
   - 7.6 Kết quả đánh giá thực tế
   - 7.7 Phân tích & Tối ưu hóa
   - 7.8 Lưu ý đặc thù cho GraphRAG
8. [Cài đặt và Chạy dự án](#8-cài-đặt-và-chạy-dự-án)
   - 8.1 Yêu cầu hệ thống
   - 8.2 Cài đặt môi trường Python
   - 8.3 Cấu hình biến môi trường
   - 8.4 Khởi chạy các services nền
   - 8.5 Nạp dữ liệu vào Knowledge Base
   - 8.6 Chạy ứng dụng Chatbot
   - 8.8 Kiểm tra hệ thống hoạt động
   - 8.9 Khác

---

## 1. Giới thiệu và Mục đích

### 1.1 Giới thiệu

VNPT Money GraphRAG Chatbot là hệ thống chatbot hỗ trợ khách hàng về các vấn đề liên quan tới chính sách, điều khoản và dịch vụ (dữ liệu công khai) của ứng dụng VNPT Money, được xây dựng trên kiến trúc **GraphRAG** (Graph-based Retrieval Augmented Generation).

Khác với kiến trúc RAG truyền thống dựa trên chunk-based retrieval, GraphRAG khai thác **đồ thị tri thức (Knowledge Graph)** với các thực thể (entities) và quan hệ (relationships) được cấu trúc hóa, cho phép truy xuất thông tin có ngữ cảnh phong phú hơn thông qua graph traversal thay vì chỉ tìm kiếm theo vector similarity trên các đoạn văn bản rời rạc.

**Các đặc điểm kỹ thuật nổi bật:**
1. Cơ chế Phản hồi và Độ chính xác
- Phản hồi dựa trên dữ liệu gốc: Chỉ trả lời dựa trên kho tri thức đã kiểm duyệt, loại bỏ ảo tưởng thông qua bộ lọc cụm từ cấm.
- Chuyển cấp dựa trên độ tin cậy: Tự động chuyển sang tổng đài viên khi điểm tin cậy thấp, sử dụng công thức phân tích đa tín hiệu thay vì một ngưỡng đơn lẻ.
- Truy xuất kiểm tra chéo: Kiểm tra lại toàn bộ kho tri thức khi kết quả tìm kiếm giới hạn không đủ độ tin cậy (ngưỡng 0,88).
2. Xử lý Ý định và Ngữ cảnh
- Phân tích ý định hỗn hợp: Kết hợp linh hoạt các quy tắc xử lý cố định (nếu câu hỏi được chấm trên ngưỡng 0.6 về độ tự tin trả lời) và LLM tạo sinh dự phòng các trường hợp hỏi phức tạp hơn
- Xử lý câu hỏi đa ý định: Tự động phát hiện và đảm bảo trả lời đầy đủ mọi khía cạnh của các câu hỏi có nhiều phần.
- Tổng hợp đa nguồn: Tổng hợp câu trả lời từ tối đa 5 ngữ cảnh liên quan thay vì chỉ dựa vào một kết quả cao nhất để xử lý các câu hỏi phức tạp.
3. Tối ưu hóa Hiệu suất và Hạ tầng
-  Tối ưu đường truyền nhanh: Bỏ qua bước tổng hợp của LLM đối với các truy vấn có độ tương đồng (similarity) cao $\ge$ 0,90, giảm độ trễ từ ~15 giây xuống ~0,5 giây.
- Lọc ngữ cảnh hợp lí: Điều chỉnh ngưỡng lọc ngữ cảnh linh hoạt theo độ tương đồng.
- Hạ tầng: Quản lý bộ nhớ đệm, lưu trữ vector đặc trưng và giám sát hệ thống thời gian thực.
4. Xử lý Ngôn ngữ Tiếng ViệtChuẩn hóa tiếng Việt: 
* Xử lý hơn 60 quy tắc viết tắt. Khôi phục dấu tiếng Việt (hơn 150 quy tắc) 

### 1.2 Mục đích

**Mục tiêu chính:**
- Cung cấp hỗ trợ khách hàng 24/7 cho dịch vụ VNPT Money
- Giải quyết các vấn đề với mức độ phức tạp ngữ cảnh nhất định
- Trả lời dựa trên knowledge base, nếu không đủ thông tin hoặc đánh giá không hữu ích thì gợi ý số tổng đài
- Giảm tải cho tổng đài viên với các câu hỏi thường gặp

### 1.3 Phạm vi hệ thống

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        VNPT MONEY CHATBOT - SCOPE                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ĐƯỢC PHÉP (IN-SCOPE)                   KHÔNG ĐƯỢC PHÉP (OUT-OF-SCOPE)      ║
║  ─────────────────────────            ──────────────────────────────────     ║
║  • Giải thích chính sách              • Truy cập dữ liệu cá nhân             ║
║  • Giải thích điều kiện dịch vụ       • Kiểm tra trạng thái giao dịch        ║
║  • Hướng dẫn quy trình thao tác       • Suy đoán kết quả giao dịch           ║
║  • Giải thích lỗi quy tắc             • Trả lời vượt knowledge base          ║
║    (OTP, hạn mức, điều kiện..)        • Sinh thông tin không có nguồn        ║
║  • Hỏi lại khi thiếu thông tin        • Tự ý đưa ra quyết định tài chính     ║
║  • Escalate đúng thời điểm            • Đoán trạng thái tài khoản            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 1.4 Các dịch vụ hỗ trợ

Hệ thống phục vụ **22 loại dịch vụ** được định nghĩa trong `ServiceEnum`, phân nhóm như sau:

| Nhóm dịch vụ | Chi tiết |
|--------------|----------|
| **Tài chính cơ bản** | Nạp tiền, rút tiền, chuyển tiền, liên kết ngân hàng, thanh toán |
| **Tài khoản & Bảo mật** | OTP/SmartOTP, hạn mức, đăng ký, định danh eKYC, bảo mật |
| **Viễn thông** | Data 3G/4G, mua thẻ, nạp tiền điện thoại, di động trả sau, hóa đơn viễn thông |
| **Tiện ích** | Tiền điện, tiền nước, dịch vụ công, học phí |
| **Tài chính - Bảo hiểm** | Bảo hiểm, vay tiêu dùng, tiết kiệm online |
| **Giải trí & Vé** | MyTV, Vietlott, vé tàu, vé máy bay, khách sạn |
| **Pháp lý** | Điều khoản sử dụng, quyền riêng tư |

### 1.5 Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Chainlit (Web Chat UI) |
| **Backend** | Python 3.11, FastAPI |
| **Database** | Neo4j 5.x (Graph + Vector Index) |
| **Cache/Session** | Redis 7.x (Connection Pooling, max 50 conn) |
| **LLM** | OpenAI GPT-4o-mini |
| **Embedding** | OpenAI text-embedding-3-small (1536 dims) |
| **Monitoring** | Prometheus + Grafana |
| **Container** | Docker Compose |

---

## 2. Kiến trúc hệ thống

### 2.1 Kiến trúc tổng quan (6 tầng)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                    (Chainlit / Web / Mobile App)                            │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: INPUT PROCESSING                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Chat History    │  │ Input Validator │  │ Session Manager │              │
│  │ Manager         │  │ + TextNormalizer│  │ (Redis+Memory)  │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: HYBRID INTENT PARSING & STRUCTURED QUERY BUILDER                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Rule-based Parser (priority-ordered keywords + action overrides)    │    │
│  │           confidence ≥ 0.6 → Use rule result                        │    │
│  │           confidence < 0.6 → Fallback to LLM Parser                 │    │
│  │  LLM Parser (gpt-4o-mini, temperature=0, slot-filling only)         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: CONSTRAINT-BASED RETRIEVAL + CROSS-CHECK FALLBACK                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐     │
│  │ Graph Constraint   │→ │ Vector Search      │→ │ Cross-Check        │     │
│  │ Filter             │  │ (Constrained)      │  │ Fallback (0.88)    │     │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘     │
│                                                   │                         │
│                                                   ▼                         │
│                                            ┌────────────────────┐           │
│                                            │ Graph Traversal    │           │
│                                            │ (Fetch Context)    │           │
│                                            └────────────────────┘           │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: MULTI-SIGNAL RANKING + CONFIDENCE SCORING                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ RRF Fusion: Vector(1.0) + Keyword(0.8) + Graph(0.6) + Intent(1.2)     │ │
│  │ Confidence: 0.35×RRF + 0.30×Intent + 0.20×Gap + 0.15×SlotPenalty      │ │
│  │ Primary Group Boost + Low Similarity Penalty                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: ADAPTIVE DECISION ENGINE                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ High Similarity Override (≥0.95) → DIRECT_ANSWER                       │ │
│  │ Confidence-First (≥0.65) → DIRECT_ANSWER                              │ │
│  │ Certainty Score: 0.75×Conf + 0.15×Gap + 0.10×RRF                      │ │
│  │ Certainty ≥ 0.55 → DIRECT | ≥ 0.45 → CLARIFY | < 0.35 → ESCALATE    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 6: RESPONSE GENERATION (Grounded)                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Fast-Path (sim ≥ 0.90): Direct KB answer → ~0.5s                       │ │
│  │ LLM Synthesis: Top 3-5 contexts → gpt-4o-mini → ~10-15s               │ │
│  │ Multi-Part: Forced LLM synthesis + 5 contexts                         │ │
│  │ No-Info Detection → Structured escalation template                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEO4J GRAPH DATABASE                               │
│    (Group) -[:HAS_TOPIC]-> (Topic) -[:HAS_PROBLEM]-> (Problem) -[:HAS_ANSWER]-> (Answer)  │
│    + Supplement Nodes (Problem/Answer Supplement)                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Cấu trúc thư mục

```
├── src/
│   ├── schema.py              # Enums, Dataclasses, Constants, Config
│   ├── intent_parser.py       # Hybrid Intent Parser (Rule + LLM)
│   ├── retrieval.py           # Graph-constrained retrieval + cross-check fallback
│   ├── ranking.py             # Multi-signal RRF ranking + confidence
│   ├── decision_engine.py     # Certainty-based decision routing
│   ├── response_generator.py  # LLM synthesis + fast-path + multi-part
│   ├── pipeline.py            # Orchestrator chính + adaptive context filtering
│   ├── app.py                 # Chainlit application + feedback system
│   ├── redis_manager.py       # Redis connection pooling + session management
│   ├── monitoring.py          # Prometheus metrics + health checks + dashboard
│   ├── metrics_server.py      # Metrics HTTP endpoint
│   ├── neo4j_config.py        # Neo4j connection config
│   ├── ragas_evaluation.py    # RAGAS evaluation framework
│   └── ingest_data_v3.py      # Data ingestion + supplement support
├── test/
│   ├── eval_dataset.json          # 20 mẫu đánh giá cơ bản
│   ├── eval_dataset_expanded.json # 50 mẫu mở rộng (9 categories)
│   └── eval_report_full_*.json    # Kết quả đánh giá qua các lần chạy
```

### 2.3 Graph Schema

```
┌─────────────┐      HAS_TOPIC      ┌─────────────┐
│   Group     │ ─────────────────→ │   Topic     │
│             │                     │             │
│ • id        │                     │ • id        │
│ • name      │                     │ • name      │
│ • description│                    │ • group_id  │
│ • order     │                     │ • keywords  │
└─────────────┘                     └──────┬──────┘
                                           │
                                    HAS_PROBLEM
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │  Problem    │
                                    │             │
                                    │ • id        │
                                    │ • title     │
                                    │ • description│
                                    │ • intent    │
                                    │ • keywords  │
                                    │ • embedding │
                                    └──────┬──────┘
                                           │
                                     HAS_ANSWER
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │   Answer    │
                                    │             │
                                    │ • id        │
                                    │ • content   │
                                    │ • steps     │
                                    │ • notes     │
                                    └─────────────┘
```




## 3. Các thuật toán

### 3.1 Phân tích ý định và ngữ cảnh

Hệ thống phân tích ý định (intent) người dùng theo chiến lược **hai bước**: 
- rule-based (câu hỏi đơn giản_dễ xử lý)
- LLM (câu hỏi đa ý, câu hỏi ngoài lề, out-of-domain,...)

**Chiến lược 2 bước:**

```python
# B1: Rule-based parsing 
rule_result = IntentParserLocal.parse(user_message)

# B2: Nếu confidence < 0.6, fallback sang LLM 
if rule_result.confidence_intent < 0.6:
    return IntentParserLLM.parse(user_message)
else:
    return rule_result
```



### 3.2 Truy hồi có ràng buộc

Quá trình truy xuất (retrieval) gồm 4 giai đoạn, trong đó **cross-check fallback** là cơ chế đảm bảo chất lượng kết quả khi intent parsing bị lệch.

**Giai đoạn 1: Graph Constraint Filter**

```cypher
MATCH (g:Group)-[:HAS_TOPIC]->(t:Topic)-[:HAS_PROBLEM]->(p:Problem)
WHERE g.id IN $allowed_groups AND ($topic IS NULL OR t.id = $topic)
RETURN DISTINCT p.id AS problem_id
```

`SERVICE_GROUP_MAP` ánh xạ 22 service → danh sách groups ưu tiên. Kết quả được cache trong bộ nhớ (`_group_cache`) để tránh lặp lại truy vấn Cypher.

**Giai đoạn 2: Vector Search (Có điều kiện)**

```python
candidates = vector_index.search(
    embedding=embed(condensed_query),
    filter_ids=constrained_problem_ids,
    top_k=10  # chọn top 10 ket quả cao nhất
)
```

Sử dụng Neo4j Vector Index với cosine similarity, model `text-embedding-3-small` (1536 chiều). Embedding được cache qua `EmbeddingCache` (thuật toán LRU, max 500 entries).

**Giai đoạn 3: Kiểm tra chéo dự phòng (Cross-Check Fallback)**

**Mục đích:** Ở giai đoạn 1-2, hệ thống chỉ tìm kiếm trong phạm vi nhóm (group) tương ứng với dịch vụ đã phân loại từ intent parsing (phân tích ý định). Tuy nhiên, nếu intent parsing phân loại sai nhóm, toàn bộ kết quả tìm kiếm sẽ bị giới hạn trong nhóm sai , bỏ sót câu trả lời đúng nằm ở nhóm khác. Cross-check giải quyết vấn đề này bằng cách **mở rộng tìm kiếm ra toàn bộ knowledge base** khi phát hiện kết quả trong phạm vi ràng buộc chưa đủ tốt.

**Điều kiện kích hoạt:**

```python
CROSS_CHECK_THRESHOLD = 0.88

should_fallback = (
    len(candidates) < 3 or            # Quá ít kết quả trong phạm vi ràng buộc
    top_similarity < CROSS_CHECK_THRESHOLD  # Kết quả tốt nhất vẫn chưa đủ khớp
)
```

**Điều kiện chấp nhận kết quả mở rộng** (chỉ dùng kết quả mới khi thực sự tốt hơn):

```python
use_expanded = (
    improvement > 0.02 or             # Cải thiện similarity > 0.02
    (full_top >= 0.85 and constrained_top < 0.85)  # Vượt ngưỡng 0.85
)
```

**Tác động thực tế (kiểm chứng trên hệ thống):**

Cross-check phát huy hiệu quả chủ yếu khi intent parsing đẩy tìm kiếm vào nhóm quá hẹp — khiến constrained search trả về **ít hoặc không có kết quả**:

| Câu hỏi | Intent → Nhóm | Constrained | Full KB | Cải thiện |
|---------|---------------|-------------|---------|-----------|
| "khiếu nại giao dịch nạp tiền bị lỗi" | `dieu_khoan` (130 problems) | **0.000** (không tìm thấy) | 0.812 | +0.812 |
| "kiểm tra lịch sử giao dịch" | `ung_dung` (35 problems) | **0.000** (không tìm thấy) | 0.811 | +0.811 |
| "tra soát giao dịch chuyển tiền" | `dieu_khoan` (130 problems) | 0.730 | 0.758 | +0.028 |

Trong trường hợp đầu, từ "khiếu nại" khiến hệ thống phân loại vào nhóm `dieu_khoan` (chỉ chứa quy định pháp lý), nhưng câu trả lời hướng dẫn xử lý lỗi nạp tiền lại nằm ở nhóm `ho_tro_khach_hang`. Nếu không có cross-check, người dùng sẽ nhận được phản hồi "không tìm thấy thông tin" dù knowledge base có sẵn câu trả lời.

**Lưu ý:** Với những dịch vụ đã map tới nhiều nhóm (ví dụ `nap_tien` → 3 nhóm, 571 problems), cross-check thường không tạo sự khác biệt vì phạm vi ràng buộc đã đủ rộng.

**Giai đoạn 4: Graph Traversal**

```cypher
MATCH (p:Problem)-[:HAS_ANSWER]->(a:Answer)
WHERE p.id IN $candidate_problem_ids
MATCH (g:Group)-[:HAS_TOPIC]->(t:Topic)-[:HAS_PROBLEM]->(p)
RETURN p.*, a.*, t.*, g.*
```

### 3.3 Thuật toán xếp hạng kết quả tính toán đa tín hiệu (Multi-Signal Ranking - RRF)

<img width="271" height="394" alt="Reciprocal_Rank_Fusion" src="https://github.com/user-attachments/assets/bbbef8c4-24d8-490e-98db-e8c80a4f6071" />

Hệ thống xếp hạng kết quả bằng **Reciprocal Rank Fusion (RRF)** kết hợp 4 tín hiệu ranking với trọng số khác nhau:

| Signal | Mô tả | Weight(mức độ quan trọng trong công thức) | Scorer |
|--------|-------|--------|--------|
| **Vector Similarity** | Cosine similarity từ embedding search | 1.0 | Vector Index |
| **Keyword Match** | BM25-style overlap (tokenized, loại stopwords) | 0.8 | `KeywordMatcher` |
| **Graph Distance** | Điểm proximity theo topic/group matching | 0.6 | `GraphDistanceScorer` |
| **Intent Alignment** | Độ phù hợp qua cross-intent similarity matrix | 1.2 | `IntentAlignmentScorer` |

**Công thức RRF:**

$$RRF\_score(d) = \sum_{s \in \{vector, keyword, graph, intent\}} \frac{w_s}{k + rank_s(d)}$$

Trong đó: $k = 60$ (mặc định smoothing parameter), $w_s$ = weight của signal $s$, $rank_s(d)$ = thứ hạng của document $d$ theo signal $s$.

**Cải tiến 1 — ưu tiên nhóm hỗ trợ chính:**

`GraphDistanceScorer` gán điểm graph distance khác nhau tùy theo mối quan hệ giữa group của candidate và service được parse:

| Loại group | Điểm | Giải thích |
|------------|-------|------------|
| Primary (group đầu tiên trong `SERVICE_GROUP_MAP`) | 0.95 | Nhóm phục vụ chính cho service này |
| Secondary (các group còn lại) | 0.65 | Nhóm liên quan nhưng không phải nguồn chính |
| Outside (không thuộc service) | 0.40 | Không liên quan trực tiếp |

**Mục đích thiết kế:** Phần lớn câu hỏi người dùng hướng tới hỗ trợ thao tác (operational — thuộc `ho_tro_khach_hang`) hơn là tra cứu điều khoản pháp lý (`dieu_khoan`). Boost này tạo tín hiệu ưu tiên phù hợp.

**Mức độ tác động thực tế:**

Kiểm nghiệm với 8 câu hỏi cross-group (kết quả từ ≥ 2 nhóm khác nhau), so sánh thứ tự ranking CÓ boost vs KHÔNG boost:

| Kết quả | Số lượng | Tỉ lệ |
|---------|----------|--------|
| Boost **thay đổi** thứ tự top-5 | 1/8 | 12.5% |
| Boost **không ảnh hưởng** thứ tự | 7/8 | 87.5% |

Trường hợp duy nhất thay đổi:

```
Q: "trách nhiệm của VNPT khi giao dịch lỗi" (service=khac, primary=dich_vu)

# CÓ boost: "Lỗi thanh toán Truyền hình K+" (dich_vu) được đẩy từ #4 → #2
# KHÔNG boost: quyen_rieng_tu giữ vị trí #2 nhờ sim=0.799
```

**Đánh giá:** Boost có tác động biên (marginal) vì vector similarity (weight=1.0) thường đã xác định đúng thứ tự. Graph distance (weight=0.6) chỉ tạo khác biệt khi các candidate có similarity rất gần nhau (chênh lệch < 0.01). Tuy nhiên, boost vẫn có giá trị như một tín hiệu bổ sung trong công thức confidence score tổng hợp.

---

**Cải tiến 2 — Ma trận tương đồng ý định chéo:**

Ma trận định nghĩa 5 cặp ý định có mối liên quan ngữ nghĩa, cho phép xếp hạng cao hơn các candidate có intent "gần" với intent người dùng:

| Intent A | Intent B | Similarity | Lý do liên quan |
|----------|----------|------------|-----------------|
| `tru_tien_chua_nhan` | `pending_lau` | 0.9 | Trừ tiền chưa nhận thường do giao dịch pending |
| `that_bai` | `loi_ket_noi` | 0.8 | Thất bại thường do lỗi kết nối |
| `that_bai` | `pending_lau` | 0.7 | Thất bại và pending cùng nhóm vấn đề giao dịch |
| `huong_dan` | `chinh_sach` | 0.7 | Hướng dẫn hay kèm nội dung chính sách |
| `khong_nhan_otp` | `that_bai` | 0.6 | Không nhận OTP gây thất bại giao dịch |

> **Lưu ý về trạng thái hiện tại:** Qua kiểm nghiệm 6 câu hỏi đại diện cho các intent trên, ma trận **chưa bao giờ kích hoạt** (tất cả candidate đều nhận điểm mặc định 0.3). Nguyên nhân: các key trong ma trận (`that_bai`, `pending_lau`...) là dạng rút gọn, trong khi `candidate.intent` từ DB là dạng đầy đủ (`chuyen_tien_that_bai_tien_co_mat`, `giao_dich_pending_lau`...). Cần cập nhật logic matching để so khớp substring hoặc chuẩn hóa intent key giữa parser và DB.

---

**Cải tiến 3 — Low Similarity Penalty (phạt khi điểm tương đồng thấp):**

Khi top candidate có similarity thấp (< 0.6), hệ thống giảm `score_gap` để tăng khả năng hệ thống đánh giá kết quả là "mơ hồ" (ambiguous) → kích hoạt trả lời thận trọng hơn:

$$score\_gap_{penalized} = score\_gap \times \frac{top\_similarity}{0.6} \quad \text{khi } top\_similarity < 0.6$$

> **Lưu ý về trạng thái hiện tại:** Qua kiểm nghiệm với các câu hỏi hoàn toàn nằm ngoài phạm vi knowledge-base, **penalty chưa bao giờ kích hoạt** vì mô hình embedding tiếng Việt (text-embedding-3-large) luôn trả về similarity ≥ 0.68 ngay cả với nội dung không liên quan. Ngưỡng 0.6 quá thấp cho đặc tính embedding này. Thay vào đó, cơ chế **Ambiguity Detection** (`gap < 0.15` hoặc `similarity < 0.55`) đang đảm nhận vai trò lọc — tất cả các câu hỏi ngoài KB đều bị đánh dấu `is_ambiguous=True` nhờ score gap rất thấp (< 0.02).

### 3.4 Thuật toán tính độ chắc chắn

Thay vì chỉ dựa vào một confidence score đơn, hệ thống sử dụng **Certainty Score** kết hợp nhiều tín hiệu để quyết định routing chính xác hơn.

**Công thức Confidence (Layer 4 - Ranking):**

$$confidence = 0.35 \times rrf\_conf + 0.30 \times intent\_conf + 0.20 \times gap\_conf + 0.15 \times slot\_penalty$$

Trong đó, mỗi thành phần đại diện cho một "tín hiệu" về độ tin cậy:

- **`rrf_conf` (Chất lượng kết quả):** Điểm RRF của kết quả top 1 đã được chuẩn hóa. Tựu trung, nó trả lời câu hỏi: *"Kết quả tìm được có khớp tốt với câu hỏi không?"* Điểm càng cao, câu trả lời càng liên quan.
- **`intent_conf` (Chất lượng phân tích):** Độ tự tin của Intent Parser khi phân tích câu hỏi ban đầu. Nó trả lời câu hỏi: *"Hệ thống có hiểu đúng ý định của người dùng không?"* Nếu câu hỏi mơ hồ, điểm này sẽ thấp.
- **`gap_conf` (Mức độ rõ ràng):** Khoảng cách điểm giữa kết quả top 1 và top 2. Nó trả lời câu hỏi: *"Kết quả top 1 có thực sự vượt trội so với các kết quả còn lại không?"* Khoảng cách lớn cho thấy một câu trả lời duy nhất, rõ ràng.
- **`slot_penalty` (Mức độ đầy đủ thông tin):** Mức phạt nếu Intent Parser không tìm thấy đủ các "slots" (thông tin cần thiết) trong câu hỏi. Nó trả lời câu hỏi: *"Câu hỏi của người dùng có đủ chi tiết để tìm câu trả lời chính xác không?"*

**Công thức Certainty (Layer 5 - Decision):**

$$certainty = 0.75 \times confidence + 0.15 \times normalized\_gap + 0.10 \times \min(2 \times top\_rrf, 1.0)$$

Trong đó:
- **`confidence`**: Điểm tin cậy tính từ Layer 4.
- **`normalized_gap`**: Khoảng cách điểm đã được chuẩn hóa, nhấn mạnh sự khác biệt giữa các kết quả hàng đầu.
- **`top_rrf`**: Điểm RRF thô của kết quả tốt nhất, đóng vai trò như một yếu tố "bonus" nếu điểm rất cao.

**Tại sao tách thành 2 công thức?**
- **Confidence** (Layer 4) đo lường chất lượng nội tại của kết quả ranking — "kết quả tìm được tốt đến đâu?"
- **Certainty** (Layer 5) kết hợp confidence với gap analysis để quyết định hành động — "có nên trả lời trực tiếp hay cần thận trọng?"

**Score Gap - tính toán:**

```python
# Sử dụng cả normalized RRF gap VÀ vector similarity gap
raw_rrf_gap = top_rrf - second_rrf
normalized_rrf_gap = raw_rrf_gap / top_rrf
vector_gap = top_similarity - second_similarity

score_gap = max(normalized_rrf_gap, vector_gap)  # Lấy gap lớn hơn
```

**Ambiguity Detection:**

```python
is_ambiguous = (
    gap_component < 0.15 or                          # Gap quá nhỏ
    (len(candidates) > 0 and top_similarity < 0.55)  # Similarity quá thấp
)
```

### 3.5 Thuật toán ra quyết định

Decision Engine sử dụng **3 lớp kiểm tra** trước khi áp dụng certainty thresholds, cho phép "shortcut" khi kết quả rõ ràng:

**Lớp 1 — High Similarity Override (≥ 0.95):**
```python
if top_similarity >= 0.95:
    return DIRECT_ANSWER  # Bỏ qua gap analysis
```
Xử lý trường hợp knowledge base có nhiều entry trùng lặp — top similarity rất cao nhưng gap thấp do có bản sao.

**Lớp 2 — Confidence-First (≥ 0.65):**
```python
if ranking.confidence_score >= 0.65:
    return DIRECT_ANSWER  # Confidence đủ cao, gap ít quan trọng
```

**Lớp 3 — Certainty-based Routing:**

| Certainty | Decision | Hành động |
|-----------|----------|-----------|
| ≥ 0.55 | `DIRECT_ANSWER` | Trả lời trực tiếp |
| ≥ 0.45 | `ANSWER_WITH_CLARIFY` | Trả lời + hỏi thêm |
| 0.35 - 0.45 + ambiguous | `CLARIFY_REQUIRED` | Hỏi làm rõ |
| 0.35 - 0.45 + missing_slots | `CLARIFY_REQUIRED` | Hỏi thông tin thiếu |
| < 0.35 | `ESCALATE_LOW_CONFIDENCE` | Chuyển tổng đài |

**Các decision đặc biệt:**

| Điều kiện | Decision | Hành động |
|-----------|----------|-----------|
| `is_out_of_domain = true` | `ESCALATE_OUT_OF_SCOPE` | Từ chối lịch sự |
| `clarify_count ≥ 10` | `ESCALATE_MAX_RETRY` | Chuyển tổng đài |
| `need_account_lookup = true` | `DIRECT_ANSWER` + Escalation Info | Hướng dẫn chung + thông tin liên hệ tổng đài |

> Khi `need_account_lookup=true`, hệ thống **không** early exit mà vẫn thực hiện retrieval để cung cấp hướng dẫn chung, sau đó kèm thông tin liên hệ tổng đài để xử lý chi tiết. Điều này đảm bảo khách hàng luôn nhận được thông tin hữu ích thay vì bị chuyển đi ngay.

### 3.6 LLM tạo sinh xử lý câu hỏi đa ý

**Chế độ LLM Synthesis:**

Sử dụng `gpt-4o-mini` (temperature=0.3, max_tokens=400) để tổng hợp câu trả lời từ nhiều contexts khi kết quả retrieval chưa đủ rõ ràng hoặc câu hỏi phức tạp.

**Cải tiến – Multi-Part Question Detection:**

Hệ thống tự động phát hiện câu hỏi đa phần bằng 3 patterns:

| Pattern | Ví dụ | Regex |
|---------|-------|-------|
| **Conditional follow-up** | "có X không? nếu có thì Y?" | `nếu (có\|được\|vậy\|rồi) thì` |
| **Multiple question marks** | "X là gì? Y ở đâu?" | `count('?') >= 2` |
| **Multiple question sentences** | "X không? Y sao?" | Nhiều câu chứa từ hỏi tiếng Việt |

**Khi phát hiện multi-part:**
- **Luôn** sử dụng LLM synthesis, kể cả khi similarity ≥ 0.90
- Mở rộng context lên tối đa **5** (thay vì 3 cho câu hỏi đơn)
- Hạ ngưỡng context filtering để thu thập đủ thông tin cho mọi phần

**Synthesis Prompt:**
```python
SYNTHESIS_PROMPT = """
CÂU HỎI KHÁCH HÀNG: {user_question}

THÔNG TIN THAM KHẢO:
{contexts}  # Top 3-5 contexts

HƯỚNG DẪN:
- Trả lời TẤT CẢ các phần của câu hỏi
- Dùng NGUYÊN VĂN thông tin từ nguồn (không diễn giải)
- Dùng semantic matching để hiểu ý định
- Trả về "chưa có thông tin" nếu không có nguồn phù hợp
"""
```

**No-Info Detection:** Nếu LLM synthesis trả về cụm như "chưa có thông tin", hệ thống tự động chuyển sang template `ESCALATE_LOW_CONFIDENCE` với thông tin liên hệ tổng đài thay vì trả lời mập mờ.

### 3.7 Tối ưu tốc độ trả lời câu đơn giản

Khi kết quả retrieval có độ tin cậy cao và câu hỏi đơn giản (không phải multi-part), hệ thống **bỏ qua hoàn toàn** LLM synthesis:

```python
if decision.top_result.similarity_score >= 0.90 and not is_multi_part:
    return format_direct_answer(decision.top_result)  # ~0.5s
```

**So sánh latency:**

| Mode | Latency | Điều kiện |
|------|---------|-----------|
| **Fast-Path** | ~0.5s | similarity ≥ 0.90 AND câu hỏi đơn |
| **LLM tạo sinh** | ~10-15s | similarity < 0.90 HOẶC multi-part |

Kết quả: Giảm latency trung bình **95%** cho các trường hợp high-similarity, trong khi vẫn đảm bảo chất lượng câu trả lời vì answer content được lấy trực tiếp từ knowledge base đã kiểm duyệt.

### 3.8 Adaptive Context Filtering 

Thay vì sử dụng ngưỡng cố định, hệ thống điều chỉnh ngưỡng lọc context **động** dựa trên phân bố điểm similarity:

```python
# Adaptive threshold theo top similarity
if top_sim >= 0.90:
    sim_threshold = max(0.82, top_sim * 0.85)   # Ngưỡng cao khi top rất tốt
elif top_sim >= 0.80:
    sim_threshold = max(0.78, top_sim * 0.82)   # Ngưỡng trung bình
else:
    sim_threshold = max(0.73, top_sim * 0.80)   # Ngưỡng thấp hơn khi top yếu

# Multi-part: hạ ngưỡng để thu thập nhiều context hơn
if is_multi_part:
    sim_threshold = min(sim_threshold, max(0.70, top_sim * 0.75))
```

**Context Deduplication:** Sau khi lọc, hệ thống loại bỏ các context trùng lặp bằng cách so sánh 100 ký tự đầu của `answer_content` — giải quyết vấn đề nhiều nhà cung cấp (VnEdu, SSC, DTSoft) có cùng nội dung trả lời trong knowledge base.

**Giới hạn context:**
- Câu hỏi đơn: tối đa **3** contexts
- Câu hỏi multi-part: tối đa **5** contexts
- Luôn bao gồm ít nhất top 1 result

### 3.9 Vietnamese Text Normalization

Module `TextNormalizer` xử lý input tiếng Việt qua 2 bước, sử dụng thuật toán **longest-match-first** để tránh partial matching sai.

**Bước 1 — Mở rộng viết tắt (60+ mapping):**

| Viết tắt | Mở rộng | Viết tắt | Mở rộng |
|----------|---------|----------|---------|
| ko, k | không | tk | tài khoản |
| dc, đc | được | ck | chuyển khoản |
| sdt | số điện thoại | vcb | Vietcombank |
| nv | nhân viên | đt | điện thoại |

**Bước 2 — Phục hồi dấu tiếng Việt (150+ mapping):**

| Không dấu | Có dấu | Không dấu | Có dấu |
|-----------|--------|-----------|--------|
| chuyen tien | chuyển tiền | ngan hang | ngân hàng |
| lien ket | liên kết | hoc phi | học phí |
| khong duoc | không được | han muc | hạn mức |

**Cơ chế longest-match-first:**
```python
# Sắp xếp cụm từ theo độ dài giảm dần
# "chuyen tien" (11 ký tự) → match trước "chuyen" (6 ký tự)
# Tránh kết quả sai: "chuyển tien" (partial match)
sorted_patterns = sorted(mapping.keys(), key=len, reverse=True)
```

**Phòng tránh false positive:** Các từ ngắn như "gi", "vay", "vi", "la", "ma" chỉ được match với word boundary (`\b`) để tránh phá vỡ các từ ghép như "digilife", "lazada", "dịch vụ".

### 3.10 Xử lý các biến thể của một câu hỏi

LLM Intent Parser tự động chuẩn hóa câu hỏi người dùng về dạng chuẩn của knowledge base, cải thiện đáng kể semantic matching:

| Cách hỏi của người dùng | Condensed Query (chuẩn) |
|------------------------|------------------------|
| "chuyển từ MB sang VNPT Money" | "nạp tiền từ ngân hàng vào ví VNPT Money" |
| "tiền bị trừ nhưng chưa cộng" | "nạp tiền bị trừ tiền nhưng ví không cộng" |
| "đã ck 21 củ rồi nhưng chưa vào" | "nạp tiền từ ngân hàng nhưng chưa nhận được" |
| "có nạp được không? nếu được thì nạp ở đâu?" | condensed query bao gồm CẢ hai phần hỏi |

**Quy tắc đặc biệt cho multi-part questions:** LLM được hướng dẫn condensed query phải bao gồm **tất cả** các phần của câu hỏi, không chỉ phần đầu tiên.

### 3.11 Embedding Caching 

**EmbeddingCache (LRU, max 500):**
```python
class EmbeddingCache:
    def _normalize_query(self, text):
        normalized = text.lower().strip()
        normalized = " ".join(normalized.split())
        return normalized
    
    def _hash_query(self, text):
        return hashlib.md5(self._normalize_query(text).encode()).hexdigest()
```
- Chuẩn hóa text trước khi hash → tránh duplicate cache cho cùng một query
- FIFO eviction (loại 10% đầu khi đầy)
- Ghi nhận hit/miss statistics

**QueryNormalizer (Retrieval-time):**

Một lớp chuẩn hóa slang bổ sung ở tầng retrieval, xử lý các viết tắt phổ biến trước khi tạo embedding: `đt→điện thoại`, `sdt→số điện thoại`, `ck→chuyển khoản`, `tk→tài khoản`.




## 4. Luồng xử lý (Pipeline Flow)

### 4.1 Main Flow

```
┌──────────────┐
│ User Message │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  1. INPUT PROCESSING                     │
│  • Lấy chat history (last 10 messages)   │
│  • Text normalization (viết tắt, dấu)   │
│  • Get session state (clarify_count)     │
│  • Dual storage: Redis + in-memory       │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  2. HYBRID INTENT PARSING                │
│  • Rule-based first (priority-ordered)   │
│  • LLM fallback if confidence < 0.6     │
│  • Output: StructuredQueryObject         │
│    - service, problem_type               │
│    - condensed_query (multi-part aware)  │
│    - need_account_lookup                 │
│    - is_out_of_domain                    │
│    - confidence_intent, missing_slots    │
└──────────────────┬───────────────────────┘
                   │
         ┌─────────┼─────────┐
         │         │         │
         ▼         ▼         ▼
    ┌─────────┐ ┌─────┐ ┌─────────┐
    │Personal │ │ OK  │ │Out of   │
    │Data     │ │     │ │Domain   │
    └────┬────┘ └──┬──┘ └────┬────┘
         │         │         │
    [RETRIEVE +    │    [ESCALATE]
     ESCALATE]     │
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  3. RETRIEVAL + CROSS-CHECK FALLBACK         │
│  • Graph Constraint Filter (cached)          │
│  • Vector Search (constrained, top_k=10)     │
│  • Cross-Check: if < 3 results OR sim < 0.88 │
│    → Search full KB, accept if +0.02 improve │
│  • Graph Traversal (fetch answers)           │
│  • Output: Candidates + Full Contexts        │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  4. RANKING & CONFIDENCE                     │
│  • 4-signal RRF fusion                       │
│  • Primary Group Boost (0.95 vs 0.65)        │
│  • Low Similarity Penalty (< 0.6)            │
│  • Confidence: 0.35×RRF + 0.30×Intent        │
│    + 0.20×Gap + 0.15×SlotPenalty              │
│  • Score Gap: max(RRF_gap, Vector_gap)        │
│  • Ambiguity Detection                       │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  5. ADAPTIVE DECISION                        │
│  • Override: sim ≥ 0.95 → DIRECT             │
│  • Override: conf ≥ 0.65 → DIRECT            │
│  • Certainty: 0.75×Conf + 0.15×Gap + 0.10×RRF│
│  • ≥ 0.55 → Direct | ≥ 0.45 → Clarify       │
│  • < 0.35 → Escalate | else → Clarify        │
│  • Max clarify: 10 lần → Escalate            │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  6. RESPONSE GENERATION                      │
│  • Detect multi-part question                │
│  • Fast-Path (sim ≥ 0.90, single): ~0.5s     │
│  • LLM Synthesis (multi-part or sim < 0.90)  │
│  • Adaptive context: 3 (single) / 5 (multi)  │
│  • Context deduplication (first 100 chars)   │
│  • No-info detection → escalation template   │
│  • Forbidden phrases validation              │
│  • Personal data escalation append           │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  7. MONITORING & LOGGING                     │
│  • Record to Prometheus (latency, decision)  │
│  • Update session state (clarify_count)      │
│  • Sample for RAGAS evaluation (10%)         │
│  • Feedback buttons (Hữu ích/Chưa hữu ích)  │
└──────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────┐
│ Response to User │
└──────────────────┘
```

### 4.2 Session Management

```python
class SessionManager:
    """Quản lý trạng thái phiên (Redis-backed + in-memory fallback)"""
    
    # Đếm số lần hỏi lại
    def get_clarify_count(session_id) -> int
    def increment_clarify_count(session_id) -> int
    def reset_clarify_count(session_id) -> None
    
    # Logic:
    # - Increment khi CLARIFY_REQUIRED
    # - Reset khi DIRECT_ANSWER hoặc ANSWER_WITH_CLARIFY
    # - Escalate khi count >= 10
    
    # Session TTL: 30 phút (1800 seconds)
```

**Dual Chat History:** Lịch sử hội thoại được lưu trữ đồng thời trên Redis (persistent, max 20 messages) và in-memory (fast access fallback), đảm bảo tính liên tục ngay cả khi Redis gián đoạn.

### 4.3 Latency Breakdown

| Component | Latency |
|-----------|---------|
| Text Normalization | ~1-2ms |
| Intent Parsing (Rule) | ~5ms |
| Intent Parsing (LLM fallback) | ~200-500ms |
| Retrieval (Constraint + Vector + Cross-Check) | ~50-150ms |
| Ranking (RRF + Confidence) | ~10ms |
| Decision Engine | ~1ms |
| Response Generation (Fast-Path) | ~50ms |
| Response Generation (LLM Synthesis) | ~1000-3000ms |
| **Total (Fast-Path, sim ≥ 0.90)** | **~0.5s** |
| **Total (LLM Synthesis)** | **~10-15s** |

---

## 5. Chi tiết từng Module

### 5.1 schema.py (454 dòng)

**Vai trò:** Định nghĩa tất cả enums, dataclasses, constants và cấu hình hệ thống.

**Thành phần chính:**

| Thành phần | Số lượng | Mô tả |
|------------|----------|-------|
| `ServiceEnum` | 22 services | Phân loại dịch vụ VNPT Money |
| `ProblemTypeEnum` | 9 loại | Phân loại vấn đề (that_bai, huong_dan, ...) |
| `DecisionType` | 7 loại | Phân loại quyết định routing |
| `Config` | 18 tham số | Tất cả thresholds và hyperparameters |
| `SERVICE_GROUP_MAP` | 22 mappings | Service → prioritized groups |
| `ESCALATION_TEMPLATES` | 7 templates | Template cho từng loại escalation |
| `FORBIDDEN_PHRASES` | 8 cụm từ | Anti-hallucination validation |

**Config — Bảng tham số đầy đủ:**

| Tham số | Giá trị | Vai trò |
|---------|---------|---------|
| `INTENT_PARSER_MODEL` | gpt-4o-mini | Model phân tích intent |
| `INTENT_PARSER_TEMPERATURE` | 0.0 | Deterministic parsing |
| `INTENT_PARSER_MAX_TOKENS` | 300 | Giới hạn output intent |
| `RESPONSE_GENERATOR_MODEL` | gpt-4o-mini | Model sinh response |
| `RESPONSE_GENERATOR_TEMPERATURE` | 0.3 | Balance factual + natural |
| `RESPONSE_GENERATOR_MAX_TOKENS` | 400 | Giới hạn output response |
| `EMBEDDING_MODEL` | text-embedding-3-small | Model embedding (1536 dims) |
| `VECTOR_SEARCH_TOP_K` | 10 | Số candidates per search |
| `RRF_K` | 60 | RRF smoothing constant |
| `RANKING_WEIGHTS` | {vector:1.0, keyword:0.8, graph:0.6, intent:1.2} | Trọng số RRF |
| `CONFIDENCE_HIGH_THRESHOLD` | 0.85 | Ngưỡng confidence cao |
| `CONFIDENCE_MEDIUM_THRESHOLD` | 0.60 | Ngưỡng confidence trung bình |
| `CONFIDENCE_LOW_THRESHOLD` | 0.40 | Ngưỡng confidence thấp |
| `SCORE_GAP_THRESHOLD` | 0.15 | Ngưỡng score gap |
| `MAX_CLARIFY_COUNT` | 10 | Tối đa số lần hỏi lại |
| `CHAT_HISTORY_MAX_MESSAGES` | 10 | Cửa sổ lịch sử hội thoại |
| `SESSION_TTL_SECONDS` | 1800 | Thời gian sống session (30 phút) |
| `LOG_SAMPLE_RATE_FOR_RAGAS` | 0.10 | Tỷ lệ sampling cho đánh giá (10%) |

### 5.2 intent_parser.py (1535 dòng)

**Vai trò:** Phân tích câu hỏi người dùng thành `StructuredQueryObject` — module lớn nhất của hệ thống.

**Classes:**

| Class | Vai trò | Đặc điểm |
|-------|---------|----------|
| `IntentParserHybrid` (=`IntentParser`) | Entry point chính | Rule-first, LLM fallback < 0.6 |
| `IntentParserLocal` | Rule-based parser | Priority-ordered keywords, action verb overrides |
| `IntentParserLLM` | LLM-based parser | System prompt chi tiết, condensed query generation |
| `TextNormalizer` | Chuẩn hóa tiếng Việt | 60+ abbreviations, 150+ no-accent mappings |

**Output:** `StructuredQueryObject` chứa:
- `service`: ServiceEnum — loại dịch vụ
- `problem_type`: ProblemTypeEnum — loại vấn đề
- `condensed_query`: str — câu hỏi chuẩn hóa cho vector search
- `need_account_lookup`: bool — cần tra cứu tài khoản cá nhân?
- `is_out_of_domain`: bool — ngoài phạm vi?
- `confidence_intent`: float (0-1) — độ tin cậy phân tích
- `missing_slots`: List[str] — thông tin còn thiếu
- `topic`, `bank`, `amount`, `error_code`: Optional slots

### 5.3 retrieval.py (332 dòng)

**Vai trò:** Truy vấn Neo4j với ràng buộc, bao gồm cross-check fallback.

**Classes:**

| Class | Vai trò |
|-------|---------|
| `EmbeddingCache` | LRU cache embeddings (max 500, FIFO eviction) |
| `GraphConstraintFilter` | Service → groups → constrained Problem IDs (Cypher, cached) |
| `ConstrainedVectorSearch` | Vector search trong tập đã lọc + cross-check fallback |
| `GraphTraversal` | Duyệt graph lấy context đầy đủ (Answer, Topic, Group) |
| `QueryNormalizer` | Chuẩn hóa slang ở tầng retrieval |
| `RetrievalPipeline` | Orchestrator cho toàn bộ retrieval pipeline |

### 5.4 ranking.py (252 dòng)

**Vai trò:** Xếp hạng candidates sử dụng RRF đa tín hiệu.

**Classes:**

| Class | Vai trò |
|-------|---------|
| `KeywordMatcher` | BM25-style tokenized overlap (loại stopwords tiếng Việt) |
| `GraphDistanceScorer` | Topic/group matching với primary group boost |
| `IntentAlignmentScorer` | Cross-intent similarity matrix |
| `MultiSignalRanker` | RRF fusion + confidence + gap + ambiguity computation |

### 5.5 decision_engine.py (256 dòng)

**Vai trò:** Quyết định routing dựa trên certainty đa tín hiệu.

**Classes:**
- `DecisionEngine`: 3 lớp kiểm tra (similarity override → confidence-first → certainty-based)
- `SessionManager`: Quản lý clarify_count, session state

### 5.6 response_generator.py (405 dòng)

**Vai trò:** Sinh response từ context đã truy vấn.

**Classes:**
- `ResponseGenerator`: LLM-based (fast-path + synthesis + multi-part handling)
- `ResponseGeneratorSimple`: Template-based fallback

**Tính năng chính:**
- Fast-path (sim ≥ 0.90 + single question): trả answer trực tiếp
- Multi-part detection → forced LLM synthesis
- No-info detection → chuyển escalation template
- Personal data escalation append
- Forbidden phrases validation

### 5.7 pipeline.py (523 dòng)

**Vai trò:** Orchestrator chính kết nối tất cả components.

**Class:** `ChatbotPipeline`

**Tính năng chính:**
- Adaptive context filtering V2 (ngưỡng động)
- Context deduplication (first 100 chars)
- Multi-part detection ở pipeline level
- `retrieve_with_fallback` handling
- Monitoring integration (Prometheus metrics)
- Dual chat history (Redis + in-memory)

### 5.8 redis_manager.py (569 dòng)

**Vai trò:** Quản lý Redis với connection pooling.

**Class:** `RedisManager` (Singleton)

**Tính năng chính:**
- Connection pooling (max 50 connections)
- Automatic reconnect (health check mỗi 30s)
- Key prefix isolation: `session:`, `cache:`, `ratelimit:`, `metrics:`, `chat_history:`
- Chat history: Redis list (`lpush`/`ltrim`), max 20 messages
- TTLs: session=30min, cache=1h, rate_limit=1min, metrics=24h, chat_history=30min

### 5.9 monitoring.py (698 dòng)

**Vai trò:** Hệ thống giám sát sức khỏe và hiệu suất.

**Classes:**

| Class | Vai trò |
|-------|---------|
| `MetricsCollector` | Thu thập counters, gauges, histograms, time series |
| `HealthChecker` | Kiểm tra Redis, Neo4j, OpenAI |
| `MonitoringDashboard` | Tổng hợp stats: requests, latency, decision distribution |

**Tính năng:**
- 4 loại metrics: Counter (tăng dần), Gauge (giá trị hiện tại), Histogram (phân bố), Time Series  
- Histogram stats: count, min, max, mean, p50, p95, p99
- Time series bucketing (configurable-minute intervals)
- Export: JSON và Prometheus-compatible format
- Thread-safe: `threading.Lock` cho in-memory operations

### 5.10 app.py (306 dòng)

**Vai trò:** Chainlit web application.

**Features:**
- Welcome message với danh sách dịch vụ
- Real-time processing với Steps UI
- Feedback buttons: "Hữu ích" / "Chưa hữu ích"
- Negative feedback flow: "Hỏi cách khác" / "Liên hệ tổng đài" / "Hỏi câu khác"
- Active sessions tracking qua Redis
- Metrics reset on startup
- Session cleanup on chat end

### 5.11 ingest_data_v3.py (329 dòng)

**Vai trò:** Nạp dữ liệu CSV vào Neo4j.

**Flow chính:**
1. Clear database (optional)
2. Create constraints & indexes
3. Ingest nodes (Groups, Topics, Problems, Answers) + supplement files
4. Create relationships + supplement relationships
5. Generate embeddings (batch 50, model text-embedding-3-small)
6. Create vector index (`problem_embedding_index`, cosine, 1536 dims)

### 5.12 ragas_evaluation.py

**Vai trò:** Khung đánh giá RAGAS tích hợp.

**Classes:**

| Class | Vai trò |
|-------|---------|
| `RAGASEvaluator` | LLM-as-Judge scoring (5 metrics) |
| `PipelineEvaluator` | End-to-end evaluation qua full pipeline |

**Hai chế độ:** Built-in (OpenAI trực tiếp) và RAGAS Library (thư viện chính thức).

---

## 6. Monitoring & Metrics

### 6.1 Kiến trúc Monitoring

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Chainlit App  │────▶│      Redis      │────▶│ Metrics Server  │
│   (Port 8000)   │     │   (Port 6379)   │     │   (Port 8001)   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │     Grafana     │◀────│   Prometheus    │
                        │   (Port 3000)   │     │   (Port 9090)   │
                        └─────────────────┘     └─────────────────┘
```

### 6.2 Metrics được thu thập

| Metric | Type | Mô tả |
|--------|------|-------|
| `chatbot_requests_total` | Counter | Tổng số requests |
| `chatbot_errors_total` | Counter | Tổng số lỗi |
| `chatbot_active_sessions` | Gauge | Số phiên đang hoạt động |
| `chatbot_latency_avg_ms` | Gauge | Latency trung bình |
| `chatbot_latency_p50_ms` | Gauge | Latency percentile 50 |
| `chatbot_latency_p95_ms` | Gauge | Latency percentile 95 |
| `chatbot_latency_p99_ms` | Gauge | Latency percentile 99 |
| `chatbot_confidence_avg` | Gauge | Confidence trung bình |
| `chatbot_neo4j_health` | Gauge | Trạng thái Neo4j (1=UP) |
| `chatbot_redis_health` | Gauge | Trạng thái Redis (1=UP) |
| `chatbot_openai_health` | Gauge | Trạng thái OpenAI (1=UP) |
| `chatbot_decision_*` | Counter | Phân bố quyết định theo loại |

### 6.3 Grafana Dashboard

Dashboard bao gồm các panel:
- **Requests per minute**: Biểu đồ tổng requests theo thời gian
- **Error Rate**: Tỷ lệ lỗi
- **Active Sessions**: Số phiên đang hoạt động
- **Response Latency**: P50, P95, P99, Average
- **Confidence Distribution**: Phân bố confidence scores
- **Decision Distribution**: Tỷ lệ các loại quyết định (Direct, Clarify, Escalate)
- **Service Health**: Trạng thái Neo4j, Redis, OpenAI

### 6.4 Endpoints

| Endpoint | Mô tả |
|----------|-------|
| `GET /health` | Health check tổng hợp |
| `GET /metrics/prometheus` | Prometheus format |
| `GET /metrics/json` | JSON format |

### 6.5 Load Testing Results

**Test Environment:**
- Machine: Local development (Windows)
- Chatbot: Chainlit on port 8000
- Test Tool: Custom Python load tester (`test/load_test.py`)
- OpenAI Rate Limit: 200,000 TPM (tokens per minute)

**Progressive Load Test Results:**

| Concurrent Users | Total Requests | RPS | Avg Latency | Success Rate | Notes |
|-----------------|----------------|-----|-------------|--------------|-------|
| 50 | 250 | 46.7 | 492ms | 100% | Stable |
| 60 | 300 | 57.1 | 633ms | 100% | Rate limit warnings |

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         LOAD TEST SUMMARY                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Maximum Throughput:     ~65 RPS (65 concurrent users)                       ║
║  Optimal Performance:    50 concurrent users                                 ║
║     - Throughput:        46.7 RPS                                            ║
║     - Latency:           492ms average                                       ║
║     - Success Rate:      100%                                                ║
║                                                                              ║
║  Bottleneck:             OpenAI API Rate Limit (200,000 TPM)                 ║
║  Success Rate:           100% (all requests completed)                       ║
║                                                                              ║
║  Capacity Estimation (at 50 concurrent):                                     ║
║     - Per minute:        ~2,800 requests                                     ║
║     - Per hour:          ~168,000 requests                                   ║
║     - Per day:           ~4,000,000 requests                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 7. Đánh giá RAGAS cho GraphRAG

### 7.1 Tổng quan về RAGAS

**RAGAS** (Retrieval-Augmented Generation Assessment) là khung đánh giá chuẩn công nghiệp để đo lường chất lượng hệ thống RAG. Đối với GraphRAG, RAGAS đặc biệt quan trọng vì cần kiểm tra cả chất lượng **graph traversal** (duyệt đúng node/edge) lẫn **grounded generation** (sinh câu trả lời trung thực từ context).

Hệ thống sử dụng phương pháp **LLM-as-Judge**: dùng GPT-4o-mini làm "trọng tài" tự động chấm điểm, phù hợp vì:
1. Đã có OpenAI API key sẵn trong hệ thống
2. Tự động hóa hoàn toàn — có thể tích hợp CI/CD pipeline
3. Không cần human annotators cho đánh giá thường xuyên
4. Sampling rate 10% cho production monitoring

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAGAS EVALUATION FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐  │
│   │ Eval     │────▶│  GraphRAG    │────▶│  LLM-as-     │────▶│  Report   │  │
│   │ Dataset  │     │  Pipeline    │     │  Judge       │     │  (JSON)   │  │
│   │          │     │              │     │              │     │           │  │
│   │ Q + GT   │     │ Q → Ctx + A  │     │ Score 5      │     │ Per-sample│  │
│   │          │     │              │     │ metrics      │     │ + Average │  │
│   └──────────┘     └──────────────┘     └──────────────┘     └───────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Chuẩn bị tập dữ liệu đánh giá

Hệ thống sử dụng **2 tập dữ liệu đánh giá** với mục đích khác nhau:

**Tập cơ bản (20 mẫu) — `eval_dataset.json`:**
- Bao phủ các dịch vụ chính: nạp tiền, rút tiền, chuyển tiền, thanh toán, OTP, hạn mức, đăng ký, eKYC, nạp điện thoại, cước trả sau, quyền riêng tư, điều khoản, tiền điện, liên kết ngân hàng, pending
- Phân bố: easy (8), medium (10), hard (2)
- Dùng cho đánh giá nhanh và regression testing

**Tập mở rộng (50 mẫu) — `eval_dataset_expanded.json` v2.0:**

Được thiết kế để stress-test toàn diện chất lượng RAG với 9 categories:

| Category | Samples | Mục đích |
|----------|---------|----------|
| `original_easy` | 5 | Câu hỏi đơn giản, khớp trực tiếp dtbase |
| `paraphrased` | 7 | Câu hỏi viết lại khác cách diễn đạt |
| `informal_slang` | 6 | Viết tắt, teencode, ngôn ngữ đời thường |
| `edge_case_ambiguous` | 6 | Câu hỏi mơ hồ, thiếu ngữ cảnh |
| `cross_domain` | 6 | Câu hỏi liên quan nhiều chủ đề/dịch vụ |
| `underrepresented_topics` | 8 | Chủ đề ít test: bảo hiểm, vé, vay, học phí |
| `negative_complaint` | 5 | Khiếu nại, thắc mắc tiêu cực |
| `policy_legal` | 5 | Chính sách, điều khoản phức tạp |
| `out_of_domain` | 2 | Câu hỏi ngoài phạm vi (Bitcoin, thời tiết) |

**Cấu trúc mẫu:**
```json
{
    "question": "Cách nạp tiền vào tài khoản Mobile Money",
    "ground_truth": "Trên ứng dụng VNPT Money, Quý khách thực hiện...",
    "metadata": {
        "service": "nap_tien",
        "group": "ho_tro_khach_hang",
        "difficulty": "easy",
        "category": "original_easy"
    }
}
```

### 7.3 Các chỉ số đánh giá cốt lõi

Hệ thống tập trung vào **5 chỉ số** chính, chia thành 2 khía cạnh:

#### A. Khía cạnh Truy xuất (Retrieval Quality)

| Chỉ số | Mô tả | Ngưỡng PASS | Phương pháp |
|--------|-------|-------------|-------------|
| **Context Precision** | Context được retrieve có chính xác và xếp hạng đúng không? | ≥ 70% | LLM-as-Judge (Average Precision) |
| **Context Recall** | Context có chứa đủ thông tin từ ground truth? | ≥ 70% | LLM-as-Judge + Text Overlap pre-check (≥ 85% → score 1.0) |

- **Context Precision**: Đo hiệu quả của thuật toán Constraint-based Retrieval + RRF ranking — context liên quan có được xếp lên đầu không.
- **Context Recall**: Kiểm tra tính đầy đủ của Knowledge Graph — tất cả thông tin cần thiết có được tìm thấy qua graph traversal không.

#### B. Khía cạnh Tạo câu trả lời (Generation Quality)

| Chỉ số | Mô tả | Ngưỡng PASS | Phương pháp |
|--------|-------|-------------|-------------|
| **Faithfulness** | Câu trả lời có hoàn toàn dựa trên context? (anti-hallucination) | ≥ 80% | LLM-as-Judge (Claim verification) |
| **Answer Relevancy** | Câu trả lời có đúng ý câu hỏi? | ≥ 75% | LLM-as-Judge (0.0-1.0) |
| **Answer Similarity** | Câu trả lời có giống ground truth về ngữ nghĩa? | ≥ 70% | Cosine similarity (Embedding) |

### 7.4 Kiến trúc module đánh giá

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     src/ragas_evaluation.py                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐    │
│  │   RAGASEvaluator    │  │  PipelineEvaluator   │  │  Utilities       │    │
│  │                     │  │                      │  │                  │    │
│  │ • evaluate_builtin()│  │ • run_evaluation()   │  │ • load_dataset() │    │
│  │   (LLM-as-Judge)    │  │   (end-to-end)       │  │ • save_report()  │    │
│  │ • evaluate_with_    │  │ • _run_pipeline()    │  │ • print_report() │    │
│  │   ragas() (RAGAS    │  │   (capture contexts) │  │                  │    │
│  │   library)          │  │                      │  │                  │    │
│  └─────────────────────┘  └──────────────────────┘  └──────────────────┘    │
│                                                                              │
│  Scoring Functions (LLM-as-Judge):                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ _score_faithfulness()    → Claim extraction + verification           │  │
│  │ _score_answer_relevancy() → Relevance assessment (0.0-1.0)          │  │
│  │ _score_context_precision()→ Average Precision calculation            │  │
│  │ _score_context_recall()   → Sentence coverage + text overlap check   │  │
│  │ _score_answer_similarity()→ Embedding cosine similarity              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Hai chế độ đánh giá:**

| Chế độ | Mô tả | Dependencies | Khi nào dùng |
|--------|-------|-------------|-------------|
| **Built-in** (`evaluate_builtin`) | LLM-as-Judge thuần, dùng OpenAI trực tiếp | Chỉ cần `openai` | Mặc định, debug nhanh |
| **RAGAS Library** (`evaluate_with_ragas`) | Dùng thư viện RAGAS chính thức | `ragas`, `langchain-openai`, `datasets` | Chuẩn hóa, benchmark |

### 7.5 Quy trình thực thi đánh giá

**Bước 1 — Sampling:** Chọn mẫu có tính đại diện cho Knowledge Graph:

```python
# Tập cơ bản: 20 mẫu (Local queries: 15, Global queries: 5)
# Tập mở rộng: 50 mẫu (9 categories, stress-testing)
```

**Bước 2 — Extraction:** Chạy từng câu hỏi qua pipeline thực tế, hook vào `response_generator.generate()` để capture contexts:

```python
# PipelineEvaluator tự động capture từ Knowledge Graph
def patched_generate(decision, context, user_question, all_contexts=None, ...):
    for ctx in all_contexts:
        ctx_text = ctx.answer_content + ctx.answer_steps + ctx.answer_notes
        captured_contexts.append(ctx_text)
    return original_generate(...)
```

**Bước 3 — Alignment:** Format dữ liệu về `EvalSample` chuẩn RAGAS.

**Bước 4 — Scoring:** Chạy 5 chỉ số song song.

**Bước 5 — Analysis:** Report JSON + console summary.

**Cách chạy (CLI):**

```bash
# Full Pipeline (cần Neo4j + OpenAI)
python test/run_ragas_eval.py --mode full

# Standalone (chỉ cần OpenAI)
python test/run_ragas_eval.py --mode standalone

# Giới hạn mẫu và metrics
python test/run_ragas_eval.py --mode full --limit 10 --metrics faithfulness answer_relevancy

# Dùng RAGAS library chính thức
python test/run_ragas_eval.py --mode standalone --use-ragas-lib
```

### 7.6 Kết quả đánh giá thực tế

**Kết quả mới nhất** (27/02/2026, Full Pipeline, 20 mẫu):

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RAGAS EVALUATION RESULTS                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Chỉ số                     Điểm        Ngưỡng      Trạng thái               ║
║  ─────────────────────── ────────── ────────── ──────────                     ║
║  Faithfulness              98.75%      ≥ 80%       PASS                      ║
║  Answer Relevancy           97.00%      ≥ 75%       PASS                      ║
║  Context Precision          90.00%      ≥ 70%       PASS                      ║
║  Context Recall             97.83%      ≥ 70%       PASS                      ║
║  Answer Similarity          87.75%      ≥ 70%       PASS                      ║
║                                                                              ║
║  Số mẫu: 20 | LLM: gpt-4o-mini | Mode: Full Pipeline                        ║
║  Thời gian đánh giá: 294 giây | Errors: 0                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Chi tiết per-sample (đầy đủ 20 mẫu):**

| # | Câu hỏi | Faith. | Relev. | Prec. | Recall | Sim. |
|---|---------|--------|--------|-------|--------|------|
| 1 | Cách nạp tiền vào tài khoản Mobile Money | 100% | 100% | 100% | 100% | 99.3% |
| 2 | Hướng dẫn rút tiền từ tài khoản Mobile Money | 100% | 100% | 100% | 100% | 99.2% |
| 3 | Hướng dẫn chuyển tiền Mobile Money | 100% | 100% | 100% | 100% | 82.9% |
| 4 | Hướng dẫn thanh toán dịch vụ bằng Mobile Money | 100% | 100% | 100% | 100% | 80.3% |
| 5 | Điều kiện kích hoạt và sử dụng SmartOTP | 100% | 100% | 100% | 100% | 92.0% |
| 6 | Hạn mức giao dịch ví điện tử VNPT Pay | 100% | 100% | 83.3% | 100% | 100% |
| 7 | Đăng ký tài khoản VNPT Money như thế nào? | 100% | 100% | 100% | 90.0% | 87.4% |
| 8 | Làm thế nào để định danh tài khoản VNPT Money? | 100% | 80% | 83.3% | 100% | 76.2% |
| 9 | Hướng dẫn nạp tiền điện thoại qua VNPT Money | 100% | 100% | 100% | 100% | 93.7% |
| 10 | Nạp nhầm số điện thoại có lấy lại được tiền không? | 100% | 100% | 100% | 100% | 98.2% |
| 11 | VNPT Money hỗ trợ nạp tiền cho nhà mạng nào? | 100% | 100% | 50.0% | 100% | 89.8% |
| 12 | Tài khoản bị trừ tiền nhưng thuê bao chưa nhận | 100% | 100% | 100% | 100% | 81.3% |
| 13 | Hướng dẫn thanh toán cước trả sau trên VNPT Money | 100% | 100% | 100% | 100% | 99.2% |
| 14 | VNPT lưu trữ thông tin cá nhân trong bao lâu? | 75% | 60% | 100% | 100% | 73.0% |
| 15 | VNPT có bán thông tin cá nhân của khách hàng không? | 100% | 100% | 33.3% | 100% | 64.5% |
| 16 | Loại dữ liệu cá nhân nào được VNPT xử lý? | 100% | 100% | 100% | 100% | 99.5% |
| 17 | Khách hàng có những quyền gì đối với dữ liệu cá nhân? | 100% | 100% | 83.3% | 100% | 85.1% |
| 18 | Cách thanh toán tiền điện trên VNPT Money | 100% | 100% | 83.3% | 66.7% | 75.0% |
| 19 | Hướng dẫn liên kết ngân hàng với VNPT Money | 100% | 100% | 100% | 100% | 95.3% |
| 20 | Giao dịch chuyển tiền bị pending lâu | 100% | 100% | 83.3% | 100% | 83.1% |

### 7.7 Phân tích & Tối ưu hóa

**Phân tích kết quả hiện tại:**

| Chỉ số | Điểm | Phân tích |
|--------|------|----------|
| **Faithfulness (98.75%)** | Gần hoàn hảo | Anti-hallucination hiệu quả: forbidden phrases + grounded-only prompt. 15/20 mẫu đạt 100%. Duy nhất mẫu #14 (75%) do LLM diễn giải thêm ở chủ đề quyền riêng tư. |
| **Answer Relevancy (97.00%)** | Rất cao | Condensed query chuẩn hóa giúp matching đúng ý định. 16/20 mẫu đạt 100%. Mẫu #8 (80%) và #14 (60%) thấp hơn do câu hỏi phức tạp về eKYC và quyền riêng tư. |
| **Context Recall (97.83%)** | Rất cao | Knowledge Graph đầy đủ thông tin. Cross-check fallback đảm bảo thu thập context từ đúng group. Mẫu #18 (66.7%) — context tiền điện bị thiếu do constraint filter giới hạn quá hẹp. |
| **Context Precision (90.00%)** | Tốt | RRF ranking + Primary Group Boost xếp đúng context liên quan lên đầu. Mẫu #15 (33.3%) — topic quyền riêng tư có nhiều context nhiễu từ group khác. |
| **Answer Similarity (87.75%)** | Khá | Sự khác biệt về cách diễn đạt giữa LLM output và ground truth. Đây là hệ quả tự nhiên khi sử dụng LLM synthesis — câu trả lời đúng ý nhưng khác lời. |

**Hướng dẫn tối ưu theo từng chỉ số:**

| Vấn đề | Nguyên nhân | Hướng tối ưu |
|--------|-------------|-------------|
| Context Recall thấp | Knowledge Graph thiếu thông tin | Bổ sung supplement nodes, mở rộng dữ liệu |
| Context Precision thấp | Graph traversal đi sai hướng | Tinh chỉnh `SERVICE_GROUP_MAP`, tối ưu RRF weights, điều chỉnh primary group boost |
| Faithfulness thấp | LLM thêm thông tin ngoài context | Thêm forbidden phrases, giảm temperature, siết prompt |
| Answer Relevancy thấp | Intent parsing sai hoặc condensed query không chuẩn | Cải thiện rule-based parser, thêm keyword priority |
| Answer Similarity thấp | Cách diễn đạt khác ground truth | Bình thường nếu các metric khác cao — có thể tinh chỉnh response format nếu cần |

**Quy trình tối ưu liên tục:**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 1. Chạy RAGAS│────▶│ 2. Phân tích │────▶│ 3. Tinh chỉnh│────▶│ 4. Chạy lại  │
│    Eval      │     │    Điểm thấp │     │    Component │     │    RAGAS      │
│              │     │              │     │              │     │              │
│ 20/50 mẫu   │     │ Xác định     │     │ Graph Index  │     │ So sánh      │
│ Full Pipeline│     │ bottleneck   │     │ Retrieval    │     │ trước/sau    │
│              │     │              │     │ Prompt       │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                        ┌─────────────┘
                                                        ▼
                                                 ┌──────────────┐
                                                 │ 5. Đạt mục   │
                                                 │    tiêu?      │
                                                 │  YES → Done   │
                                                 │  NO  → Lặp 1  │
                                                 └──────────────┘
```

### 7.8 Lưu ý đặc thù cho GraphRAG

**1. Chuyển đổi Graph Context → Text:**

RAGAS yêu cầu context dạng text. Hệ thống tự động chuyển đổi:
```python
# (Problem) -[:HAS_ANSWER]-> (Answer) → "answer_content + answer_steps + answer_notes"
```

**2. Local vs Global Queries:**

| Loại Query | Ví dụ | Đặc điểm |
|------------|-------|----------|
| **Local** (Entity-specific) | "Cách nạp tiền Mobile Money" | Context tập trung 1-2 nodes, precision thường cao |
| **Global** (Cross-topic) | "Khách hàng có quyền gì với dữ liệu?" | Context từ nhiều nodes, cần recall cao |

**3. Đánh giá bổ sung ngoài RAGAS:**

| Phương pháp | Mục đích |
|------------|----------|
| **Aspect Critique** | Đánh giá tính hữu ích (helpfulness) và tính vô hại (harmlessness) — đặc biệt quan trọng cho domain tài chính |
| **Escalation Accuracy** | Tỷ lệ chuyển tổng đài đúng lúc (không quá sớm/muộn) |
| **Anti-hallucination Check** | Validate response không chứa forbidden phrases |
| **Multi-Part Coverage** | Kiểm tra tất cả phần của câu hỏi đa phần đều được trả lời |

**Bảng tóm tắt quy trình:**

| Bước | Hoạt động | Mục tiêu |
|------|----------|----------|
| 1. Sampling | Chọn mẫu Local & Global, 9 categories | Tính đại diện cho đồ thị |
| 2. Extraction | Pipeline → capture Context + Answer | Thu thập nguyên liệu đánh giá |
| 3. Alignment | Format → EvalSample chuẩn RAGAS | Chuẩn bị cho scoring |
| 4. Scoring | 5 chỉ số (Faith, Relev, Prec, Recall, Sim) | Định lượng chất lượng |
| 5. Optimization | Tinh chỉnh dựa trên điểm thấp | Cải thiện liên tục |

---

## 8. Cài đặt và Chạy dự án

### 8.1 Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|------------|---------|
| **Python** | 3.10 trở lên (khuyến nghị 3.11) |
| **Docker Desktop** | Để chạy Neo4j, Redis, Prometheus, Grafana |
| **RAM** | Tối thiểu 4 GB (services Docker chiếm ~2 GB) |
| **OpenAI API Key** | Cần cho embedding (text-embedding-3-large) và LLM (gpt-4o-mini) |

### 8.2 Cài đặt môi trường Python

**Bước 1 — Tạo môi trường Conda (tải anaconda về để quản lý môi trường - Optional):**

```bash
conda create -n vnpt-chatbot python=3.11 -y
conda activate vnpt-chatbot
```

**Bước 2 — Cài đặt thư viện:**

```bash
pip install -r requirements.txt
```

Các thư viện chính bao gồm: `chainlit`, `openai`, `neo4j`, `redis`, `fastapi`, `pydantic`, `pandas`.

### 8.3 Cấu hình biến môi trường

Tạo file `.env` ở thư mục gốc dự án với nội dung sau:

```env
# Bắt buộc
OPENAI_API_KEY=sk-...

# Neo4j (mặc định khớp với docker-compose.yml)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=

# Redis (mặc định khớp với docker-compose.yml)
REDIS_URL=redis://localhost:6379/0

# Tùy chọn
USE_LLM=true
ENABLE_MONITORING=false
```

> **Lưu ý:** Docker Compose cấu hình Neo4j với `NEO4J_AUTH=none`, do đó `NEO4J_PASSWORD` để trống.

### 8.4 Khởi chạy các services nền

**Bước 1 — Đảm bảo Docker Desktop đang chạy.**

**Bước 2 — Khởi tạo containers:**

```bash
docker-compose up -d
```

Lệnh này khởi chạy 4 services:

| Service | Container | Port | Mô tả |
|---------|-----------|------|--------|
| **Neo4j** | `vnpt-money-neo4j` | `7474` (Browser), `7687` (Bolt) | Graph database chính |
| **Redis** | `vnpt-money-redis` | `6379` | Cache kết quả tìm kiếm |
| **Prometheus** | `vnpt-money-prometheus` | `9090` | Thu thập metrics |
| **Grafana** | `vnpt-money-grafana` | `3000` | Dashboard giám sát |

**Bước 3 — Kiểm tra services đã sẵn sàng:**

```bash
docker-compose ps
```

Đợi tất cả services có trạng thái `healthy` (Neo4j có thể mất ~90 giây để khởi động).

Truy cập Neo4j Browser tại `http://localhost:7474` để kiểm tra kết nối.

### 8.5 Nạp dữ liệu vào Knowledge Base

Sau khi Neo4j đã sẵn sàng, nạp dữ liệu CSV vào graph database:

```bash
conda activate vnpt-chatbot
python src/ingest_data_v3.py
```

Script này thực hiện các bước:
1. Xóa dữ liệu cũ trong database (nếu có)
2. Tạo constraints và indexes trên các node (Group, Topic, Problem, Answer)
3. Nạp nodes và relationships từ thư mục `external_data_v3/`
4. Tạo vector embeddings cho tất cả Problem nodes bằng `text-embedding-3-large`
5. Tạo Neo4j vector index để hỗ trợ similarity search



### 8.6 Chạy ứng dụng Chatbot

```bash
conda activate vnpt-chatbot
chainlit run src/app.py -w
```

- Cờ `-w` bật chế độ watch — tự động reload khi thay đổi code.
- Ứng dụng khởi chạy tại: **http://localhost:8000**



**Prometheus và Grafana (đã chạy sẵn từ Docker Compose):**

Nếu đã chạy `docker-compose up -d` ở bước 8.4, Prometheus và Grafana đã tự động khởi chạy. Prometheus được cấu hình sẵn để scrape metrics từ `host.docker.internal:8001` (file `monitoring/prometheus.yml`).

**Kiểm tra monitoring:**

| Service | URL | Kiểm tra |
|---------|-----|----------|
| **Prometheus** | `http://localhost:9090` | Vào Status → Targets, xác nhận target `vnpt-chatbot` có trạng thái `UP` |
| **Grafana** | `http://localhost:3000` | Đăng nhập (mặc định: `admin` / `admin123`), xem dashboard |


### 8.8 Kiểm tra hệ thống hoạt động

Sau khi khởi chạy, thực hiện các bước kiểm tra:

| Bước | Hành động | Kết quả mong đợi |
|------|-----------|-------------------|
| 1 | Mở `http://localhost:8000` | Giao diện chat Chainlit hiển thị |
| 2 | Gửi: "Hướng dẫn chuyển tiền Mobile Money..." | Nhận câu trả lời hướng dẫn chi tiết |
| 3 | Gửi: "abc xyz 123" | Nhận mẫu chuyển tổng đài (escalation) |
| 4 | Mở `http://localhost:7474` | Neo4j Browser — chạy `MATCH (n) RETURN count(n)` để xác nhận có dữ liệu |



### 8.9 Khác

```bash
# Xem logs của services
docker-compose logs -f neo4j
docker-compose logs -f redis
docker-compose logs -f prometheus
docker-compose logs -f grafana

# Dừng tất cả services
docker-compose down

# Dừng và xóa toàn bộ dữ liệu (volumes)
docker-compose down -v

# Chạy metrics server (nếu muốn xem ở terminal riêng bên ngoài docker)
cd src && uvicorn metrics_server:app --host 0.0.0.0 --port 8001

# Chạy evaluation
python src/ragas_evaluation.py
```
