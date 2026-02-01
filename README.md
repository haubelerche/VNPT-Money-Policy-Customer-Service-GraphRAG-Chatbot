# VNPT Money Policy Customer Service GraphRAG Chatbot 
# Dự án thực tập cá nhân 
> **Thời gian thực hiện**: 15/12/2025 - 01/02/2026 | **Phiên bản**: 3.1

---
![Neo4j Vector Retrieval Flow-2026-02-01-052744.png](action_graphs/Neo4j%20Vector%20Retrieval%20Flow-2026-02-01-052744.png)

## MỤC LỤC

1. [Giới thiệu và Mục đích](#1-giới-thiệu-và-mục-đích)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Các thuật toán sử dụng](#3-các-thuật-toán-sử-dụng)
   - 3.1 Hybrid Intent Parsing
   - 3.2 Constraint-based Retrieval
   - 3.3 Multi-Signal Ranking (RRF)
   - 3.4 Certainty Score
   - 3.5 Decision Logic
   - 3.6 LLM Synthesis *(NEW)*
   - 3.7 Embedding Caching
   - 3.8 Vietnamese Text Normalization *(NEW)*
4. [Luồng xử lý (Pipeline Flow)](#4-luồng-xử-lý-pipeline-flow)
5. [Chi tiết từng Module](#5-chi-tiết-từng-module)
6. [Monitoring & Metrics](#6-monitoring--metrics)

---

## 1. Giới thiệu và Mục đích

### 1.1 Giới thiệu

VNPT Money GraphRAG Chatbot là hệ thống chatbot hỗ trợ khách hàng về các vấn đề liên quan tới chính sách, điều khoản và dịch vụ (dữ liệu công khai) của app VNPT Money sử dụng kiến trúc **GraphRAG** (Graph-based Retrieval Augmented Generation). 

**Điểm nổi bật của hệ thống:**
- **Grounded Responses**: Chỉ trả lời dựa trên knowledge base đã được kiểm duyệt, không hallucination
- **LLM Synthesis**: Tổng hợp câu trả lời từ nhiều nguồn contexts thay vì single-context
- **Intelligent Escalation**: Tự động chuyển tổng đài khi không chắc chắn thay vì đoán sai
- **Vietnamese Text Normalization**: Xử lý tốt input có dấu và không dấu
- **Certainty-based Decision**: Sử dụng "Certainty Score" kết hợp nhiều yếu tố để quyết định chính xác
- **Real-time Monitoring**: Dashboard Grafana theo dõi hiệu suất và sức khỏe hệ thống

### 1.2 Mục đích

**Mục tiêu chính:**
- Cung cấp hỗ trợ khách hàng 24/7 cho dịch vụ VNPT Money
- Giải quyết các vấn đề với mức độ phức tạp từ cơ bản tới trung bình
- Trả lời chính xác dựa trên knowledge base, tự động escalate khi không chắc chắn
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
║    (OTP, hạn mức, điều kiện)          • Sinh thông tin không có nguồn        ║
║  • Hỏi lại khi thiếu thông tin        • Tự ý đưa ra quyết định tài chính     ║
║  • Escalate đúng thời điểm            • Đoán trạng thái tài khoản            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 1.4 Các dịch vụ hỗ trợ

| Nhóm dịch vụ | Chi tiết |
|--------------|----------|
| **Tài chính cơ bản** | Nạp tiền, rút tiền, chuyển tiền, liên kết ngân hàng, thanh toán |
| **Tài khoản & Bảo mật** | OTP/SmartOTP, hạn mức, đăng ký, định danh eKYC, bảo mật |
| **Viễn thông** | Data 3G/4G, mua thẻ, di động trả sau, hóa đơn viễn thông |
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
| **Cache/Session** | Redis 7.x |
| **LLM** | OpenAI GPT-4o-mini |
| **Embedding** | OpenAI text-embedding-3-small |
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
│  LAYER 1: INPUT PROCESSING                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Chat History    │  │ Input Validator │  │ Session Manager │              │
│  │ Manager         │  │                 │  │                 │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: INTENT PARSING & STRUCTURED QUERY BUILDER                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    LLM/Rule-based Intent Parser                      │    │
│  │                    (Slot Filling - KHÔNG sinh answer)                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: CONSTRAINT-BASED RETRIEVAL                                         │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐     │
│  │ Graph Constraint   │→ │ Vector Search      │→ │ Graph Traversal    │     │
│  │ Filter             │  │ (Constrained)      │  │ (Fetch Context)    │     │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘     │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: RANKING & CONFIDENCE SCORING                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Multi-Signal Ranking (RRF): Vector + Keyword + Graph + Intent          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: DECISION ENGINE                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Decision Router: Confidence-based Routing & Escalation Logic           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 6: RESPONSE GENERATION (Grounded)                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LLM Answer Formatter (CHỈ format, KHÔNG thêm thông tin mới)            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEO4J GRAPH DATABASE                               │
│    (Group) -[:HAS_TOPIC]-> (Topic) -[:HAS_PROBLEM]-> (Problem) -[:HAS_ANSWER]-> (Answer)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Cấu trúc thư mục

```
VNPT-MONEY-CHATBOT/
├── docs/
│   ├── ARCHITECTURE_GRAPHRAG_V2.md   # Tài liệu kiến trúc chi tiết
│   └── PROJECT_DOCUMENTATION.md      # Tài liệu này
├── src/
│   ├── schema.py              # Định nghĩa Enums, Dataclasses, Constants
│   ├── intent_parser.py       # Phân tích intent (Hybrid: Rule + LLM)
│   ├── retrieval.py           # Truy vấn Neo4j có ràng buộc
│   ├── ranking.py             # Xếp hạng đa tín hiệu (RRF)
│   ├── decision_engine.py     # Quyết định routing
│   ├── response_generator.py  # Sinh response grounded
│   ├── pipeline.py            # Orchestrator chính
│   ├── app.py                 # Chainlit application
│   └── ingest_data_v3.py      # Nạp dữ liệu vào Neo4j
├── external_data_v3/          # CSV data files
│   ├── nodes_group.csv        # Nhóm dịch vụ
│   ├── nodes_topic.csv        # Chủ đề
│   ├── nodes_problem.csv      # Vấn đề/câu hỏi
│   ├── nodes_answer.csv       # Câu trả lời
│   └── rels_*.csv             # Quan hệ giữa các node
├── db/                        # Neo4j database files
├── test/                      # Test files
├── requirements.txt           # Python dependencies
└── docker-compose.yml         # Docker configuration
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

---

## 3. Các thuật toán sử dụng

### 3.1 Hybrid Intent Parsing

**Chiến lược 2 bước:**

```python
# Bước 1: Rule-based parsing (fast, no latency)
rule_result = rule_parser.parse(user_message)

# Bước 2: Nếu confidence < 0.7, fallback to LLM
if rule_result.confidence_intent < 0.7:
    return llm_parser.parse(user_message)
else:
    return rule_result
```

**Rule-based Parser:**
- Sử dụng regex patterns và keyword matching
- Xác định service, problem_type từ từ khóa
- Trích xuất entities: ngân hàng, số tiền, mã lỗi

**LLM Parser:**
- Model: `gpt-4o-mini` với temperature = 0 (deterministic)
- Output: JSON với schema cố định (StructuredQueryObject)
- Chỉ làm slot-filling, KHÔNG sinh câu trả lời

### 3.2 Constraint-based Retrieval

**Bước 1: Graph Constraint Filter**

```cypher
-- Cypher query DETERMINISTIC (không phải LLM-generated)
MATCH (g:Group)-[:HAS_TOPIC]->(t:Topic)-[:HAS_PROBLEM]->(p:Problem)
WHERE g.id IN $allowed_groups AND p.status = 'active'
RETURN DISTINCT p.id AS problem_id
```

- `SERVICE_GROUP_MAP` ánh xạ service → list of groups
- Ví dụ: `chuyen_tien → ["ho_tro_khach_hang", "dieu_khoan"]`

**Bước 2: Vector Search (Constrained)**

```python
# Vector search CHỈ trên Problem nodes đã filter
query_embedding = embed(condensed_query)
candidates = vector_index.search(
    embedding=query_embedding,
    filter_ids=constrained_problem_ids,
    top_k=10
)
```

- Model embedding: `text-embedding-3-small` (1536 dimensions)
- Sử dụng Neo4j Vector Index với cosine similarity
- Embedding được cache để giảm API calls

**Bước 3: Graph Traversal**

```cypher
MATCH (p:Problem)-[:HAS_ANSWER]->(a:Answer)
WHERE p.id IN $candidate_problem_ids
MATCH (g:Group)-[:HAS_TOPIC]->(t:Topic)-[:HAS_PROBLEM]->(p)
RETURN p.*, a.*, t.*, g.*
```

### 3.3 Multi-Signal Ranking (RRF - Reciprocal Rank Fusion)

**4 tín hiệu ranking:**

| Signal | Mô tả | Weight |
|--------|-------|--------|
| **Vector Similarity** | Cosine similarity từ embedding search | 1.0 |
| **Keyword Match** | BM25-style overlap giữa query và document | 0.8 |
| **Graph Distance** | Điểm dựa trên topic/group matching | 0.6 |
| **Intent Alignment** | Độ phù hợp giữa query intent và problem intent | 1.2 |

> **Note:** Weights được áp dụng trong công thức RRF, không phải normalized weights.

**Công thức RRF:**

$$RRF\_score(d) = \sum_{i \in \{vector, keyword, graph, intent\}} \frac{w_i}{k + rank_i(d)}$$

Trong đó:
- $k = 60$ (RRF smoothing parameter)
- $w_i$ = weight của signal $i$
- $rank_i(d)$ = thứ hạng của document $d$ theo signal $i$

**Keyword Matcher (BM25-style):**

```python
def compute_overlap_score(query_tokens, doc_tokens):
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    overlap = len(query_set & doc_set)
    return overlap / len(query_set)
```

### 3.4 Certainty Score (Decision Making)

**Công thức tính Certainty Score:**

Thay vì chỉ dựa vào confidence, hệ thống sử dụng **Certainty Score** kết hợp nhiều yếu tố:

$$certainty = 0.60 \times confidence + 0.30 \times normalized\_gap + 0.10 \times rrf\_boost$$

Trong đó:
- $confidence$ = confidence score từ ranking (0-1)
- $normalized\_gap$ = min(score_gap / 0.15, 1.0) - khoảng cách giữa top 1 và top 2
- $rrf\_boost$ = min(top_rrf × 2, 1.0) - chất lượng của kết quả tốt nhất

**Tại sao cần Certainty Score?**
- **Confidence cao + Gap thấp** = Có nhiều kết quả giống nhau → Cần thận trọng
- **Confidence cao + Gap cao** = Kết quả rõ ràng → Trả lời trực tiếp  
- **Confidence thấp** = Không chắc chắn → Escalate

### 3.5 Decision Logic (Certainty-based)

**Decision Thresholds:**

| Threshold | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| `CONFIDENCE_HIGH` | 0.85 | Rất chắc chắn → Direct Answer |
| `CONFIDENCE_MEDIUM` | 0.60 | Khá chắc → Answer with Clarify |
| `CONFIDENCE_LOW` | 0.40 | Ngưỡng escalate |

**Decision Matrix:**

| Điều kiện | Decision Type | Hành động |
|-----------|--------------|-----------|
| `need_account_lookup = true` | ESCALATE_PERSONAL | Chuyển tổng đài |
| `is_out_of_domain = true` | ESCALATE_OUT_OF_SCOPE | Từ chối lịch sự |
| `clarify_count >= 10` | ESCALATE_MAX_RETRY | Chuyển tổng đài |
| `confidence < 0.40` | ESCALATE_LOW_CONFIDENCE | Chuyển tổng đài |
| `confidence >= 0.85` | DIRECT_ANSWER | Trả lời trực tiếp |
| `confidence >= 0.60` | ANSWER_WITH_CLARIFY | Trả lời + hỏi thêm |
| `is_ambiguous AND confidence < 0.60` | CLARIFY_REQUIRED | Hỏi làm rõ |

### 3.6 LLM Synthesis (Response Generation)

**Mô tả:** Khi có nhiều contexts liên quan từ retrieval, hệ thống sử dụng LLM để tổng hợp câu trả lời từ top 5 kết quả thay vì chỉ dùng kết quả đầu tiên.

**Cấu hình:**
- Model: `gpt-4o-mini`
- Temperature: `0.3` (low để đảm bảo factual responses)
- Input: Top 5 contexts từ ranking

**Quy tắc synthesis:**
```python
SYNTHESIS_PROMPT = """
CÂU HỎI: {user_question}

THÔNG TIN THAM KHẢO:
{contexts}  # Top 5 contexts

QUY TẮC:
1. Nếu có thông tin PHÙ HỢP → Trả lời dựa trên đó
2. Nếu KHÔNG có thông tin → Trả lời: "Mình chưa có thông tin về vấn đề này..."
3. KHÔNG bịa đặt, KHÔNG trả lời nửa vời
4. KHÔNG liệt kê những gì không biết
"""
```

**Ưu điểm:**
- Kết hợp thông tin từ nhiều nguồn liên quan
- Trả lời tự nhiên hơn single-context approach
- Fallback rõ ràng khi không có thông tin

### 3.7 Embedding Caching

```python
class EmbeddingCache:
    """LRU Cache cho embeddings để giảm API calls"""
    
    def __init__(self, max_size=500):
        self.cache = {}
        self.max_size = max_size
    
    def _normalize_query(self, text):
        # Chuẩn hóa text trước khi hash
        normalized = text.lower().strip()
        normalized = " ".join(normalized.split())
        return normalized
    
    def _hash_query(self, text):
        normalized = self._normalize_query(text)
        return hashlib.md5(normalized.encode()).hexdigest()
```

### 3.8 Vietnamese Text Normalization

**Mô tả:** Chuẩn hóa input tiếng Việt trước khi processing, xử lý cả text có dấu và không dấu.

**Hai dictionary chính:**
- `ABBREVIATIONS`: Mở rộng viết tắt phổ biến (vd: "tk" → "tài khoản")
- `NO_ACCENT_MAP`: Map từ không dấu → có dấu (100+ cụm từ)

**Thuật toán: Longest-match-first**
```python
# Sắp xếp theo độ dài giảm dần để match cụm từ dài trước
sorted_patterns = sorted(mapping.keys(), key=len, reverse=True)

# Ví dụ: "chuyen tien" được match trước "chuyen"
# Tránh: "chuyển tien" (partial match sai)
```

**Ví dụ:**
- Input: "toi khong chuyen tien duoc"
- Output: "tôi không chuyển tiền được"

---

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
│  • Lấy chat history (last N messages)    │
│  • Sanitize input                        │
│  • Get session state (clarify_count)     │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  2. INTENT PARSING                       │
│  • Rule-based first (fast)               │
│  • LLM fallback if confidence < 0.7      │
│  • Output: StructuredQueryObject         │
│    - service, problem_type               │
│    - condensed_query                     │
│    - need_account_lookup (early exit?)   │
│    - is_out_of_domain (early exit?)      │
└──────────────────┬───────────────────────┘
                   │
         ┌─────────┼─────────┐
         │         │         │
         ▼         ▼         ▼
    ┌─────────┐ ┌─────┐ ┌─────────┐
    │Personal │ │OK   │ │Out of   │
    │Data     │ │     │ │Domain   │
    └────┬────┘ └──┬──┘ └────┬────┘
         │         │         │
         ▼         │         ▼
    [ESCALATE]     │    [ESCALATE]
                   │
                   ▼
┌──────────────────────────────────────────┐
│  3. RETRIEVAL                            │
│  • Graph Constraint Filter               │
│    (service → allowed groups)            │
│  • Vector Search (constrained scope)     │
│  • Graph Traversal (fetch answers)       │
│  • Output: Candidates + Contexts         │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  4. RANKING                              │
│  • Compute 4 signals per candidate       │
│  • RRF fusion → final ranking            │
│  • Compute confidence & score_gap        │
│  • Output: RankingOutput                 │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  5. DECISION                             │
│  • Apply decision matrix                 │
│  • Consider clarify_count                │
│  • Output: Decision                      │
│    - type: DIRECT/CLARIFY/ESCALATE       │
│    - top_result, clarification_slots     │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  6. RESPONSE GENERATION                  │
│  • Format answer from retrieved context  │
│  • Add clarification if needed           │
│  • Add source citation                   │
│  • Output: FormattedResponse             │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  7. LOGGING                              │
│  • Log full interaction                  │
│  • Update session state                  │
│  • Sample for RAGAS evaluation (10%)     │
└──────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────┐
│ Response to User │
└──────────────────┘
```

### 4.2 Session Management

```python
class SessionManager:
    """Quản lý trạng thái phiên"""
    
    # Đếm số lần hỏi lại
    def get_clarify_count(session_id) -> int
    def increment_clarify_count(session_id) -> int
    def reset_clarify_count(session_id) -> None
    
    # Logic
    # - Increment khi CLARIFY_REQUIRED
    # - Reset khi DIRECT_ANSWER hoặc ANSWER_WITH_CLARIFY
    # - Escalate khi count >= 3
```

### 4.3 Latency Breakdown (Typical)

| Component | Latency |
|-----------|---------|
| Intent Parsing (Rule) | ~5ms |
| Intent Parsing (LLM) | ~200-500ms |
| Retrieval (Graph + Vector) | ~50-100ms |
| Ranking | ~10ms |
| Decision | ~1ms |
| Response Generation | ~100-300ms |
| **Total (Rule-based)** | **~200ms** |
| **Total (with LLM)** | **~500-800ms** |

---

## 5. Chi tiết từng Module

### 5.1 schema.py

**Vai trò:** Định nghĩa tất cả enums, dataclasses, và constants

**Thành phần chính:**
- `ServiceEnum`: 25+ services (nap_tien, rut_tien, data_3g_4g, ...)
- `ProblemTypeEnum`: 9 loại vấn đề (that_bai, huong_dan, ...)
- `DecisionType`: 7 loại quyết định
- `StructuredQueryObject`: Core query object
- `Config`: Tất cả thresholds và parameters
- `SERVICE_GROUP_MAP`: Ánh xạ service → groups
- `ESCALATION_TEMPLATES`: Template cho các loại escalation
- `FORBIDDEN_PHRASES`: Danh sách cụm từ bị cấm (anti-hallucination)

### 5.2 intent_parser.py

**Vai trò:** Phân tích câu hỏi người dùng thành StructuredQueryObject

**Classes:**
- `IntentParserLocal`: Rule-based parser (regex + keywords)
- `IntentParserLLM`: LLM-based parser (gpt-4o-mini)
- `IntentParserHybrid`: Kết hợp cả hai (mặc định)
- `IntentParser`: Alias cho IntentParserHybrid

**Output:** StructuredQueryObject chứa:
- service, problem_type
- condensed_query (cho vector search)
- need_account_lookup, is_out_of_domain
- confidence_intent, missing_slots

### 5.3 retrieval.py

**Vai trò:** Truy vấn Neo4j với ràng buộc

**Classes:**
- `EmbeddingCache`: Cache embeddings (LRU, max 500)
- `GraphConstraintFilter`: Lọc problems theo service/group
- `ConstrainedVectorSearch`: Vector search trên tập đã lọc
- `GraphTraversal`: Duyệt graph lấy context đầy đủ
- `RetrievalPipeline`: Orchestrator cho retrieval

**Flow:**
1. Filter problems by allowed groups
2. Vector search on filtered problems
3. Fetch full context (answers, topics, groups)

### 5.4 ranking.py

**Vai trò:** Xếp hạng candidates sử dụng RRF

**Classes:**
- `KeywordMatcher`: BM25-style keyword matching
- `GraphDistanceScorer`: Điểm dựa trên topic/group
- `IntentAlignmentScorer`: Điểm dựa trên intent matching
- `MultiSignalRanker`: RRF fusion của 4 signals

**Output:** RankingOutput chứa:
- results: List[RankedResult] đã sắp xếp
- confidence_score, score_gap, is_ambiguous

### 5.5 decision_engine.py

**Vai trò:** Quyết định routing dựa trên confidence

**Classes:**
- `DecisionEngine`: Logic quyết định
- `SessionManager`: Quản lý session state (clarify_count)

**Thresholds:**
- HIGH: 0.85
- MEDIUM: 0.60
- LOW: 0.40
- GAP_THRESHOLD: 0.15
- MAX_CLARIFY: 10

### 5.6 response_generator.py

**Vai trò:** Sinh response từ context đã truy vấn

**Classes:**
- `ResponseGenerator`: Sử dụng LLM để format và tổng hợp
- `ResponseGeneratorSimple`: Không dùng LLM (template-based)

**Tính năng chính:**
- **LLM Synthesis Mode**: Tổng hợp câu trả lời từ top 5 contexts khi có nhiều nguồn liên quan
- Temperature 0.3 cho synthesis (factual responses)
- Fallback escalation khi không đủ thông tin

**Nguyên tắc:**
- CHỈ trả lời dựa trên context có sẵn, KHÔNG thêm thông tin mới
- Nếu không có thông tin phù hợp → escalate với message rõ ràng
- KHÔNG trả lời "nửa vời" (liệt kê những gì không biết)
- Validate response không chứa forbidden phrases

### 5.7 pipeline.py

**Vai trò:** Orchestrator chính kết nối tất cả components

**Class:** `ChatbotPipeline`

**Methods:**
- `process(user_message, session_id) → FormattedResponse`
- Internal: _get_chat_history, _handle_early_exit, _log_interaction

### 5.8 app.py

**Vai trò:** Chainlit web application

**Features:**
- Welcome message
- Real-time processing với Steps
- Feedback buttons (Hữu ích / Chưa hữu ích)
- Follow-up actions (Hỏi cách khác, Liên hệ tổng đài)

### 5.9 ingest_data_v3.py

**Vai trò:** Nạp dữ liệu CSV vào Neo4j

**Flow:**
1. Clear database (optional)
2. Create constraints & indexes
3. Ingest nodes (Groups, Topics, Problems, Answers)
4. Create relationships
5. Generate embeddings (OpenAI)
6. Create vector index

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
| `chatbot_confidence_avg` | Gauge | Confidence trung bình |
| `chatbot_neo4j_health` | Gauge | Trạng thái Neo4j (1=UP) |
| `chatbot_redis_health` | Gauge | Trạng thái Redis (1=UP) |
| `chatbot_openai_health` | Gauge | Trạng thái OpenAI (1=UP) |

### 6.3 Grafana Dashboard

Dashboard bao gồm các panel:
- **Requests per minute**: Biểu đồ tổng requests theo thời gian
- **Error Rate**: Tỷ lệ lỗi 
- **Active Sessions**: Số phiên đang hoạt động
- **Response Latency**: P50, P95, Average latency
- **Confidence Distribution**: Phân bố confidence scores
- **Service Health**: Trạng thái Neo4j, Redis, OpenAI

### 6.4 Endpoints

| Endpoint | Mô tả |
|----------|-------|
| `GET /health` | Health check |
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
| 50 | 250 | 46.7 | 492ms | 100% |  Stable |
| 60 | 300 | 57.1 | 633ms | 100% |  Rate limit warnings |

**Key Findings:**

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         LOAD TEST SUMMARY                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 Maximum Throughput:     ~65 RPS (65 concurrent users)                    ║
║  ⚡ Optimal Performance:    50 concurrent users                              ║
║     - Throughput:           46.7 RPS                                         ║
║     - Latency:              492ms average                                    ║
║     - Success Rate:         100%                                             ║
║                                                                              ║
║  🚧 Bottleneck:            OpenAI API Rate Limit (200,000 TPM)               ║
║  ✅ Success Rate:          100% (all requests completed)                     ║
║                                                                              ║
║  📈 Capacity Estimation (at 50 concurrent):                                  ║
║     - Per minute:          ~2,800 requests                                   ║
║     - Per hour:            ~168,000 requests                                 ║
║     - Per day:             ~4,000,000 requests                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Latency Breakdown (at 50 concurrent users):**

| Phase | Latency Range |
|-------|---------------|
| Start (0-50 requests) | 240-303ms |
| Mid (50-200 requests) | 274-305ms |
| End (200-250 requests) | 263-492ms |



