# VNPT Money Policy Customer Service GraphRAG Chatbot 
# Dự án thực tập cá nhân 
> **Thời gian thực hiện**: 15/12/2025 - ../02/2026 | **Phiên bản**: 3.2

---
<img width="8192" height="7768" alt="Neo4j Vector Retrieval Flow-2026-02-05-095816" src="https://github.com/user-attachments/assets/0406d53e-095e-4119-8a09-def62545794e" />

<img width="4009" height="8192" alt="flowchart" src="https://github.com/user-attachments/assets/a976afad-7dd2-4b85-910f-06434d96018f" />

https://github.com/user-attachments/assets/243ca33e-4ef8-4d73-bd56-2e1aa2f3ee28





## MỤC LỤC

1. [Giới thiệu và Mục đích](#1-giới-thiệu-và-mục-đích)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Các thuật toán sử dụng](#3-các-thuật-toán-sử-dụng)
   - 3.1 Hybrid Intent Parsing
   - 3.2 Constraint-based Retrieval
   - 3.3 Multi-Signal Ranking (RRF)
   - 3.4 Certainty Score
   - 3.5 Decision Logic 
   - 3.6 LLM Synthesis 
   - 3.7 Embedding Caching
   - 3.8 Vietnamese Text Normalization
   - 3.9 Smart Condensed Query Generation 
   - 3.10 Fast-Path Response Optimization 
4. [Luồng xử lý (Pipeline Flow)](#4-luồng-xử-lý-pipeline-flow)
5. [Chi tiết từng Module](#5-chi-tiết-từng-module)
6. [Monitoring & Metrics](#6-monitoring--metrics)

---

## 1. Giới thiệu và Mục đích

### 1.1 Giới thiệu

VNPT Money GraphRAG Chatbot là hệ thống chatbot hỗ trợ khách hàng về các vấn đề liên quan tới chính sách, điều khoản và dịch vụ (dữ liệu công khai) của app VNPT Money sử dụng kiến trúc **GraphRAG** (Graph-based Retrieval Augmented Generation). 

**Điểm nổi bật của hệ thống:**
- Chỉ trả lời dựa trên knowledge base đã được kiểm duyệt, không hallucination
- Tổng hợp câu trả lời từ nhiều nguồn contexts thay vì single-context
- Tự động chuyển tổng đài khi không chắc chắn thay vì đoán sai
- Xử lý tốt input có dấu và không dấu
- Sử dụng "Certainty Score" kết hợp nhiều yếu tố để quyết định chính xác
- Dashboard Grafana theo dõi hiệu suất và sức khỏe hệ thống
- Chuẩn hóa câu hỏi người dùng về dạng chuẩn để matching tốt hơn
- Bỏ qua LLM khi similarity >= 0.85 để giảm latency xuống ~6s

### 1.2 Mục đích

**Mục tiêu chính:**
- Cung cấp hỗ trợ khách hàng 24/7 cho dịch vụ VNPT Money
- Giải quyết các vấn đề với mức độ phức tạp về ngữ cảnh từ cơ bản tới trung bình
- Biết gì nói đó dựa trên knowledge-base, nếu không biết hoặc bị đánh giá không hữu ích thì gợi ý số tổng đài để người dùng được tổng đài viên giúp đỡ
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
║                                                                              ║
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
...
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
│  LAYER 1: INPUT PROCESSING                                                  │
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
│  LAYER 3: CONSTRAINT-BASED RETRIEVAL                                        │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐     │
│  │ Graph Constraint   │→ │ Vector Search      │→ │ Graph Traversal    │     │
│  │ Filter             │  │ (Constrained)      │  │ (Fetch Context)    │     │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘     │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: RANKING & CONFIDENCE SCORING                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Multi-Signal Ranking (RRF): Vector + Keyword + Graph + Intent          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: DECISION ENGINE                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Decision Router: Confidence-based Routing & Escalation Logic           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 6: RESPONSE GENERATION (Grounded)                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ LLM điều chỉnh format câu trả lời                                      │ │
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

├── src/
│   ├── schema.py              # Định nghĩa Enums, Dataclasses, Constants
│   ├── intent_parser.py       # Phân tích intent, xét cả rule và llm
│   ├── retrieval.py           # Truy vấn Neo4j có ràng buộc
│   ├── ranking.py             # Xếp hạng kết quả đa mô hình (RRF)
│   ├── decision_engine.py     # Quyết định routing
│   ├── response_generator.py  # Sinh response 
│   ├── pipeline.py            # điều phối chính
│   ├── app.py                 # Chainlit application
│   └── ingest_data_v3.py      # Nạp dữ liệu vào Neo4j

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
# B1: Rule-based phan tích ý định 
rule_result = rule_parser.parse(user_message)

# B2: Nếu ý định chưa được rõ ràng hoặc hỏi lạc đề..., confidence < 0.6, fallback sang llm
if rule_result.confidence_intent < 0.6:
    return llm_parser.parse(user_message)
else:
    return rule_result
```

**1. Phân tích bằng Rule-based:**
- Sử dụng regex patterns và keyword matching
- Xác định service, problem_type từ từ khóa

**2. Phân tích bằng LLM:**
- Model: `gpt-4o-mini` với temperature = 0 (deterministic)
- Output: JSON với schema cố định (StructuredQueryObject)
- Chỉ làm slot-filling, không sinh câu trả lời

### 3.2 Constraint-based Retrieval

**Bước 1: Graph Constraint Filter**

```cypher
-- Cypher query DETERMINISTIC 
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

<img width="271" height="394" alt="Reciprocal_Rank_Fusion" src="https://github.com/user-attachments/assets/bbbef8c4-24d8-490e-98db-e8c80a4f6071" />

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

**Decision Matrix (Updated v3.2):**

| Điều kiện | Decision Type | Hành động |
|-----------|--------------|-----------|
| `is_out_of_domain = true` | ESCALATE_OUT_OF_SCOPE | Từ chối lịch sự |
| `clarify_count >= 10` | ESCALATE_MAX_RETRY | Chuyển tổng đài |
| `confidence < 0.40` | ESCALATE_LOW_CONFIDENCE | Chuyển tổng đài |
| `confidence >= 0.85` | DIRECT_ANSWER | Trả lời trực tiếp |
| `confidence >= 0.60` | ANSWER_WITH_CLARIFY | Trả lời + hỏi thêm |
| `is_ambiguous AND confidence < 0.60` | CLARIFY_REQUIRED | Hỏi làm rõ |
| `need_account_lookup = true` | DIRECT_ANSWER + Escalation Info | **Trả lời hướng dẫn + kèm thông tin liên hệ tổng đài** |

>  Khi `need_account_lookup=true`, hệ thống không còn early exit mà vẫn tiến hành retrieval để cung cấp hướng dẫn chung cho khách hàng, sau đó kèm thông tin liên hệ tổng đài để xử lý chi tiết. Điều này đảm bảo khách hàng luôn nhận được thông tin hữu ích.

### 3.6 LLM Synthesis (Response Generation)

**Mô tả:** Xử lý câu hỏi có độ phức tạp cao chứa đa dạng khía cạnh hỏi hoặc hỏi chưa được rõ ràng vì dùng từ chưa tường minh, hệ thống sử dụng LLM để tổng hợp câu trả lời từ top 5 kết quả thay vì chỉ dùng kết quả đầu tiên.

**Cấu hình:**
- Model: `gpt-4o-mini`
- Temperature: `0.3` 
- Input: Top 3 contexts từ ranking 
- Max tokens: 400 

**Quy tắc synthesis:**
```python
SYNTHESIS_PROMPT = """
CÂU HỎI KHÁCH HÀNG: {user_question}

THÔNG TIN THAM KHẢO:
{contexts}  # Top 3 contexts

HƯỚNG DẪN: Trả lời ngắn gọn dựa trên thông tin tham khảo. Dùng semantic matching
để hiểu ý định khách hàng (ví dụ: "chuyển từ ngân hàng" = "nạp tiền từ ngân hàng").
Không bịa thông tin.
"""
```

**Ưu điểm:**
- Kết hợp thông tin từ nhiều nguồn liên quan
- Semantic matching: Hiểu các cách diễn đạt khác nhau của cùng một vấn đề
- Generic prompt: Không hard-code case cụ thể, linh hoạt với mọi câu hỏi
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

### 3.9 Cơ chế chuẩn hóa lại các biến thể của cùng một câu hỏi

**Mô tả:** Chuẩn hóa câu hỏi người dùng về dạng chuẩn của knowledge base để cải thiện semantic matching. Giải quyết vấn đề người dùng hỏi theo nhiều cách khác nhau nhưng cùng một ý hoặc hỏi dùng từ chưa tường minh

**Ví dụ mapping:**
| Cách hỏi của người dùng | Condensed Query (chuẩn) |
|------------------------|------------------------|
| "chuyển từ MB sang VNPT Money" | "nạp tiền từ ngân hàng vào ví VNPT Money" |
| "tiền bị trừ nhưng chưa cộng" | "nạp tiền bị trừ tiền nhưng ví không cộng" |
| "đã chuyển 21 củ rồi nhưng chưa vào" | "nạp tiền từ ngân hàng nhưng chưa nhận được" |
| "làm sao để lấy lại tiền" | "hoàn tiền giao dịch thất bại" |

**Quy tắc:**
```python
QUY_TAC_CONDENSED_QUERY = """
1. "chuyển từ [ngân hàng] sang VNPT Money" → "nạp tiền từ ngân hàng vào ví"
2. "bị trừ tiền nhưng chưa cộng/nhận" → "nạp tiền bị trừ nhưng ví không cộng"  
3. "[số tiền] củ/triệu/k" → bỏ qua số cụ thể, giữ ngữ cảnh
4. Ưu tiên dùng từ khóa chuẩn: "nạp tiền", "rút tiền", "chuyển tiền"
"""
```

**Tác dụng:**
- Tăng similarity score khi vector search
- Giảm mismatch giữa user input và database entries
- Hỗ trợ tốt các biến thể ngôn ngữ tự nhiên

### 3.10 Tối ưu hóa tốc độ trả lời các câu hỏi đơn giản / độ rõ ràng cao (Fast-Path)

**Mô tả:** Bỏ qua LLM synthesis khi kết quả retrieval có độ tin cậy cao, giảm đáng kể latency.

**Điều kiện kích hoạt Fast-Path:**
```python
# Sử dụng trực tiếp answer từ database khi:
if decision.top_result.similarity_score >= 0.85:
    use_direct_answer = True  # Bỏ qua LLM synthesis
```

**So sánh latency:**

| Mode | Latency | Khi nào sử dụng |
|------|---------|-----------------|
| **Fast-Path** | ~6s | similarity >= 0.85 |
| **LLM Synthesis** | ~15-40s | similarity < 0.85 hoặc multi-context |

**Kết quả:**
- Giảm latency từ ~40s xuống ~6s (giảm 85%)
- Vẫn đảm bảo chất lượng câu trả lời với high-similarity matches
- LLM chỉ được gọi khi cần tổng hợp từ nhiều nguồn hoặc similarity thấp

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

### 4.3 Latency Breakdown 

| Component | Latency |
|-----------|---------|
| Intent Parsing (Rule) | ~5ms |
| Intent Parsing (LLM) | ~200-500ms |
| Retrieval (Graph + Vector) | ~50-100ms |
| Ranking | ~10ms |
| Decision | ~1ms |
| Response Generation (Fast-Path) | ~50ms |
| Response Generation (LLM Synthesis) | ~1000-3000ms |
| **Total (Fast-Path, similarity ≥ 0.85)** | **~6s** |
| **Total (LLM Synthesis)** | **~15-40s** |

> Với Fast-Path optimization, latency ~6s (giảm 85%) cho các trường hợp có kết quả matching tốt (similarity ≥ 0.85).

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

**Fast-Path (v3.2):**
```python
# Bỏ qua LLM synthesis khi similarity cao
if decision.top_result.similarity_score >= 0.85:
    return decision.top_result.answer_content  # Direct answer
```

### 5.7 pipeline.py

**Vai trò:** Orchestrator chính kết nối tất cả components

**Class:** `ChatbotPipeline`

**Methods:**
- `process(user_message, session_id) → FormattedResponse`
- Internal: _get_chat_history, _handle_early_exit, _log_interaction

**Cải tiến v3.2:**
- Sử dụng `retrieve_with_fallback` để xử lý các trường hợp không tìm thấy kết quả
- Truyền `need_account_lookup` đến response generator để thêm thông tin escalation

### 5.8 app.py

**Vai trò:** Chainlit web application

**Features:**
- Welcome message
- Real-time processing với Steps
- Feedback buttons (Hữu ích / Chưa hữu ích)
- Follow-up actions (Hỏi cách khác, Liên hệ tổng đài)

### 5.9 ingest_data_v3.py

**Vai trò:** Nạp dữ liệu CSV vào Neo4j

**Flow chính:**
1. Clear database (optional)
2. Create constraints & indexes
3. Ingest nodes (Groups, Topics, Problems, Answers)
4. Create relationships
5. Generate embeddings (OpenAI)
6. Create vector index

**Supplement Data Ingestion (v3.2):**
```python
# Nạp dữ liệu bổ sung mà không ảnh hưởng database hiện tại
def ingest_supplement_only():
    # Load từ db/import/nodes_problem_supplement.csv
    # Load từ db/import/nodes_answer_supplement.csv  
    # Load từ db/import/rels_has_problem_supplement.csv
    # Tạo embedding cho nodes mới
```

**Lưu ý:** File supplement được đặt trong `db/import/` để Neo4j có thể import trực tiếp khi cần.

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

---

