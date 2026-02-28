import logging
import hashlib
from typing import List, Optional, Dict

from schema import (
    StructuredQueryObject,
    CandidateProblem,
    RetrievedContext,
    SERVICE_GROUP_MAP,
    Config,
)

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache embedding để giảm API calls."""
    
    def __init__(self, max_size: int = 500):
        self.cache: Dict[str, List[float]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _normalize_query(self, text: str) -> str:
        normalized = text.lower().strip()
        normalized = " ".join(normalized.split())
        return normalized
    
    def _hash_query(self, text: str) -> str:
        normalized = self._normalize_query(text)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        key = self._hash_query(text)
        if key in self.cache:
            self.hits += 1
            logger.debug(f"Embedding cache HIT (hits={self.hits}, misses={self.misses})")
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        if len(self.cache) >= self.max_size:
            keys_to_remove = list(self.cache.keys())[:self.max_size // 10]
            for k in keys_to_remove:
                del self.cache[k]
        key = self._hash_query(text)
        self.cache[key] = embedding
    
    def stats(self) -> Dict[str, int]:
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


_embedding_cache = EmbeddingCache(max_size=500)


class GraphConstraintFilter:
    """Lọc không gian tìm kiếm dựa trên service/topic."""
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self._group_cache: Dict[tuple, List[str]] = {}
    
    def get_constrained_problems(self, query: StructuredQueryObject) -> List[str]:
        service_value = query.service.value
        allowed_groups = SERVICE_GROUP_MAP.get(service_value, [])
        
        if not allowed_groups:
            allowed_groups = list(SERVICE_GROUP_MAP.get("khac", []))
        
        cache_key = tuple(sorted(allowed_groups))
        if cache_key in self._group_cache:
            problem_ids = self._group_cache[cache_key]
            logger.info(f"Constrained to {len(problem_ids)} Problems (cached)")
            return problem_ids
        
        cypher = """
        MATCH (g:Group)-[:HAS_TOPIC]->(t:Topic)-[:HAS_PROBLEM]->(p:Problem)
        WHERE g.id IN $allowed_groups AND p.status = 'active'
        RETURN DISTINCT p.id AS problem_id
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, {"allowed_groups": allowed_groups})
            problem_ids = [record["problem_id"] for record in result]
        
        self._group_cache[cache_key] = problem_ids
        logger.info(f"Constrained to {len(problem_ids)} Problems from groups: {allowed_groups}")
        return problem_ids
    
    def get_all_active_problems(self) -> List[str]:
        cypher = "MATCH (p:Problem) WHERE p.status = 'active' RETURN p.id AS problem_id"
        with self.driver.session() as session:
            result = session.run(cypher)
            return [record["problem_id"] for record in result]


class ConstrainedVectorSearch:
    """Vector search trên tập Problem đã được lọc."""
    
    def __init__(self, neo4j_driver, embedding_client):
        self.driver = neo4j_driver
        self.embedding_client = embedding_client
        self.embedding_model = Config.EMBEDDING_MODEL
        self.top_k = Config.VECTOR_SEARCH_TOP_K
        self.cache = _embedding_cache
    
    def embed(self, text: str) -> List[float]:
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        response = self.embedding_client.embeddings.create(model=self.embedding_model, input=text)
        embedding = response.data[0].embedding
        self.cache.set(text, embedding)
        return embedding
    
    def search(self, query: str, constrained_ids: List[str], top_k: Optional[int] = None) -> List[CandidateProblem]:
        if not constrained_ids:
            logger.warning("Không có constrained IDs")
            return []
        
        top_k = top_k or self.top_k
        query_embedding = self.embed(query)
        
        cypher = """
        CALL db.index.vector.queryNodes('problem_embedding_index', $top_k * 5, $embedding)
        YIELD node, score
        WHERE node.id IN $constrained_ids
        RETURN node.id AS problem_id, node.title AS title, node.description AS description,
               node.intent AS intent, node.keywords AS keywords, score AS similarity_score
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, {"embedding": query_embedding, "constrained_ids": constrained_ids, "top_k": top_k})
            candidates = []
            for record in result:
                keywords = record["keywords"]
                if isinstance(keywords, str):
                    keywords = keywords.split(",") if keywords else []
                candidates.append(CandidateProblem(
                    problem_id=record["problem_id"],
                    title=record["title"],
                    description=record["description"],
                    intent=record["intent"],
                    keywords=keywords,
                    similarity_score=record["similarity_score"]
                ))
            return candidates
    
    def search_with_fallback(self, query: str, constrained_ids: List[str], all_problem_ids: List[str], top_k: Optional[int] = None) -> List[CandidateProblem]:
        candidates = self.search(query, constrained_ids, top_k)
        
        top_similarity = candidates[0].similarity_score if candidates else 0.0
        
        # Cross-check threshold: when constrained results aren't a strong match,
        # always verify against the full KB. This catches intent misclassification
        # (e.g., "bỏ tiền vào ví" parsed as dieu_khoan instead of nap_tien).
        # Threshold raised from 0.75→0.88: wrong-group content can score 0.80+
        # due to partial semantic overlap (e.g., "Mobile Money" appears in many groups).
        CROSS_CHECK_THRESHOLD = 0.88
        
        should_fallback = (
            len(candidates) < 3 or
            top_similarity < CROSS_CHECK_THRESHOLD
        )
        
        if should_fallback and len(all_problem_ids) > len(constrained_ids):
            logger.info(
                f"Cross-check triggered: {len(candidates)} candidates, "
                f"top_similarity={top_similarity:.3f} < {CROSS_CHECK_THRESHOLD}, "
                f"expanding to all {len(all_problem_ids)} problems"
            )
            fallback_candidates = self.search(query, all_problem_ids, top_k)
            
            # Use fallback if it finds a notably better match (>0.02 improvement)
            # or if it finds a match above 0.85 that the constrained search missed
            if fallback_candidates:
                fallback_top = fallback_candidates[0].similarity_score
                improvement = fallback_top - top_similarity
                
                if improvement > 0.02 or (fallback_top >= 0.85 and top_similarity < 0.85):
                    logger.info(
                        f"Cross-check improved: {fallback_top:.3f} vs {top_similarity:.3f} "
                        f"(+{improvement:.3f})"
                    )
                    candidates = fallback_candidates
                else:
                    logger.info(
                        f"Cross-check: no significant improvement "
                        f"({fallback_top:.3f} vs {top_similarity:.3f}), keeping constrained"
                    )
        
        return candidates


class GraphTraversal:
    """Duyệt graph để lấy context đầy đủ."""
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    
    def fetch_context(self, problem_ids: List[str]) -> List[RetrievedContext]:
        if not problem_ids:
            return []
        
        cypher = """
        MATCH (p:Problem)-[:HAS_ANSWER]->(a:Answer)
        WHERE p.id IN $problem_ids
        MATCH (g:Group)-[:HAS_TOPIC]->(t:Topic)-[:HAS_PROBLEM]->(p)
        RETURN p.id AS problem_id, p.title AS problem_title, a.id AS answer_id,
               a.content AS answer_content, a.steps AS answer_steps, a.notes AS answer_notes,
               t.id AS topic_id, t.name AS topic_name, g.id AS group_id, g.name AS group_name
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, {"problem_ids": problem_ids})
            contexts = []
            for record in result:
                steps = record["answer_steps"]
                if isinstance(steps, str):
                    steps = steps.split("\n") if steps else None
                contexts.append(RetrievedContext(
                    problem_id=record["problem_id"],
                    problem_title=record["problem_title"],
                    answer_id=record["answer_id"],
                    answer_content=record["answer_content"],
                    answer_steps=steps,
                    answer_notes=record["answer_notes"],
                    topic_id=record["topic_id"],
                    topic_name=record["topic_name"],
                    group_id=record["group_id"],
                    group_name=record["group_name"]
                ))
            return contexts
    
    def get_context_for_problem(self, problem_id: str) -> Optional[RetrievedContext]:
        contexts = self.fetch_context([problem_id])
        return contexts[0] if contexts else None


class QueryNormalizer:
    """Chuẩn hóa câu hỏi informal/slang thành dạng chuẩn để cải thiện vector search.
    
    Giải quyết vấn đề: câu hỏi viết tắt, ngôn ngữ đời thường có similarity
    thấp hơn so với câu hỏi chuẩn → giảm Context Precision & Recall khi
    mở rộng test data.
    """
    
    SLANG_MAP = {
        "đt": "điện thoại",
        "sdt": "số điện thoại",
        "ck": "chuyển khoản",
        "tk": "tài khoản",
        "nk": "nạp khoản",
        "đky": "đăng ký",
        "đk": "đăng ký",
        "ko": "không",
        "k": "không",
        "dc": "được",
        "đc": "được",
        "lk": "liên kết",
        "nh": "ngân hàng",
        "app": "ứng dụng",
        "ib": "inbox",
        "tks": "cảm ơn",
        "mk": "mật khẩu",
        "otp": "OTP",
        "gd": "giao dịch",
        "tt": "thanh toán",
    }
    
    @classmethod
    def normalize(cls, query: str) -> str:
        """Chuẩn hóa query: thay thế viết tắt, chuẩn hóa khoảng trắng."""
        import re
        normalized = query.strip()
        
        # Replace slang/abbreviations (word boundary matching)
        for slang, full in cls.SLANG_MAP.items():
            pattern = r'\b' + re.escape(slang) + r'\b'
            normalized = re.sub(pattern, full, normalized, flags=re.IGNORECASE)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized


class RetrievalPipeline:
    """Pipeline retrieval hoàn chỉnh."""
    
    def __init__(self, neo4j_driver, embedding_client):
        self.constraint_filter = GraphConstraintFilter(neo4j_driver)
        self.vector_search = ConstrainedVectorSearch(neo4j_driver, embedding_client)
        self.graph_traversal = GraphTraversal(neo4j_driver)
        self.query_normalizer = QueryNormalizer()
    
    def retrieve(self, query: StructuredQueryObject, top_k: Optional[int] = None) -> tuple[List[CandidateProblem], List[RetrievedContext]]:
        constrained_ids = self.constraint_filter.get_constrained_problems(query)
        logger.info(f"Constrained to {len(constrained_ids)} Problems")
        
        candidates = self.vector_search.search(query.condensed_query, constrained_ids, top_k)
        logger.info(f"Found {len(candidates)} candidates")
        
        problem_ids = [c.problem_id for c in candidates]
        contexts = self.graph_traversal.fetch_context(problem_ids)
        logger.info(f"Retrieved {len(contexts)} contexts")
        
        return candidates, contexts
    
    def retrieve_with_fallback(self, query: StructuredQueryObject, top_k: Optional[int] = None) -> tuple[List[CandidateProblem], List[RetrievedContext]]:
        constrained_ids = self.constraint_filter.get_constrained_problems(query)
        all_ids = self.constraint_filter.get_all_active_problems()
        
        # Normalize query for better matching with informal/slang input
        search_query = self.query_normalizer.normalize(query.condensed_query)
        if search_query != query.condensed_query:
            logger.info(f"Query normalized: '{query.condensed_query}' -> '{search_query}'")
        
        candidates = self.vector_search.search_with_fallback(search_query, constrained_ids, all_ids, top_k)
        problem_ids = [c.problem_id for c in candidates]
        contexts = self.graph_traversal.fetch_context(problem_ids)
        return candidates, contexts
