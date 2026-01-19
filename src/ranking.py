from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RerankConfig:
    """Configuration for ranking thresholds (tier-aware)."""
    # Tier-aware thresholds (RAISED for better precision - avoid wrong answers)
    tier4_min_score: float = 75.0  # Full match: require strong signal
    tier3_min_score: float = 85.0  # Partial match: very strict
    tier2_min_score: float = 95.0  # Base match: extremely strict
    tier1_min_score: float = 100.0  # Fallback: almost impossible (force clarify)
    
    # Minimum confidence from slot extraction (from LLM)
    min_extraction_confidence: float = 0.70  # If slot confidence < 0.7, clarify
    
    # CRITICAL: Semantic relevance threshold (prevent totally wrong answers)
    min_semantic_threshold: float = 0.12  # If semantic < 0.12, answer is IRRELEVANT
    
    # Tie-breaking
    close_score_threshold: float = 5.0  # If scores within 5 pts, use tie-breaker
    
    # Answer verification (use LLM to verify answer relevance)
    enable_answer_verification: bool = True  # Verify answer matches question
    
    # Debug mode
    debug: bool = False  # Enable detailed logging for top 3 candidates


@dataclass
class RankedSolution:
    """Single ranked solution with transparent scoring."""
    solution_id: str
    title: str
    content: str
    
    # Scoring components (transparent and traceable)
    tier: int  # Graph match tier from Neo4j (5=best, 0=fallback)
    type_match: float  # How well solution type matches expected (0-1)
    bank_match: float  # Bank match score (1=exact, 0.6=generic, 0=mismatch)
    semantic: float  # Semantic similarity to query (0-1)
    
    # Final score (simple sum, easy to understand)
    score: float
    
    # Metadata
    solution_type: str = "faq"
    bank: Optional[str] = None
    confidence: float = 0.8
    priority: float = 0


@dataclass
class GateDecision:
    """Decision whether to answer or ask for clarification."""
    decision: str  # "answer" | "clarify" | "out_of_scope"
    reason: str
    best: Optional[RankedSolution]
    topk: List[RankedSolution]
    clarifying_questions: List[str]


def simple_tfidf_similarity(query: str, doc: str) -> float:
    """
    Enhanced TF-IDF with critical keyword matching and exact phrase detection.
    
    Priority order:
    1. Critical keywords (OTP, hạn mức, bị trừ, etc.) - heavily weighted
    2. Exact phrase matches (trigrams, bigrams)
    3. Action word matching (hủy, xem, mua, etc.)
    4. Base TF-IDF similarity
    
    Deterministic: same inputs always give same output.
    """
    if not query or not doc:
        return 0.0
    
    query_lower = query.lower()
    doc_lower = doc.lower()
    
    # Critical domain keywords - these MUST match for specific problems
    critical_keywords = {
        'otp': ['otp', 'ma xac thuc', 'xac thuc', 'khong nhan duoc otp', 'khong nhan otp'],
        'money_issue': ['bi tru tien', 'da tru tien', 'mat tien', 'chua nhan tien', 'tien chua ve', 'khong nhan duoc tien', 
                        'bi tru', 'da bi tru', 'ngan hang da tru', 'vi da bi tru', 'chua duoc cong', 'khong duoc cong',
                        'ngan hang tru', 'tru roi', 'bi tru 2 lan', 'that bai nhung', 'that bai ma', 'bi tru',
                        'da tru', 'tru tien'],
        'duplicate_charge': ['2 lan', 'hai lan', 'trung', 'gach no 2 lan', 'tru 2 lan', 'bi tru 2 lan', 'thanh toan 2 lan',
                            '1 giao dich thanh cong va 1 giao dich dang xu ly'],
        'limit': ['han muc', 'qua han muc', 'vuot han muc', 'toi da', 'toi thieu', 'bao nhieu', 'khong du so du', 
                  'so du khong du', 'tai khoan khong du so du', 'giao dich qua han muc'],
        'invalid': ['khong hop le', 'thong tin khong hop le', 'the khong hop le', 'tai khoan khong hop le',
                    'ma the cao bi loi', 'the bi loi', 'sai thong tin', 'thong tin sai'],
        'voucher': ['voucher', 'ma giam gia', 'uu dai', 'khong su dung duoc voucher', 'voucher khong dung',
                    'mat voucher', 'bi mat voucher'],
        'history': ['lich su', 'xem giao dich', 'tra cuu', 'trong thang', 'giao dich trong thang', 'lich su giao dich', 'xem lich su'],
        'code': ['ma the', 'chua nhan ma', 'ma khong ve', 'khong nhan duoc ma the', 'thong tin ma the'],
        'link_issue': ['khong lien ket', 'lien ket khong duoc', 'khong the lien ket', 'loi lien ket',
                       'khong lien ket duoc', 'lien ket that bai'],
        'transfer_error': ['chuyen nham', 'chuyen sai', 'nham tai khoan', 'sai so tien', 'chuyen nhom'],
        'condition': ['dieu kien', 'yeu cau', 'can gi', 'phai co gi'],
        'transportation': ['ve tau', 've may bay', 'khach san', 'dat ve', 'tau hoa'],
        'status_processing': ['dang xu ly', 'dang cho xu ly', 'dang cho', 'trang thai dang xu ly', 
                             'bao dang xu ly', 'hien thi dang xu ly', 'giao dich dang xu ly'],
        'status_failed': ['that bai', 'khong thanh cong', 'bi loi', 'bao loi', 'loi', 'khong duoc', 
                         'khong su dung duoc', 'khong nap duoc', 'giao dich that bai', 'nap that bai'],
        'status_success': ['thanh cong', 'da thanh cong', 'hoan thanh', 'trang thai thanh cong'],
        'not_registered': ['chua dang ky', 'khong dang ky', 'chua dang ky dich vu', 
                          'chua dang ky thanh toan truc tuyen', 'chua dang ky sms banking', 'tai khoan chua dang ky'],
        'not_received': ['chua nhan', 'khong nhan duoc', 'chua duoc cong', 'chua ve', 'khong nhan duoc thong tin',
                        'thuong huong chua nhan', 'chua nhan duoc uu dai'],
        'mismatch': ['khong trung khop', 'khong trung', 'so dien thoai khong trung khop', 
                    'khuon mat khong trung khop', 'khong khop'],
        'verification': ['dinh danh', 'sinh trac', 'sinh trac hoc', 'xac thuc khuon mat', 'cccd', 'cmnd'],
        'cancel_close': ['huy vi', 'khoa tai khoan', 'huy tai khoan', 'khoa vi', 'huy giao dich', 'huy dich vu'],
        'password': ['mat khau', 'quen mat khau', 'lay lai mat khau', 'doi mat khau', 'thay doi mat khau'],
        'bill_payment': ['thanh toan hoa don', 'hoa don', 'gach no', 'chua gach no', 'chua duoc gach no',
                        'da thanh toan nhung chua gach no'],
        'package_service': ['goi cuoc', 'mua goi cuoc', 'goi mytv', 'goi cuoc truyen hinh', 'dang ky goi'],
        'card_reload': ['mua ma the', 'nap dien thoai', 'ma the cao', 'mua the cao', 'the dien thoai'],
        'not_work': ['khong su dung duoc', 'khong hoat dong', 'khong chay', 'khong dung duoc'],
        'bank_errors': ['giao dich qua han muc', 'vuot qua so tien toi da', 'tai khoan chua dang ky dich vu',
                       'chua dang ky thanh toan truc tuyen', 'tai khoan da duoc lien ket voi vi dien tu khac',
                       'chua mo dich vu', 'chua kich hoat'],
        'auto_service': ['tu dong', 'nap tu dong', 'dang ky tu dong', 'huy tu dong'],
        'display_issue': ['khong hien thi', 'khong thay', 'so du khong cap nhat', 'khong cap nhat',
                         'chua cap nhat', 'hien thi sai', 'sai so du', 'so du chua', 'chua duoc cap nhat',
                         'khong duoc cap nhat', 'so du khong'],
        'account_issue': ['khoa tai khoan', 'mo khoa', 'khoa vi', 'dong vi', 'huy vi'],
        'mytv': ['mytv', 'goi cuoc mytv', 'truyen hinh', 'goi truyen hinh'],
        'reward': ['thuong', 'uu dai', 'hoan tien', 'cashback', 'tich diem'],
        'investigation': ['tra soat', 'khieu nai', 'xu ly', 'giai quyet'],
    }
    
    # Check critical keywords
    critical_boost = 0.0
    critical_penalty = 0.0
    
    for category, keywords in critical_keywords.items():
        query_has = any(kw in query_lower for kw in keywords)
        doc_has = any(kw in doc_lower for kw in keywords)
        
        if query_has and doc_has:
            critical_boost += 1.0  # Strong boost for matching critical keywords (increased from 0.8)
        elif query_has and not doc_has:
            critical_penalty -= 0.6  # Heavy penalty for missing critical keywords (increased from 0.5)
    
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
        X = vectorizer.fit_transform([query_lower, doc_lower])
        sim = cosine_similarity(X[0:1], X[1:2])[0][0]
        
        # Critical action words that must match (these change meaning completely)
        action_words = ['huy', 'hủy', 'thanh toan', 'thanh toán', 'dang ky', 'đăng ký', 
                       'xem', 'mua', 'ban', 'bán', 'tao', 'tạo', 'xoa', 'xóa',
                       'nap', 'nạp', 'rut', 'rút', 'chuyen', 'chuyển']
        
        action_penalty = 0.0
        for action in action_words:
            if action in query_lower and action not in doc_lower:
                action_penalty -= 0.3  # Heavy penalty for missing critical action words
        
        # Boost for exact phrase matches (2-3 word phrases)
        q_words = query_lower.split()
        phrase_boost = 0.0
        
        # Check bigrams
        for i in range(len(q_words) - 1):
            bigram = q_words[i] + " " + q_words[i+1]
            if bigram in doc_lower:
                phrase_boost += 0.15
        
        # Check trigrams (even more important)
        for i in range(len(q_words) - 2):
            trigram = q_words[i] + " " + q_words[i+1] + " " + q_words[i+2]
            if trigram in doc_lower:
                phrase_boost += 0.30  # Increased from 0.25
        
        # Combine: critical keywords (highest priority) + base similarity + phrase boost + penalties
        final_score = critical_boost + sim + phrase_boost + action_penalty + critical_penalty
        
        return float(max(min(final_score, 2.0), 0.0))  # Allow scores up to 2.0 for strong matches
        
    except Exception:
        # Fallback: simple word overlap with keyword checks
        q_words = set(query_lower.split())
        d_words = set(doc_lower.split())
        if not q_words:
            return 0.0
        overlap = len(q_words & d_words)
        base = overlap / len(q_words)
        
        # Apply critical keyword boost/penalty in fallback too
        return max(base + critical_boost + critical_penalty, 0.0)


def expected_solution_type(service: str, problem_type: str) -> str:
    """
    Map service+problem to expected solution type.
    Simplified heuristic: policy questions get "policy", others get "faq".
    
    CRITICAL: Use actual slot_keys from ProblemTypeTaxonomy, not accented keywords.
    """
    p = (problem_type or "").lower()
    
    # Policy questions (using actual slot_keys)
    if any(kw in p for kw in ["chinh_sach", "loi_han_muc", "loi_so_du"]):
        return "policy"
    
    # Procedure questions (using actual slot_keys)
    if any(kw in p for kw in ["huong_dan"]):
        return "procedure"
    
    # Default
    return "faq"


def calculate_type_match(solution_type: str, service: str, problem_type: str) -> float:
    """
    Score how well solution type matches the query.
    Simple binary: exact match = 1.0, mismatch = 0.5 (not 0, to allow flexibility).
    """
    expected = expected_solution_type(service, problem_type)
    sol_type = (solution_type or "").lower()
    
    if sol_type == expected:
        return 1.0
    if sol_type in ["faq", "procedure"]:  # These are often interchangeable
        return 0.7
    return 0.5


def calculate_bank_match(solution_bank: Optional[str], slot_bank: Optional[str]) -> float:
    """
    Score bank matching.
    - Exact match: 1.0
    - Generic (null): 0.6
    - Mismatch: 0.0
    """
    if not slot_bank:
        return 0.8  # User didn't specify bank, so any bank is OK
    
    if not solution_bank:
        return 0.6  # Generic solution, applicable to any bank
    
    # Normalize and compare
    sol_bank_norm = solution_bank.lower().strip()
    slot_bank_norm = slot_bank.lower().strip()
    
    return 1.0 if sol_bank_norm == slot_bank_norm else 0.0


def calculate_slot_agreement(solution: Dict[str, Any], slots: Any) -> float:
    """
    Calculate how well solution's metadata agrees with extracted slots.
    This is critical for tie-breaking when scores are close.
    
    Returns score 0-7 (weighted):
    - +2.0 for service match (critical)
    - +2.0 for problem match (critical)  
    - +1.0 for bank match (nice-to-have)
    - +1.0 for state match (nice-to-have)
    - +1.0 for outcome match (nice-to-have)
    """
    agreement = 0.0
    
    # Service agreement (CRITICAL - 2.0 weight)
    sol_service = solution.get('service', '')
    slot_service = getattr(slots, 'service', None) or ''
    if sol_service and slot_service and sol_service.lower() == slot_service.lower():
        agreement += 2.0
    
    # Problem agreement (CRITICAL - 2.0 weight)
    sol_problem = solution.get('problem', '')
    slot_problem = getattr(slots, 'problem_type', None) or ''
    if sol_problem and slot_problem and sol_problem.lower() == slot_problem.lower():
        agreement += 2.0
    
    # Bank agreement (nice-to-have - 1.0 weight)
    sol_bank = solution.get('bank_id', '')
    slot_bank = getattr(slots, 'bank_id', None) or ''
    if sol_bank and slot_bank:
        if sol_bank.lower() == slot_bank.lower():
            agreement += 1.0
    elif not sol_bank and not slot_bank:
        agreement += 0.8  # Both generic
    elif not sol_bank:  # Solution is generic, slot has specific bank
        agreement += 0.5  # Generic solution can still be useful
    
    # State agreement (optional bonus - 0.5 weight)
    sol_state = solution.get('state', '')
    slot_state = getattr(slots, 'state', None) or ''
    if sol_state and slot_state:
        if sol_state.lower() == slot_state.lower():
            agreement += 0.5
    elif not sol_state or not slot_state or sol_state == 'unknown' or slot_state == 'unknown':
        agreement += 0.3  # Unknown state is common, don't penalize
    
    # Outcome agreement (optional bonus - 0.5 weight)
    sol_outcome = solution.get('outcome', '')
    slot_outcome = getattr(slots, 'outcome', None) or ''
    if sol_outcome and slot_outcome:
        if sol_outcome.lower() == slot_outcome.lower():
            agreement += 0.5
    elif not sol_outcome or not slot_outcome or sol_outcome == 'unknown' or slot_outcome == 'unknown':
        agreement += 0.3  # Unknown outcome is common, don't penalize
    
    return agreement


def check_lexical_anchors(query: str, doc: str) -> int:
    """
    Check if critical lexical anchors appear in both query and doc.
    Returns count of matching anchor phrases.
    
    Critical for tie-breaking when scores are close.
    """
    query_lower = query.lower()
    doc_lower = doc.lower()
    
    # Critical phrases that signal specific problems
    anchors = [
        "không nhận được otp", "không nhận otp", "chưa nhận otp",
        "bị trừ tiền", "đã trừ tiền", "ngân hàng đã trừ",
        "gạch nợ 2 lần", "trừ 2 lần", "thanh toán 2 lần",
        "quá hạn mức", "vượt hạn mức", "hạn mức",
        "đang xử lý", "trạng thái đang xử lý",
        "chuyển nhầm", "chuyển sai",
        "chưa đăng ký sms banking", "chưa đăng ký thanh toán trực tuyến",
        "không lien kết", "lỗi liên kết",
        "định danh", "sinh trắc", "ekyc",
        "voucher", "mã giảm giá",
        "thẻ cao", "mã thẻ",
    ]
    
    matches = 0
    for anchor in anchors:
        if anchor in query_lower and anchor in doc_lower:
            matches += 1
    
    return matches


def rerank_solutions(
    *,
    user_text: str,
    slots: Any,
    neo4j_rows: List[Dict[str, Any]],
    cfg: Optional[RerankConfig] = None
) -> GateDecision:
    """
    Template-first ranking with state/outcome as bonuses (not filters).
    
    Scoring formula (transparent):
    - Neo4j base_score already includes: +30 service, +30 problem, +30 state (if match), +30 outcome (if match), +10 bank
    - We add: title_semantic * 10.0 + content_semantic * 0.3
    
    Key principles:
    - State/outcome are BONUSES in Neo4j scoring, not filters (unknown doesn't hurt recall)
    - Tier reflects match quality: 4 (full match) > 3 (partial) > 2 (base) > 1/0 (fallback)
    - Semantic similarity confirms relevance but doesn't override database matches
    
    Max possible score: ~100 (Neo4j base_score) + 20 (title semantic) + 0.6 (content) = ~121
    """
    cfg = cfg or RerankConfig()
    
    service = getattr(slots, "service", None) or ""
    problem_type = getattr(slots, "problem_type", None) or ""
    slot_bank = getattr(slots, "bank", None) or getattr(slots, "bank_id", None)
    
    ranked: List[RankedSolution] = []
    
    for row in neo4j_rows:
        # Extract fields from Neo4j (matching neo4j_config.py schema)
        solution_id = row.get("qa_id", "") or row.get("title", "")
        title = row.get("question", "") or row.get("title", "")
        content = row.get("answer", "") or row.get("solution", "")
        
        # Parse tier (0-4) from Neo4j
        tier_raw = row.get("tier", "0")
        try:
            if isinstance(tier_raw, str):
                import re
                match = re.search(r'\\d+', tier_raw)
                tier = int(match.group()) if match else 0
            elif isinstance(tier_raw, (int, float)):
                tier = int(tier_raw)
            else:
                tier = 0
        except (ValueError, AttributeError, TypeError):
            tier = 0
        
        # Get base_score from Neo4j (already includes state/outcome bonuses)
        base_score = float(row.get("score", 0))
        
        solution_type = row.get("solution_type", "qa")
        solution_bank = row.get("bank_name") or row.get("bank_id")
        confidence = float(row.get("confidence", 0.8))
        priority = float(row.get("priority", 1.0))
        
        # Calculate additional components
        type_match = calculate_type_match(solution_type, service, problem_type)
        bank_match = calculate_bank_match(solution_bank, slot_bank)
        
        # Enhanced semantic similarity: prioritize title, de-emphasize content
        title_semantic = simple_tfidf_similarity(user_text, title) if title else 0.0
        content_semantic = simple_tfidf_similarity(user_text, content)
        
        # Semantic: title 12x, content 0.3x (balanced emphasis)
        # Max contribution: 2.0*12 + 2.0*0.3 = 24.6
        semantic = (title_semantic * 12.0) + (content_semantic * 0.3)
        
        # Calculate lexical anchors (for tie-breaking)
        lexical_anchors = check_lexical_anchors(user_text, title + " " + content)
        
        # Get solution metadata for scoring
        sol_service = row.get('service', '')
        sol_problem = row.get('problem', '')
        
        # CRITICAL BOOST: Exact state+outcome match (highly specific queries)
        exact_state_outcome_boost = 0.0
        slot_state = getattr(slots, 'state', None) or ''
        slot_outcome = getattr(slots, 'outcome', None) or ''
        sol_state = row.get('state', '')
        sol_outcome = row.get('outcome', '')
        
        # If query has specific state AND outcome, reward exact matches heavily
        if slot_state and slot_state != 'unknown' and slot_outcome and slot_outcome != 'unknown':
            if sol_state == slot_state and sol_outcome == slot_outcome:
                exact_state_outcome_boost = 20.0  # Reduced from 30 to 20 to avoid over-weighting
            elif sol_state == slot_state or sol_outcome == slot_outcome:
                exact_state_outcome_boost = 8.0  # Reduced from 10 to 8
        
        # Calculate slot agreement (weighted: service+problem=critical, others=bonus)
        slot_agreement = calculate_slot_agreement(row, slots)
        
        # Final score: base + semantic + agreement boost + exact state/outcome boost
        # Max agreement: 7.0 (2+2+1+0.5+0.5+bonuses), but critical is service+problem=4.0
        # Use 8x multiplier (reduced from 15x) to prevent score inflation
        # Max contribution: 7.0*8 = 56 (reduced from 105)
        agreement_boost = slot_agreement * 8.0
        
        # CRITICAL: Semantic penalty for weak matches (prevents irrelevant high-agreement answers)
        # If semantic is very low (<0.5), heavily penalize even if slot agreement is good
        semantic_penalty = 0.0
        if semantic < 0.5:
            semantic_penalty = -30.0  # Strong penalty for irrelevant answers
        elif semantic < 1.0:
            semantic_penalty = -15.0  # Moderate penalty for weak matches
        
        # Penalty for missing critical service/problem match
        critical_penalty = 0.0
        if service and not (sol_service and sol_service.lower() == service.lower()):
            critical_penalty -= 20.0  # Heavy penalty for service mismatch
        if problem_type and not (sol_problem and sol_problem.lower() == problem_type.lower()):
            critical_penalty -= 20.0  # Heavy penalty for problem mismatch
        
        # Perfect slot match bonus: if service+problem both match exactly, add extra bonus
        perfect_match_bonus = 0.0
        if (service and sol_service and sol_service.lower() == service.lower() and
            problem_type and sol_problem and sol_problem.lower() == problem_type.lower()):
            perfect_match_bonus = 8.0  # Modest boost for perfect service+problem match (reduced from 10)
        
        # Lexical anchor boost: strong signal for relevance
        lexical_boost = lexical_anchors * 3.0  # Each anchor adds 3 points
        
        score = (base_score + semantic + agreement_boost + exact_state_outcome_boost + 
                 semantic_penalty + critical_penalty + perfect_match_bonus + lexical_boost)
        
        ranked_sol = RankedSolution(
            solution_id=solution_id,
            title=title,
            content=content,
            tier=tier,
            type_match=type_match,
            bank_match=bank_match,
            semantic=semantic,
            score=score,
            solution_type=solution_type,
            bank=solution_bank,
            confidence=min(slot_agreement / 7.0, 1.0),  # Normalize to 0-1 (max 7.0 agreement)
            priority=float(lexical_anchors)  # Store for tie-breaking
        )
        ranked.append(ranked_sol)
        
        # Store debug info separately for logging
        if cfg.debug:
            if not hasattr(ranked_sol, '_debug_info'):
                ranked_sol._debug_info = {}
            ranked_sol._debug_info = {
                'base_score': base_score,
                'semantic': semantic,
                'agreement_boost': agreement_boost,
                'slot_agreement': slot_agreement,
                'exact_state_outcome_boost': exact_state_outcome_boost,
                'semantic_penalty': semantic_penalty,
                'critical_penalty': critical_penalty,
                'perfect_match_bonus': perfect_match_bonus,
                'lexical_boost': lexical_boost,
                'title_semantic': title_semantic,
                'content_semantic': content_semantic,
                'lexical_anchors': lexical_anchors,
                'sol_service': sol_service,
                'sol_problem': sol_problem,
                'slot_service': service,
                'slot_problem': problem_type
            }
    
    # Sort by score (descending), then by confidence (tie-breaker), then by priority
    ranked.sort(key=lambda x: (x.score, x.confidence, x.priority), reverse=True)
    
    # ========== DEBUG LOGGING: Print top 3 candidates ==========
    if cfg.debug and ranked:
        print("\n" + "="*80)
        print("RANKING DEBUG - TOP 3 CANDIDATES")
        print("="*80)
        for idx, sol in enumerate(ranked[:3], 1):
            print(f"\n[Rank #{idx}] qa_id={sol.solution_id}")
            print(f"  Title: {sol.title[:80]}...")
            print(f"  FINAL SCORE: {sol.score:.3f} (tier={sol.tier})")
            
            if hasattr(sol, '_debug_info'):
                debug = sol._debug_info
                print(f"\n  Score Breakdown:")
                print(f"    Base (Neo4j):        {debug['base_score']:>6.2f}")
                print(f"    Semantic:            {debug['semantic']:>6.2f} (title={debug['title_semantic']:.2f} × 12, content={debug['content_semantic']:.2f} × 0.3)")
                print(f"    Agreement boost:     {debug['agreement_boost']:>6.2f} (slot_agreement={debug['slot_agreement']:.2f} × 8)")
                print(f"    Semantic penalty:    {debug.get('semantic_penalty', 0):>6.2f}")
                print(f"    Critical penalty:    {debug['critical_penalty']:>6.2f}")
                print(f"    State/outcome boost: {debug['exact_state_outcome_boost']:>6.2f}")
                print(f"    Perfect match bonus: {debug.get('perfect_match_bonus', 0):>6.2f}")
                print(f"    Lexical boost:       {debug.get('lexical_boost', 0):>6.2f} ({debug['lexical_anchors']} anchors × 3)")
                
                print(f"\n  Slot Matching:")
                print(f"    Service: {debug['sol_service']} vs {debug['slot_service']} → {'✓' if debug['sol_service'] == debug['slot_service'] else '✗'}")
                print(f"    Problem: {debug['sol_problem']} vs {debug['slot_problem']} → {'✓' if debug['sol_problem'] == debug['slot_problem'] else '✗'}")
                print(f"    Confidence: {sol.confidence:.2f}")
        print("\n" + "="*80 + "\n")
    
    # Gate decision (tier-aware with slot agreement tie-breaker)
    if not ranked:
        return GateDecision(
            decision="clarify",
            reason="Không tìm thấy giải pháp phù hợp.",
            best=None,
            topk=[],
            clarifying_questions=_generate_clarifying_questions(slots)
        )
    
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    
    # ========== TIE-BREAKER: If top 2 scores are close, use multiple criteria ==========
    reason_suffix = ""
    if second and abs(best.score - second.score) <= cfg.close_score_threshold:
        # Priority 1: Lexical anchors (strongest signal of relevance)
        if second.priority > best.priority + 1:  # At least 2 more anchors
            best = second
            reason_suffix = " (promoted via lexical anchors)"
        # Priority 2: Slot agreement (metadata match)
        elif second.confidence > best.confidence + 0.1:  # Significantly better agreement
            best = second
            reason_suffix = " (promoted via slot agreement)"
        # Priority 3: Semantic similarity (from debug info if available)
        elif hasattr(second, '_debug_info') and hasattr(best, '_debug_info'):
            if second._debug_info['semantic'] > best._debug_info['semantic'] + 2.0:
                best = second
                reason_suffix = " (promoted via semantic similarity)"
    
    # ========== TIER-AWARE GATE LOGIC ==========
    # No hard semantic requirement - trust DB matches + slot agreement
    
    # Tier 4 (full match): Accept easily
    if best.tier >= 4 and best.score >= cfg.tier4_min_score:
        return GateDecision(
            decision="answer",
            reason=f"Tier 4 full match (score={best.score:.1f}, agreement={best.confidence:.2f}){reason_suffix}",
            best=best,
            topk=ranked[:3],
            clarifying_questions=[]
        )
    
    # CRITICAL: Check if out-of-scope was detected during extraction
    inference_evidence = getattr(slots, 'inference_evidence', {})
    if inference_evidence.get('method') == 'out_of_scope_detection':
        return GateDecision(
            decision="out_of_scope",
            reason=f"Out of scope: {inference_evidence.get('reason', 'Service not supported')}",
            best=None,
            topk=[],
            clarifying_questions=[]
        )
    
    # Check extraction confidence first (from LLM)
    extraction_conf = getattr(slots, 'confidence_score', 1.0)
    if extraction_conf < cfg.min_extraction_confidence:
        return GateDecision(
            decision="clarify",
            reason=f"Low extraction confidence ({extraction_conf:.2f} < {cfg.min_extraction_confidence}). Need more context.",
            best=best,
            topk=ranked[:3],
            clarifying_questions=_generate_clarifying_questions(slots)
        )
    
    # CRITICAL: Check semantic relevance (prevent totally wrong answers)
    # If best answer has very low semantic similarity, it's likely irrelevant
    if best.semantic < cfg.min_semantic_threshold:
        # Check if this is an out-of-scope query (no service match)
        if not slots.service or slots.service_confidence < 0.3:
            return GateDecision(
                decision="out_of_scope",
                reason=f"Service not supported or out of scope (semantic={best.semantic:.3f} < {cfg.min_semantic_threshold})",
                best=None,
                topk=ranked[:3],
                clarifying_questions=[]
            )
        else:
            # Service detected but no good match - need clarification
            return GateDecision(
                decision="clarify",
                reason=f"Low semantic relevance (semantic={best.semantic:.3f} < {cfg.min_semantic_threshold}). Answer may not match question.",
                best=best,
                topk=ranked[:3],
                clarifying_questions=_generate_clarifying_questions(slots)
            )
    
    # Tier 3 (partial match): Moderate threshold
    if best.tier >= 3 and best.score >= cfg.tier3_min_score:
        return GateDecision(
            decision="answer",
            reason=f"Tier 3 partial match (score={best.score:.1f}, agreement={best.confidence:.2f}){reason_suffix}",
            best=best,
            topk=ranked[:3],
            clarifying_questions=[]
        )
    
    # Tier 2 (base match): Stricter threshold
    if best.tier >= 2 and best.score >= cfg.tier2_min_score:
        return GateDecision(
            decision="answer",
            reason=f"Tier 2 base match (score={best.score:.1f}, agreement={best.confidence:.2f}){reason_suffix}",
            best=best,
            topk=ranked[:3],
            clarifying_questions=[]
        )
    
    # Tier 0-1 (fallback): Very strict, requires good slot agreement
    if best.tier < 2 and best.score >= cfg.tier1_min_score:
        if best.confidence >= 0.4:  # At least 2/5 slots agree
            return GateDecision(
                decision="answer",
                reason=f"Tier {best.tier} fallback (score={best.score:.1f}, agreement={best.confidence:.2f}){reason_suffix}",
                best=best,
                topk=ranked[:3],
                clarifying_questions=[]
            )
    
    # Default: clarify
    return GateDecision(
        decision="clarify",
        reason=f"Low confidence (tier={best.tier}, score={best.score:.1f}, agreement={best.confidence:.2f}, extraction_conf={extraction_conf:.2f})",
        best=best,
        topk=ranked[:3],
        clarifying_questions=_generate_clarifying_questions(slots)
    )


def _verify_answer_relevance_with_llm(question: str, answer_title: str, answer_content: str, api_key: str) -> Tuple[bool, float, str]:
    """Use LLM to verify if answer is relevant to question.
    
    Returns:
        (is_relevant, confidence, reason)
    """
    from openai import OpenAI
    import json
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""Kiểm tra xem câu trả lời có phù hợp với câu hỏi không.

Câu hỏi: {question}

Câu trả lời: {answer_title}
{answer_content[:300]}...

Hãy phân tích:
1. Câu hỏi hỏi về gì?
2. Câu trả lời nói về gì?
3. Có liên quan không? (có/không)
4. Độ tin cậy (0.0-1.0)

Trả về JSON:
{{
    "question_topic": "...",
    "answer_topic": "...",
    "is_relevant": true/false,
    "confidence": 0.0-1.0,
    "reason": "..."
}}

Lưu ý: Nếu câu hỏi về dịch vụ KHÔNG có trong VNPT Money (ví dụ: đặt đồ ăn, gọi xe, mua sắm online) thì is_relevant = false.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là hệ thống kiểm tra độ phù hợp của câu trả lời."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        is_relevant = result.get("is_relevant", False)
        confidence = float(result.get("confidence", 0.0))
        reason = result.get("reason", "Unknown")
        
        return (is_relevant, confidence, reason)
    
    except Exception as e:
        print(f"[WARNING] LLM verification failed: {e}")
        # Fallback: assume relevant if LLM fails
        return (True, 0.5, "LLM verification failed")


def _generate_clarifying_questions(slots: Any) -> List[str]:
    """Generate smart clarifying questions based on missing/ambiguous slots."""
    questions = []
    
    # Check if service is missing or low confidence
    service_conf = getattr(slots, 'service_confidence', 1.0)
    if not slots.service or service_conf < 0.6:
        questions.append("Bạn muốn thực hiện dịch vụ gì? (ví dụ: nạp tiền, rút tiền, chuyển tiền, thanh toán hóa đơn, liên kết ngân hàng, ...)")
    
    # Check if problem_type is missing or low confidence
    problem_conf = getattr(slots, 'problem_confidence', 1.0)
    if not slots.problem_type or problem_conf < 0.6:
        # Give context-specific examples based on service
        if slots.service in ["nap_tien", "rut_tien", "chuyen_tien"]:
            questions.append("Bạn gặp vấn đề gì? (ví dụ: giao dịch thất bại, chưa nhận tiền, bị trừ tiền, muốn hướng dẫn, hỏi chính sách, ...)")
        elif slots.service:
            questions.append("Bạn cần hướng dẫn hay đang gặp lỗi/vấn đề gì?")
        else:
            questions.append("Bạn cần hỗ trợ về vấn đề gì?")
    
    # Ask for bank if needed for financial transactions
    if not slots.bank and slots.service in ["nap_tien", "rut_tien", "lien_ket_ngan_hang", "huy_lien_ket_ngan_hang"]:
        questions.append("Bạn đang giao dịch với ngân hàng nào? (ví dụ: Vietcombank, BIDV, Techcombank, ...)")
    
    # Ask for error details if state is 'failed' but no error message
    if getattr(slots, 'state', None) == 'failed' and not getattr(slots, 'error_message', None):
        questions.append("App báo lỗi cụ thể là gì? (ví dụ: 'Thẻ không hợp lệ', 'Giao dịch thất bại', 'Hết hạn mức', ...)")
    
    # If no specific questions, ask for general clarification
    if not questions:
        questions.append("Bạn có thể mô tả chi tiết hơn về vấn đề bạn đang gặp phải không?")
    
    return questions
