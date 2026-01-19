from typing import List, Optional, Tuple, Any, Dict
import os
from dataclasses import dataclass
import time
import json

import chainlit as cl
from neo4j import GraphDatabase
from dotenv import load_dotenv
from openai import OpenAI

from cases_handling import CaseTriage, CaseSlots
from ranking import rerank_solutions, GateDecision, RankedSolution
from normalize_content import _norm_text


load_dotenv()

# Test mode flag - set to True to use deterministic inference only
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"


# =============================================================================
# History Management Functions
# =============================================================================

def enrich_query_with_history(
    user_text: str,
    history: List[Dict[str, Any]],
    user_info: Dict[str, Any]
) -> str:
    query_lower = user_text.lower()
    
    # Check if query is a follow-up (short query or uses pronouns)
    followup_indicators = [
        "nó", "đó", "này", "kia", "vừa rồi", "lần trước", 
        "bank này", "ngân hàng này", "dịch vụ này", "còn",
        "thế", "sao", "như vậy", "tương tự"
    ]
    
    is_followup = (
        len(user_text.split()) <= 5 or
        any(indicator in query_lower for indicator in followup_indicators)
    )
    
    if not is_followup or len(history) == 0:
        return user_text  # No enrichment needed
    
    # Build context from user_info and recent history
    context_parts = []
    
    # Add persistent user info
    if user_info.get("bank"):
        context_parts.append(f"Ngân hàng: {user_info['bank']}")
    if user_info.get("service"):
        context_parts.append(f"Dịch vụ: {user_info['service']}")
    if user_info.get("problem"):
        context_parts.append(f"Vấn đề: {user_info['problem']}")
    
    # Add context from last user question (for "còn X thì sao?" pattern)
    recent_user_msgs = [msg for msg in history[-6:] if msg["role"] == "user"]
    if recent_user_msgs:
        last_user_query = recent_user_msgs[-1]["content"]
        # Extract key context from previous question
        if len(last_user_query) > 10:
            context_parts.append(f"Câu hỏi trước: {last_user_query[:80]}")
    
    # Build enriched query
    if context_parts:
        context_str = " | ".join(context_parts)
        enriched = f"[Context: {context_str}] {user_text}"
        return enriched
    
    return user_text


def resolve_coreference_with_llm(
    current_query: str,
    history: List[Dict[str, Any]],
    user_info: Dict[str, Any],
    api_key: str
) -> str:
  
    if len(history) == 0:
        return current_query
    
    # Only resolve if query seems to need context
    query_lower = current_query.lower()
    needs_resolution = any([
        "còn" in query_lower and "thì sao" in query_lower,
        "nó" in query_lower or "đó" in query_lower,
        "bank này" in query_lower or "ngân hàng này" in query_lower,
        "dịch vụ này" in query_lower,
        len(current_query.split()) <= 5
    ])
    
    if not needs_resolution:
        return current_query
    
    # Build conversation context
    context_messages = []
    for msg in history[-4:]:  # Last 4 messages
        context_messages.append({
            "role": msg["role"],
            "content": msg["content"][:200]  # Truncate long messages
        })
    
    # Add user_info to context
    context_info = ""
    if user_info.get("bank"):
        context_info += f"\nNgân hàng hiện tại: {user_info['bank']}"
    if user_info.get("service"):
        context_info += f"\nDịch vụ hiện tại: {user_info['service']}"
    if user_info.get("problem"):
        context_info += f"\nVấn đề hiện tại: {user_info['problem']}"
    
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f"""Dựa vào lịch sử hội thoại, hãy viết lại câu hỏi để bao gồm đầy đủ context (ngân hàng, dịch vụ, vấn đề).

Lịch sử hội thoại:
{json.dumps(context_messages, ensure_ascii=False, indent=2)}
{context_info}

Câu hỏi hiện tại: {current_query}

Yêu cầu:
- Thay thế đại từ (nó, đó, này, bank này) bằng tên cụ thể
- Bổ sung thông tin ngân hàng/dịch vụ nếu thiếu
- Giữ nguyên ý nghĩa câu hỏi gốc
- Chỉ trả về câu hỏi đã viết lại, không giải thích

Câu hỏi đã viết lại:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là trợ lý viết lại câu hỏi với context đầy đủ. Chỉ trả về câu hỏi, không giải thích."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        enriched = response.choices[0].message.content.strip()
        
        # Validate that enriched query is reasonable (not too different)
        if len(enriched) > len(current_query) * 3:
            print("[WARNING] LLM enrichment too verbose, using original")
            return current_query
        
        print(f"[COREFERENCE] Original: {current_query}")
        print(f"[COREFERENCE] Resolved: {enriched}")
        
        return enriched
        
    except Exception as e:
        print(f"[WARNING] Coreference resolution failed: {e}")
        return current_query


# =============================================================================
# Step-by-step pipeline following readme.md Section 3
# =============================================================================

def step1_extract_slots(user_text: str, use_llm: bool = True) -> CaseSlots:
    """
    Step 1: Intent + Slot extraction.
    ALWAYS use the deterministic version for reliability.
    """
    triage = CaseTriage(api_key=os.getenv("OPENAI_API_KEY"))
    # Force deterministic extraction to ensure bank_id is always found correctly
    slots = triage.extract_slots(user_text)
    return slots


def step2_retrieve_from_graph(slots: CaseSlots, enable_fallback: bool = True) -> List[dict]:
    """
    Step 2: Deterministic graph retrieval with fallback support.
    Execute Cypher query to get solutions (with tier scores).
    
    Args:
        slots: Extracted case slots
        enable_fallback: If True and primary query returns 0 results, retry with relaxed filters
    
    Returns:
        List of solution records from Neo4j
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    triage = CaseTriage(api_key=os.getenv("OPENAI_API_KEY"))
    cypher = triage.build_cypher_query(slots)
    params = triage.build_cypher_params(slots)
    
    # DEBUG: Log query structure for troubleshooting
    print("\n[DEBUG] Primary Cypher Query:")
    print(f"Service: {params.get('service')}, Problem: {params.get('problem')}")
    print(f"Query length: {len(cypher)} chars")
    print(f"UNION count: {cypher.count('UNION')}")
    print(f"Tier 0 count: {cypher.count('Tier 0')}")
    print(f"Tier 1 count: {cypher.count('Tier 1')}")
    print(f"Tier 2 count: {cypher.count('Tier 2')}")
    print(f"Tier 3 count: {cypher.count('Tier 3')}")
    
    rows = []
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # Primary query
            result = session.run(cypher, **params)
            rows = [dict(record) for record in result]
            
            # FALLBACK: If 0 results and fallback enabled, retry with relaxed query
            if len(rows) == 0 and enable_fallback:
                print("  [FALLBACK] Primary query returned 0 results, trying fallback query...")
                fallback_cypher = triage.build_fallback_cypher_query(slots)
                fallback_result = session.run(fallback_cypher, **params)
                rows = [dict(record) for record in fallback_result]
                print(f"  [FALLBACK] Retrieved {len(rows)} results from fallback query")
        
    except Exception as e:
        print(f"  Neo4j query failed: {e}")
    finally:
        if driver:
            driver.close()
    
    return rows


# Step 3 and 4 removed - policy facts pipeline not implemented yet


def step2_5_check_specific_errors(user_text: str, slots: CaseSlots) -> Optional[str]:
    """
    Step 2.5: Check for specific error patterns and return direct answers.
    This handles known specific errors that should bypass general retrieval.
    
    CRITICAL: This function must run BEFORE general Neo4j retrieval to prevent
    returning wrong answers for specific error messages.
    """
    import unicodedata
    import re
    
    def normalize(text):
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    narrative = normalize(user_text)
    
    # Pattern 1: Invalid card/account info error
    # Must match: "thông tin thẻ/tài khoản không hợp lệ" or similar variations
    # Must have BOTH parts: (1) card/account info mention + (2) invalid/error phrase
    has_card_account_info = any(pattern in narrative for pattern in [
        "thong tin the", "thong tin tai khoan", "the/tai khoan",
        "tai khoan ngan hang", "so the", "so tai khoan"
    ])
    
    has_invalid_error = any(pattern in narrative for pattern in [
        "khong hop le", "sai thong tin", "khong dung", "bi loi", "loi",
        "that bai", "khong thanh cong"
    ])
    
    # Only trigger if BOTH parts are present AND it's nap_tien service
    is_invalid_account_error = (
        has_card_account_info and 
        has_invalid_error and 
        slots.service == "nap_tien"
    )
    
    if is_invalid_account_error:
        # Return the specific answer for invalid account/card info
        return "Bạn vui lòng thực hiện hủy liên kết hiện tại và liên kết ngân hàng lại, sau đó thực hiện nạp tiền"
    
    return None


def step3_check_policy_intent(user_text: str, slots: CaseSlots) -> Optional[str]:
    """
    Step 3: Check if this is a policy question and retrieve policy data.
    Returns formatted policy answer or None.
    
    NOTE: Does NOT overwrite slots - respects original inference.
    """
    # Normalize text for better keyword matching
    import unicodedata
    import re
    from cases_handling import PolicyStore
    
    def normalize(text):
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    narrative = normalize(user_text)
    
    policy_keywords = [
        "han muc", "toi da", "toi thieu", "phi", "bieu phi", "mat phi",
        "dieu kien", "bao nhieu tien", "gioi han", "bao nhieu",
        "co mat phi", "mien phi", "ton bao nhieu", "chi phi"
    ]
    
    is_policy_question = any(kw in narrative for kw in policy_keywords)
    
    # CRITICAL: For "chuyen_tien" service, check if it's about fees/limits
    # Many policy questions about transfer don't have exact policy data,
    # so we should let Neo4j handle them when no policy exists
    if slots.service == "chuyen_tien" and is_policy_question:
        # Check if we have policy data for this bank
        if slots.bank_id:
            policy_store = PolicyStore(os.path.join(os.getcwd(), "external_data", "banks_policy.json"))
            try:
                # Try to get transfer-related policies
                policies = policy_store.get_policy_for_service(slots.bank_id, "chuyen_tien")
                if not policies:
                    # No policy data - let Neo4j handle it (skip policy check)
                    return None
            except Exception:
                return None
    
    # Services that don't need bank_id (internal VNPT services)
    services_without_bank = [
        "sieu_tich_luy", "mobile_money", "vnpt_pay", "ung_dung",
        "xac_thuc_dinh_danh", "hcc_dvc", "huy_dich_vu"
    ]
    
    # Skip policy check for services that don't need bank
    if slots.service and slots.service in services_without_bank:
        return None
    
    # If policy question but missing bank_id, ask user for bank
    if is_policy_question and not slots.bank_id and slots.service:
        return f"Để biết chính xác về phí/hạn mức cho dịch vụ {slots.service.replace('_', ' ')}, bạn vui lòng cho mình biết bạn đang dùng ngân hàng nào nhé? (Ví dụ: Vietcombank, Techcombank, BIDV...)"
    
    # Only proceed if policy question AND we have both bank_id + service
    if not is_policy_question or not slots.bank_id or not slots.service:
        return None
    
    # Get policy from PolicyStore
    policy_store = PolicyStore(os.path.join(os.getcwd(), "external_data", "banks_policy.json"))
    
    try:
        policies = policy_store.get_policy_for_service(slots.bank_id, slots.service)
        
        if not policies:
            return None
        
        # Format policy data
        service_code, policy = policies[0]
        
        response_parts = []
        response_parts.append(f"**Thông tin về {slots.service.replace('_', ' ')} từ {slots.bank}:**\n")
        
        if 'min_per_tx' in policy and policy['min_per_tx']:
            response_parts.append(f"- Số tiền tối thiểu: {policy['min_per_tx']:,} VNĐ")
        
        if 'max_per_tx' in policy and policy['max_per_tx']:
            response_parts.append(f"- Hạn mức mỗi giao dịch: {policy['max_per_tx']:,} VNĐ")
        
        if 'max_per_day' in policy and policy['max_per_day']:
            response_parts.append(f"- Hạn mức mỗi ngày: {policy['max_per_day']:,} VNĐ")
        
        if 'max_per_month' in policy and policy['max_per_month']:
            response_parts.append(f"- Hạn mức mỗi tháng: {policy['max_per_month']:,} VNĐ")
        
        if 'fee' in policy:
            response_parts.append(f"- Phí giao dịch: {policy['fee']}")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        print(f"  Policy retrieval failed: {e}")
        return None





def step4_llm_synthesis(user_text: str, slots: CaseSlots, top_candidates: List[Any]) -> Optional[str]:
    """
    Step 4: Use LLM to synthesize answer from top candidates when no exact match.
    This handles edge cases and multi-source reasoning.
    """
    if not top_candidates or len(top_candidates) == 0:
        return None
    
    # Only synthesize if we have reasonable candidates (tier >= 2)
    if all(c.tier < 2 for c in top_candidates):
        return None
    
    # Build context from top 3 candidates
    context_parts = []
    for i, cand in enumerate(top_candidates[:3], 1):
        context_parts.append(f"Nguồn {i} (tier={cand.tier}, score={cand.score:.2f}):")
        if cand.title:
            context_parts.append(f"Tiêu đề: {cand.title}")
        context_parts.append(f"Nội dung: {cand.content[:500]}...\n")
    
        context = "\n".join(context_parts)
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""Người dùng hỏi: "{user_text}"

Thông tin từ cơ sở dữ liệu:
{context}

Hãy trả lời câu hỏi của người dùng dựa trên thông tin trên. Nếu thông tin không đủ để trả lời, trả về "NO_ANSWER".
Nếu có thể trả lời, hãy tổng hợp và trả lời đầy đủ, chính xác bằng tiếng Việt."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là trợ lý VNPT Money. Trả lời chính xác, ngắn gọn."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content.strip()
        
        if answer and answer != "NO_ANSWER" and len(answer) > 20:
            return answer
        
    except Exception as e:
        print(f"  LLM synthesis error: {e}")
    
    return None


def step5_rank_and_gate(
    user_text: str,
    slots: CaseSlots,
    neo4j_rows: List[dict]
) -> GateDecision:
    """
    Step 5: Rank solutions and apply confidence gate.
    Returns decision to answer or clarify.
    """
    decision = rerank_solutions(
        user_text=user_text,
        slots=slots,
        neo4j_rows=neo4j_rows
    )
    return decision


def step6_assemble_response(
    user_text: str,
    slots: CaseSlots,
    decision: GateDecision
) -> str:
    """
    Step 6: Assemble final response using simple formatting.
    Enhanced with better clarification messages.
    """
    # If out of scope or service not supported
    if decision.decision == "out_of_scope":
        return (
            "Xin lỗi, dịch vụ bạn hỏi hiện chưa được hỗ trợ trên VNPT Money.\n\n"
            "VNPT Money hỗ trợ các dịch vụ sau:\n"
            "• Nạp tiền, rút tiền, chuyển tiền\n"
            "• Liên kết/hủy liên kết ngân hàng\n"
            "• Thanh toán hóa đơn (điện, nước, viễn thông)\n"
            "• Nạp điện thoại, mua mã thẻ\n"
            "• Đặt vé máy bay, vé tàu, vé tham quan\n"
            "• Siêu tích lũy\n\n"
            "Nếu bạn cần hỗ trợ về các dịch vụ khác, vui lòng liên hệ tổng đài 1800 1091."
        )
    
    # If clarification needed
    if decision.decision == "clarify":
        # Check if we have any candidates at all
        has_candidates = decision.topk and len(decision.topk) > 0
        
        if not has_candidates:
            # No data from Neo4j - don't ask for clarification, inform user
            return "Xin lỗi, mình chưa tìm thấy thông tin phù hợp trong cơ sở dữ liệu. Bạn có thể liên hệ tổng đài 1800 1091 của VNPT Money để được hỗ trợ trực tiếp."
        
        # Generate friendly clarification message
        questions = decision.clarifying_questions or []
        if questions:
            clarification_intro = "Mình cần thêm một số thông tin để hỗ trợ bạn tốt hơn:"
            questions_text = "\n".join(f"• {q}" for q in questions)
            return f"{clarification_intro}\n\n{questions_text}"
        else:
            # Fallback generic clarification
            return (
                "Mình chưa hiểu rõ câu hỏi của bạn. "
                "Bạn có thể mô tả chi tiết hơn về vấn đề bạn đang gặp phải không?\n\n"
                "Ví dụ:\n"
                "• Bạn muốn làm gì? (nạp tiền, rút tiền, chuyển tiền, ...)\n"
                "• Bạn gặp lỗi gì? (app báo lỗi cụ thể như thế nào?)\n"
                "• Ngân hàng nào? (nếu có)"
            )
    
    # Build response from best solution
    best = decision.best
    if not best:
        return "Xin lỗi, mình chưa tìm thấy câu trả lời phù hợp. Bạn có thể liên hệ tổng đài 1800 1091 để được hỗ trợ."
    
    # Format response based on available information
    response_parts = []
    
    if best.title:
        response_parts.append(f"**{best.title}**\n")
    
    if best.content:
        response_parts.append(best.content)
    
    response = "\n".join(response_parts) if response_parts else "Xin lỗi, mình chưa có thông tin chi tiết. Vui lòng liên hệ tổng đài 1800 1091."
    
    # Add disclaimer if confidence is borderline (score < 90)
    if best.score < 90:
        response += "\n\n💡 *Nếu câu trả lời này chưa đúng với vấn đề của bạn, vui lòng cung cấp thêm chi tiết để mình hỗ trợ tốt hơn.*"
    
    return response


# =============================================================================
# Main pipeline
# =============================================================================

def process_user_query(
    user_text: str, 
    history: Optional[List[Dict[str, Any]]] = None,
    user_info: Optional[Dict[str, Any]] = None,
    verbose: bool = True, 
    log_latency: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    Complete pipeline following readme.md flow with conversation history support.
    
    Flow:
    0. Enrich query from conversation history
    1. Extract slots (LLM with guardrails)
    2. Retrieve from graph (deterministic Cypher)
    3. Rank and gate
    4. Assemble response
    
    Args:
        user_text: User's query text
        history: List of previous exchanges [{"role": "user/assistant", "content": "..."}]
        user_info: Persistent user context (bank, service, problem)
        verbose: If True, print debug info. Set to False when running in Chainlit to avoid WebSocket issues.
        log_latency: If True, always log latency info to console/file even when verbose=False
    
    Returns:
        Tuple[str, Dict]: (response_text, updated_user_info)
    """
    latencies = {}  # Store latency for each step
    total_start = time.time()
    
    # Initialize history and user_info if not provided
    if history is None:
        history = []
    if user_info is None:
        user_info = {}
    
    if verbose:
        print("\n" + "="*60)
        print(" User Query:", user_text)
        print(" History Length:", len(history))
        print(" User Info:", user_info)
        print("="*60)
    
    # ===== STEP 0: CONTEXT ENRICHMENT FROM HISTORY =====
    start_time = time.time()
    enriched_query = user_text
    
    # Step 0.1: Simple pattern-based enrichment
    if len(history) > 0:
        enriched_query = enrich_query_with_history(user_text, history, user_info)
        
        # Step 0.2: LLM-based coreference resolution for complex cases
        if enriched_query != user_text or len(user_text.split()) <= 5:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                enriched_query = resolve_coreference_with_llm(
                    current_query=enriched_query,
                    history=history,
                    user_info=user_info,
                    api_key=api_key
                )
    
    latencies['step0_enrich_context'] = time.time() - start_time
    
    if verbose and enriched_query != user_text:
        print(f"\n Step 0: Context enrichment...")
        print(f"   Original: {user_text}")
        print(f"   Enriched: {enriched_query}")
        print(f"   Latency: {latencies['step0_enrich_context']:.3f}s")
    
    # ===== STEP 1: EXTRACT SLOTS (with enriched query) =====
    if verbose:
        print("\n Step 1: Extracting slots...")
    start_time = time.time()
    slots = step1_extract_slots(enriched_query)
    
    # Fill missing slots from user_info (persistent context)
    if not slots.bank and user_info.get("bank"):
        slots.bank = user_info["bank"]
        slots.bank_id = user_info.get("bank_id")
        if verbose:
            print(f"   [SLOT FILL] Bank from context: {slots.bank}")
    
    if not slots.service and user_info.get("service"):
        slots.service = user_info["service"]
        if verbose:
            print(f"   [SLOT FILL] Service from context: {slots.service}")
    
    if not slots.problem_type and user_info.get("problem"):
        slots.problem_type = user_info["problem"]
        if verbose:
            print(f"   [SLOT FILL] Problem from context: {slots.problem_type}")
    
    latencies['step1_extract_slots'] = time.time() - start_time
    
    # Update user_info with newly extracted slots
    if slots.bank:
        user_info["bank"] = slots.bank
        user_info["bank_id"] = slots.bank_id
    if slots.service:
        user_info["service"] = slots.service
    if slots.problem_type:
        user_info["problem"] = slots.problem_type
    
    if verbose:
        print(f"   Service: {slots.service}")
        print(f"   Problem: {slots.problem_type}")
        print(f"   Bank: {slots.bank} ({slots.bank_id})")
        print(f"   Latency: {latencies['step1_extract_slots']:.3f}s")
    
    # Step 2: Retrieve from graph
    if verbose:
        print("\n Step 2: Querying Neo4j...")
    start_time = time.time()
    neo4j_rows = step2_retrieve_from_graph(slots)
    latencies['step2_retrieve_from_graph'] = time.time() - start_time
    if verbose:
        print(f"   Found {len(neo4j_rows)} candidates")
        if len(neo4j_rows) > 0:
            print(f"   Top candidate: tier={neo4j_rows[0].get('tier', '?')}, score={neo4j_rows[0].get('score', '?')}")
        elif len(neo4j_rows) == 0:
            print("   [WARNING] Neo4j returned 0 results - possible query generation error")
        print(f"   Latency: {latencies['step2_retrieve_from_graph']:.3f}s")
    
    # Step 2.5: Check for specific error patterns
    if verbose:
        print("\n Step 2.5: Checking specific error patterns...")
    start_time = time.time()
    specific_error_answer = step2_5_check_specific_errors(user_text, slots)
    latencies['step2_5_check_specific_errors'] = time.time() - start_time
    if specific_error_answer:
        if verbose:
            print("   Specific error pattern matched - returning direct answer")
            print(f"   Latency: {latencies['step2_5_check_specific_errors']:.3f}s")
        return specific_error_answer
    else:
        if verbose:
            print("   No specific error pattern matched")
            print(f"   Latency: {latencies['step2_5_check_specific_errors']:.3f}s")
    
    # Step 3: Check for policy intent (no slot overwriting)
    if verbose:
        print("\n Step 3: Checking policy intent...")
    start_time = time.time()
    policy_answer = step3_check_policy_intent(user_text, slots)
    latencies['step3_check_policy_intent'] = time.time() - start_time
    if policy_answer:
        if verbose:
            print("   Policy answer found - skipping ranking")
            print(f"   Latency: {latencies['step3_check_policy_intent']:.3f}s")
        return policy_answer
    else:
        if verbose:
            print("   Not a policy question or missing bank/service")
            print(f"   Latency: {latencies['step3_check_policy_intent']:.3f}s")
    
    # Step 4: Rank and gate (use enriched query for better matching)
    if verbose:
        print("\n Step 4: Ranking and gating...")
    start_time = time.time()
    decision = step5_rank_and_gate(enriched_query, slots, neo4j_rows)
    latencies['step5_rank_and_gate'] = time.time() - start_time
    if verbose:
        print(f"   Decision: {decision.decision}")
        print(f"   Reason: {decision.reason}")
        if decision.best:
            print(f"   Best score: {decision.best.score:.2f} (tier={decision.best.tier})")
        print(f"   Latency: {latencies['step5_rank_and_gate']:.3f}s")
    
    # Step 4.5: Use best candidate even with medium confidence (disabled LLM synthesis to prevent hallucination)
    if decision.decision == "clarify" and decision.topk and len(decision.topk) > 0:
        if verbose:
            print("\n Step 4.5: Checking if we can use best candidate despite low confidence...")
        best_candidate = decision.topk[0]
        # Accept tier >= 2, OR tier=1 with high semantic score (>20)
        should_use = (
            (best_candidate.tier >= 2 and best_candidate.content) or
            (best_candidate.tier >= 1 and best_candidate.score > 20 and best_candidate.content)
        )
        if should_use:
            if verbose:
                print(f"   Using best candidate (tier={best_candidate.tier}, score={best_candidate.score:.2f}) instead of clarifying")
            decision.decision = "answer"
            decision.best = best_candidate
            decision.reason = f"(tier={best_candidate.tier}, score={best_candidate.score:.2f})"
        else:
            if verbose:
                print(f"   Best candidate not good enough (tier={best_candidate.tier}, score={best_candidate.score:.2f}), will ask for clarification")
    
    # Step 5: Assemble response (use enriched query)
    if verbose:
        print("\n Step 5: Assembling response...")
    start_time = time.time()
    response = step6_assemble_response(enriched_query, slots, decision)
    latencies['step6_assemble_response'] = time.time() - start_time
    total_latency = time.time() - total_start
    latencies['total'] = total_latency
    
    # Log latency info
    if log_latency:
        print(f"[LATENCY] Query: '{user_text[:50]}...' | Total: {total_latency:.3f}s | Steps: " + 
              " | ".join([f"{step}: {lat:.3f}s" for step, lat in latencies.items() if step != 'total']))
    
    if verbose:
        print("   Response generated")
        print(f"   Latency: {latencies['step6_assemble_response']:.3f}s")
        print(f"\n Total Latency: {total_latency:.3f}s")
        print(" Step Latencies:")
        for step, lat in latencies.items():
            if step != 'total':
                print(f"   {step}: {lat:.3f}s")
        print("\n" + "="*60)
    
    return response, user_info


# =============================================================================
# Chainlit handlers
# =============================================================================

@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session with conversation history and user context."""
    # Initialize conversation history in session
    cl.user_session.set("conversation_history", [])
    
    # Initialize persistent user context (bank, service, problem)
    cl.user_session.set("user_info", {
        "bank": None,
        "bank_id": None,
        "service": None,
        "problem": None,
    })
    
    welcome = (
        "Xin chào! Mình là VNPT Money chatbot. "
        "Mình có thể giúp bạn giải đáp các thắc mắc về dịch vụ tài chính số.\n\n"
        "Bạn cần hỗ trợ gì ạ?"
    )
    await cl.Message(content=welcome).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user message with conversation history."""
    user_text = message.content
    
    # Get conversation history and user info from session
    history = cl.user_session.get("conversation_history", [])
    user_info = cl.user_session.get("user_info", {})
    
    # Show processing indicator
    processing_msg = cl.Message(content="🤔 Đang xử lý...")
    await processing_msg.send()
    
    try:
        # Process query with history context (verbose=False to avoid WebSocket issues, but log_latency=True for monitoring)
        response, updated_user_info = process_user_query(
            user_text=user_text,
            history=history,
            user_info=user_info,
            verbose=False,
            log_latency=True
        )
        
        # Update conversation history
        history.append({
            "role": "user",
            "content": user_text,
            "timestamp": time.time()
        })
        history.append({
            "role": "assistant",
            "content": response,
            "timestamp": time.time()
        })
        
        # Keep only last 20 messages (10 exchanges) to avoid memory issues
        if len(history) > 20:
            history = history[-20:]
        
        # Save updated history and user_info back to session
        cl.user_session.set("conversation_history", history)
        cl.user_session.set("user_info", updated_user_info)
        
        # Update message with response
        processing_msg.content = response
        await processing_msg.update()
        
        # Add feedback actions
        actions = [
            cl.Action(name="helpful", value="yes", label="👍 Hữu ích", payload={"feedback": "positive"}),
            cl.Action(name="not_helpful", value="no", label="👎 Chưa hữu ích", payload={"feedback": "negative"}),
        ]
        processing_msg.actions = actions
        await processing_msg.update()
        
    except Exception as e:
        error_msg = (
            "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn. "
            "Vui lòng thử lại hoặc liên hệ tổng đài 1800 1091 của VNPT Money để được hỗ trợ."
        )
        processing_msg.content = error_msg
        await processing_msg.update()
        print(f"❌ Error: {e}")


@cl.action_callback("helpful")
async def on_helpful(action: cl.Action):
    """Handle positive feedback."""
    await cl.Message(content="Cảm ơn phản hồi của bạn! 🎉").send()


@cl.action_callback("not_helpful")
async def on_not_helpful(action: cl.Action):
    """Handle negative feedback."""
    await cl.Message(
        content="Xin lỗi câu trả lời chưa hữu ích. Bạn có muốn được chuyển đến tổng đài 1800 1091 không?"
    ).send()


if __name__ == "__main__":
    # Add src to path for standalone execution
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Test the pipeline
    test_queries = [
        "Tôi thực hiện nạp tiền thất bại nhưng ngân hàng đã trừ tiền",
        "Tôi muốn biết hạn mức rút tiền từ Vietcombank",
        "Hướng dẫn nạp tiền vào ví",
        "Giao dịch của tôi bị lỗi",
        "Tôi thanh toán hóa đơn bị trừ tiền và gạch nợ 2 lần",
    ]
    
    for query in test_queries:
        response = process_user_query(query)
        print("\n📤 Response:")
        print(response)
        print("\n")
