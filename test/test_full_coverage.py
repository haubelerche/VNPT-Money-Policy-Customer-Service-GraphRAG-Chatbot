import os
import sys
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import re

# Add src path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from openai import OpenAI
from schema import ServiceEnum

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def load_all_problems() -> List[Dict]:
    """Load all problems from CSV."""
    data_dir = PROJECT_ROOT / "db" / "import"
    filepath = data_dir / "nodes_problem.csv"
    
    if not filepath.exists():
        # Fallback to external_data_v3
        data_dir = PROJECT_ROOT / "external_data_v3"
        filepath = data_dir / "nodes_problem.csv"
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze_problem_distribution(problems: List[Dict]) -> Dict[str, List[Dict]]:
    """Analyze problems by service group."""
    groups = defaultdict(list)
    
    for p in problems:
        problem_id = p['id']
        group = problem_id.split('__')[0]
        groups[group].append(p)
    
    return groups


def extract_keywords_from_titles(problems: List[Dict]) -> Dict[str, Set[str]]:
    """Extract potential keywords from problem titles for each group."""
   
    
    groups = analyze_problem_distribution(problems)
    group_keywords = {}
    stopwords = {
        "và", "của", "là", "các", "được", "có", "khi", "trong", "cho", "như",
        "về", "nào", "với", "đối", "những", "theo", "này", "để", "ra", "quy",
        "định", "việc", "thì", "hay", "hoặc", "bị", "từ", "đến", "đã", "sẽ",
        "phải", "trường", "hợp", "một", "trên", "qua", "mà", "còn", "vậy",
        "the", "gì", "sao", "bao", "làm", "thế", "nếu", "ai", "đây", "đó"
    }
    
    for group, group_problems in groups.items():
        keywords = set()
        
        for p in group_problems:
            title = p['title'].lower()
            
            # Extract meaningful phrases
            # Split by common delimiters
            words = re.split(r'[,?!.\-/()]', title)
            
            for w in words:
                w = w.strip()
                if len(w) > 2 and w not in stopwords:
                    keywords.add(w)
        
        group_keywords[group] = keywords
    
    return group_keywords


def test_intent_parser_coverage():
    """Test IntentParserLocal coverage for all problems."""
    from intent_parser import IntentParserLocal
    
    parser = IntentParserLocal()
    problems = load_all_problems()
    groups = analyze_problem_distribution(problems)
    
    print("=" * 80)
    print("INTENT PARSER COVERAGE TEST")
    print("=" * 80)
    
    # Map group names to expected ServiceEnum
    GROUP_TO_SERVICE = {
        "quyen_rieng_tu": ServiceEnum.QUYEN_RIENG_TU,
        "ho_tro_khach_hang": None,  # Can map to multiple services
        "dieu_khoan": ServiceEnum.DIEU_KHOAN,
        "dich_vu": None,  # Maps to specific service based on topic
    }
    
    # Track results
    total = 0
    matched = 0
    failed_cases = []
    
    # Define acceptable alternative services for overlapping cases
    # Key: (group, expected_service), Value: list of acceptable alternatives
    ACCEPTABLE_ALTERNATIVES = {
        # QUYEN_RIENG_TU and DIEU_KHOAN overlap for "khieu nai", "boi thuong"
        ("quyen_rieng_tu", ServiceEnum.QUYEN_RIENG_TU): [ServiceEnum.DIEU_KHOAN],
        ("dieu_khoan", ServiceEnum.DIEU_KHOAN): [ServiceEnum.QUYEN_RIENG_TU],
        # DICH_VU overlaps - same question can apply to multiple services
        ("dich_vu", ServiceEnum.HOA_DON_VIEN_THONG): [ServiceEnum.DI_DONG_TRA_SAU, ServiceEnum.TIEN_DIEN, ServiceEnum.THANH_TOAN, ServiceEnum.KHAC],
        ("dich_vu", ServiceEnum.DI_DONG_TRA_SAU): [ServiceEnum.HOA_DON_VIEN_THONG, ServiceEnum.THANH_TOAN, ServiceEnum.TIEN_DIEN],
        ("dich_vu", ServiceEnum.TIEN_DIEN): [ServiceEnum.THANH_TOAN, ServiceEnum.HOA_DON_VIEN_THONG, ServiceEnum.KHAC],
        ("dich_vu", ServiceEnum.DIEN_NUOC_KHAC): [ServiceEnum.TIEN_DIEN, ServiceEnum.THANH_TOAN, ServiceEnum.KHAC],
        ("dich_vu", ServiceEnum.CHUYEN_TIEN): [ServiceEnum.THANH_TOAN],
        ("dich_vu", ServiceEnum.KHAC): [ServiceEnum.THANH_TOAN],  # VinaID can match thanh_toan
    }
    
    for group, group_problems in groups.items():
        print(f"\n--- GROUP: {group} ({len(group_problems)} problems) ---")
        
        for p in group_problems:
            problem_id = p['id']
            title = p['title']
            total += 1
            
            # Parse intent
            result = parser.parse(title)
            
            # Check if service matches expected
            expected_service = GROUP_TO_SERVICE.get(group)
            
            # For dich_vu group, extract specific service from ID
            if group == "dich_vu":
                # e.g., "dich_vu__nap_dien_thoai__prob_1" -> "nap_dien_thoai" -> NAP_TIEN
                topic = problem_id.split('__')[1] if '__' in problem_id else None
                expected_service = _infer_service_from_topic(topic)
            
            # For ho_tro_khach_hang, it's general support
            if group == "ho_tro_khach_hang":
                # These are general support - should NOT be out of domain
                if result.is_out_of_domain:
                    failed_cases.append({
                        "id": problem_id,
                        "title": title,
                        "reason": "Marked as out_of_domain",
                        "service_got": result.service.value
                    })
                    print(f"  [FAIL] {title[:60]}...")
                    print(f"     -> Out of domain!")
                else:
                    matched += 1
                    print(f"  [OK] {title[:60]}... -> {result.service.value}")
                continue
            
            # Check match - including acceptable alternatives
            if expected_service:
                # Get acceptable alternatives for this case
                alternatives = ACCEPTABLE_ALTERNATIVES.get((group, expected_service), [])
                acceptable_services = [expected_service] + alternatives
                
                if result.service in acceptable_services and not result.is_out_of_domain:
                    matched += 1
                    print(f"  [OK] {title[:50]}... -> {result.service.value}")
                else:
                    failed_cases.append({
                        "id": problem_id,
                        "title": title,
                        "expected": expected_service.value if expected_service else "N/A",
                        "got": result.service.value,
                        "is_out_of_domain": result.is_out_of_domain
                    })
                    print(f"  [FAIL] {title[:50]}...")
                    print(f"     Expected: {expected_service.value if expected_service else 'N/A'}, Got: {result.service.value}")
            else:
                # No expected service, just check not out of domain
                if not result.is_out_of_domain:
                    matched += 1
                    print(f"  [OK] {title[:50]}... -> {result.service.value}")
                else:
                    failed_cases.append({
                        "id": problem_id,
                        "title": title,
                        "reason": "Marked as out_of_domain"
                    })
                    print(f"  [FAIL] {title[:50]}... -> Out of domain!")
    
    # Summary
    print("\n" + "=" * 80)
    print("COVERAGE SUMMARY")
    print("=" * 80)
    print(f"Total problems: {total}")
    print(f"Matched: {matched}")
    print(f"Failed: {total - matched}")
    print(f"Coverage: {matched/total*100:.1f}%")
    
    if failed_cases:
        print("\n" + "=" * 80)
        print("FAILED CASES (need keyword expansion)")
        print("=" * 80)
        for case in failed_cases[:20]:  # Show first 20
            print(f"\nID: {case['id']}")
            print(f"Title: {case['title']}")
            if 'expected' in case:
                print(f"Expected: {case['expected']}, Got: {case['got']}")
            if 'reason' in case:
                print(f"Reason: {case['reason']}")
    
    return matched, total, failed_cases


def _infer_service_from_topic(topic: str) -> ServiceEnum:
    """Infer ServiceEnum from topic name."""
    if not topic:
        return ServiceEnum.KHAC
    
    topic_lower = topic.lower()
    
    # Mapping topic prefixes to services
    # NOTE: Some services share similar keywords, so we accept multiple valid matches
    TOPIC_MAPPINGS = {
        "nap_dien_thoai": ServiceEnum.NAP_TIEN,
        "nap_tien": ServiceEnum.NAP_TIEN,
        "hoa_don_vien_thong": ServiceEnum.HOA_DON_VIEN_THONG,
        "data": ServiceEnum.DATA_3G_4G,
        "mua_ma_the": ServiceEnum.MUA_THE,
        "mua_the": ServiceEnum.MUA_THE,
        "goi_cuoc_mytv": ServiceEnum.GIAI_TRI,
        "goi_cuoc_digilife": ServiceEnum.GIAI_TRI,
        "don_hang_vnpt": ServiceEnum.THANH_TOAN,
        "di_dong_tra_sau": ServiceEnum.DI_DONG_TRA_SAU,
        "dien_thoai_co_dinh": ServiceEnum.HOA_DON_VIEN_THONG,
        "thanh_toan_tra_truoc": ServiceEnum.THANH_TOAN,
        "nap_tien_tu_dong": ServiceEnum.NAP_TIEN,
        "tien_dien": ServiceEnum.TIEN_DIEN,
        "tien_nuoc": ServiceEnum.TIEN_NUOC,
        "ve_sinh_moi_truong": ServiceEnum.DIEN_NUOC_KHAC,
        "phi_chung_cu": ServiceEnum.DIEN_NUOC_KHAC,
        "dien_nuoc_khac": ServiceEnum.DIEN_NUOC_KHAC,
        "vien_thong": ServiceEnum.HOA_DON_VIEN_THONG,
        "internet": ServiceEnum.HOA_DON_VIEN_THONG,
        "hoc_phi": ServiceEnum.HOC_PHI,
        "thu_phi_dh_cd": ServiceEnum.HOC_PHI,
        "nop_phi_xet_tuyen": ServiceEnum.HOC_PHI,
        "hoc_mai": ServiceEnum.HOC_PHI,
        "truyen_hinh": ServiceEnum.GIAI_TRI,
        "vtvcab": ServiceEnum.GIAI_TRI,
        "mytv": ServiceEnum.GIAI_TRI,
        "sieu_tich_luy": ServiceEnum.TIET_KIEM,
        "bao_hiem": ServiceEnum.BAO_HIEM,
        "vay": ServiceEnum.VAY,
        "mirae_asset": ServiceEnum.VAY,
        "msb_credit": ServiceEnum.VAY,
        "fe_credit": ServiceEnum.VAY,
        "aeon_finance": ServiceEnum.VAY,
        "tiet_kiem": ServiceEnum.TIET_KIEM,
        "hoa_don_tai_chinh": ServiceEnum.BAO_HIEM,
        "manulife": ServiceEnum.BAO_HIEM,
        "cong_dich_vu_cong": ServiceEnum.DICH_VU_CONG,
        "dong_bhyt_bhxh": ServiceEnum.DICH_VU_CONG,
        "nop_thue": ServiceEnum.DICH_VU_CONG,
        "nop_phat_giao_thong": ServiceEnum.DICH_VU_CONG,
        "thanh_toan_tu_dong": ServiceEnum.THANH_TOAN,
        "nhan_tien_kieu_hoi": ServiceEnum.CHUYEN_TIEN,
        "diem_thanh_toan": ServiceEnum.THANH_TOAN,
        "mua_ve": ServiceEnum.MUA_VE,
        "ve_tau": ServiceEnum.MUA_VE,
        "ve_xe": ServiceEnum.MUA_VE,
        "ve_may_bay": ServiceEnum.MUA_VE,
        "dat_phong": ServiceEnum.MUA_VE,
        "khach_san": ServiceEnum.MUA_VE,
        "thu_phi_ben_xe": ServiceEnum.MUA_VE,
        "dat_xe_taxi": ServiceEnum.MUA_VE,
        "pvoil": ServiceEnum.THANH_TOAN,
        "phuong_tien": ServiceEnum.MUA_VE,
        "vietlott": ServiceEnum.GIAI_TRI,
        "ve_su_kien": ServiceEnum.MUA_VE,
        "dealtoday": ServiceEnum.GIAI_TRI,
        "ve_vui_choi": ServiceEnum.MUA_VE, 
        "lazada": ServiceEnum.THANH_TOAN,
        "sendo": ServiceEnum.THANH_TOAN,
        "vong_quay": ServiceEnum.GIAI_TRI,
        "vnpt_money_du_hi": ServiceEnum.GIAI_TRI,
        "nap_the_vinaid": ServiceEnum.KHAC, 
    }
    
    for prefix, service in TOPIC_MAPPINGS.items():
        if prefix in topic_lower:
            return service
    
    return ServiceEnum.KHAC


def test_retrieval_for_all_problems():
    """Test vector retrieval for all problems."""
    
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # Use Neo4j driver and OpenAI directly
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    openai_client = OpenAI(api_key=openai_key)
    
    problems = load_all_problems()
    
    print("\n" + "=" * 80)
    print("RETRIEVAL TEST")
    print("=" * 80)
    
    total = 0
    matched = 0
    failed = []
    
    # Test a sample of problems (first problem from each topic)
    tested_topics = set()
    
    for p in problems:
        problem_id = p['id']
        title = p['title']
        
        # Extract topic
        parts = problem_id.split('__')
        if len(parts) >= 2:
            topic = '__'.join(parts[:2])
        else:
            topic = problem_id
        
        # Skip if already tested this topic
        if topic in tested_topics:
            continue
        tested_topics.add(topic)
        
        total += 1
        
        # Test retrieval
        try:
            # Get embedding
            response = openai_client.embeddings.create(model="text-embedding-3-small", input=[title])
            query_emb = response.data[0].embedding
            
            # Vector search
            with driver.session() as session:
                result = session.run("""
                    CALL db.index.vector.queryNodes('problem_embedding_index', 3, $embedding)
                    YIELD node, score
                    RETURN node.id as id, node.title as title, score
                """, {"embedding": query_emb})
                results = list(result)
            
            if results:
                # Check if exact match or high similarity
                best_result = results[0]
                best_id = best_result['id']
                best_score = best_result['score']
                
                # Check if it matches the expected problem
                if problem_id in best_id or best_score >= 0.85:
                    matched += 1
                    print(f"[OK] [{topic}] Score: {best_score:.2f}")
                else:
                    failed.append({
                        "id": problem_id,
                        "title": title,
                        "best_match": best_id,
                        "score": best_score
                    })
                    print(f"[FAIL] [{topic}] Expected: {problem_id}")
                    print(f"   Got: {best_id} (score: {best_score:.2f})")
            else:
                failed.append({
                    "id": problem_id,
                    "title": title,
                    "reason": "No results"
                })
                print(f"[FAIL] [{topic}] No results!")
                
        except Exception as e:
            failed.append({
                "id": problem_id,
                "title": title,
                "error": str(e)
            })
            print(f"[ERROR] [{topic}] Error: {e}")
    
    driver.close()
    
    print("\n" + "=" * 80)
    print("RETRIEVAL SUMMARY")
    print("=" * 80)
    print(f"Topics tested: {total}")
    print(f"Matched: {matched}")
    print(f"Failed: {total - matched}")
    print(f"Success rate: {matched/total*100:.1f}%")
    
    return matched, total, failed


def generate_keyword_recommendations():
    """Generate keyword recommendations based on problem analysis."""
    problems = load_all_problems()
    group_keywords = extract_keywords_from_titles(problems)
    
    print("\n" + "=" * 80)
    print("KEYWORD RECOMMENDATIONS")
    print("=" * 80)
    
    for group, keywords in group_keywords.items():
        print(f"\n--- {group.upper()} ---")
        print(f"Sample keywords: {', '.join(list(keywords)[:15])}")


def main():
    """Run all coverage tests."""
    print("[TEST] Starting Full Coverage Test...\n")
    
    # Test 1: Intent Parser Coverage
    intent_matched, intent_total, intent_failed = test_intent_parser_coverage()
    
    # Test 2: Retrieval Coverage (optional - requires Neo4j)
    try:
        ret_matched, ret_total, ret_failed = test_retrieval_for_all_problems()
    except Exception as e:
        print(f"\n[SKIP] Retrieval test skipped: {e}")
        ret_matched, ret_total, ret_failed = 0, 0, []
    
    # Generate recommendations
    generate_keyword_recommendations()
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Intent Parser Coverage: {intent_matched}/{intent_total} ({intent_matched/intent_total*100:.1f}%)")
    if ret_total > 0:
        print(f"Retrieval Coverage: {ret_matched}/{ret_total} ({ret_matched/ret_total*100:.1f}%)")
    
    if intent_matched < intent_total:
        print("\n[WARNING] Some problems need additional keywords in IntentParserLocal!")
        print("Check the failed cases above to expand keywords.")


if __name__ == "__main__":
    main()
