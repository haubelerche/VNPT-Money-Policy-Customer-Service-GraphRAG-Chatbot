"""
Test Intent Parser với các câu hỏi thực tế
==========================================

Verify rằng Intent Parser KHÔNG đánh câu hỏi hợp lệ thành out_of_domain
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from schema import ServiceEnum
from intent_parser import IntentParserLocal


def test_intent_parser_local():
    """Test IntentParserLocal với các câu hỏi thực tế"""
    print("=" * 60)
    print("TEST: IntentParserLocal - Keyword-based Intent Detection")
    print("=" * 60)
    
    parser = IntentParserLocal()
    
    # Test cases: (question, expected_service, expected_is_out_of_domain)
    test_cases = [
        # === Các câu hỏi về Data 3G/4G ===
        ("Gói data có tự động gia hạn không?", ServiceEnum.DATA_3G_4G, False),
        ("Mua data cho người khác được không?", ServiceEnum.DATA_3G_4G, False),
        ("Tại sao mua gói data báo lỗi?", ServiceEnum.DATA_3G_4G, False),
        ("Hướng dẫn mua gói cước Data 3G/4G", ServiceEnum.DATA_3G_4G, False),
        
        # === Các câu hỏi về nạp tiền ===
        ("Nạp nhầm số điện thoại có lấy lại được tiền không?", ServiceEnum.NAP_TIEN, False),
        ("Hướng dẫn nạp tiền điện thoại", ServiceEnum.NAP_TIEN, False),
        
        # === Các câu hỏi về tiền điện/nước ===
        ("Hướng dẫn thanh toán tiền điện", ServiceEnum.TIEN_DIEN, False),
        ("Tôi không nhớ mã khách hàng điện thì làm sao?", ServiceEnum.TIEN_DIEN, False),
        ("Hướng dẫn thanh toán tiền nước", ServiceEnum.TIEN_NUOC, False),
        
        # === Các câu hỏi về bảo hiểm ===
        ("Quy trình bồi thường bảo hiểm", ServiceEnum.BAO_HIEM, False),
        ("Tra cứu hợp đồng bảo hiểm đã mua", ServiceEnum.BAO_HIEM, False),
        
        # === Các câu hỏi về vay ===
        ("Hướng dẫn thanh toán khoản vay tiêu dùng", ServiceEnum.VAY, False),
        ("Thanh toán FE Credit", ServiceEnum.VAY, False),
        
        # === Các câu hỏi về học phí ===
        ("Hướng dẫn đóng học phí VnEdu", ServiceEnum.HOC_PHI, False),
        ("Không tìm thấy thông tin học sinh", ServiceEnum.HOC_PHI, False),
        
        # === Các câu hỏi về vé ===
        ("Hướng dẫn đặt vé tàu", ServiceEnum.MUA_VE, False),
        ("Đặt phòng khách sạn", ServiceEnum.MUA_VE, False),
        
        # === Các câu hỏi về dịch vụ công ===
        ("Hướng dẫn nộp phạt giao thông", ServiceEnum.DICH_VU_CONG, False),
        ("Đóng BHXH trên VNPT Money", ServiceEnum.DICH_VU_CONG, False),
        
        # === Các câu hỏi về giải trí ===
        ("Hướng dẫn mua Vietlott", ServiceEnum.GIAI_TRI, False),
        ("Thanh toán MyTV", ServiceEnum.GIAI_TRI, False),
        
        # === Các câu hỏi về tài khoản ===
        ("Hướng dẫn đăng ký tài khoản VNPT Money", ServiceEnum.DANG_KY, False),
        ("Định danh eKYC", ServiceEnum.DINH_DANH, False),
        ("Liên kết ngân hàng Vietcombank", ServiceEnum.LIEN_KET_NGAN_HANG, False),
        
        # === Câu hỏi NGOÀI PHẠM VI ===
        # (parser local luôn trả is_out_of_domain = False vì không có logic detect)
        # Việc detect out_of_domain chủ yếu do LLM parser thực hiện
    ]
    
    passed = 0
    failed = 0
    
    for question, expected_service, expected_out_of_domain in test_cases:
        result = parser.parse(question)
        
        service_match = result.service == expected_service
        # Local parser luôn trả is_out_of_domain = False
        out_of_domain_match = result.is_out_of_domain == expected_out_of_domain
        
        if service_match and out_of_domain_match:
            passed += 1
            print(f"  ✅ \"{question}\"")
            print(f"       → service={result.service.value}, out_of_domain={result.is_out_of_domain}")
        else:
            failed += 1
            print(f"  ❌ \"{question}\"")
            print(f"       Expected: service={expected_service.value}, out_of_domain={expected_out_of_domain}")
            print(f"       Got:      service={result.service.value}, out_of_domain={result.is_out_of_domain}")
    
    print(f"\n  Results: {passed}/{len(test_cases)} passed, {failed} failed")
    
    return failed == 0


def test_service_group_retrieval_simulation():
    """Simulate retrieval với SERVICE_GROUP_MAP để verify mapping"""
    print("\n" + "=" * 60)
    print("TEST: Service → Group Retrieval Simulation")
    print("=" * 60)
    
    from schema import SERVICE_GROUP_MAP
    
    # Test cases: (service, expected_groups_to_contain)
    test_cases = [
        ("data_3g_4g", ["dich_vu"]),  # Câu hỏi "Gói data có tự động gia hạn không?" → phải tìm trong dich_vu
        ("tien_dien", ["dich_vu"]),
        ("bao_hiem", ["dich_vu"]),
        ("hoc_phi", ["dich_vu"]),
        ("nap_tien", ["ho_tro_khach_hang"]),
        ("dang_ky", ["ho_tro_khach_hang"]),
        ("dieu_khoan", ["dieu_khoan"]),
        ("quyen_rieng_tu", ["quyen_rieng_tu"]),
    ]
    
    passed = 0
    failed = 0
    
    for service, expected_groups in test_cases:
        actual_groups = SERVICE_GROUP_MAP.get(service, [])
        
        all_found = all(g in actual_groups for g in expected_groups)
        
        if all_found:
            passed += 1
            print(f"  ✅ {service} → {actual_groups}")
        else:
            failed += 1
            print(f"  ❌ {service}: Expected {expected_groups} in {actual_groups}")
    
    print(f"\n  Results: {passed}/{len(test_cases)} passed, {failed} failed")
    
    return failed == 0


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print(" INTENT PARSER & RETRIEVAL TESTS")
    print("=" * 60)
    
    results = []
    
    results.append(("IntentParserLocal", test_intent_parser_local()))
    results.append(("Service→Group Retrieval", test_service_group_retrieval_simulation()))
    
    # Summary
    print("\n" + "=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n🚨 SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
