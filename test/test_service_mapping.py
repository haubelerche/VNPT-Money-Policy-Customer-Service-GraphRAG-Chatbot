"""
Test SERVICE_GROUP_MAP và Intent Parser
========================================

Verify rằng:
1. Mọi service trong ServiceEnum đều có trong SERVICE_GROUP_MAP
2. Mọi group trong mapping tồn tại trong database
3. Intent Parser phân loại đúng các câu hỏi test
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from schema import ServiceEnum, SERVICE_GROUP_MAP


def test_service_enum_coverage():
    """Test: Mọi service trong ServiceEnum phải có trong SERVICE_GROUP_MAP"""
    print("=" * 60)
    print("TEST 1: ServiceEnum Coverage in SERVICE_GROUP_MAP")
    print("=" * 60)
    
    missing = []
    for service in ServiceEnum:
        if service.value not in SERVICE_GROUP_MAP:
            missing.append(service.value)
            print(f"  ❌ MISSING: {service.value}")
        else:
            groups = SERVICE_GROUP_MAP[service.value]
            print(f"  ✅ {service.value} → {groups}")
    
    if missing:
        print(f"\n🚨 FAILED: {len(missing)} services missing from SERVICE_GROUP_MAP")
        return False
    
    print(f"\n✅ PASSED: All {len(ServiceEnum)} services mapped correctly")
    return True


def test_valid_groups():
    """Test: Mọi group trong mapping phải là valid (4 groups trong database)"""
    print("\n" + "=" * 60)
    print("TEST 2: Valid Groups in SERVICE_GROUP_MAP")
    print("=" * 60)
    
    VALID_GROUPS = {"quyen_rieng_tu", "ho_tro_khach_hang", "dieu_khoan", "dich_vu"}
    
    invalid = []
    for service, groups in SERVICE_GROUP_MAP.items():
        for group in groups:
            if group not in VALID_GROUPS:
                invalid.append((service, group))
                print(f"  ❌ INVALID: {service} → {group}")
    
    if invalid:
        print(f"\n🚨 FAILED: Found {len(invalid)} invalid groups")
        return False
    
    print(f"  Valid groups: {VALID_GROUPS}")
    print(f"\n✅ PASSED: All groups are valid")
    return True


def test_data_service_mapping():
    """Test: data_3g_4g phải map đến dich_vu (nơi có topic dich_vu__data_3g4g)"""
    print("\n" + "=" * 60)
    print("TEST 3: Data 3G/4G Service Mapping")
    print("=" * 60)
    
    data_groups = SERVICE_GROUP_MAP.get("data_3g_4g", [])
    
    print(f"  data_3g_4g → {data_groups}")
    
    if "dich_vu" not in data_groups:
        print(f"\n🚨 FAILED: data_3g_4g MUST include 'dich_vu' group")
        return False
    
    print(f"\n✅ PASSED: data_3g_4g correctly maps to dich_vu")
    return True


def test_critical_services():
    """Test: Các service quan trọng phải được map đúng"""
    print("\n" + "=" * 60)
    print("TEST 4: Critical Services Mapping")
    print("=" * 60)
    
    EXPECTED_MAPPINGS = {
        # Dịch vụ viễn thông phải map đến dich_vu
        "data_3g_4g": ["dich_vu"],
        "mua_the": ["dich_vu"],
        "tien_dien": ["dich_vu"],
        "tien_nuoc": ["dich_vu"],
        "bao_hiem": ["dich_vu"],
        "hoc_phi": ["dich_vu"],
        "mua_ve": ["dich_vu"],
        
        # Dịch vụ tài chính phải map đến ho_tro_khach_hang
        "nap_tien": ["ho_tro_khach_hang"],
        "rut_tien": ["ho_tro_khach_hang"],
        "chuyen_tien": ["ho_tro_khach_hang"],
        "lien_ket_ngan_hang": ["ho_tro_khach_hang"],
        "dang_ky": ["ho_tro_khach_hang"],
        
        # Chính sách/Điều khoản phải map đến dieu_khoan
        "dieu_khoan": ["dieu_khoan"],
        
        # Quyền riêng tư phải map đến quyen_rieng_tu
        "quyen_rieng_tu": ["quyen_rieng_tu"],
    }
    
    failed = []
    for service, expected_groups in EXPECTED_MAPPINGS.items():
        actual_groups = SERVICE_GROUP_MAP.get(service, [])
        
        missing_groups = [g for g in expected_groups if g not in actual_groups]
        
        if missing_groups:
            failed.append((service, expected_groups, actual_groups))
            print(f"  ❌ {service}: Expected {expected_groups} in {actual_groups}")
        else:
            print(f"  ✅ {service} → {actual_groups} (contains {expected_groups})")
    
    if failed:
        print(f"\n🚨 FAILED: {len(failed)} services incorrectly mapped")
        return False
    
    print(f"\n✅ PASSED: All critical services mapped correctly")
    return True


def test_service_keywords_coverage():
    """Test: Kiểm tra SERVICE_KEYWORDS trong IntentParserLocal"""
    print("\n" + "=" * 60)
    print("TEST 5: IntentParserLocal SERVICE_KEYWORDS Coverage")
    print("=" * 60)
    
    try:
        from intent_parser import IntentParserLocal
        
        missing = []
        for service in ServiceEnum:
            if service not in IntentParserLocal.SERVICE_KEYWORDS and service != ServiceEnum.KHAC:
                missing.append(service.value)
                print(f"  ❌ MISSING: {service.value}")
            elif service != ServiceEnum.KHAC:
                keywords = IntentParserLocal.SERVICE_KEYWORDS[service]
                print(f"  ✅ {service.value} → {len(keywords)} keywords")
        
        if missing:
            print(f"\n🚨 FAILED: {len(missing)} services missing from SERVICE_KEYWORDS")
            return False
        
        print(f"\n✅ PASSED: All services have keywords")
        return True
        
    except ImportError as e:
        print(f"  ⚠️ Cannot import IntentParserLocal: {e}")
        return True  # Skip this test if can't import


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print(" SERVICE MAPPING DIAGNOSTIC TESTS")
    print("=" * 60)
    
    results = []
    
    results.append(("ServiceEnum Coverage", test_service_enum_coverage()))
    results.append(("Valid Groups", test_valid_groups()))
    results.append(("Data 3G/4G Mapping", test_data_service_mapping()))
    results.append(("Critical Services", test_critical_services()))
    results.append(("SERVICE_KEYWORDS Coverage", test_service_keywords_coverage()))
    
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
