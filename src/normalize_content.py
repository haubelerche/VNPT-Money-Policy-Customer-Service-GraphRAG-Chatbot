from __future__ import annotations
import json
import os
import red
from typing import Dict, List, Optional, Tuple


_WS_RE = re.compile(r"\s+")

def _strip_accents(s: str) -> str:
    """Remove Vietnamese diacritics and accents."""
    # Vietnamese character mapping
    vietnamese_map = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'đ': 'd',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    }
    
    result = []
    for ch in s:
        result.append(vietnamese_map.get(ch, ch))
    
    return ''.join(result)


def _norm_text(s: str) -> str:

    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _strip_accents(s)
    s = _WS_RE.sub(" ", s)
    return s


class ServiceTaxonomy:
    
    TIER_A_SERVICES = {
        "nap_tien", "rut_tien", "chuyen_tien", "lien_ket_ngan_hang",
        "huy_lien_ket_ngan_hang", "nap_tien_dien_thoai", "vien_thong",
        "thanh_toan_hoa_don", "thanh_toan_dich_vu", "thanh_toan_khoan_vay",
        "hoa_don_vnpt", "ve_may_bay", "ve_tau", "ve_tham_quan", "sieu_tich_luy",
        "ctkm_voucher", "hcc_dvc"
    }
    
    TIER_B_SERVICES = {
        "tai_khoan_vi", "mobile_money", "ung_dung", "vnpt_pay",
        "thong_tin_chung", "xac_thuc_dinh_danh"
    }
    
    # All valid service slot_keys (for validation)
    VALID_SERVICES = TIER_A_SERVICES | TIER_B_SERVICES
    
    SERVICES: Dict[str, Dict] = {
        "TOPUP_WALLET": {
            "display": "Nạp tiền (ví)",
            "slot_key": "nap_tien",
            "graph_label": "Service",
            "domain": "wallet_core",
            "priority": 1,
            "tier": "A",
            "anchors": ["nạp ví", "nạp vào ví", "nạp tiền vào ví", "nạp từ ngân hàng", "nạp tiền từ ngân hàng"],
            "keywords": ["nạp tiền", "top up", "topup", "nạp"],
            "aliases": ["nạp tiền", "nap tien", "top up", "topup", "nạp ví", "nap vi", "nạp vào ví", "nap vao vi"],
            "negative_keywords": ["tự động", "nạp tự động", "đơn hàng", "thuê bao", "điện thoại", "mã thẻ", "thẻ cào", "gói cước", "data", "voucher", "khuyến mại"],
            "excludes": ["VNPT_ORDER", "PHONE_TOPUP", "TELECOM_SERVICE"]
        },
        "WITHDRAW_WALLET": {
            "display": "Rút tiền",
            "slot_key": "rut_tien",
            "graph_label": "Service",
            "domain": "wallet_core",
            "priority": 1,
            "tier": "A",
            "anchors": ["rút tiền", "rút về ngân hàng", "rút về tk", "rút về tài khoản"],
            "keywords": ["rút tiền", "withdraw", "rút"],
            "aliases": ["rút", "rút tiền", "withdraw", "rut", "rut tien", "rút về ngân hàng", "rút ví", "rut vi"],
            "negative_keywords": ["chuyển tiền", "chuyển khoản", "nạp tiền"]
        },
        "TRANSFER_INTERNAL": {
            "display": "Chuyển tiền",
            "slot_key": "chuyen_tien",
            "graph_label": "Service",
            "domain": "wallet_core",
            "priority": 1,
            "tier": "A",
            "anchors": ["chuyển tiền", "chuyển khoản", "gửi tiền", "chuyển cho", "chuyển tới"],
            "keywords": ["chuyển tiền", "chuyển khoản", "gửi tiền", "transfer"],
            "aliases": ["chuyển", "chuyển tiền", "chuyển khoản", "transfer", "chuyen tien", "chuyen khoan", "gửi tiền", "gui tien", "chuyển cho bạn"],
            "negative_keywords": ["rút tiền", "nạp tiền", "thuê bao", "mã thẻ"]
        },
        "BANK_LINK": {
            "display": "Liên kết ngân hàng",
            "slot_key": "lien_ket_ngan_hang",
            "graph_label": "Service",
            "domain": "wallet_core",
            "priority": 1,
            "tier": "A",
            "anchors": ["liên kết ngân hàng", "kết nối ngân hàng", "lknh", "lk nh", "hủy liên kết", "bỏ liên kết", "xóa liên kết", "gỡ liên kết"],
            "keywords": ["liên kết ngân hàng", "kết nối ngân hàng", "thêm ngân hàng", "link bank", "hủy liên kết", "bỏ liên kết", "xóa liên kết", "unlink"],
            "aliases": ["liên kết ngân hàng", "lien ket ngan hang", "liên kết", "lien ket", "kết nối", "ket noi", "lk nh", "lknh", "link bank", "liên kết bank", "them ngan hang", "hủy liên kết", "huy lien ket", "bỏ liên kết", "bo lien ket", "xóa liên kết", "xoa lien ket", "gỡ liên kết", "go lien ket", "unlink", "hủy kết nối", "bỏ kết nối"],
            "negative_keywords": []
        },
        "FLIGHT_TICKET": {
            "display": "Vé máy bay",
            "slot_key": "ve_may_bay",
            "graph_label": "Service",
            "domain": "transportation",
            "priority": 2,
            "anchors": ["vé máy bay", "ve may bay", "chuyến bay", "hãng bay", "đặt chỗ", "booking"],
            "keywords": ["vé máy bay", "máy bay", "chuyến bay", "hãng bay", "đổi vé", "hủy vé", "đặt chỗ", "booking"],
            "aliases": ["vé máy bay", "ve may bay", "máy bay", "chuyến bay", "vé mb", "đặt vé máy bay", "mua vé máy bay", "đặt chỗ máy bay", "kiểm tra đặt chỗ", "booking"]
        },
        "TRAIN_TICKET": {
            "display": "Vé tàu",
            "slot_key": "ve_tau",
            "graph_label": "Service",
            "domain": "transportation",
            "priority": 2,
            "anchors": ["vé tàu", "ve tau", "tàu hỏa", "tau hoa"],
            "keywords": ["vé tàu", "tàu hỏa", "đổi vé tàu", "hủy vé tàu", "mua vé tàu"],
            "aliases": ["vé tàu", "ve tau", "tàu hỏa", "vé tàu hỏa", "đặt vé tàu", "dat ve tau", "tau hoa"]
        },
        "TICKET": {
            "display": "Vé tham quan",
            "slot_key": "ve_tham_quan",
            "graph_label": "Service",
            "domain": "entertainment",
            "priority": 3,
            "tier": "A",
            "anchors": ["vé tham quan", "ve tham quan", "vé vào cổng", "ve vao cong", "tham quan", "khu du lịch", "khu du lich"],
            "keywords": ["vé tham quan", "vào cổng", "khu du lịch"],
            "aliases": ["vé tham quan", "ve tham quan", "vé vào cổng", "ve vao cong", "tham quan", "khu du lịch", "khu du lich"]
        },
        "BILL_PAYMENT": {
            "display": "Thanh toán hóa đơn",
            "slot_key": "thanh_toan_hoa_don",
            "graph_label": "Service",
            "domain": "bill_payment",
            "priority": 2,
            "parent": "SERVICE_PAYMENT",
            "anchors": ["tiền điện", "tiền nước", "hóa đơn", "hoa don", "hóa đơn cước", "cước trả sau", "cước trả trước"],
            "keywords": ["thanh toán hóa đơn", "hóa đơn", "tiền điện", "tiền nước", "cước internet", "hóa đơn cước", "cước trả trước", "cước trả sau"],
            "aliases": ["thanh toán hóa đơn", "thanh toan hoa don", "hóa đơn", "hoa don", "bill", "tiền điện", "tiền nước", "cước internet", "hóa đơn cước", "cước trả trước", "cước trả sau", "hóa đơn vnpt"],
            "negative_keywords": ["khoản vay", "trả nợ", "thanh toán khoản vay"],
            "excludes": ["LOAN_PAYMENT"]
        },
        "LOAN_PAYMENT": {
            "display": "Thanh toán khoản vay",
            "slot_key": "thanh_toan_khoan_vay",
            "graph_label": "Service",
            "domain": "bill_payment",
            "priority": 2,
            "parent": "SERVICE_PAYMENT",
            "anchors": ["khoản vay", "thanh toán khoản vay", "thanh toan khoan vay", "trả nợ", "tra no"],
            "keywords": ["khoản vay", "trả nợ", "thanh toán khoản vay"],
            "aliases": ["thanh toán khoản vay", "thanh toan khoan vay", "trả nợ", "tra no", "khoản vay"]
        },
        "SERVICE_PAYMENT": {
            "display": "Thanh toán dịch vụ",
            "slot_key": "thanh_toan_dich_vu",
            "graph_label": "Service",
            "domain": "bill_payment",
            "priority": 10,
            "anchors": ["thanh toán dịch vụ", "thanh toan dich vu"],
            "keywords": ["thanh toán dịch vụ"],
            "aliases": ["thanh toán dịch vụ", "thanh toan dich vu"],
            "negative_keywords": ["hóa đơn", "hoa don", "tiền điện", "tiền nước", "mytv", "truyền hình", "tv", "khoản vay", "trả nợ"]
        },
        "GOVERNMENT_SERVICE": {
            "display": "Dịch vụ công/Hành chính công",
            "slot_key": "hcc_dvc",
            "graph_label": "Service",
            "domain": "government",
            "priority": 3,
            "anchors": ["dịch vụ công", "dich vu cong", "hành chính công", "hanh chinh cong", "dvcqg", "bhyt", "bhxh", "bảo hiểm", "bao hiem", "thuế", "thue", "phí", "phi"],
            "keywords": ["dịch vụ công", "hành chính công", "dvcqg", "thuế", "phí", "bảo hiểm", "bhyt", "bhxh", "gia hạn"],
            "aliases": ["dịch vụ công", "dich vu cong", "hành chính công", "hanh chinh cong", "hcc_dvc", "dvcqg", "thuế", "thue", "phí", "phi", "bảo hiểm", "bao hiem", "bhyt", "bhxh", "bảo hiểm y tế", "bảo hiểm xã hội", "gia hạn bảo hiểm"],
            "negative_keywords": ["tích lũy", "tich luy", "siêu tích lũy", "sieu tich luy"]
        },
        "TELECOM_SERVICE": {
            "display": "Dịch vụ viễn thông",
            "slot_key": "vien_thong",
            "graph_label": "Service",
            "domain": "telecom",
            "priority": 3,
            "anchors": ["mã thẻ", "ma the", "thẻ cào", "the cao", "gói cước", "goi cuoc", "gói data", "goi data"],
            "keywords": ["mua mã thẻ", "mã thẻ cào", "thẻ cào", "gói cước", "data 3g", "data 4g", "gói data", "nạp data"],
            "aliases": ["viễn thông", "vien thong", "mã thẻ", "ma the", "thẻ cào", "the cao", "gói cước", "goi cuoc", "data", "3g", "4g", "gói data", "goi data"],
            "negative_keywords": ["thuê bao", "nạp điện thoại", "nap dien thoai", "nạp tiền điện thoại"],
            "excludes": ["PHONE_TOPUP"]
        },
        "PHONE_TOPUP": {
            "display": "Nạp tiền điện thoại",
            "slot_key": "nap_tien_dien_thoai",
            "graph_label": "Service",
            "domain": "telecom",
            "priority": 2,
            "anchors": ["nạp điện thoại", "nap dien thoai", "nạp tiền điện thoại", "nap tien dien thoai", "nạp thuê bao", "nap thue bao", "vào thuê bao", "vao thue bao"],
            "keywords": ["nạp điện thoại", "nạp tiền điện thoại", "nạp thuê bao", "topup điện thoại", "tiền vào thuê bao"],
            "aliases": ["nạp tiền điện thoại", "nap tien dien thoai", "nạp đt", "nap dt", "nạp thuê bao", "nap thue bao", "topup điện thoại", "topup dien thoai"],
            "negative_keywords": ["mã thẻ", "ma the", "thẻ cào", "the cao", "gói cước", "goi cuoc"],
            "excludes": ["TELECOM_SERVICE", "TOPUP_WALLET"]
        },
        "INVESTMENT": {
            "display": "Siêu tích lũy",
            "slot_key": "sieu_tich_luy",
            "graph_label": "Service",
            "domain": "investment",
            "priority": 3,
            "anchors": ["siêu tích lũy", "sieu tich luy", "tích lũy tự động", "tich luy tu dong"],
            "keywords": ["siêu tích lũy", "tích lũy", "tiết kiệm", "gửi tiết kiệm", "chu kỳ"],
            "aliases": ["siêu tích lũy", "sieu tich luy", "tích lũy", "tich luy", "đầu tư", "dau tu", "tích lũy tự động", "tich luy tu dong"]
        },
        "PROMO_VOUCHER": {
            "display": "Khuyến mại/Voucher",
            "slot_key": "ctkm_voucher",
            "graph_label": "Service",
            "domain": "promotion",
            "priority": 3,
            "anchors": ["voucher", "khuyến mại", "khuyen mai", "ưu đãi", "uu dai", "hoàn tiền", "hoan tien", "cashback"],
            "keywords": ["voucher", "khuyến mại", "ưu đãi", "mã giảm giá", "hoàn tiền", "cashback"],
            "aliases": ["voucher", "khuyến mại", "khuyen mai", "ưu đãi", "uu dai", "giảm giá", "giam gia", "quà tặng", "qua tang", "ctkm", "ctkm_voucher", "chương trình khuyến mại", "chuong trinh khuyen mai", "hoàn tiền", "hoan tien", "cashback"]
        },
        "APP_USAGE": {
            "display": "Sử dụng ứng dụng",
            "slot_key": "ung_dung",
            "graph_label": "Service",
            "domain": "app",
            "priority": 5,
            "tier": "B",
            "anchors": ["lỗi app", "loi app", "không vào được", "khong vao duoc", "đăng nhập", "dang nhap", "đăng ký", "dang ky"],
            "keywords": ["đăng nhập", "đăng ký", "lỗi app", "không vào được", "không đăng nhập được"],
            "aliases": ["ứng dụng", "ung dung", "app", "đăng nhập", "dang nhap", "đăng ký", "dang ky", "lỗi app", "loi app", "không vào được", "khong vao duoc"],
            "negative_keywords": ["mobile money", "vnpt pay", "mytv", "sinh trắc", "face id", "vân tay", "định danh", "ekyc"]
        },
        "WALLET_ACCOUNT": {
            "display": "Tài khoản ví",
            "slot_key": "tai_khoan_vi",
            "graph_label": "Service",
            "domain": "wallet_core",
            "priority": 3,
            "tier": "B",
            "anchors": ["hủy ví", "huy vi", "khóa tài khoản", "khoa tai khoan", "mở khóa", "mo khoa", "đổi mật khẩu", "doi mat khau", "quên mật khẩu", "quen mat khau", "lấy lại mật khẩu", "lay lai mat khau", "số dư", "so du"],
            "keywords": ["tài khoản ví", "thông tin ví", "số dư", "khóa", "mở khóa", "hủy ví", "đổi mật khẩu", "quên mật khẩu"],
            "aliases": ["tài khoản ví", "tai khoan vi", "thông tin tài khoản", "thong tin tai khoan", "mở khóa tài khoản", "mo khoa tai khoan", "khóa tài khoản", "khoa tai khoan", "hủy ví", "huy vi", "đổi mật khẩu", "doi mat khau", "lấy lại mật khẩu", "lay lai mat khau", "quên mật khẩu", "quen mat khau", "số dư", "so du"],
            "negative_keywords": ["mobile money", "tk mobile money"],
            "excludes": ["MOBILE_MONEY"]
        },
        "MOBILE_MONEY": {
            "display": "Mobile Money",
            "slot_key": "mobile_money",
            "graph_label": "Service",
            "domain": "wallet_core",
            "priority": 2,
            "tier": "B",
            "anchors": ["mobile money", "tk mobile money", "tài khoản mobile money", "tai khoan mobile money", "ví mobile money", "vi mobile money"],
            "keywords": ["mobile money", "đăng ký mobile money", "hủy mobile money", "mở khóa mobile money"],
            "aliases": ["mobile money", "ví mobile money", "vi mobile money", "tài khoản mobile money", "tai khoan mobile money", "tk mobile money", "đăng ký mobile money", "dang ky mobile money", "hủy mobile money", "huy mobile money", "mở khóa mobile money", "mo khoa mobile money"],
            "excludes": ["WALLET_ACCOUNT"]
        },
        "IDENTITY_VERIFICATION": {
            "display": "Xác thực định danh (eKYC)",
            "slot_key": "xac_thuc_dinh_danh",
            "graph_label": "Service",
            "domain": "app",
            "priority": 2,
            "anchors": ["ekyc", "định danh", "dinh danh", "cccd", "cmnd", "hộ chiếu", "ho chieu", "xác thực định danh", "xac thuc dinh danh", "sinh trắc", "sinh trac", "vân tay", "van tay", "face id", "khuôn mặt", "khuon mat"],
            "keywords": ["định danh", "ekyc", "xác thực", "cccd", "cmnd", "hộ chiếu", "sinh trắc học", "vân tay", "face id"],
            "aliases": ["xác thực định danh", "xac thuc dinh danh", "định danh", "dinh danh", "ekyc", "cccd", "cmnd", "hộ chiếu", "ho chieu", "sinh trắc học", "sinh trac hoc", "sinh trắc", "sinh trac", "vân tay", "van tay", "face id", "khuôn mặt", "khuon mat"],
            "negative_keywords": [],
            "excludes": []
        },
        "VNPT_PAY": {
            "display": "VNPT Pay",
            "slot_key": "vnpt_pay",
            "graph_label": "Service",
            "domain": "wallet_core",
            "priority": 2,
            "tier": "B",
            "anchors": ["vnpt pay", "ví vnpt pay", "vi vnpt pay"],
            "keywords": ["vnpt pay"],
            "aliases": ["vnpt pay", "ví vnpt pay", "vi vnpt pay"]
        },
        "VNPT_ORDER": {
            "display": "Hóa đơn VNPT / Nạp tự động",
            "slot_key": "hoa_don_vnpt",
            "graph_label": "Service",
            "domain": "other",
            "priority": 2,
            "anchors": ["nạp tiền tự động", "nap tien tu dong", "nạp tự động", "nap tu dong", "đơn hàng", "don hang", "mytv", "gói cước mytv"],
            "keywords": ["đơn hàng", "nạp tiền tự động", "nạp tự động", "hủy nạp tiền tự động", "đặt dịch vụ nạp", "mua gói mytv"],
            "aliases": ["hóa đơn vnpt", "hoa don vnpt", "đơn hàng", "don hang", "nạp tiền tự động", "nap tien tu dong", "nạp tự động", "nap tu dong", "đặt dịch vụ nạp tiền tự động", "hủy nạp tiền tự động", "gói cước mytv", "goi cuoc mytv"],
            "excludes": ["TOPUP_WALLET"]
        },
        "THONG_TIN_CHUNG": {
            "display": "Thông tin chung",
            "slot_key": "thong_tin_chung",
            "graph_label": "Service",
            "domain": "app",
            "priority": 50,
            "anchors": ["thông tin chung", "thong tin chung"],
            "keywords": ["thông tin chung"],
            "aliases": ["thông tin chung", "thong tin chung"]
        },
        "OTHER_SERVICE": {
            "display": "Dịch vụ khác",
            "slot_key": "khac",
            "graph_label": "Service",
            "domain": "other",
            "priority": 99,
            "anchors": [],
            "keywords": [],
            "aliases": ["khác", "khac", "other"]
        }
    }

    _by_slot: Dict[str, str] = {}
    _by_display: Dict[str, str] = {}
    _by_alias: Dict[str, str] = {}

    @classmethod
    def _build_index(cls) -> None:
        if cls._by_slot:
            return
        for cid, data in cls.SERVICES.items():
            cls._by_slot[_norm_text(data["slot_key"])] = cid
            cls._by_display[_norm_text(data["display"])] = cid
            for a in data.get("aliases", []):
                cls._by_alias[_norm_text(a)] = cid

    @classmethod
    def from_slot_key(cls, slot_key: str) -> Optional[str]:
        cls._build_index()
        return cls._by_slot.get(_norm_text(slot_key))

    @classmethod
    def from_display(cls, display_name: str) -> Optional[str]:
        cls._build_index()
        return cls._by_display.get(_norm_text(display_name))

    @classmethod
    def from_alias(cls, alias: str) -> Optional[str]:
        cls._build_index()
        return cls._by_alias.get(_norm_text(alias))

    @classmethod
    def to_slot_key(cls, canonical_id: str) -> Optional[str]:
        return cls.SERVICES.get(canonical_id, {}).get("slot_key")

    @classmethod
    def to_display(cls, canonical_id: str) -> Optional[str]:
        return cls.SERVICES.get(canonical_id, {}).get("display")

 

class ProblemTypeTaxonomy:

    PROBLEMS: Dict[str, Dict] = {
        "INSTRUCTION": {
            "display": "Hướng dẫn",
            "slot_key": "huong_dan",
            "graph_label": "ProblemType",
            "aliases": ["hướng dẫn", "huong dan", "cách", "lam sao", "làm sao", "thế nào", "như thế nào", "how to", "hướng dẫn giúp", "chỉ giúp", "hướng dẫn cách"],
            "intent": "HOW_TO"
        },
        "POLICY": {
            "display": "Chính sách",
            "slot_key": "chinh_sach",
            "graph_label": "ProblemType",
            "aliases": ["chính sách", "chinh sach", "quy định", "quy dinh", "điều khoản", "dieu khoan", "phí", "phi", "biểu phí", "bieu phi", "mức phí", "han muc quy dinh"],
            "intent": "INFO"
        },
        "OTHER": {
            "display": "Khác",
            "slot_key": "khac",
            "graph_label": "ProblemType",
            "aliases": ["khác", "khac", "thông tin", "thong tin", "có không", "được không", "duoc khong", "hỏi", "hoi"],
            "intent": "INFO"
        },
        "FAILED": {
            "display": "Thất bại",
            "slot_key": "that_bai",
            "graph_label": "ProblemType",
            "aliases": ["thất bại", "that bai", "không thành công", "khong thanh cong", "failed", "fail", "báo lỗi", "bao loi", "giao dịch lỗi", "giao dich loi"],
            "intent": "TROUBLESHOOT"
        },
        "INVESTIGATION": {
            "display": "Tra soát",
            "slot_key": "tra_soat",
            "graph_label": "ProblemType",
            "aliases": ["tra soát", "tra soat", "tra cứu giao dịch", "tra cuu giao dich", "kiểm tra giao dịch", "kiem tra giao dich", "xác minh giao dịch", "xac minh giao dich", "soát giao dịch", "soat giao dich", "khiếu nại", "khieu nai"],
            "intent": "TROUBLESHOOT"
        },
        "LINK_ERROR": {
            "display": "Lỗi liên kết",
            "slot_key": "loi_lien_ket",
            "graph_label": "ProblemType",
            "aliases": ["lỗi liên kết", "loi lien ket", "không liên kết được", "khong lien ket duoc", "liên kết thất bại", "lien ket that bai", "không thể liên kết", "khong the lien ket", "lỗi lknh", "loi lknh"],
            "intent": "TROUBLESHOOT"
        },
        "LIMIT_ERROR": {
            "display": "Lỗi hạn mức",
            "slot_key": "loi_han_muc",
            "graph_label": "ProblemType",
            "aliases": ["lỗi hạn mức", "loi han muc", "vượt hạn mức", "vuot han muc", "quá hạn mức", "qua han muc", "giao dịch quá hạn mức", "qua han muc giao dich"],
            "intent": "TROUBLESHOOT"
        },
        "BALANCE_ERROR": {
            "display": "Lỗi số dư",
            "slot_key": "loi_so_du",
            "graph_label": "ProblemType",
            "aliases": ["lỗi số dư", "loi so du", "số dư sai", "so du sai", "không đủ số dư", "khong du so du", "thiếu số dư", "thieu so du"],
            "intent": "TROUBLESHOOT"
        },
        "DEVICE_ERROR": {
            "display": "Lỗi thiết bị",
            "slot_key": "loi_thiet_bi",
            "graph_label": "ProblemType",
            "aliases": ["lỗi thiết bị", "loi thiet bi", "thiết bị lỗi", "thiet bi loi", "không hỗ trợ nfc", "khong ho tro nfc", "nfc lỗi", "nfc loi"],
            "intent": "TROUBLESHOOT"
        },
        "DISPLAY_ERROR": {
            "display": "Sự cố hiển thị",
            "slot_key": "su_co_hien_thi",
            "graph_label": "ProblemType",
            "aliases": ["sự cố hiển thị", "su co hien thi", "không hiển thị", "khong hien thi", "hiển thị sai", "hien thi sai", "không thấy", "khong thay", "không hiện", "khong hien"],
            "intent": "TROUBLESHOOT"
        },
        "BANK_NOT_SUPPORTED": {
            "display": "Không hỗ trợ ngân hàng",
            "slot_key": "khong_ho_tro_ngan_hang",
            "graph_label": "ProblemType",
            "aliases": ["không hỗ trợ ngân hàng", "khong ho tro ngan hang", "ngân hàng không hỗ trợ", "ngan hang khong ho tro", "không có ngân hàng", "khong co ngan hang"],
            "intent": "INFO"
        },
        "AUTH_ERROR": {
            "display": "Lỗi xác thực/OTP",
            "slot_key": "loi_xac_thuc",
            "graph_label": "ProblemType",
            "aliases": ["lỗi xác thực", "loi xac thuc", "không nhận otp", "khong nhan otp", "otp không về", "otp khong ve", "chưa nhận otp", "chua nhan otp", "otp không đến", "otp khong den", "không có otp", "khong co otp", "mã otp", "ma otp", "mã xác thực", "ma xac thuc", "tài khoản chưa đăng ký dịch vụ thanh toán", "tai khoan chua dang ky dich vu thanh toan", "chưa đăng ký dịch vụ thanh toán", "chua dang ky dich vu thanh toan"],
            "intent": "TROUBLESHOOT"
        }
    }

    _by_slot: Dict[str, str] = {}
    _by_display: Dict[str, str] = {}
    _by_alias: Dict[str, str] = {}

    @classmethod
    def _build_index(cls) -> None:
        if cls._by_slot:
            return
        for cid, data in cls.PROBLEMS.items():
            cls._by_slot[_norm_text(data["slot_key"])] = cid
            cls._by_display[_norm_text(data["display"])] = cid
            for a in data.get("aliases", []):
                cls._by_alias[_norm_text(a)] = cid

    @classmethod
    def from_slot_key(cls, slot_key: str) -> Optional[str]:
        cls._build_index()
        return cls._by_slot.get(_norm_text(slot_key))

    @classmethod
    def from_display(cls, display_name: str) -> Optional[str]:
        cls._build_index()
        return cls._by_display.get(_norm_text(display_name))

    @classmethod
    def from_alias(cls, alias: str) -> Optional[str]:
        cls._build_index()
        return cls._by_alias.get(_norm_text(alias))

    @classmethod
    def to_slot_key(cls, canonical_id: str) -> Optional[str]:
        return cls.PROBLEMS.get(canonical_id, {}).get("slot_key")

    @classmethod
    def to_display(cls, canonical_id: str) -> Optional[str]:
        return cls.PROBLEMS.get(canonical_id, {}).get("display")

  


class StateTaxonomy:
    """
    Transaction Status UI - what the system/UI displays as transaction status.
    
    This represents the status shown in transaction history/UI:
    - pending: Transaction is being processed
    - success: Transaction completed successfully (shown in UI)
    - failed: Transaction failed (shown in UI with error)
    - unknown: Status not mentioned or unclear
    
    Note: State should only be inferred for transactional problems (chua_nhan_tien, 
    bi_tru_tien, that_bai, tra_soat). For procedural/policy problems (huong_dan, 
    chinh_sach, khac), state should default to 'unknown'.
    """

    STATES: Dict[str, Dict] = {
        "UNKNOWN": {
            "display": "Không rõ trạng thái",
            "slot_key": "unknown",
            "graph_label": "State",
            "aliases": ["không rõ", "chưa biết", "unknown", "không nói"],
            "description": "User did not mention transaction status or it's unclear from UI"
        },
        "PENDING": {
            "display": "Đang xử lý",
            "slot_key": "pending",
            "graph_label": "State",
            "aliases": ["đang xử lý", "pending", "chờ xử lý", "đang chờ", "chờ", "dang xu ly", "dang cho"],
            "description": "Transaction is in processing state (shown in UI as 'đang xử lý')"
        },
        "FAILED": {
            "display": "Thất bại",
            "slot_key": "failed",
            "graph_label": "State",
            "aliases": ["thất bại", "không thành công", "failed", "báo lỗi thất bại", "hiển thị thất bại", "that bai", "khong thanh cong", "không được", "khong duoc", "không thực hiện được", "khong thuc hien duoc", "bị lỗi", "bi loi"],
            "description": "Transaction failed (shown in UI as 'thất bại')"
        },
        "SUCCESS": {
            "display": "Thành công",
            "slot_key": "success",
            "graph_label": "State",
            "aliases": ["thành công", "success", "đã thành công", "hoàn thành", "hiển thị thành công", "thanh cong", "da thanh cong"],
            "description": "Transaction succeeded (shown in UI as 'thành công')"
        },
    }

    _by_slot: Dict[str, str] = {}
    _by_display: Dict[str, str] = {}
    _by_alias: Dict[str, str] = {}

    @classmethod
    def _build_index(cls) -> None:
        if cls._by_slot:
            return
        for cid, data in cls.STATES.items():
            cls._by_slot[_norm_text(data["slot_key"])] = cid
            cls._by_display[_norm_text(data["display"])] = cid
            for a in data.get("aliases", []):
                cls._by_alias[_norm_text(a)] = cid

    @classmethod
    def from_slot_key(cls, slot_key: str) -> Optional[str]:
        cls._build_index()
        return cls._by_slot.get(_norm_text(slot_key))

    @classmethod
    def from_alias(cls, alias: str) -> Optional[str]:
        cls._build_index()
        return cls._by_alias.get(_norm_text(alias))

    @classmethod
    def to_slot_key(cls, canonical_id: str) -> Optional[str]:
        return cls.STATES.get(canonical_id, {}).get("slot_key")


# ---------------------------
# Outcome taxonomy (transaction outcome)
# ---------------------------

class OutcomeTaxonomy:
    OUTCOMES: Dict[str, Dict] = {
        "UNKNOWN": {
            "display": "Không rõ triệu chứng",
            "slot_key": "unknown",
            "graph_label": "Outcome",
            "aliases": ["không rõ", "chưa biết", "unknown", "none"],
            "description": "Symptom unclear or not explicitly mentioned"
        },
        "NEED_INSTRUCTION": {
            "display": "Cần hướng dẫn",
            "slot_key": "need_instruction",
            "graph_label": "Outcome",
            "aliases": ["cần hướng dẫn", "cần chỉ dẫn", "cần giúp đỡ", "hướng dẫn", "chỉ dẫn", "làm sao", "làm thế nào", "need_instruction", "need instruction"],
            "description": "User needs procedural guidance or how-to instructions"
        },
        "MONEY_DEDUCTED": {
            "display": "Tiền đã bị trừ",
            "slot_key": "money_deducted",
            "graph_label": "Outcome",
            "aliases": ["tiền đã bị trừ", "đã trừ tiền", "bị trừ tiền", "mất tiền", "trừ tiền", "money_deducted", "money deducted"],
            "description": "Money was deducted/charged"
        },
        "MONEY_NOT_RECEIVED": {
            "display": "Chưa nhận được tiền/dịch vụ",
            "slot_key": "money_not_received",
            "graph_label": "Outcome",
            "aliases": ["chưa nhận được tiền", "tiền chưa về", "không nhận được tiền", "chưa cộng tiền", "chưa nhận", "không thấy tiền", "chưa thấy tiền", "chua thay tien", "không có tiền", "khong co tien", "money_not_received", "money not received"],
            "description": "User hasn't received money or service yet"
        },
        "OTP_NOT_RECEIVED": {
            "display": "Không nhận OTP",
            "slot_key": "otp_not_received",
            "graph_label": "Outcome",
            "aliases": ["không nhận otp", "otp không về", "chưa nhận otp", "otp không đến", "không có otp", "mã otp không về", "otp_not_received"],
            "description": "OTP was not received"
        },
        "CANNOT_CANCEL": {
            "display": "Không hủy được",
            "slot_key": "cannot_cancel",
            "graph_label": "Outcome",
            "aliases": ["không hủy được", "khong huy duoc", "không thể hủy", "khong the huy", "cannot_cancel", "cannot cancel"],
            "description": "User cannot cancel a subscription/auto service"
        }
    }

    _by_slot: Dict[str, str] = {}
    _by_display: Dict[str, str] = {}
    _by_alias: Dict[str, str] = {}

    @classmethod
    def _build_index(cls) -> None:
        if cls._by_slot:
            return
        for cid, data in cls.OUTCOMES.items():
            cls._by_slot[_norm_text(data["slot_key"])] = cid
            cls._by_display[_norm_text(data["display"])] = cid
            for a in data.get("aliases", []):
                cls._by_alias[_norm_text(a)] = cid

    @classmethod
    def from_slot_key(cls, slot_key: str) -> Optional[str]:
        cls._build_index()
        return cls._by_slot.get(_norm_text(slot_key))

    @classmethod
    def from_alias(cls, alias: str) -> Optional[str]:
        cls._build_index()
        return cls._by_alias.get(_norm_text(alias))

    @classmethod
    def to_slot_key(cls, canonical_id: str) -> Optional[str]:
        return cls.OUTCOMES.get(canonical_id, {}).get("slot_key")


# ---------------------------
# Bank taxonomy (loaded from JSON)
# ---------------------------

class BankTaxonomy:

    BANKS: Dict[str, Dict] = {}
    _by_code: Dict[str, str] = {}
    _by_display: Dict[str, str] = {}
    _by_alias: Dict[str, str] = {}
    _loaded: bool = False

    @classmethod
    def _default_banks_path(cls) -> str:
        here = os.path.dirname(os.path.abspath(__file__))  # src/ folder absolute path
        parent = os.path.dirname(here)  # project root
        
        # Try external_data folder (standard location)
        p1 = os.path.join(parent, "external_data", "supported_banks.json")
        if os.path.exists(p1):
            return p1
        
        # Try src folder
        p2 = os.path.join(here, "supported_banks.json")
        if os.path.exists(p2):
            return p2
        
        # Try project root
        p3 = os.path.join(parent, "supported_banks.json")
        if os.path.exists(p3):
            return p3
        
        # Last resort: return expected path (will fail with clear error)
        return p1

    @classmethod
    def load_from_json(cls, path: Optional[str] = None) -> None:
        if cls._loaded:
            return
        path = path or cls._default_banks_path()
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        banks: Dict[str, Dict] = {}
        for it in items:
            code = it.get("bank_id") 
            name = it.get("name")
            if not code or not name:
                continue
            banks[code] = {
                "display": name,
                "slot_key": name,         
                "aliases": list({*(it.get("aliases") or []), code, name}),
                "supported": bool(it.get("supported", True)),
            }

        cls.BANKS = banks
        cls._build_index()
        cls._loaded = True

    @classmethod
    def _build_index(cls) -> None:
        cls._by_code.clear()
        cls._by_display.clear()
        cls._by_alias.clear()

        for cid, data in cls.BANKS.items():
            cls._by_code[_norm_text(cid)] = cid
            cls._by_display[_norm_text(data["display"])] = cid
            for a in data.get("aliases", []):
                cls._by_alias[_norm_text(a)] = cid

    @classmethod
    def from_any(cls, text: str) -> Optional[str]:
        """Try code/display/alias → canonical bank code."""
        if not cls._loaded:
            cls.load_from_json()
        t = _norm_text(text)
        return cls._by_code.get(t) or cls._by_display.get(t) or cls._by_alias.get(t)

    @classmethod
    def to_display(cls, canonical_id: str) -> Optional[str]:
        if not cls._loaded:
            cls.load_from_json()
        return cls.BANKS.get(canonical_id, {}).get("display")

    

    @classmethod
    def is_supported(cls, canonical_id: str) -> bool:
        if not cls._loaded:
            cls.load_from_json()
        return bool(cls.BANKS.get(canonical_id, {}).get("supported", True))

    @classmethod
    def get_all_banks(cls) -> list:
        """Return list of all banks with their info."""
        if not cls._loaded:
            cls.load_from_json()
        return [
            {"bank_id": bid, "name": data["display"]}
            for bid, data in cls.BANKS.items()
        ]


try:
    BankTaxonomy.load_from_json()
except Exception:
    pass


# ---------------------------
# Unified normalizer
# ---------------------------

class UnifiedNormalizer:
    """Single deterministic normalizer using taxonomy system."""

    AMOUNT_RE = re.compile(
        r"(?P<num>[0-9]+(?:[.,][0-9]+)?)\s*(?P<unit>k|nghin|ngan|tr|trieu|cu|canh|m|ty|tyr|ti|t)?\b",
        re.IGNORECASE,
    )

    @staticmethod
    def normalize_service(text: str) -> Optional[str]:
        if not text:
            return None

        # if already slot_key
        if ServiceTaxonomy.from_slot_key(text):
            return ServiceTaxonomy.to_slot_key(ServiceTaxonomy.from_slot_key(text))

        cid = ServiceTaxonomy.from_display(text) or ServiceTaxonomy.from_alias(text)
        return ServiceTaxonomy.to_slot_key(cid) if cid else None

    @staticmethod
    def normalize_problem(text: str) -> Optional[str]:
        if not text:
            return None

        if ProblemTypeTaxonomy.from_slot_key(text):
            return ProblemTypeTaxonomy.to_slot_key(ProblemTypeTaxonomy.from_slot_key(text))

        cid = ProblemTypeTaxonomy.from_display(text) or ProblemTypeTaxonomy.from_alias(text)
        return ProblemTypeTaxonomy.to_slot_key(cid) if cid else None

    @staticmethod
    def normalize_bank_pair(text: str) -> Optional[Tuple[str, str]]:

        if not text:
            return None
        cid = BankTaxonomy.from_any(text)
        if not cid:
            return None
        display = BankTaxonomy.to_display(cid) or cid
        return (cid, display)

    @staticmethod
    def normalize_bank(text: str) -> Optional[str]:
        """Backward-compatible: normalize bank text → bank_display.

        Prefer `normalize_bank_pair()` in new code to avoid bank_id/display drift.
        """
        pair = UnifiedNormalizer.normalize_bank_pair(text)
        return pair[1] if pair else None
    @staticmethod
    def normalize_amount(text: str) -> Optional[int]:
      
        if not text:
            return None

        t = _norm_text(text)
        m = UnifiedNormalizer.AMOUNT_RE.search(t)
        if not m:
            return None

        num_raw = m.group("num").replace(",", ".")
        try:
            val = float(num_raw)
        except ValueError:
            return None

        unit = (m.group("unit") or "").lower()
        if unit in ("k", "nghin", "ngan", "canh", "cành"):
            val *= 1_000
        elif unit in ("tr", "trieu", "m", "cu", "củ"):
            val *= 1_000_000
        elif unit in ("ty", "tyr", "ti", "t"):
            val *= 1_000_000_000

        return int(round(val))

    @staticmethod
    def infer_service_from_text(text: str) -> Tuple[Optional[str], float, List[str]]:
        """
        Infer service from natural language text using taxonomy.
        
        Returns:
            (service_slot_key, confidence_score, evidence_list)
            
        Algorithm:
        1. Longest match wins (phrases > single words)
        2. Anchor tokens have highest priority
        3. Score = weighted sum of matches
        4. Negative keywords disqualify candidates
        """
        if not text:
            return (None, 0.0, [])
        
        text_norm = _norm_text(text)
        candidates: Dict[str, Dict] = {}  # service_id -> {score, evidence, has_anchor}
        
        for service_id, service_data in ServiceTaxonomy.SERVICES.items():
            score = 0.0
            evidence = []
            has_anchor = False
            has_negative = False
            
            # Check negative keywords first (disqualify immediately)
            negative_kws = service_data.get("negative_keywords", [])
            for neg_kw in negative_kws:
                if _norm_text(neg_kw) in text_norm:
                    has_negative = True
                    break
            
            if has_negative:
                continue  # Skip this service
            
            # Check anchors first (highest priority)
            anchors = service_data.get("anchors", [])
            for anchor in anchors:
                anchor_norm = _norm_text(anchor)
                if anchor_norm and anchor_norm in text_norm:
                    # Anchor match: high weight (10 points per anchor)
                    score += 10.0 * len(anchor_norm.split())
                    evidence.append(f"anchor:{anchor}")
                    has_anchor = True
            
            # Check keywords (phrases weighted by length)
            keywords = service_data.get("keywords", [])
            for kw in keywords:
                kw_norm = _norm_text(kw)
                if kw_norm and kw_norm in text_norm:
                    # Longer phrases = higher weight
                    word_count = len(kw_norm.split())
                    kw_score = word_count * 2.0  # 2 points per word
                    score += kw_score
                    evidence.append(f"keyword:{kw}")
            
            # Check aliases (lower weight than keywords)
            aliases = service_data.get("aliases", [])
            for alias in aliases:
                alias_norm = _norm_text(alias)
                if alias_norm and alias_norm in text_norm:
                    word_count = len(alias_norm.split())
                    alias_score = word_count * 1.5  # 1.5 points per word
                    score += alias_score
                    evidence.append(f"alias:{alias}")
            
            if score > 0:
                candidates[service_id] = {
                    "score": score,
                    "evidence": evidence,
                    "has_anchor": has_anchor,
                    "slot_key": service_data["slot_key"],
                    "priority": service_data.get("priority", 99),
                    "excludes": service_data.get("excludes", []),
                }
        
        if not candidates:
            return ("khac", 0.20, ["fallback:no_service_match"])

        # STEP 1: Apply TIER filtering (CRITICAL for accuracy)
        # If any TIER_A service matched, remove all TIER_B services
        tier_a_candidates = {k: v for k, v in candidates.items() if v.get("slot_key") in ServiceTaxonomy.TIER_A_SERVICES}
        tier_b_candidates = {k: v for k, v in candidates.items() if v.get("slot_key") in ServiceTaxonomy.TIER_B_SERVICES}
        
        if tier_a_candidates:
            # Action services found - ignore platform services
            candidates = tier_a_candidates
        else:
            # No action services - use platform services
            candidates = tier_b_candidates if tier_b_candidates else candidates

        # STEP 2: If any candidate has an anchor match, restrict to anchored candidates.
        if any(v.get("has_anchor") for v in candidates.values()):
            candidates = {k: v for k, v in candidates.items() if v.get("has_anchor")}

        # STEP 3: Sort by: anchor presence (desc), then score (desc), then priority (asc)
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: (x[1]["has_anchor"], x[1]["score"], -x[1].get("priority", 99)),
            reverse=True
        )
        
        best_service_id, best_data = sorted_candidates[0]
        
        # CONTEXTUAL OVERRIDES (fixes remaining service confusions)
        # Override 1: xac_thuc_dinh_danh (identity verification including biometric)
        biometric_kws = ["sinh trac", "van tay", "face id", "khuon mat"]
        auth_error_kws = ["ngay sinh", "dinh danh", "xac thuc", "cccd", "cmnd"]
        error_kws = ["loi", "bao", "khong hop le", "sai", "that bai"]
        
        has_biometric = any(kw in text_norm for kw in biometric_kws)
        has_auth_context = any(kw in text_norm for kw in auth_error_kws)
        has_error = any(kw in text_norm for kw in error_kws)
        
        if has_biometric or (has_auth_context and has_error):
            # Force xac_thuc_dinh_danh if biometric or identity context detected
            return ("xac_thuc_dinh_danh", 0.85, ["override:identity_verification_context"])
        
        # Override 2: vien_thong (telecom service) - boost if strong anchors present
        if best_data["slot_key"] == "vien_thong":
            strong_telecom = ["ma the", "the cao", "goi cuoc", "data", "3g", "4g"]
            if any(kw in text_norm for kw in strong_telecom):
                # Boost confidence if strong telecom keywords present
                best_data["score"] *= 2
        
        # Calculate confidence: normalize score to 0-1 range
        # High anchor + keywords = 0.9+, keywords only = 0.6-0.8, single match = 0.4-0.5
        raw_score = best_data["score"]
        if best_data["has_anchor"]:
            confidence = min(0.95, 0.75 + (raw_score / 100))
        else:
            confidence = min(0.85, 0.40 + (raw_score / 50))
        
        return (best_data["slot_key"], confidence, best_data["evidence"])

    @staticmethod
    def infer_problem_from_text(text: str, service: Optional[str] = None) -> Tuple[Optional[str], float, List[str]]:
        """
        Infer problem type from natural language text using taxonomy.
        
        Args:
            text: User's natural language input
            service: Optional service context to improve inference
            
        Returns:
            (problem_slot_key, confidence_score, evidence_list)
            
        Algorithm:
        1. Longest phrase match wins
        2. Context-aware inference (e.g., "trừ tiền" + "pending" = tra_soat)
        3. Score by keyword match count and length
        """
        if not text:
            return (None, 0.0, [])
        
        text_norm = _norm_text(text)
        candidates: Dict[str, Dict] = {}  # problem_id -> {score, evidence}
        
        # =============================================================================
        # CRITICAL PRIORITY ORDER (DO NOT CHANGE):
        # 1. OTP/Authentication issues (loi_xac_thuc)
        # 2. Transaction symptoms (tra_soat) - MUST come before how-to
        # 3. How-to questions (huong_dan)
        # 4. Policy questions (chinh_sach)
        # =============================================================================
        
        # PRIORITY 1: OTP/Authentication issues (HIGHEST - override everything)
        otp_kws = ["otp", "ma xac thuc", "xac thuc"]
        has_otp = any(kw in text_norm for kw in otp_kws)
        
        if has_otp:
            # Any OTP-related issue is loi_xac_thuc
            return ("loi_xac_thuc", 0.90, ["priority:otp_authentication"])
        
        # PRIORITY 2: Transaction symptoms (MUST be before how-to)
        # This fixes: tra_soat → huong_dan confusion (10 cases)
        transaction_symptom_kws = [
            "chua nhan", "khong nhan", "bi tru tien", "mat tien",
            "khong thay tien", "khong cong", "chua ve", "chua duoc",
            "chua toi", "khong thanh cong", "that bai", "pending",
            "chuyen nham", "chuyen sai", "gui nham"
        ]
        transaction_context_kws = ["tien", "giao dich", "thanh toan", "nap", "rut", "chuyen"]
        
        has_symptom = any(kw in text_norm for kw in transaction_symptom_kws)
        has_transaction = any(kw in text_norm for kw in transaction_context_kws)
        
        if has_symptom and has_transaction:
            # Transaction problem detected - tra_soat even if has "làm sao"
            return ("tra_soat", 0.85, ["priority:transaction_symptom"])
        
        # Specific error patterns
        if any(kw in text_norm for kw in ["khong lien ket", "lien ket that bai", "loi lien ket"]):
            return ("loi_lien_ket", 0.90, ["pattern:link_error"])
        
        # Invalid card/account info (linking error)
        if any(kw in text_norm for kw in ["thong tin the khong hop le", "thong tin tai khoan khong hop le", "thong tin the/tai khoan khong hop le", "the khong hop le", "tai khoan khong hop le", "thong tin khong hop le"]):
            return ("loi_lien_ket", 0.85, ["pattern:invalid_card_info"])
        
        # Pattern 5: Investigation context (money deducted + failed/pending)
        state_indicators = {
            "pending": ["dang xu ly", "cho xu ly", "pending"],
            "failed": ["that bai", "khong thanh cong", "failed", "bao loi"],
            "success": ["thanh cong", "da thanh cong", "hoan thanh"]
        }
        
        detected_state = None
        for state, indicators in state_indicators.items():
            if any(ind in text_norm for ind in indicators):
                detected_state = state
                break
        
        money_deducted_kws = [
            "bi tru tien", "da tru tien", "ngan hang da tru", "mat tien",
            "tru oan", "da tru", "bi tru", "ngan hang tru"
        ]
        has_money_deducted = any(kw in text_norm for kw in money_deducted_kws)
        
        if has_money_deducted and detected_state in ["failed", "pending"]:
            # Money deducted but transaction failed/pending = needs investigation
            return ("tra_soat", 0.90, ["pattern:investigation_needed"])
        elif has_money_deducted:
            # Simple money deduction without clear state
            return ("tra_soat", 0.85, ["pattern:money_deducted"])
        
        # Pattern 6: Money not received
        money_not_received_kws = [
            "chua nhan duoc tien", "tien chua ve", "khong nhan duoc tien",
            "chua thay tien", "chua cong", "chua vao", "chua duoc cong"
        ]
        if any(kw in text_norm for kw in money_not_received_kws):
            return ("tra_soat", 0.85, ["pattern:money_not_received"])
        
        # General matching using taxonomy
        for problem_id, problem_data in ProblemTypeTaxonomy.PROBLEMS.items():
            score = 0.0
            evidence = []
            
            # Check aliases (weighted by phrase length)
            aliases = problem_data.get("aliases", [])
            for alias in aliases:
                alias_norm = _norm_text(alias)
                if alias_norm and alias_norm in text_norm:
                    word_count = len(alias_norm.split())
                    alias_score = word_count * 2.0  # 2 points per word
                    score += alias_score
                    evidence.append(f"alias:{alias}")
            
            if score > 0:
                candidates[problem_id] = {
                    "score": score,
                    "evidence": evidence,
                    "slot_key": problem_data["slot_key"]
                }
        
        # If we have strong candidates (score >= 8), use them
        if candidates:
            sorted_candidates = sorted(
                candidates.items(),
                key=lambda x: x[1]["score"],
                reverse=True
            )
            
            best_problem_id, best_data = sorted_candidates[0]
            raw_score = best_data["score"]
            
            # Strong match - use it
            if raw_score >= 8:
                confidence = min(0.85, 0.40 + (raw_score / 30))
                return (best_data["slot_key"], confidence, best_data["evidence"])
        
        # FALLBACK RULES (only when no strong taxonomy match)
        # How-to questions
        howto_kws = [
            "lam the nao", "nhu the nao", "cach", "lam sao",
            "thao tac", "huong dan", "chi dan", "phai lam gi"
        ]
        if any(kw in text_norm for kw in howto_kws):
            return ("huong_dan", 0.75, ["fallback:howto_question"])
        
        # Policy questions
        policy_kws = [
            "co duoc khong", "co the", "co cho phep", "dieu kien",
            "yeu cau", "han muc", "phi", "mat phi", "bao nhieu",
            "toi da", "toi thieu"
        ]
        if any(kw in text_norm for kw in policy_kws):
            return ("chinh_sach", 0.70, ["fallback:policy_question"])
        
        # Use weak candidates if available
        if candidates:
            best_problem_id, best_data = sorted_candidates[0]
            raw_score = best_data["score"]
            confidence = min(0.85, 0.40 + (raw_score / 30))
            return (best_data["slot_key"], confidence, best_data["evidence"])
        
        return ("khac", 0.20, ["fallback:no_problem_match"])

    @staticmethod
    def infer_state_from_text(text: str) -> Tuple[Optional[str], float, List[str]]:
        """
        Infer state from natural language text using taxonomy.
        
        Returns:
            (state_slot_key, confidence_score, evidence_list)
        """
        if not text:
            return ("unknown", 0.0, [])
        
        text_norm = _norm_text(text)
        
        # Enhanced failed detection (fixes: expected='failed' → predicted='unknown')
        failed_indicators = [
            "khong duoc", "that bai", "bao loi", "khong thanh cong",
            "failed", "error", "khong the", "bi loi"
        ]
        if any(ind in text_norm for ind in failed_indicators):
            return ("failed", 0.85, ["rule:failed_indicator"])
        
        candidates: Dict[str, Dict] = {}  # state_id -> {score, evidence}
        
        for state_id, state_data in StateTaxonomy.STATES.items():
            score = 0.0
            evidence = []
            
            # Check aliases (weighted by phrase length)
            aliases = state_data.get("aliases", [])
            for alias in aliases:
                alias_norm = _norm_text(alias)
                if alias_norm and alias_norm in text_norm:
                    word_count = len(alias_norm.split())
                    alias_score = word_count * 2.0  # 2 points per word
                    score += alias_score
                    evidence.append(f"alias:{alias}")
            
            if score > 0:
                candidates[state_id] = {
                    "score": score,
                    "evidence": evidence,
                    "slot_key": state_data["slot_key"]
                }
        
        if not candidates:
            return ("unknown", 0.20, ["fallback:no_state_match"])
        
        # Sort by score (highest first)
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        best_state_id, best_data = sorted_candidates[0]
        
        # Calculate confidence
        raw_score = best_data["score"]
        confidence = min(0.85, 0.40 + (raw_score / 30))
        
        return (best_data["slot_key"], confidence, best_data["evidence"])

    @staticmethod
    def infer_outcome_from_text(text: str, problem: Optional[str] = None) -> Tuple[Optional[str], float, List[str]]:
        """
        Infer outcome from natural language text using taxonomy.
        
        Args:
            text: User's natural language input
            problem: Optional problem context for default mapping
        
        Returns:
            (outcome_slot_key, confidence_score, evidence_list)
        """
        if not text:
            return ("unknown", 0.0, [])
        
        text_norm = _norm_text(text)
        
        # DEFAULT MAPPING: problem → outcome (TIGHTENED)
        # Only huong_dan → need_instruction (not chinh_sach)
        # This fixes over-prediction of need_instruction (21 cases unknown → need_instruction)
        if problem == "huong_dan":
            # How-to questions need instruction
            return ("need_instruction", 0.80, ["rule:howto_needs_instruction"])
        
        # chinh_sach (policy) → unknown by default
        if problem == "chinh_sach":
            return ("unknown", 0.60, ["rule:policy_outcome_unknown"])
        
        candidates: Dict[str, Dict] = {}  # outcome_id -> {score, evidence}
        
        for outcome_id, outcome_data in OutcomeTaxonomy.OUTCOMES.items():
            score = 0.0
            evidence = []
            
            # Check aliases (weighted by phrase length)
            aliases = outcome_data.get("aliases", [])
            for alias in aliases:
                alias_norm = _norm_text(alias)
                if alias_norm and alias_norm in text_norm:
                    word_count = len(alias_norm.split())
                    alias_score = word_count * 2.0  # 2 points per word
                    score += alias_score
                    evidence.append(f"alias:{alias}")
            
            if score > 0:
                candidates[outcome_id] = {
                    "score": score,
                    "evidence": evidence,
                    "slot_key": outcome_data["slot_key"]
                }
        
        # Fallback: Only huong_dan gets need_instruction (tightened)
        if problem == "huong_dan":
            if not candidates or max(v["score"] for v in candidates.values()) < 5:
                return ("need_instruction", 0.75, ["fallback:howto_instruction"])
        
        # chinh_sach stays unknown if no strong outcome
        if problem == "chinh_sach":
            if not candidates or max(v["score"] for v in candidates.values()) < 5:
                return ("unknown", 0.70, ["fallback:policy_unknown"])
        
        if not candidates:
            return ("unknown", 0.20, ["fallback:no_outcome_match"])
        
        # Sort by score (highest first)
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        best_outcome_id, best_data = sorted_candidates[0]
        
        # Calculate confidence
        raw_score = best_data["score"]
        confidence = min(0.85, 0.40 + (raw_score / 30))
        
        return (best_data["slot_key"], confidence, best_data["evidence"])