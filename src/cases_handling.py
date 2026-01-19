from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from normalize_content import (
    ServiceTaxonomy, 
    ProblemTypeTaxonomy, 
    StateTaxonomy,
    OutcomeTaxonomy,
    BankTaxonomy, 
    UnifiedNormalizer, 
    _norm_text
)
from dotenv import load_dotenv
from openai import OpenAI
from normalize_content import UnifiedNormalizer
load_dotenv()

bank_path = os.path.join(os.getcwd(), "external_data", "supported_banks.json")
policy_path_default = os.path.join(os.getcwd(), "external_data", "banks_policy.json")
policy_path = policy_path_default
@dataclass
class CaseSlots:
    service: Optional[str] = None            # slot_key from ServiceTaxonomy
    service_id: Optional[str] = None         # canonical ID from ServiceTaxonomy

    problem_type: Optional[str] = None       # slot_key from ProblemTypeTaxonomy
    problem_id: Optional[str] = None         # canonical ID from ProblemTypeTaxonomy

    state: Optional[str] = None              # slot_key from StateTaxonomy (unknown/pending/failed/success)
    state_id: Optional[str] = None           # canonical ID from StateTaxonomy
    
    outcome: Optional[str] = None            # slot_key from OutcomeTaxonomy
    outcome_id: Optional[str] = None         # canonical ID from OutcomeTaxonomy

    amount: Optional[str] = None            
    time: Optional[str] = None               

    bank: Optional[str] = None               
    bank_id: Optional[str] = None            

    transaction_status: Optional[str] = None 
    account_info: Optional[str] = None
    error_message: Optional[str] = None

    raw_narrative: str = ""
    confidence_score: float = 0.0
    
    # Individual confidence scores for filtering decisions
    service_confidence: float = 0.0         # Confidence in service detection
    problem_confidence: float = 0.0         # Confidence in problem detection
    inference_evidence: Dict[str, List[str]] = None  # Evidence for debugging

    missing_slots: List[str] = None
    contradictions: List[str] = None

    def __post_init__(self):
        if self.missing_slots is None:
            self.missing_slots = []
        if self.contradictions is None:
            self.contradictions = []
        if self.inference_evidence is None:
            self.inference_evidence = {}
    
    def to_dict(self):
        """Convert CaseSlots to dictionary for JSON serialization"""
        return {
            'service': self.service,
            'service_id': self.service_id,
            'problem_type': self.problem_type,
            'problem_id': self.problem_id,
            'state': self.state,
            'state_id': self.state_id,
            'outcome': self.outcome,
            'outcome_id': self.outcome_id,
            'amount': self.amount,
            'time': self.time,
            'bank': self.bank,
            'bank_id': self.bank_id,
            'transaction_status': self.transaction_status,
            'account_info': self.account_info,
            'error_message': self.error_message,
            'confidence_score': self.confidence_score,
            'missing_slots': self.missing_slots,
            'contradictions': self.contradictions
        }

TRANSACTIONAL_PROBLEMS = {
    "chua_nhan_tien",      # Money not received - state determines if pending/success/failed
    "bi_tru_tien",         # Money deducted - state for investigation
    "that_bai",            # Failed transaction - state for retry logic
    "tra_soat",            # Investigation - state for case handling
}


CRITICAL_SLOTS: Dict[str, List[str]] = {
    "nap_tien": ["service", "problem_type", "bank"],
    "rut_tien": ["service", "problem_type", "bank"],  # amount là nice-to-have
    "chuyen_tien": ["service", "problem_type"],  # amount + account_info là nice-to-have
    "lien_ket_ngan_hang": ["service", "problem_type", "bank"],
    "huy_lien_ket_ngan_hang": ["service", "problem_type", "bank"],
    "thanh_toan": ["service", "problem_type"],  # amount là nice-to-have
    "ung_dung": ["service", "problem_type"],
}


CONTRADICTIONS: Dict[Tuple[str, str], bool] = {
    ("thành_công", "giao_dịch_thất_bại"): True,
    ("thành_công", "chưa_nhận_dịch_vụ"): True,
    ("thành_công", "chưa_nhận_được_tiền"): True,
    ("thất_bại", "đang_xử_lý"): True,
}


class PolicyStore:

    SERVICE_TO_POLICY_CODES: Dict[str, List[str]] = {
        "nap_tien": ["DEPOSIT"],
        "rut_tien": ["WITHDRAW", "TRANSFER_TO_BANK"],
        "thanh_toan": ["PAYMENT"],
        "chuyen_tien": ["TRANSFER_TO_BANK", "TRANSFER"],  # Support transfer fee/limit questions
    }

    def __init__(self, policy_path: str):
        self.path = policy_path
        self._index: Dict[Tuple[str, str, str], Dict] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"banks_policy.json not found at: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            items = json.load(f)

        for it in items:
            if (it.get("wallet") or "").strip() != "VNPT_MONEY":
                continue

            bank_raw = it.get("bank")
            service_code = it.get("service")
            policy = it.get("policy") or {}
            if not bank_raw or not service_code:
                continue

            bank_id = BankTaxonomy.from_any(str(bank_raw))
            if not bank_id:
                continue

            self._index[("VNPT_MONEY", bank_id, str(service_code))] = policy

        self._loaded = True

    def get_policy_for_service(self, bank_id: str, service_slot_key: str) -> List[Tuple[str, Dict]]:
      
        self.load()
        codes = self.SERVICE_TO_POLICY_CODES.get(service_slot_key, [])
        out: List[Tuple[str, Dict]] = []
        for code in codes:
            pol = self._index.get(("VNPT_MONEY", bank_id, code))
            if pol:
                out.append((code, pol))
        return out

    def get_policies(self, bank_id: str) -> List[Tuple[str, Dict]]:
        """
        Return all policies for a given bank_id (across all service codes) for this wallet.
        Output: list of (service_code, policy_dict)
        """
        self.load()
        out: List[Tuple[str, Dict]] = []
        for (w, b, svc), pol in self._index.items():
            if w == "VNPT_MONEY" and b == bank_id:
                out.append((svc, pol))
        return out

    @staticmethod
    def is_policy_intent(text: str) -> bool:
        t = (text or "").lower()
        keywords = [
            "hạn mức", "tối đa", "tối thiểu", "mỗi lần", "mỗi ngày", "mỗi tháng",
            "phí", "mất phí", "bao nhiêu tiền", "fee", "limit"
        ]
        return any(k in t for k in keywords)



class CaseTriage:

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        policy_path: Optional[str] = None, 
  
    ):
        self.client = OpenAI(api_key=api_key)
        
        self.policy_store = PolicyStore(policy_path or policy_path_default)
        self.normalizer = UnifiedNormalizer()
        
        # Load service and problem taxonomies for rule-based extraction
        self.service_taxonomy = ServiceTaxonomy()
        self.problem_taxonomy = ProblemTypeTaxonomy()
        self.state_taxonomy = StateTaxonomy()
        self.outcome_taxonomy = OutcomeTaxonomy()
        self.bank_taxonomy = BankTaxonomy()

    def _rule_based_service_extraction(self, text: str) -> Tuple[Optional[str], float]:
        """
        Extract service using keyword patterns (fast, covers 80%+ cases).
        Returns: (service_key, confidence)
        """
        text_lower = _norm_text(text)
        
        # High-precision overrides for common confusions
        # CRITICAL: "phí nạp/rút tích lũy" -> sieu_tich_luy (NOT hcc_dvc)
        # MUST check tích lũy patterns BEFORE any other service checks
        if "sieu tich luy" in text_lower or "siêu tích lũy" in text_lower:
            return "sieu_tich_luy", 0.98
        if "tich luy" in text_lower or "tích lũy" in text_lower:
            # ANY mention of nạp/rút with tích lũy -> sieu_tich_luy
            if any(kw in text_lower for kw in ["nap/rut", "nạp/rút", "nap", "nạp", "rut", "rút"]):
                return "sieu_tich_luy", 0.97
            # Questions about fees/costs related to tích lũy -> sieu_tich_luy
            if any(kw in text_lower for kw in ["phi", "phí", "chi phi", "chi phí", "khoan phi", "khoản phí", "le phi", "lệ phí"]):
                return "sieu_tich_luy", 0.97
            # General tích lũy questions (unless clearly about other services)
            return "sieu_tich_luy", 0.95
        # CRITICAL: Ticketing services must be detected early
        if "ve vui choi" in text_lower or "ve tham quan" in text_lower or ("mua ve" in text_lower and "vui choi" in text_lower):
            return "ve_tham_quan", 0.95
        if "co the mua ve" in text_lower and "vui choi" in text_lower:
            return "ve_tham_quan", 0.95
        if "co the mua ve" in text_lower and "o dau" in text_lower:
            # Check context to determine ticket type
            if "tau" in text_lower:
                return "ve_tau", 0.95
            if "may bay" in text_lower:
                return "ve_may_bay", 0.95
            if "vui choi" in text_lower or "tham quan" in text_lower:
                return "ve_tham_quan", 0.95
        if "ve tau" in text_lower and "gia ve" in text_lower:
            return "ve_tau", 0.95
        if "ve tau" in text_lower and ("thue" in text_lower or "phi" in text_lower):
            return "ve_tau", 0.95
        if "luu y" in text_lower and "dat ve" in text_lower and "tau" in text_lower:
            return "ve_tau", 0.95
        
        # CRITICAL: MyTV purchase must be detected BEFORE general telecom patterns
        if ("mua goi" in text_lower or "mua gói" in text_lower) and "mytv" in text_lower:
            return "hoa_don_vnpt", 0.96
        if "mua goi cuoc" in text_lower or "goi cuoc" in text_lower and "that bai" in text_lower:
            return "hoa_don_vnpt", 0.90
        if "huy giao dich" in text_lower and "thanh toan" in text_lower and ("sai" in text_lower or "nham" in text_lower or "trung" in text_lower):
            return "thanh_toan_khoan_vay", 0.90
        
        # Highest-priority contextual patterns
        if "rút tiền" in text_lower and "siêu tích lũy" in text_lower:
            return "rut_tien", 0.99
        if "nạp tiền" in text_lower and "mobile money" in text_lower:
            return "nap_tien", 0.99
        if "chuyển tiền" in text_lower and "mobile money" in text_lower:
            return "chuyen_tien", 0.99
        if "rút tiền" in text_lower and "mobile money" in text_lower:
            return "rut_tien", 0.99

        # High-priority keywords (check first)
        # CRITICAL: Check for "tích lũy" patterns (avoid confusion with hcc_dvc)
        if any(kw in text_lower for kw in ["siêu tích lũy", "sieu tich luy"]):
            return "sieu_tich_luy", 0.99
        
        # Check for "tích lũy" but NOT in context of fees/policies
        if any(kw in text_lower for kw in ["tích lũy", "tich luy", "gửi tiết kiệm", "lãi suất tích lũy"]):
            # But NOT if it's about fees/policies in general context
            if not any(kw in text_lower for kw in ["chi phí", "phí", "biểu phí", "mất phí"]):
                return "sieu_tich_luy", 0.95
            # If asking about fees FOR tích lũy specifically, still map to sieu_tich_luy
            elif any(kw in text_lower for kw in ["nạp tích lũy", "rút tích lũy", "nạp/rút tích lũy"]):
                return "sieu_tich_luy", 0.95
            
        if "mobile money" in text_lower:
            return "mobile_money", 0.98

        if any(kw in text_lower for kw in ["hủy liên kết", "xóa liên kết", "gỡ liên kết"]):
            return "huy_lien_ket_ngan_hang", 0.95
            
        if any(kw in text_lower for kw in ["liên kết ngân hàng", "liên kết thẻ", "thêm thẻ"]):
            return "lien_ket_ngan_hang", 0.95

        if any(kw in text_lower for kw in ["nạp tiền điện thoại", "nạp card", "nạp thẻ cào", "mua mã thẻ", "mua the dt", "the dien thoai"]):
            return "nap_tien_dien_thoai", 0.95
            
        if any(kw in text_lower for kw in ["rút tiền", "rút về"]):
            return "rut_tien", 0.95
            
        if any(kw in text_lower for kw in ["nạp tiền", "nạp vào"]):
            if any(kw in text_lower for kw in ["điện thoại", "card", "thẻ cào", "ma the"]):
                return "nap_tien_dien_thoai", 0.9
            return "nap_tien", 0.95

        if any(kw in text_lower for kw in ["chuyển tiền", "chuyển khoản", "bắn tiền"]):
            return "chuyen_tien", 0.95
        
        # CRITICAL: Barcode/payment code queries - MUST be before thanh_toan patterns
        if any(kw in text_lower for kw in ["mã bạn", "ma ban", "mã khách hàng", "ma khach hang", "barcode"]):
            return "vien_thong", 0.96
        # "vào mục nào kiểm tra mã" for payment codes
        if any(kw in text_lower for kw in ["vao muc nao", "vào mục nào"]) and any(kw in text_lower for kw in ["kiem tra", "kiểm tra", "xem"]):
            if any(kw in text_lower for kw in ["ma", "mã", "barcode", "thanh toan", "thanh toán"]):
                return "vien_thong", 0.95
            
        if any(kw in text_lower for kw in ["thanh toán hóa đơn", "đóng tiền điện", "đóng tiền nước", "thanh toán cước"]):
            return "thanh_toan_hoa_don", 0.95
            
        if "thanh toán" in text_lower:
            if any(kw in text_lower for kw in ["khoản vay", "vay", "hủy giao dịch", "thanh toán sai", "hủy thanh toán", "sai số hợp đồng"]):
                return "thanh_toan_khoan_vay", 0.92
            return "thanh_toan_dich_vu", 0.9

        if any(kw in text_lower for kw in ["viễn thông", "goi cuoc data", "3g/4g", "goi cuoc mytv", "truyen hinh", "gói cước", "mytv"]):
             # CRITICAL: Check if it's about purchasing packages (hoa_don_vnpt) vs general telecom
             if any(kw in text_lower for kw in ["mua goi", "mua gói", "goi cuoc lan", "gói cước lần", "thanh toan goi", "thanh toán gói", "goi cuoc tra truoc", "gói cước trả trước"]):
                 return "hoa_don_vnpt", 0.93
             # MyTV is a special case of telecom service, but handled under hoa_don_vnpt for package purchases
             if "mytv" in text_lower:
                 if any(kw in text_lower for kw in ["mua goi", "mua gói", "dang ky goi", "đăng ký gói"]):
                     return "hoa_don_vnpt", 0.93  # Purchasing MyTV packages
                 return "vien_thong", 0.85 # General MyTV questions
             return "vien_thong", 0.8
             
        # CONSOLIDATED: All identity verification maps to xac_thuc_dinh_danh
        # PRIORITY 1: CCCD/CMND verification questions (especially policy/quantity questions)
        if any(kw in text_lower for kw in ["cccd", "cmnd", "chung minh nhan dan", "can cuoc cong dan"]):
            if any(kw in text_lower for kw in ["may vi", "mấy ví", "bao nhieu vi", "bao nhiêu ví"]):
                return "xac_thuc_dinh_danh", 0.98 # Policy question about account limits
            return "xac_thuc_dinh_danh", 0.95
        # PRIORITY 2: General identity verification
        if any(kw in text_lower for kw in ["xác thực định danh", "định danh tài khoản", "cccd", "sinh trắc học", "định danh", "sinh trac hoc", "xac thuc", "kiem tra khuon mat", "kiểm tra khuôn mặt"]):
            # If it's about "how to", it's a guide
            if any(kw in text_lower for kw in ["lam sao", "làm sao", "huong dan", "hướng dẫn", "cach", "cách"]):
                return "xac_thuc_dinh_danh", 0.96
            return "xac_thuc_dinh_danh", 0.94

        if any(kw in text_lower for kw in ["hủy dịch vụ", "hủy gói", "hủy tài khoản", "khóa tài khoản", "huy vi", "hủy ví", "khoa vi", "khóa ví"]):
            # CRITICAL: "tự khóa/hủy ví" = tai_khoan_vi (policy question)
            if any(kw in text_lower for kw in ["tu khoa", "tự khóa", "tu huy", "tự hủy", "co the", "có thể"]):
                return "tai_khoan_vi", 0.92
            return "ung_dung", 0.9
        
        if any(kw in text_lower for kw in ["voucher", "khuyến mại", "ưu đãi", "quà tặng"]):
            return "ctkm_voucher", 0.9

        if any(kw in text_lower for kw in ["vé máy bay", "hãng bay"]):
            return "ve_may_bay", 0.9
        
        if any(kw in text_lower for kw in ["vé tàu", "tau hoa"]):
            return "ve_tau", 0.9
        
        if any(kw in text_lower for kw in ["vé vui chơi", "ve vui choi", "vé tham quan", "ve tham quan", "khu vui chơi", "khu vui choi"]):
            return "ve_tham_quan", 0.9
            
        if "vnpt pay" in text_lower:
            if any(kw in text_lower for kw in ["rút tiền", "nạp tiền", "chuyển tiền"]):
                return None, 0.4 # Ambiguous
            return "vnpt_pay", 0.85
            
        if "ứng dụng" in text_lower or "app" in text_lower:
            return "ung_dung", 0.8
            
        return None, 0.0

    def _rule_based_problem_extraction(self, text: str, service: Optional[str]) -> Tuple[Optional[str], float]:
        """
        Extract problem using keyword patterns. Order is critical.
        """
        text_lower = _norm_text(text)
        
        # High-priority problem patterns (tra_soat vs that_bai vs huong_dan)
        # CRITICAL: Policy/comparison questions FIRST (highest priority)
        # "có điểm gì hơn", "khác gì", "so với" = chinh_sach (comparison)
        if any(kw in text_lower for kw in ["co diem gi hon", "có điểm gì hơn", "khac gi", "khác gì", "so voi", "so với", "hon gi", "hơn gì"]):
            return "chinh_sach", 0.96
        
        # "bao nhiêu tài khoản", "bao nhiêu ví" = chinh_sach (quantity limits)
        if any(kw in text_lower for kw in ["bao nhieu", "bao nhiêu"]):
            if any(kw in text_lower for kw in ["tai khoan", "tài khoản", "vi", "ví", "cccd", "cmnd"]):
                return "chinh_sach", 0.95
        
        # "muốn kiểm tra", "muốn xem" = huong_dan (how to check)
        if any(kw in text_lower for kw in ["muon kiem tra", "muốn kiểm tra", "muon xem", "muốn xem"]):
            return "huong_dan", 0.94
        
        # CRITICAL: MyTV/gói cước purchase failures
        if any(kw in text_lower for kw in ["mua goi", "mua gói"]) and any(kw in text_lower for kw in ["that bai", "thất bại", "bi loi", "bị lỗi"]):
            return "that_bai", 0.93
        if any(kw in text_lower for kw in ["mua goi cuoc", "mua gói cước", "mua truyen hinh", "mua truyền hình"]):
            if any(kw in text_lower for kw in ["thanh cong nhung", "thành công nhưng", "tru tien", "trừ tiền"]):
                if any(kw in text_lower for kw in ["khong su dung", "không sử dụng", "chua su dung", "chưa sử dụng", "khong hoat dong", "không hoạt động"]):
                    return "tra_soat", 0.94
        if "khong tim duoc hoa don" in text_lower or "khong thay hoa don" in text_lower:
            return "that_bai", 0.92
        if "khong du so du" in text_lower or "tai khoan khong du" in text_lower or "so du khong du" in text_lower:
            return "loi_so_du", 0.95
        if "ma the cao bi loi" in text_lower or "ma the bi loi" in text_lower:
            return "tra_soat", 0.94
        if "da thanh toan" in text_lower and ("van" in text_lower or "nhung" in text_lower) and ("nhac no" in text_lower or "chua gach no" in text_lower):
            return "tra_soat", 0.95
        if "giao dich hien trang thai" in text_lower or "trang thai dang xu ly" in text_lower:
            return "tra_soat", 0.92
        if "thanh cong nhung" in text_lower and ("khong nhan" in text_lower or "chua nhan" in text_lower):
            return "tra_soat", 0.93
        if "co the" in text_lower and ("thanh toan" in text_lower or "mua" in text_lower or "dat ve" in text_lower) and ("khu vuc nao" in text_lower or "o dau" in text_lower or "hang bay nao" in text_lower):
            return "chinh_sach", 0.93
        if "sau khi dat ve" in text_lower and "su dung" in text_lower:
            return "huong_dan", 0.93
        if "khong su dung duoc voucher" in text_lower or "voucher khong dung" in text_lower:
            return "that_bai", 0.93
        if "khong nhan duoc uu dai" in text_lower or "chua nhan uu dai" in text_lower:
            return "tra_soat", 0.92
        if "mat voucher" in text_lower or "bi mat voucher" in text_lower:
            return "tra_soat", 0.93

        # --- Highest Priority & Specific ---
        # CRITICAL: Display issues - success but not showing (MUST be before tra_soat patterns)
        if any(kw in text_lower for kw in ["thanh cong nhung", "thành công nhưng", "thanh cong ma", "thành công mà"]):
            if any(kw in text_lower for kw in ["khong duoc cong", "không được cộng", "chua duoc cong", "chưa được cộng",
                                                 "so du khong duoc cong", "số dư không được cộng",
                                                 "kiem tra so du khong", "kiểm tra số dư không",
                                                 "chua cap nhat", "chưa cập nhật", "khong cap nhat", "không cập nhật"]):
                return "su_co_hien_thi", 0.98
        
        # CRITICAL: Money deducted + failed/pending ALWAYS maps to tra_soat (investigation)
        if any(kw in text_lower for kw in ["bị trừ tiền", "đã trừ tiền", "ngân hàng đã trừ", "ngân hàng trừ", "ví đã bị trừ", "tài khoản đã bị trừ"]):
            return "tra_soat", 0.98
        
        # CRITICAL: Reward/cashback not received should map to tra_soat
        if any(kw in text_lower for kw in ["không nhận được ưu đãi", "khong nhan duoc uu dai", "không nhận được hoàn tiền", "chưa nhận ưu đãi", "chua nhan uu dai"]):
            return "tra_soat", 0.97
        
        if any(kw in text_lower for kw in ["không đủ số dư", "không đủ tiền", "tài khoản không đủ"]):
            return "loi_so_du", 0.98
        
        if any(kw in text_lower for kw in ["thông tin thẻ không hợp lệ", "thẻ đã được liên kết với ví khác"]):
             return "loi_lien_ket", 0.98

        if "lỗi liên kết" in text_lower or "không liên kết được" in text_lower or "khong lien ket duoc" in text_lower:
            # CRITICAL: "tại sao không liên kết được với thẻ quốc tế" = chinh_sach (policy)
            if any(kw in text_lower for kw in ["tai sao", "tại sao", "vi sao", "vì sao"]):
                if any(kw in text_lower for kw in ["the quoc te", "thẻ quốc tế", "visa", "mastercard"]):
                    return "chinh_sach", 0.92
                # "tại sao không liên kết được" without specific error = policy question
                return "chinh_sach", 0.88
            return "loi_lien_ket", 0.97

        # Explicit investigation keywords
        if any(kw in text_lower for kw in ["tra soát", "khiếu nại", "hoàn tiền", "hoàn trả", "chuyển nhầm", "chuyển sai"]):
            return "tra_soat", 0.96
        
        # Service not received patterns (also investigation)
        if any(kw in text_lower for kw in ["chưa nhận được", "chưa gạch nợ", "không nhận được thông tin", "mã thẻ cào bị lỗi", "chưa nhận tiền", "không nhận tiền"]):
            return "tra_soat", 0.95

        # Voucher issues should map to that_bai (failed to use/redeem)
        # CRITICAL: "voucher bị khóa" should be tra_soat (investigation)
        if any(kw in text_lower for kw in ["voucher bi khoa", "voucher bị khóa", "ma giam gia bi khoa", "mã giảm giá bị khóa"]):
            return "tra_soat", 0.94
        if any(kw in text_lower for kw in ["voucher không dùng được", "voucher khong dung duoc", "không sử dụng được voucher", "khong su dung duoc voucher", "mất voucher", "mat voucher", "voucher không hoạt động", "voucher khong hoat dong"]):
            return "that_bai", 0.95

        # CRITICAL: Unsupported bank questions - "tại sao không ... ngân hàng X"
        # BUT: "tại sao đã ... nhưng vẫn ..." should be tra_soat or chinh_sach
        if any(kw in text_lower for kw in ["tai sao da", "tại sao đã"]) and any(kw in text_lower for kw in ["nhung", "nhưng", "van", "vẫn"]):
            # "tại sao đã ... nhưng vẫn ..." could be tra_soat or chinh_sach
            # If about account status, it's chinh_sach
            if any(kw in text_lower for kw in ["chua kich hoat", "chưa kích hoạt", "chua active", "chưa active", "chua hieu luc", "chưa hiệu lực"]):
                return "chinh_sach", 0.92
            # Otherwise investigation
            return "tra_soat", 0.90
        
        if any(kw in text_lower for kw in ["tai sao", "tại sao", "vi sao", "vì sao"]) and any(kw in text_lower for kw in ["khong", "không"]):
            if any(kw in text_lower for kw in ["ngan hang", "ngân hàng", "techcombank", "vietcombank", "bidv", "vcb", "acb"]):
                return "khong_ho_tro_ngan_hang", 0.95
            # "tại sao ... không thấy mục" = policy/feature availability
            if any(kw in text_lower for kw in ["khong thay", "không thấy", "khong co", "không có"]):
                return "chinh_sach", 0.90
        
        if any(kw in text_lower for kw in ["không hỗ trợ", "không dùng được với"]) and "ngân hàng" in text_lower:
            return "khong_ho_tro_ngan_hang", 0.95

        # --- Medium Priority ---
        # CRITICAL: "thông tin thẻ/tài khoản không hợp lệ" in nạp tiền context = loi_xac_thuc
        # But in liên kết context = loi_lien_ket
        if any(kw in text_lower for kw in ["không hợp lệ", "khong hop le", "thông tin sai", "thong tin sai", "sai thông tin", "sai thong tin", "không trùng khớp", "khong trung khop"]):
            if "lien ket" in text_lower or "liên kết" in text_lower:
                return "loi_lien_ket", 0.90
            if "nap tien" in text_lower or "nạp tiền" in text_lower:
                return "loi_xac_thuc", 0.88
            # Device-specific errors = loi_thiet_bi
            if any(kw in text_lower for kw in ["thiet bi", "thiết bị", "nfc", "camera", "may khong ho tro", "máy không hỗ trợ"]):
                return "loi_thiet_bi", 0.92
            if any(kw in text_lower for kw in ["khuôn mặt", "khuon mat", "cccd", "ngày sinh", "ngay sinh", "họ tên", "ho ten"]):
                return "loi_xac_thuc", 0.85
            return "loi_xac_thuc", 0.80

        # CRITICAL: Policy questions about conditions/requirements BEFORE huong_dan
        # "điều kiện để", "yêu cầu để" = chinh_sach (what conditions needed)
        if any(kw in text_lower for kw in ["dieu kien de", "điều kiện để", "yeu cau de", "yêu cầu để", "dieu kien", "điều kiện"]):
            if any(kw in text_lower for kw in ["thanh cong", "thành công", "duoc", "được", "la gi", "là gì"]):
                return "chinh_sach", 0.95
        
        if any(kw in text_lower for kw in ["hướng dẫn", "làm sao", "làm thế nào", "cách để", "cần làm gì", "vào mục nào", "ở đâu", "cách mua", "thao tác", "bước"]):
            if "tra soát" in text_lower or "khiếu nại" in text_lower:
                return "tra_soat", 0.8
            # CRITICAL: "tôi có thể ... ở đâu" = huong_dan (where to do X)
            if any(kw in text_lower for kw in ["o dau", "ở đâu", "mua ve o dau", "mua ... o dau"]):
                return "huong_dan", 0.93
            # If user mentions an error, it's less likely a pure "how-to" question
            if "lỗi" in text_lower or "vấn đề" in text_lower or "sự cố" in text_lower:
                return None, 0.4 # Lower confidence for huong_dan if error words are present
            return "huong_dan", 0.9

        # "Có thể ... nào" patterns (which ones) -> chinh_sach (policy/scope questions)
        # CRITICAL: "đặt vé ... hãng bay nào" = chinh_sach (which airlines supported)
        if any(kw in text_lower for kw in ["có thể", "co the"]):
            if any(kw in text_lower for kw in ["những nào", "nhung nao", "hãng nào", "hang nao", "ngân hàng nào", "ngan hang nao", "dịch vụ nào", "dich vu nao", "khu vực nào", "khu vuc nao"]):
                return "chinh_sach", 0.93
        # Direct "... nào" questions about supported services
        if any(kw in text_lower for kw in ["hang bay nao", "hãng bay nào", "khu vuc nao", "khu vực nào", "ngan hang nao", "ngân hàng nào"]):
            return "chinh_sach", 0.92
        
        # CRITICAL: "tổng giá vé ... đã bao gồm thuế/phí chưa" = chinh_sach
        if "tong gia ve" in text_lower or "tổng giá vé" in text_lower:
            if any(kw in text_lower for kw in ["bao gom", "bao gồm", "da bao", "đã bao", "chua", "chưa", "thue", "thuế", "phi", "phí"]):
                return "chinh_sach", 0.94
        
        if any(kw in text_lower for kw in ["chính sách", "quy định", "điều kiện", "có được không", "có mất phí không", "phí là bao nhiêu", "chi phí"]):
            if "hạn mức" in text_lower:
                return "chinh_sach", 0.85  # Policy about limits (not han_muc_phi)
            return "chinh_sach", 0.9
            
        if any(kw in text_lower for kw in ["hạn mức", "tối đa", "tối thiểu"]):
            return "chinh_sach", 0.85  # Policy about limits (was han_muc_phi)

        # --- General Errors & States ---
        if any(kw in text_lower for kw in ["lỗi", "bị lỗi", "gặp sự cố", "vấn đề"]):
            # Context-specific error mapping
            # CRITICAL: "thiết bị không hỗ trợ NFC" = loi_thiet_bi (device error)
            if any(kw in text_lower for kw in ["thiet bi", "thiết bị"]) and any(kw in text_lower for kw in ["khong ho tro", "không hỗ trợ", "nfc", "NFC"]):
                return "loi_thiet_bi", 0.93
            if "lien ket" in text_lower or "liên kết" in text_lower:
                return "loi_lien_ket", 0.9
            if "thanh toan" in text_lower or "thanh toán" in text_lower:
                return "giao_dich_that_bai", 0.85
            if any(kw in text_lower for kw in ["xác thực", "xac thuc", "sinh trắc học", "sinh trac hoc", "khuôn mặt", "khuon mat", "cccd", "định danh", "dinh danh"]):
                return "loi_xac_thuc", 0.88
            if "thẻ cào" in text_lower or "the cao" in text_lower or "mã thẻ" in text_lower or "ma the" in text_lower:
                return "tra_soat", 0.8
            # Check if it's actually an investigation case (money deducted)
            if any(kw in text_lower for kw in ["bị trừ", "bi tru", "đã trừ", "da tru", "ngân hàng trừ", "ngan hang tru"]):
                return "tra_soat", 0.90
            # Generic error - return None to let taxonomy handle it properly
            return None, 0.5
            
        # that_bai: Only if NO investigation signals (money deducted, not received, etc.)
        if any(kw in text_lower for kw in ["thất bại", "không thành công"]):
            # Check if it's actually investigation (money deducted + failed)
            if any(kw in text_lower for kw in ["bị trừ", "đã trừ", "ngân hàng trừ", "chưa nhận", "không nhận được"]):
                return "tra_soat", 0.90
            # Pure failure without money issue
            return "that_bai", 0.85
            
        if any(kw in text_lower for kw in ["trừ 2 lần", "trừ hai lần", "trừ tiền rồi"]):
            return "tra_soat", 0.95
            
        if any(kw in text_lower for kw in ["thành công", "hoàn thành"]):
            return "thanh_cong", 0.7
            
        if any(kw in text_lower for kw in ["đang xử lý", "chờ xử lý"]):
            return "dang_xu_ly", 0.7
            
        return None, 0.0

    def _rule_based_state_extraction(self, text: str) -> Tuple[Optional[str], float]:
        """
        Extract transaction state from UI status description.
        Returns: (state_key, confidence)
        
        CRITICAL: Only extract state when user is describing an actual transaction,
        not when asking how-to questions.
        """
        text_lower = _norm_text(text)
        
        # Don't extract state for how-to questions
        if any(kw in text_lower for kw in ['lam sao', 'the nao', 'huong dan', 'cach thuc', 'vao muc nao']):
            return 'unknown', 0.3
        
        # State requires transaction context
        if not any(kw in text_lower for kw in ['giao dich', 'trang thai', 'he thong bao', 'bao', 'hien thi']):
            return 'unknown', 0.3
        
        # State patterns (very specific to avoid false positives)
        state_patterns = {
            'pending': [
                'dang xu ly', 'dang cho xu ly', 'trang thai dang xu ly',
                'bao dang xu ly', 'hien thi dang xu ly', 'giao dich dang xu ly',
            ],
            'failed': [
                'that bai', 'khong thanh cong', 'giao dich that bai',
                'trang thai that bai', 'bao that bai', 'hien thi that bai',
            ],
            'success': [
                'thanh cong', 'da thanh cong', 'hoan thanh',
                'trang thai thanh cong', 'hien thi thanh cong', 'giao dich thanh cong',
            ],
        }
        
        for state_key, patterns in state_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return state_key, 0.85
        
        return 'unknown', 0.3  # Default to unknown

    def _rule_based_outcome_extraction(self, text: str) -> Tuple[Optional[str], float]:
        """
        Extract user symptom/outcome.
        Returns: (outcome_key, confidence)
        """
        text_lower = _norm_text(text)
        
        outcome_patterns = {
            'money_not_received': [
                'chua nhan tien', 'chua nhan duoc tien', 'tien chua ve',
                'chua duoc cong', 'khong nhan duoc tien', 'khong nhan tien',
            ],
            'money_deducted': [
                'bi tru tien', 'da tru tien', 'ngan hang da tru', 'ngan hang tru',
                'vi da bi tru', 'tru 2 lan', 'bi tru 2 lan',
            ],
            'need_instruction': [
                'lam sao', 'the nao', 'huong dan', 'cach thuc', 'phai lam gi',
                'can lam gi', 'thao tac nhu the nao',
            ],
        }
        
        for outcome_key, patterns in outcome_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    confidence = 0.85
                    return outcome_key, confidence
        
        return 'unknown', 0.5

    def _extract_entities_ner(self, text: str) -> Dict[str, Any]:
        """
        Extract entities using regex and keyword matching (lightweight NER).
        Returns: dict with bank, amount, time, error_message
        """
        entities = {
            'bank': None,
            'amount': self._extract_amount(text),
            'time': self._extract_time(text),
            'error_message': self._extract_error_message(text),
        }
        
        # Extract bank using normalizer
        bank_pair = UnifiedNormalizer.normalize_bank_pair(text)
        if bank_pair:
            entities['bank'] = bank_pair[1]  # Display name
        
        return entities

    @staticmethod
    def _detect_contradictions(transaction_status: Optional[str], problem_type: Optional[str]) -> List[str]:
        if not transaction_status or not problem_type:
            return []
        if CONTRADICTIONS.get((transaction_status, problem_type)):
            return [f"transaction_status={transaction_status} mâu thuẫn với problem_type={problem_type}"]
        return []

    def extract_slots_from_narrative(self, narrative: str) -> CaseSlots:
        return self.extract_slots(narrative)
  
    def _validate_taxonomy_compliance(self, slots: CaseSlots) -> CaseSlots:
        """
        CRITICAL VALIDATOR: Ensure all slot_keys are valid per taxonomy.
        Maps unknown keys to 'unknown' or valid alternatives.
        This prevents "Expected qa_id NOT in Neo4j candidates" errors.
        """
        # Validate service
        if slots.service and slots.service not in self.service_taxonomy.VALID_SERVICES:
            print(f"⚠️  TAXONOMY VIOLATION: service='{slots.service}' not in taxonomy, setting to 'unknown'")
            slots.service = None
            slots.service_confidence = 0.0
        
        # Validate problem_type
        valid_problems = {
            'huong_dan', 'chinh_sach', 'loi_xac_thuc', 'loi_lien_ket', 'loi_han_muc', 
            'loi_so_du', 'tra_soat', 'that_bai', 'su_co_hien_thi', 'khac', 'unknown',
            'khong_ho_tro_ngan_hang', 'loi_thiet_bi', 'loi_voucher'
        }
        if slots.problem_type and slots.problem_type not in valid_problems:
            print(f"⚠️  TAXONOMY VIOLATION: problem_type='{slots.problem_type}' not in taxonomy, mapping to 'unknown'")
            # Try to map common mistakes
            if slots.problem_type in ['han_muc_phi', 'dieu_kien']:
                slots.problem_type = 'chinh_sach'
                print(f"   → Auto-mapped to 'chinh_sach'")
            else:
                slots.problem_type = 'unknown'
            slots.problem_confidence = 0.3
        
        # Validate state
        valid_states = {'pending', 'failed', 'success', 'unknown'}
        if slots.state and slots.state not in valid_states:
            print(f"⚠️  TAXONOMY VIOLATION: state='{slots.state}' not in taxonomy, setting to 'unknown'")
            slots.state = 'unknown'
        
        # Validate outcome
        valid_outcomes = {'money_not_received', 'money_deducted', 'need_instruction', 'unknown'}
        if slots.outcome and slots.outcome not in valid_outcomes:
            print(f"⚠️  TAXONOMY VIOLATION: outcome='{slots.outcome}' not in taxonomy, setting to 'unknown'")
            slots.outcome = 'unknown'
        
        return slots
  
    def extract_slots(self, narrative: str) -> CaseSlots:
        """
        Hybrid slot extraction: Rule-based first (fast, accurate), GPT only for ambiguous cases.
        
        Strategy:
        1. Check for out-of-scope keywords FIRST (before any extraction)
        2. Try rule-based extraction (80%+ cases, < 10ms)
        3. If confidence is low or critical slots are missing, use GPT as a fallback.
        4. Apply post-processing to fix common mistakes.
        5. VALIDATE taxonomy compliance (prevent slot_key mismatches)
        """
        
        # ===== STEP 0: OUT-OF-SCOPE DETECTION (BEFORE EXTRACTION) =====
        # Detect services NOT supported by VNPT Money
        out_of_scope_patterns = [
            # Food delivery (flexible patterns)
            ('do an', ['goi', 'dat', 'giao', 'mua']),  # gọi/đặt/giao/mua đồ ăn
            ('grab food', []),
            ('shopee food', []),
            ('now', ['goi', 'dat']),  # gọi/đặt now
            
            # Ride-hailing
            ('xe', ['goi', 'dat', 'grab', 'book']),  # gọi/đặt xe, grab xe
            ('grab', ['xe', 'car', 'bike']),
            
            # E-commerce
            ('shopee', []),
            ('lazada', []),
            ('tiki', []),
            ('sendo', []),
            
            # Hotels & Travel
            ('khach san', ['dat', 'book']),
            ('phong', ['dat']),
            ('booking', []),
            ('agoda', []),
            
            # Restaurants
            ('nha hang', ['dat ban', 'book']),
        ]
        
        narrative_norm = _norm_text(narrative)
        
        # Check patterns
        for main_keyword, modifiers in out_of_scope_patterns:
            if main_keyword in narrative_norm:
                # If no modifiers, direct match
                if not modifiers:
                    return CaseSlots(
                        raw_narrative=narrative,
                        confidence_score=0.0,
                        service_confidence=0.0,
                        problem_confidence=0.0,
                        inference_evidence={
                            'method': 'out_of_scope_detection',
                            'keyword': main_keyword,
                            'reason': f'Service not supported by VNPT Money: {main_keyword}'
                        }
                    )
                # Check if any modifier is present
                for modifier in modifiers:
                    if modifier in narrative_norm:
                        return CaseSlots(
                            raw_narrative=narrative,
                            confidence_score=0.0,
                            service_confidence=0.0,
                            problem_confidence=0.0,
                            inference_evidence={
                                'method': 'out_of_scope_detection',
                                'keyword': f'{modifier} {main_keyword}',
                                'reason': f'Service not supported by VNPT Money: {modifier} {main_keyword}'
                            }
                        )
        
        # ===== STEP 1: RULE-BASED EXTRACTION (FAST PATH) =====
        # Use UnifiedNormalizer as single source of truth for service
        # Rule-based extraction is replaced by UnifiedNormalizer's comprehensive inference
        service_slot, service_conf, service_evidence = UnifiedNormalizer.infer_service_from_text(narrative)
        
        problem_slot, problem_conf = self._rule_based_problem_extraction(narrative, service_slot)
        state_slot, state_conf = self._rule_based_state_extraction(narrative)
        outcome_slot, outcome_conf = self._rule_based_outcome_extraction(narrative)
        entities = self._extract_entities_ner(narrative)
        
        # Calculate overall confidence from rule-based extraction
        rule_based_confidence = (service_conf + problem_conf) / 2 if service_conf > 0 else 0.0
        
        # ===== STEP 2: DETERMINE IF GPT FALLBACK IS NEEDED =====
        use_gpt = False
        # Trigger GPT if service or problem is unknown, or confidence is low
        if not service_slot or not problem_slot or rule_based_confidence < 0.85:
            use_gpt = True
        
        gpt_result = {}
        if use_gpt:
            # Use GPT for ambiguous cases
            gpt_result = self._gpt_extraction(narrative)
            
            # Merge: prefer GPT for missing/low-confidence slots
            gpt_service = gpt_result.get('service')
            gpt_service_conf = gpt_result.get('service_confidence', 0.0)
            if gpt_service and (not service_slot or service_conf < gpt_service_conf):
                service_slot = gpt_service
                service_conf = gpt_service_conf

            gpt_problem = gpt_result.get('problem_type')
            gpt_problem_conf = gpt_result.get('problem_confidence', 0.0)
            if gpt_problem and (not problem_slot or problem_conf < gpt_problem_conf):
                problem_slot = gpt_problem
                problem_conf = gpt_problem_conf

            # Use GPT for state/outcome if not detected by rules
            if state_slot == 'unknown' and gpt_result.get('issue_state') != 'unknown':
                state_slot = gpt_result.get('issue_state', 'unknown')
                state_conf = gpt_result.get('state_confidence', 0.5)
            
            if outcome_slot == 'unknown' and gpt_result.get('issue_outcome') != 'unknown':
                outcome_slot = gpt_result.get('issue_outcome', 'unknown')
                outcome_conf = gpt_result.get('outcome_confidence', 0.5)

            # Use GPT's entities if NER missed them
            if not entities['bank'] and gpt_result.get('bank'):
                entities['bank'] = gpt_result.get('bank')
            if not entities['amount'] and gpt_result.get('amount'):
                entities['amount'] = gpt_result.get('amount')
            if not entities['time'] and gpt_result.get('time'):
                entities['time'] = gpt_result.get('time')
            if not entities['error_message'] and gpt_result.get('error_message'):
                entities['error_message'] = gpt_result.get('error_message')

        # ===== STEP 3: POST-PROCESSING (FIX COMMON MISTAKES) =====
        service_slot, problem_slot, state_slot, outcome_slot = self._post_process_slots(
            narrative, service_slot, problem_slot, state_slot, outcome_slot
        )
        
        # ===== STEP 4: NORMALIZE AND VALIDATE =====
        # Normalize using taxonomy
        service_id = ServiceTaxonomy.from_slot_key(service_slot) if service_slot else None
        problem_id = ProblemTypeTaxonomy.from_slot_key(problem_slot) if problem_slot else None
        state_id = StateTaxonomy.from_slot_key(state_slot) if state_slot else None
        outcome_id = OutcomeTaxonomy.from_slot_key(outcome_slot) if outcome_slot else None
        
        # Normalize bank using BankTaxonomy
        bank_slot = None
        bank_id = None
        if entities['bank']:
            bank_pair = UnifiedNormalizer.normalize_bank_pair(entities['bank'])
            if bank_pair:
                bank_id = bank_pair[0]
                bank_slot = bank_pair[1]
        
        # Build CaseSlots
        slots = CaseSlots(
            service=service_slot,
            service_id=service_id,
            problem_type=problem_slot,
            problem_id=problem_id,
            state=state_slot if state_slot != 'unknown' else None,
            state_id=state_id if state_id else None,
            outcome=outcome_slot if outcome_slot != 'unknown' else None,
            outcome_id=outcome_id if outcome_id else None,
            bank=bank_slot,
            bank_id=bank_id,
            amount=entities.get('amount'),
            time=entities.get('time'),
            error_message=entities.get('error_message'),
            raw_narrative=narrative,
            confidence_score=(service_conf + problem_conf) / 2,
            service_confidence=service_conf,
            problem_confidence=problem_conf,
            inference_evidence={
                'method': 'gpt_fallback' if use_gpt else 'rule_based',
                'service_source': 'unified_normalizer',
                'service_evidence': service_evidence,
                'rule_problem_conf': self._rule_based_problem_extraction(narrative, service_slot)[1],
                'gpt_service_conf': gpt_result.get('service_confidence', 0.0),
                'gpt_problem_conf': gpt_result.get('problem_confidence', 0.0),
            }
        )
        
        # Validate and detect issues
        slots.missing_slots = self.identify_missing_slots(slots)
        slots.contradictions = self._detect_contradictions(slots.state, slots.problem_type)
        
        # ===== STEP 5: TAXONOMY COMPLIANCE VALIDATION (CRITICAL) =====
        slots = self._validate_taxonomy_compliance(slots)
        
        return slots

    def _gpt_extraction(self, narrative: str) -> Dict[str, Any]:
        """
        Call GPT for slot extraction (fallback only).
        Returns simplified result dict for merging.
        """
        prompt = self._build_gpt_prompt(narrative)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Bạn là hệ thống trích xuất thông tin. Luôn trả về đúng JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=600,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            
            # Normalize GPT output
            service_raw = result.get("service")
            service_norm = UnifiedNormalizer.normalize_service(service_raw) if service_raw else None
            
            # Get confidence
            conf_assessment = result.get("confidence_assessment", {})
            service_confidence = float(conf_assessment.get("service_confidence", 0.5))
            problem_confidence = float(conf_assessment.get("problem_confidence", 0.5))
            state_confidence = float(conf_assessment.get("state_confidence", 0.5))
            outcome_confidence = float(conf_assessment.get("outcome_confidence", 0.5))
            
            return {
                'service': service_norm,
                'service_confidence': service_confidence,
                'problem_type': result.get("problem_type"),
                'problem_confidence': problem_confidence,
                'issue_state': result.get("issue_state", "unknown"),
                'state_confidence': state_confidence,
                'issue_outcome': result.get("issue_outcome", "unknown"),
                'outcome_confidence': outcome_confidence,
                'bank': result.get("bank"),
                'amount': result.get("amount"),
                'time': result.get("time"),
                'error_message': result.get("error_message"),
            }

        except Exception as e:
            print(f"[ERROR] GPT extraction failed: {e}")
            return {}

    def _post_process_slots(
        self, 
        text: str, 
        service: Optional[str],
        problem: Optional[str],
        state: Optional[str],
        outcome: Optional[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Post-processing to fix common slot extraction mistakes.
        
        Common mistakes:
        - "nạp tiền điện thoại" misclassified as nap_tien (should be vien_thong)
        - "chưa nhận tiền" + "thành công" contradiction
        - State inferred when user only asks how-to
        """
        text_lower = _norm_text(text)
        
        # SERVICE IS NOW CONTROLLED BY UnifiedNormalizer - DO NOT OVERRIDE HERE
        # Fix 1-3 REMOVED: Service normalization is handled by single source of truth
        
        # Fix 4: If user asks "làm sao", "như thế nào" → problem should be huong_dan
        # BUT: ONLY if no error keywords present AND no specific problem detected
        # SPECIAL: "vào mục nào" is always huong_dan (navigation question)
        # SPECIAL: "xem ... như thế nào", "kiểm tra ... như thế nào" = huong_dan
        def has_error_keywords(txt):
            return any(kw in txt for kw in ['loi', 'van de', 'su co', 'that bai', 'khong duoc', 'bi tru', 'chua nhan', 'tra soat'])
        
        if 'vao muc nao' in text_lower or 'vào mục nào' in text_lower or 'vao dau' in text_lower or 'o dau' in text_lower:
            if not has_error_keywords(text_lower):
                problem = 'huong_dan'
        elif any(kw in text_lower for kw in ['xem', 'kiem tra', 'kiểm tra']) and any(kw in text_lower for kw in ['nhu the nao', 'như thế nào', 'the nao']):
            if not has_error_keywords(text_lower):
                problem = 'huong_dan'
        elif problem is None or problem == 'unknown':
            if any(kw in text_lower for kw in ['lam sao', 'the nao', 'huong dan', 'hướng dẫn', 'chi tiet', 'thu tuc', 'cach de']) and not has_error_keywords(text_lower):
                problem = 'huong_dan'
        
        # Fix 5: Fee/policy questions should typically be chinh_sach
        # CRITICAL: "có mất phí hay không" should be chinh_sach, not han_muc_phi
        # "hạn mức" questions without specific "vượt" context are also chinh_sach
        if problem is None or problem == 'unknown':
            if any(kw in text_lower for kw in ['co mat phi', 'có mất phí', 'mat phi hay khong', 'mất phí hay không', 'mat phi khong', 'mất phí không', 'co phi khong', 'có phí không']):
                problem = 'chinh_sach'  # Policy question about fees
            elif any(kw in text_lower for kw in ['han muc', 'hạn mức']):
                # If asking "what is the limit", it's chinh_sach; if "exceeded limit error", handled elsewhere
                if 'qua' not in text_lower and 'vuot' not in text_lower and 'vượt' not in text_lower:
                    problem = 'chinh_sach'
        
        # Fix 6: "điều kiện", "yêu cầu", "có thể ... hay không" → problem should be chinh_sach
        # CRITICAL: Policy questions have HIGHEST priority in post-processing
        # ALWAYS override huong_dan/unknown if policy keywords present
        if problem is None or problem == 'unknown' or problem == 'huong_dan':
            # "có thể ... không" / "có thể ... hay không" = policy question
            if any(kw in text_lower for kw in ['co the', 'có thể']):
                if any(kw in text_lower for kw in ['hay khong', 'hay không', 'duoc khong', 'được không', 'duoc bao nhieu', 'được bao nhiêu']):
                    problem = 'chinh_sach'
                # "có thể ... hộ" (can do for someone else) = policy
                elif 'ho' in text_lower or 'hộ' in text_lower:
                    problem = 'chinh_sach'
            # "điều kiện", "yêu cầu" = policy
            elif any(kw in text_lower for kw in ['dieu kien', 'điều kiện', 'yeu cau', 'yêu cầu']):
                problem = 'chinh_sach'
            # Comparison questions = policy
            elif any(kw in text_lower for kw in ['co diem gi', 'có điểm gì', 'khac gi', 'khác gì', 'hon', 'hơn']):
                problem = 'chinh_sach'
        
        # CRITICAL OVERRIDE: "tại sao ... không liên kết được" should be chinh_sach (policy)
        # This overrides loi_lien_ket when it's a "why" question
        if problem == 'loi_lien_ket':
            if any(kw in text_lower for kw in ['tai sao', 'tại sao', 'vi sao', 'vì sao']):
                problem = 'chinh_sach'
        
        # CRITICAL OVERRIDE: "hệ thống báo ... thiết bị" should be loi_thiet_bi not chinh_sach
        if problem == 'chinh_sach' and any(kw in text_lower for kw in ['he thong bao', 'hệ thống báo', 'bao loi', 'báo lỗi']):
            if any(kw in text_lower for kw in ['thiet bi', 'thiết bị', 'nfc', 'khong ho tro', 'không hỗ trợ']):
                problem = 'loi_thiet_bi'
        
        # Fix 7: If user mentions state explicitly, use it; otherwise default to unknown
        state_keywords = ['dang xu ly', 'that bai', 'thanh cong', 'trang thai', 'hien thi', 'bao', 'loi']
        if state and state != 'unknown':
            # Keep state only if explicitly mentioned
            if not any(kw in text_lower for kw in state_keywords):
                state = 'unknown'
        
        # Fix 8: Contradiction detection: "thành công" but "chưa nhận tiền"
        if state == 'success' and outcome == 'money_not_received':
            # This is common: transaction shows success but money not received → investigation needed
            pass
        
        # Fix 9: "failed" + "bị trừ" should map to tra_soat
        if state == 'failed' and any(kw in text_lower for kw in ['bi tru', 'bitru', 'da tru', 'datru', 'ngan hang tru', 'nganhang tru']):
            problem = 'tra_soat'
        
        # Fix 10: "đang xử lý" + "bị trừ" should also map to tra_soat
        if state == 'pending' and any(kw in text_lower for kw in ['bi tru', 'bitru', 'da tru', 'datru', 'ngan hang tru', 'nganhang tru']):
            problem = 'tra_soat'
        
        # Fix 10b: "thất bại" + "bị trừ" should map to tra_soat (even without explicit state)
        if any(kw in text_lower for kw in ['that bai', 'thatbai', 'bi loi', 'biloi']) and any(kw in text_lower for kw in ['bi tru', 'bitru', 'da tru', 'datru', 'ngan hang tru']):
            problem = 'tra_soat'
        
        # Fix 10c: "chuyển nhầm", "sai tài khoản" should always be tra_soat
        if any(kw in text_lower for kw in ['chuyen tien nham', 'tien nham', 'chuyen nham', 'chuyennham', 'nham tai khoan', 'sai tai khoan', 'chuyen sai']):
            problem = 'tra_soat'
        
        # Fix 10d: "đã trừ tiền" + "vẫn còn nợ" / "chưa gạch nợ" should be tra_soat
        if any(kw in text_lower for kw in ['da tru tien', 'bi tru tien', 'tru tien']) and any(kw in text_lower for kw in ['van con no', 'chua gach no', 'khong gach no', 'chua duoc gach no']):
            problem = 'tra_soat'
        
        # Fix 10e: "đang chờ xử lý" or "giao dịch đang chờ" + "bị trừ tiền" = tra_soat
        if any(kw in text_lower for kw in ['dang cho xu ly', 'dang cho', 'dang xu ly', 'cho xu ly']) and any(kw in text_lower for kw in ['bi tru', 'da tru', 'tru tien']):
            problem = 'tra_soat'
        
        # Fix 11: "quá hạn mức" should be problem=vuot_han_muc OR loi_han_muc
        # ONLY if problem not already detected as more specific issue
        if problem is None or problem not in ['tra_soat', 'bi_tru_tien', 'su_co_hien_thi', 'loi_xac_thuc']:
            if any(kw in text_lower for kw in ['qua han muc', 'vuot han muc', 'vuot qua han muc', 'quahanmuc', 'vuothanmuc']):
                problem = 'loi_han_muc'
        
        # Fix 12: "không nhận được OTP" should be problem=khong_nhan_duoc_otp OR loi_xac_thuc
        if any(kw in text_lower for kw in ['khong nhan otp', 'khongnhanotp', 'khong nhan duoc otp', 'khongnhanduocotp', 'chua nhan otp', 'chuanhanotp', 'otp khong ve', 'otpkhongve']):
            problem = 'loi_xac_thuc'
        
        # Fix 13: "chưa đăng ký" patterns
        if any(kw in text_lower for kw in ['chua dang ky', 'chuadangky', 'chua mo dich vu', 'chua kich hoat', 'khong dang ky']):
            if 'sms' in text_lower or 'thanh toan truc tuyen' in text_lower:
                problem = 'loi_xac_thuc'
            else:
                problem = 'tai_khoan_chua_dang_ky'
        
        # Fix 14: "voucher" or "mã giảm giá" → voucher problem
        if any(kw in text_lower for kw in ['voucher', 'ma giam gia', 'uu dai']):
            if 'khong su dung' in text_lower or 'khong dung' in text_lower or 'het han' in text_lower:
                problem = 'voucher_khong_su_dung_duoc'
        
        # Fix 15: "hủy" or "hoàn" → cancellation/refund problem
        # BUT: Don't override navigation questions (huong_dan)
        if problem != 'huong_dan':  # Preserve navigation questions
            if any(kw in text_lower for kw in ['huy giao dich', 'hoan tien', 'hoan lai', 'muon huy']):
                problem = 'chinh_sach'  # Policy question about cancellation
        
        # Fix 16: Display issue patterns
        if any(kw in text_lower for kw in ['khong hien thi', 'khong thay', 'so du khong cap nhat', 'chua cap nhat']):
            if state == 'success':
                problem = 'su_co_hien_thi'
        
        return service, problem, state, outcome
    
    def _extract_amount(self, text: str) -> Optional[str]:
        """Extract amount from text using regex."""
        patterns = [
            r'(\d+(?:[.,]\d+)?)\s*(?:triệu|tr|m)',  # 5 triệu, 1.5 triệu
            r'(\d+(?:[.,]\d+)?)\s*(?:k|nghìn|ngàn)',  # 500k, 100 nghìn
            r'(\d+(?:[.,]\d+)?)\s*(?:đ|đồng|vnd)',  # 100000đ
            r'(\d{3,}(?:[.,]\d{3})*)',  # 1.000.000
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return None
    
    def _extract_time(self, text: str) -> Optional[str]:
        """Extract time/date from text using regex."""
        patterns = [
            r'(hôm\s+(?:nay|qua|kia))',
            r'(ngày\s+\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)',
            r'(\d{1,2}\s+giờ)',
            r'(tuần\s+(?:này|trước|sau))',
            r'(tháng\s+(?:này|trước|sau))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return None
    
    def _extract_error_message(self, text: str) -> Optional[str]:
        """Extract error message from text."""
        # Look for quoted text or specific error patterns
        patterns = [
            r'"([^"]+)"',  # Quoted text
            r"'([^']+)'",  # Single quoted
            r'lỗi\s+([^\.,;]+)',  # "lỗi X"
            r'báo\s+([^\.,;]+)',  # "báo X"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()
        
        return None
    
    def _build_gpt_prompt(self, narrative: str) -> str:
        return f"""
Trích xuất thông tin từ câu hỏi của user: "{narrative}"

CRITICAL: VNPT Money CHỈ hỗ trợ các dịch vụ tài chính sau:
- Nạp tiền vào ví (nap_tien)
- Rút tiền từ ví về ngân hàng (rut_tien)
- Chuyển tiền (chuyen_tien)
- Liên kết/Hủy liên kết ngân hàng (lien_ket_ngan_hang, huy_lien_ket_ngan_hang)
- Thanh toán hóa đơn điện/nước/internet (thanh_toan_hoa_don)
- Nạp điện thoại, mua mã thẻ (nap_tien_dien_thoai, vien_thong)
- Đặt vé máy bay/tàu/tham quan (ve_may_bay, ve_tau, ve_tham_quan)
- Siêu tích lũy (sieu_tich_luy)
- Xác thực định danh (xac_thuc_dinh_danh)
- Quản lý tài khoản ví (tai_khoan_vi, mobile_money)

Nếu user hỏi về dịch vụ KHÔNG có trong danh sách trên (ví dụ: gọi đồ ăn, gọi xe, mua sắm online, đặt phòng khách sạn), 
thì:
- service = null
- service_confidence = 0.0
- problem_type = "chinh_sach" (hỏi về chính sách hỗ trợ)

Trả về JSON với các trường sau (QUAN TRỌNG: các giá trị phải là snake_case và không dấu):
- service: Dịch vụ chính mà user đề cập (CHỈ các service trong danh sách trên, nếu không có thì null).
  Ví dụ: nap_tien, rut_tien, chuyen_tien, lien_ket_ngan_hang, thanh_toan_hoa_don, mobile_money, sieu_tich_luy, xac_thuc_dinh_danh, tai_khoan_vi, ve_may_bay, vien_thong.
  
  QUY TẮC:
  - "nạp tiền điện thoại", "mua mã thẻ", "thẻ cào" -> service: "nap_tien_dien_thoai"
  - "nạp tiền vào ví" -> service: "nap_tien"
  - "rút tiền từ ví" -> service: "rut_tien"
  - "chuyển tiền" -> service: "chuyen_tien"
  - "liên kết ngân hàng" -> service: "lien_ket_ngan_hang"
  - "hủy liên kết" -> service: "huy_lien_ket_ngan_hang"
  - "thanh toán hóa đơn" (điện, nước, internet) -> service: "thanh_toan_hoa_don"
  - "thanh toán khoản vay" -> service: "thanh_toan_khoan_vay"
  - "siêu tích lũy", "tích lũy" -> service: "sieu_tich_luy"
  - "mobile money" -> service: "mobile_money"
  - "định danh", "xác thực", "cccd", "sinh trắc học" -> service: "xac_thuc_dinh_danh"
  - "mật khẩu", "khóa/mở tài khoản", "hủy ví" -> service: "tai_khoan_vi"
  - "voucher", "khuyến mại", "ưu đãi" -> service: "ctkm_voucher"
  - "gọi đồ ăn", "gọi xe", "mua sắm", "đặt phòng" -> service: null (KHÔNG hỗ trợ)

- problem_type: Vấn đề hoặc mục đích của user.
  Ví dụ: huong_dan, tra_soat, chinh_sach, han_muc_phi, loi_lien_ket, loi_xac_thuc, that_bai, khong_ho_tro_ngan_hang, thong_tin_khong_hop_le.
  QUY TẮC:
  - "làm sao", "hướng dẫn", "cách nào", "vào mục nào" -> problem_type: "huong_dan"
  - "tra soát", "khiếu nại", "chưa nhận được tiền", "bị trừ tiền", "giao dịch lỗi" -> problem_type: "tra_soat"
  - "chính sách", "quy định", "có được không", "có mất phí không" -> problem_type: "chinh_sach"
  - "hạn mức", "tối đa", "tối thiểu" -> problem_type: "han_muc_phi"
  - "lỗi liên kết", "không liên kết được" -> problem_type: "loi_lien_ket"
  - "lỗi xác thực", "sai thông tin", "không trùng khớp" -> problem_type: "loi_xac_thuc"
  - "giao dịch thất bại", "không thành công" -> problem_type: "that_bai"

- issue_state: Trạng thái giao dịch được user đề cập RÕ RÀNG.
  - "pending": nếu user nói "đang xử lý", "chờ xử lý".
  - "failed": nếu user nói "thất bại", "không thành công".
  - "success": nếu user nói "thành công", "hoàn thành".
  - "unknown": nếu không đề cập hoặc không chắc chắn.

- issue_outcome: Triệu chứng mà user gặp phải.
  - "need_instruction": nếu user cần hướng dẫn.
  - "money_not_received": nếu user nói "chưa nhận được tiền".
  - "money_deducted": nếu user nói "bị trừ tiền".
  - "unknown": nếu không rõ.

- bank: Tên ngân hàng (nếu có).
- amount: Số tiền (nếu có).
- time: Thời gian (nếu có).
- error_message: Thông báo lỗi trong ngoặc kép (nếu có).

- confidence_assessment:
  - service_confidence: Độ tin cậy của `service` (0.0 đến 1.0). Nếu service = null thì = 0.0
  - problem_confidence: Độ tin cậy của `problem_type` (0.0 đến 1.0).
  - state_confidence: Độ tin cậy của `issue_state` (0.0 đến 1.0).
  - outcome_confidence: Độ tin cậy của `issue_outcome` (0.0 đến 1.0).

VÍ DỤ:
1. User: "Tôi muốn gọi đồ ăn qua VNPT Money"
   -> {{ "service": null, "problem_type": "chinh_sach", "issue_state": "unknown", "issue_outcome": "unknown", "confidence_assessment": {{ "service_confidence": 0.0, "problem_confidence": 0.8 }} }}

2. User: "Làm sao để nạp tiền điện thoại?"
   -> {{ "service": "nap_tien_dien_thoai", "problem_type": "huong_dan", "issue_state": "unknown", "issue_outcome": "need_instruction", "confidence_assessment": {{ "service_confidence": 0.95, "problem_confidence": 0.95 }} }}

3. User: "Giao dịch báo đang xử lý nhưng ngân hàng đã trừ tiền."
   -> {{ "service": null, "problem_type": "tra_soat", "issue_state": "pending", "issue_outcome": "money_deducted", "confidence_assessment": {{ "service_confidence": 0.0, "problem_confidence": 0.9 }} }}

Chỉ trả về JSON.
"""

    def _calculate_confidence(self, conf_assessment: dict) -> float:
        """
        Convert confidence_assessment to numeric score.
        Supports both:
        - String values: "high"/"medium"/"low"
        - Numeric values: 0.0-1.0
        
        Logic:
        - All high: 0.90
        - All medium: 0.70
        - All low: 0.40
        - Mixed: weighted average
        """
        def normalize_confidence(val):
            """Convert string or numeric confidence to float."""
            if isinstance(val, (int, float)):
                return float(val)
            
            confidence_values = {
                "high": 0.90,
                "medium": 0.70,
                "low": 0.40
            }
            return confidence_values.get(str(val).lower(), 0.40)
        
        service_conf = normalize_confidence(conf_assessment.get("service_confidence", 0.40))
        state_conf = normalize_confidence(conf_assessment.get("state_confidence", 0.40))
        outcome_conf = normalize_confidence(conf_assessment.get("outcome_confidence", 0.40))
        
        # Weighted: service (40%), outcome (35%), state (25%)
        score = (
            service_conf * 0.40 +
            outcome_conf * 0.35 +
            state_conf * 0.25
        )
        
        return round(score, 2)

    def identify_missing_slots(self, slots: CaseSlots) -> List[str]:
        missing: List[str] = []

        if slots.service:
            critical = CRITICAL_SLOTS.get(slots.service, ["service", "problem_type"])
        else:
            critical = ["service", "problem_type"]

        for slot_name in critical:
            if not getattr(slots, slot_name, None):
                missing.append(slot_name)

        return missing

    def generate_clarification_questions(self, slots: CaseSlots) -> List[str]:
        questions: List[str] = []

        if not slots.service:
            questions.append("Bạn đang gặp vấn đề với dịch vụ nào? (Nạp tiền / Rút tiền / Chuyển tiền / Liên kết ngân hàng / Thanh toán / Ứng dụng)")

        if not slots.problem_type:
            questions.append("Vấn đề cụ thể là gì? (Bị trừ tiền / Chưa nhận dịch vụ / Không nhận OTP / Giao dịch thất bại / Đang xử lý / Chưa nhận được tiền)")

        if not slots.bank and slots.service in {"nap_tien", "rut_tien", "lien_ket_ngan_hang", "huy_lien_ket_ngan_hang"}:
            questions.append("Bạn đang thực hiện với ngân hàng nào? (Ví dụ: Vietcombank, BIDV, Techcombank...)")

        if not slots.amount and slots.service in {"rut_tien", "chuyen_tien", "thanh_toan"}:
            questions.append("Số tiền giao dịch là bao nhiêu? (Ví dụ: 500k, 1 triệu)")

        if not slots.time and slots.problem_type in {"bi_tru_tien", "chua_nhan_dich_vu", "chua_nhan_duoc_tien"}:
            questions.append("Giao dịch xảy ra lúc nào? (Hôm nay / Hôm qua / Ngày cụ thể)")

        if not slots.error_message and any(kw in slots.raw_narrative for kw in ["lỗi", "không được", "bị lỗi", "gặp sự cố"]):
            questions.append("Có vẻ như có lỗi xảy ra. Vui lòng cung cấp thêm thông tin lỗi nếu có.")

        if slots.contradictions:
            questions.append("Có vẻ thông tin đang mâu thuẫn. Hãy xác nhận lại giúp mình: " + "; ".join(slots.contradictions))

        return questions

    def should_clarify(self, slots: CaseSlots) -> bool:
        # KHÔNG bắt buộc service - cho phép fallback
        # Chỉ clarify khi thực sự thiếu thông tin quan trọng
        
        # Nếu có problem_type hoặc có thể infer -> không cần clarify
        if slots.problem_type:
            slots.missing_slots = []
            return False
            
        inferred = self._infer_problem_type(slots)
        if inferred:
            slots.problem_type = inferred
            slots.missing_slots = []
            return False

        # Nếu có service nhưng không có problem -> vẫn OK, để graph tự tìm
        if slots.service:
            slots.missing_slots = []
            return False
        
        # Chỉ clarify khi THẬT SỰ không có gì
        if not slots.service and not slots.problem_type and not slots.bank:
            slots.missing_slots = ["service", "problem_type"]
            return True
            
        slots.missing_slots = []
        return False

    def _fallback_service_detection(self, narrative: str) -> Optional[str]:
        """
        Fallback service detection using UnifiedNormalizer inference.
        This is now a thin wrapper around the single source of truth.
        
        Returns slot_key or None.
        """
        from normalize_content import UnifiedNormalizer
        
        slot_key, confidence, evidence = UnifiedNormalizer.infer_service_from_text(narrative)
        
        if slot_key and confidence >= 0.40:  # Accept if confidence is reasonable
            return slot_key
        
        return None
    
    def _detect_problem_from_keywords(self, narrative: str) -> Optional[str]:
        """
        Detect specific problem types from keywords in user text.
        This now delegates to UnifiedNormalizer for single source of truth.
        Returns slot_key or None.
        
        Priority order: specific problems first, then generic.
        """
       
        
        problem_slot, confidence, evidence = UnifiedNormalizer.infer_problem_from_text(narrative)
        
        if problem_slot and confidence >= 0.40:  # Accept if confidence is reasonable
            return problem_slot
        
        return None  # No specific problem detected
    
    def _map_to_problem_type(self, issue_state: str, issue_outcome: str) -> Optional[str]:
        """Map LLM outcome to problem_type slot.
        
        SIMPLE FALLBACK: Chỉ map outcome cơ bản từ LLM khi inference không tìm thấy gì.
        Keyword matching đã được xử lý trong UnifiedNormalizer.infer_problem_from_text().
        """
        outcome = (issue_outcome or "").lower().strip()
        
        # Simple outcome mapping
        outcome_map = {
            "money_deducted": "bi_tru_tien",
            "money_not_received": "chua_nhan_tien",
            "need_instruction": "huong_dan",
            "asking_policy": "huong_dan",
            "no_otp": "otp_xac_thuc",
            "voucher_issue": "voucher_khong_dung",
            "other_action": "khac"
        }
        
        return outcome_map.get(outcome, "khac")
    
    def _infer_problem_type(self, slots: CaseSlots) -> Optional[str]:
        """Suy luận vấn đề - DEPRECATED, sử dụng UnifiedNormalizer.infer_problem_from_text() thay thế.
        
        Hàm này chỉ còn để backward compatibility và sẽ được remove trong tương lai.
        Tất cả logic inference đã được move vào normalize_content.py (single source of truth).
        """
        if slots.problem_type:
            return slots.problem_type

        # Delegate to UnifiedNormalizer (single source of truth)
        problem_slot, confidence, evidence = UnifiedNormalizer.infer_problem_from_text(
            slots.raw_narrative, 
            service=slots.service
        )
        
        return problem_slot if confidence >= 0.40 else None
    
    def is_out_of_scope(self, narrative: str, slots: CaseSlots) -> Tuple[bool, str]:
    
        t = narrative.lower().strip()
        
        # === TIER 1: RÕ RÀNG OOS (reject ngay) ===
        clear_oos_patterns = [
            # Greeting đơn thuần (không có context gì)
            (r'^(xin chào|chào|hello|hi|hey|yo)\s*[!.?]*$', "pure_greeting"),
            
            # Chủ đề hoàn toàn không liên quan
            (r'(thời tiết|dự báo thời tiết|trời|mưa|nắng)', "weather"),
            (r'(giờ|mấy giờ|bây giờ|thời gian hiện tại)(?!.*giao.?dịch)', "time_query"),
    
            
            # Banking services KHÔNG phải VNPT Money (rõ ràng)
            (r'(mở.*tài khoản.*ngân hàng.*(?!vnpt)|vay.*tiêu dùng.*ngân hàng|thẻ.*tín dụng.*(?!vnpt))(?!.*(vnpt|money|ví))', "other_banking"),
        ]
        
        for pattern, reason in clear_oos_patterns:
            if re.search(pattern, t):
                return True, reason
        
        # === TIER 2: UNCERTAIN - Có thể liên quan hoặc không ===
        # Trả False + để clarify xử lý sau (không reject)
        uncertain_patterns = [
            (r'(tên.*gì|bạn.*tên|bạn.*ai)', "identity_question"),
            (r'^(cảm ơn|thanks|thank you|ok|được)\s*[!.?]*$', "acknowledgment"),
        ]
        
        for pattern, reason in uncertain_patterns:
            if re.search(pattern, t):
                # Không reject - để flow clarify xử lý
                return False, f"uncertain:{reason}"
        
        # === TIER 3: KEYWORD-BASED IN-SCOPE CHECK ===
        # Mở rộng keyword list để giảm false positive
        vnpt_keywords = [
            # Brand & app
            "vnpt", "money", "vnpt money", "vnptmoney",
            
            # Services (mở rộng)
            "ví", "ví điện tử", "nạp", "rút", "chuyển", "thanh toán", "giao dịch",
            "liên kết", "hủy liên kết", "mở tài khoản", "đăng ký",
            
            # Transaction terms
            "tiền", "số dư", "balance", "transaction", "transfer",
            "deposit", "withdraw", "payment",
            
            # Problems
            "lỗi", "không được", "thất bại", "treo", "pending", "chưa nhận",
            "bị trừ", "mất tiền", "otp", "xác thực",
            
            # Features
            "hạn mức", "phí", "limit", "fee", "mất phí",
            "tài khoản", "account", "login", "đăng nhập",
            
        ]
        
        # Nếu có bất kỳ keyword nào -> coi như in-scope
        if any(k in t for k in vnpt_keywords):
            return False, "has_vnpt_keywords"
        
        # Nếu có slot nào được extract -> chắc chắn in-scope
        if slots.service or slots.problem_type or slots.bank:
            return False, "has_extracted_slots"
        
        # === TIER 4: CUỐI CÙNG - Uncertain ===
        # Không có keyword và không có slot -> có thể OOS NHƯNG không reject
        # Để clarify hỏi lại: "Anh/chị đang gặp vấn đề gì với VNPT Money?"
        # (Tránh reject nhầm paraphrase hoặc cách diễn đạt lạ)
        
        return False, "uncertain:no_clear_signal"

    def decide_action(self, slots: CaseSlots) -> Dict[str, Any]:
        if self.should_clarify(slots):
            return {
                "action_type": "clarify",
                "reason": "Chưa xác định được dịch vụ (service).",
                "clarification_questions": [
                    "Bạn đang muốn thực hiện tác vụ tài chính nào trong VNPT Money?"
                ],
                "policy_results": [],
                "search_query": "",
                "cypher": "",
                "cypher_params": {},
                "confidence": slots.confidence_score,
            }

# Nếu câu hỏi mang tính chính sách (hạn mức/phí) và đã có bank_id thì trả chính sách trước
        if PolicyStore.is_policy_intent(slots.raw_narrative) and slots.bank_id:
            policies = self.policy_store.get_policy_for_service(slots.bank_id, slots.service or "")
            if policies:
                return {
                    "action_type": "policy",
                    "reason": "Truy vấn chính sách theo ngân hàng liên kết.",
                    "clarification_questions": [],
                    "policy_results": policies,
                    "search_query": "",
                    "cypher": "",
                    "cypher_params": {},
                    "confidence": slots.confidence_score,
                }

        cypher = self.build_cypher_query(slots)
        cypher_params = self.build_cypher_params(slots)

        return {
            "action_type": "search",
            "reason": "Đủ điều kiện truy xuất trong graph.",
            "clarification_questions": [],
            "policy_results": [],
            "search_query": f"{slots.service} {slots.problem_type or ''} {slots.bank or ''}".strip(),
            "cypher": cypher,
            "cypher_params": cypher_params,
            "confidence": slots.confidence_score,
        }

    def build_search_query(self, slots: CaseSlots) -> str:
        parts: List[str] = []
        if slots.service:
            parts.append(slots.service.replace("_", " "))
        if slots.problem_type:
            parts.append(slots.problem_type.replace("_", " "))
        if slots.bank:
            parts.append(f"ngân hàng {slots.bank}")
        if slots.error_message:
            parts.append(f'lỗi "{slots.error_message}"')
        return " ".join(parts)

    def build_cypher_query(self, slots: CaseSlots) -> str:
        """
        Build a tiered Cypher query with SOFT PROBLEM FILTERING.
        
        KEY PRINCIPLE: Only use SERVICE as hard filter. Problem/state/outcome are SCORING bonuses.
        This prevents "lọc chết" (filtering out correct QAs due to mis-classification).
        
        Strategy:
        - Service: Hard filter (WHERE clause) if confidence >= 0.40
        - Problem: NO hard filter - used for scoring/ranking only
        - Bank: Optional filter (tier bonus)
        - State/Outcome: Scoring bonuses only
        
        This ensures we retrieve ALL relevant QAs for the service, then let rerank
        select the best match based on problem/state/outcome similarity.
        """
        
        # Only use service for hard filtering
        use_service_filter = slots.service and slots.service_confidence >= 0.40
        use_bank_filter = slots.bank_id is not None
        
        # Build conditional WHERE clause - SERVICE ONLY
        service_where = "s.name = $service" if use_service_filter else "1=1"
        # Problem is NOT used in WHERE - only for scoring
        
        return f"""
        // SOFT PROBLEM FILTERING: Service as hard filter, Problem for scoring only
        // use_service_filter: {use_service_filter} (conf: {slots.service_confidence:.2f})
        // problem: {slots.problem_type} (conf: {slots.problem_confidence:.2f}) - SCORING ONLY, NOT FILTERED
        // use_bank_filter: {use_bank_filter}
        
        CALL {{
          // Tier 3: Service + Problem (exact match)
          WITH $service AS service, $problem AS problem
          MATCH (sol:Solution)-[:OF_SERVICE]->(s:Service)
          WHERE {service_where}
          MATCH (sol)-[:OF_PROBLEM]->(p:Problem)
          WHERE p.name = $problem
          RETURN 
            s.name AS service,
            p.name AS problem,
            sol.answer AS solution,
            sol.question AS title,
            'qa' AS solution_type,
            sol.bank_id AS bank_id,
            sol.id AS qa_id,
            3 AS tier
          LIMIT 10
          
          UNION
          
          // Tier 2: Service only (broad recall, problem for scoring)
          WITH $service AS service
          MATCH (sol:Solution)-[:OF_SERVICE]->(s:Service)
          WHERE {service_where}
          OPTIONAL MATCH (sol)-[:OF_PROBLEM]->(p:Problem)
          RETURN 
            s.name AS service,
            p.name AS problem,
            sol.answer AS solution,
            sol.question AS title,
            'qa' AS solution_type,
            sol.bank_id AS bank_id,
            sol.id AS qa_id,
            2 AS tier
          LIMIT 20
          
          UNION
          
          // Tier 1: Problem only (if service failed)
          WITH $problem AS problem
          MATCH (sol:Solution)-[:OF_PROBLEM]->(p:Problem)
          WHERE p.name = $problem
          OPTIONAL MATCH (sol)-[:OF_SERVICE]->(s:Service)
          RETURN 
            s.name AS service,
            p.name AS problem,
            sol.answer AS solution,
            sol.question AS title,
            'qa' AS solution_type,
            sol.bank_id AS bank_id,
            sol.id AS qa_id,
            1 AS tier
          LIMIT 10
          
          UNION
          
          // Tier 0: All solutions (fallback)
          MATCH (sol:Solution)
          OPTIONAL MATCH (sol)-[:OF_SERVICE]->(s:Service)
          OPTIONAL MATCH (sol)-[:OF_PROBLEM]->(p:Problem)
          RETURN 
            s.name AS service,
            p.name AS problem,
            sol.answer AS solution,
            sol.question AS title,
            'qa' AS solution_type,
            sol.bank_id AS bank_id,
            sol.id AS qa_id,
            0 AS tier
          LIMIT 5
        }}
        RETURN DISTINCT
          service,
          problem,
          solution,
          title,
          solution_type,
          bank_id,
          qa_id,
          tier,
          (tier * 30) AS score,
          CASE 
            WHEN $problem IS NOT NULL AND $problem <> '' AND $problem <> 'unknown' AND problem = $problem 
            THEN 1 
            ELSE 0 
          END AS problem_match
        ORDER BY tier DESC, problem_match DESC, score DESC
        LIMIT 80
        """
    
    def build_fallback_cypher_query(self, slots: CaseSlots) -> str:
        """
        Build a relaxed fallback query when primary query returns 0 results.
        
        Strategy:
        1. If primary had both service+problem: try service only
        2. If primary had service only: try problem only (if available)
        3. Final fallback: get top generic QAs sorted by priority
        """
        
        # Fallback 1: Service only (ignore problem)
        if slots.service and slots.service_confidence >= 0.40:
            return f"""
            // FALLBACK QUERY: Service only (no problem filter)
            MATCH (sol:Solution)-[:OF_SERVICE]->(s:Service {{name: $service}})
            OPTIONAL MATCH (sol)-[:OF_PROBLEM]->(p:Problem)
            RETURN 
              s.name AS service,
              p.name AS problem,
              sol.answer AS solution,
              sol.question AS title,
              'qa' AS solution_type,
              sol.bank_id AS bank_id,
              sol.id AS qa_id,
              2 AS tier
            LIMIT 30
            """
        
        # Fallback 2: Problem only (if service inference failed)
        if slots.problem_type and slots.problem_confidence >= 0.40:
            return f"""
            // FALLBACK QUERY: Problem only (no service filter)
            MATCH (sol:Solution)-[:OF_PROBLEM]->(p:Problem {{name: $problem}})
            OPTIONAL MATCH (sol)-[:OF_SERVICE]->(s:Service)
            RETURN 
              s.name AS service,
              p.name AS problem,
              sol.answer AS solution,
              sol.question AS title,
              'qa' AS solution_type,
              sol.bank_id AS bank_id,
              sol.id AS qa_id,
              1 AS tier
            LIMIT 30
            """
        
        # Final fallback: Top generic QAs (high priority, no filters)
        return f"""
        // FINAL FALLBACK: Top generic QAs (no filters)
        MATCH (sol:Solution)
        OPTIONAL MATCH (sol)-[:OF_SERVICE]->(s:Service)
        OPTIONAL MATCH (sol)-[:OF_PROBLEM]->(p:Problem)
        RETURN 
          s.name AS service,
          p.name AS problem,
          sol.answer AS solution,
          sol.question AS title,
          'qa' AS solution_type,
          sol.bank_id AS bank_id,
          sol.id AS qa_id,
          0 AS tier
        LIMIT 20
        """
       
    def build_cypher_params(self, slots: CaseSlots) -> Dict[str, Any]:
        """Build parameters for parameterized Cypher query.
        
        NEW: Support state + outcome (primary) with problem (fallback)
        
        Hàm này chỉ còn để backward compatibility và sẽ được remove trong tương lai.
        Tất cả logic inference đã được move vào normalize_content.py (single source of truth).
        
        Matches Neo4j schema:
        - service: NORMALIZED SLOT_KEY (e.g., "nap_tien", "rut_tien")
        - state: NORMALIZED slot_key (e.g., "failed", "pending", "success")  
        - outcome: NORMALIZED slot_key (e.g., "money_deducted", "money_not_received")
        - problem: NORMALIZED slot_key (e.g., "huong_dan", "bi_tru_tien") - LEGACY
        - bank_id: code like "VCB", "TCB" (used for APPLIES_TO relationship)
        
        PRIORITY: state + outcome > problem
        """
        # Normalize service
        service = slots.service or ""
        if service:
            service = _norm_text(service)
        
        # Normalize state (PRIMARY)
        state = slots.state or ""
        if state:
            state = _norm_text(state)
        
        # Normalize outcome (PRIMARY)
        outcome = slots.outcome or ""
        if outcome:
            outcome = _norm_text(outcome)
        
        # Normalize problem (LEGACY fallback)
        problem = slots.problem_type or ""
        if problem:
            problem = _norm_text(problem)
        
        # Set problem to None if "unknown" to avoid false matches in Neo4j
        if problem in ["", "unknown"]:
            problem = None
        
        return {
            "service": service,
            "state": state,
            "outcome": outcome,
            "problem": problem,
            "bank_id": slots.bank_id or "",
        }

def demo_run():
    api_key = os.getenv("OPENAI_API_KEY")
    
    triage = CaseTriage(
        api_key=api_key, 
        policy_path=policy_path_default,
       
    )

    text = "Tôi cần đặt dịch vụ nạp tiền tự động nhưng không biết làm sao"
    print("USER:", text)

    slots = triage.extract_slots(text)

    print(slots)
    
    if hasattr(slots, '_debug_info'):
        print("\n[DEBUG INFO]")
        print(json.dumps(slots._debug_info, ensure_ascii=False, indent=2))

    action = triage.decide_action(slots)
    print("\n[ACTION]")
   

    if action["action_type"] == "search":
       
        print("Params:")
        print(action["cypher_params"])



if __name__ == "__main__":
    demo_run()
