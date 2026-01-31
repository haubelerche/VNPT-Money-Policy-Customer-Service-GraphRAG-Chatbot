import logging
from typing import Optional, List

from schema import (
    Decision,
    DecisionType,
    RetrievedContext,
    FormattedResponse,
    ESCALATION_TEMPLATES,
    CLARIFICATION_QUESTIONS,
    FORBIDDEN_PHRASES,
    Config,
)

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Tạo câu trả lời dựa trên context và decision."""
    
    SYSTEM_PROMPT = """Bạn là Answer Formatter cho dịch vụ hỗ trợ VNPT Money.

QUY TẮC:
1. CHỈ ĐƯỢC sử dụng nội dung trong <answer_content>
2. KHÔNG ĐƯỢC thêm thông tin mới
3. KHÔNG ĐƯỢC suy đoán trạng thái giao dịch cá nhân
4. Giữ nguyên ý nghĩa của câu trả lời gốc
5. Format tự nhiên, thân thiện nhưng chuyên nghiệp
6. Nếu có steps, format dạng numbered list
7. Không bắt đầu bằng lời chào
8. Đi thẳng vào nội dung trả lời"""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.model = Config.RESPONSE_GENERATOR_MODEL
        self.temperature = Config.RESPONSE_GENERATOR_TEMPERATURE
        self.max_tokens = Config.RESPONSE_GENERATOR_MAX_TOKENS
    
    def generate(self, decision: Decision, context: Optional[RetrievedContext], user_question: str) -> FormattedResponse:
        if decision.type == DecisionType.DIRECT_ANSWER:
            return self._generate_direct_answer(decision, context, user_question)
        elif decision.type == DecisionType.ANSWER_WITH_CLARIFY:
            return self._generate_answer_with_clarify(decision, context, user_question)
        elif decision.type == DecisionType.CLARIFY_REQUIRED:
            return self._generate_clarification(decision, user_question)
        elif decision.type == DecisionType.ESCALATE_PERSONAL:
            return self._generate_escalation_personal(user_question)
        elif decision.type == DecisionType.ESCALATE_OUT_OF_SCOPE:
            return self._generate_escalation_out_of_scope()
        elif decision.type == DecisionType.ESCALATE_MAX_RETRY:
            return self._generate_escalation_max_retry()
        elif decision.type == DecisionType.ESCALATE_LOW_CONFIDENCE:
            return self._generate_escalation_low_confidence()
        else:
            logger.error(f"Unknown decision type: {decision.type}")
            return self._generate_escalation_low_confidence()
    
    def _generate_direct_answer(self, decision: Decision, context: Optional[RetrievedContext], user_question: str) -> FormattedResponse:
        if not context:
            logger.warning("DIRECT_ANSWER không có context, fallback")
            return self._generate_escalation_low_confidence()
        response_text = self._format_answer_fast(context)
        return FormattedResponse(message=response_text, source_citation="", decision_type=DecisionType.DIRECT_ANSWER)
    
    def _format_answer_fast(self, context: RetrievedContext) -> str:
        parts = []
        if context.answer_content:
            parts.append(context.answer_content.strip())
        if context.answer_steps:
            steps_formatted = "\n".join(f"{i+1}. {step}" for i, step in enumerate(context.answer_steps))
            parts.append(f"\n**Các bước thực hiện:**\n{steps_formatted}")
        if context.answer_notes:
            parts.append(f"\n**Lưu ý:** {context.answer_notes}")
        return "\n".join(parts)
    
    def _generate_answer_with_clarify(self, decision: Decision, context: Optional[RetrievedContext], user_question: str) -> FormattedResponse:
        if not context:
            return self._generate_clarification(decision, user_question)
        answer_text = self._format_answer_fast(context)
        return FormattedResponse(message=answer_text, source_citation="", decision_type=DecisionType.ANSWER_WITH_CLARIFY)
    
    def _generate_clarification(self, decision: Decision, user_question: str) -> FormattedResponse:
        response_text = """Mình chưa chắc chắn về vấn đề bạn đang gặp phải.

Bạn có thể cho mình biết thêm:
- Bạn đang thực hiện giao dịch gì? (nạp tiền, chuyển tiền, thanh toán...)
- Có thông báo lỗi cụ thể nào không?
- Hoặc mô tả chi tiết hơn tình huống của bạn

Mình sẽ cố gắng hỗ trợ bạn tốt nhất!"""
        return FormattedResponse(message=response_text, source_citation="", decision_type=DecisionType.CLARIFY_REQUIRED)
    
    def _generate_escalation_personal(self, user_question: str) -> FormattedResponse:
        message = ESCALATION_TEMPLATES["TEMPLATE_PERSONAL_DATA"]
        return FormattedResponse(message=message, source_citation="", decision_type=DecisionType.ESCALATE_PERSONAL)
    
    def _generate_escalation_out_of_scope(self) -> FormattedResponse:
        message = ESCALATION_TEMPLATES["TEMPLATE_OUT_OF_SCOPE"]
        return FormattedResponse(message=message, source_citation="", decision_type=DecisionType.ESCALATE_OUT_OF_SCOPE)
    
    def _generate_escalation_max_retry(self) -> FormattedResponse:
        message = ESCALATION_TEMPLATES["TEMPLATE_MAX_RETRY"]
        return FormattedResponse(message=message, source_citation="", decision_type=DecisionType.ESCALATE_MAX_RETRY)
    
    def _generate_escalation_low_confidence(self) -> FormattedResponse:
        message = ESCALATION_TEMPLATES["TEMPLATE_LOW_CONFIDENCE"]
        return FormattedResponse(message=message, source_citation="", decision_type=DecisionType.ESCALATE_LOW_CONFIDENCE)
    
    def _build_clarification_text(self, slots: List[str]) -> str:
        if not slots:
            return "Bạn có thể mô tả chi tiết hơn vấn đề bạn gặp phải không?"
        questions = []
        for slot in slots[:2]:
            if slot in CLARIFICATION_QUESTIONS:
                questions.append(CLARIFICATION_QUESTIONS[slot])
        if not questions:
            return "Bạn có thể mô tả chi tiết hơn vấn đề bạn gặp phải không?"
        return "\n".join(f"- {q}" for q in questions)
    
    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call thất bại: {e}")
            return "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại hoặc liên hệ hotline 18001091 (nhánh 3)"
    
    def _validate_response(self, response: str, source_content: str) -> str:
        response_lower = response.lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in response_lower:
                logger.warning(f"Phát hiện cụm từ bị cấm: {phrase}")
                return source_content
        return response


class ResponseGeneratorSimple:
    """Response generator đơn giản không dùng LLM."""
    
    def generate(self, decision: Decision, context: Optional[RetrievedContext], user_question: str) -> FormattedResponse:
        if decision.type == DecisionType.DIRECT_ANSWER and context:
            message = context.answer_content
            if context.answer_steps:
                steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(context.answer_steps))
                message += f"\n\n**Các bước thực hiện:**\n{steps}"
            citation = f"[Ref: {context.problem_id}/{context.answer_id}]"
        elif decision.type == DecisionType.ANSWER_WITH_CLARIFY and context:
            message = context.answer_content
            message += "\n\nBạn có thể cho mình biết thêm chi tiết về vấn đề bạn gặp phải không?"
            citation = f"[Ref: {context.problem_id}/{context.answer_id}]"
        elif decision.type == DecisionType.CLARIFY_REQUIRED:
            clarify = self._build_clarification(decision.clarification_slots)
            message = f"Để hỗ trợ bạn tốt hơn, bạn có thể cho mình biết:\n\n{clarify}"
            citation = ""
        elif decision.type == DecisionType.ESCALATE_PERSONAL:
            message = ESCALATION_TEMPLATES["TEMPLATE_PERSONAL_DATA"]
            citation = ""
        elif decision.type == DecisionType.ESCALATE_OUT_OF_SCOPE:
            message = ESCALATION_TEMPLATES["TEMPLATE_OUT_OF_SCOPE"]
            citation = ""
        elif decision.type == DecisionType.ESCALATE_MAX_RETRY:
            message = ESCALATION_TEMPLATES["TEMPLATE_MAX_RETRY"]
            citation = ""
        else:
            message = ESCALATION_TEMPLATES["TEMPLATE_LOW_CONFIDENCE"]
            citation = ""
        return FormattedResponse(message=message, source_citation=citation, decision_type=decision.type)
    
    def _build_clarification(self, slots: List[str]) -> str:
        if not slots:
            return "- Bạn đang thực hiện giao dịch gì?\n- Bạn gặp vấn đề cụ thể gì?"
        questions = []
        for slot in slots[:2]:
            if slot in CLARIFICATION_QUESTIONS:
                questions.append(f"- {CLARIFICATION_QUESTIONS[slot]}")
        return "\n".join(questions) if questions else "- Bạn có thể mô tả chi tiết hơn không?"
