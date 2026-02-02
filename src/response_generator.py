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

    SYNTHESIS_PROMPT = """Trợ lý VNPT Money. Trả lời dựa trên thông tin tham khảo.

CÂU HỎI: {user_question}

THÔNG TIN THAM KHẢO:
{contexts}

QUY TẮC:
- CHỈ dùng thông tin từ nguồn tham khảo, KHÔNG bịa đặt
- So sánh ngữ nghĩa để tìm nguồn phù hợp (VD: "chuyển từ ngân hàng" = "nạp tiền vào ví")
- Nếu KHÔNG có thông tin phù hợp → trả lời: "Mình chưa có thông tin về vấn đề này. Vui lòng liên hệ hotline 18001091 (nhánh 3)."
- Format: numbered list cho bước, bullet cho lưu ý

Trả lời:"""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.model = Config.RESPONSE_GENERATOR_MODEL
        self.temperature = Config.RESPONSE_GENERATOR_TEMPERATURE
        self.max_tokens = Config.RESPONSE_GENERATOR_MAX_TOKENS
    
    def generate(
        self, 
        decision: Decision, 
        context: Optional[RetrievedContext], 
        user_question: str,
        all_contexts: Optional[List[RetrievedContext]] = None,
        need_account_lookup: bool = False
    ) -> FormattedResponse:
        # OPTIMIZATION: Skip LLM synthesis khi có context tốt để giảm latency
        if decision.type in [DecisionType.DIRECT_ANSWER, DecisionType.ANSWER_WITH_CLARIFY]:
            # Kiểm tra nếu top result có similarity cao (>= 0.85) → dùng direct answer (nhanh)
            use_direct = False
            similarity = 0.0
            
            if decision.top_result:
                similarity = decision.top_result.similarity_score
                if similarity >= 0.85:
                    use_direct = True
            
            if use_direct and context:
                # Fast path: Direct answer without LLM synthesis (~0.5s thay vì 10-15s)
                logger.info(f"Fast path: Direct answer (similarity={similarity:.2f})")
                response = self._generate_direct_answer(decision, context, user_question)
                if need_account_lookup:
                    response = self._append_personal_escalation(response)
                return response
            elif all_contexts and len(all_contexts) > 0:
                # Slow path: LLM synthesis khi cần tổng hợp nhiều nguồn
                response = self._generate_synthesized_answer(decision, all_contexts, user_question)
                if need_account_lookup:
                    response = self._append_personal_escalation(response)
                return response
            elif context:
                response = self._generate_direct_answer(decision, context, user_question)
                if need_account_lookup:
                    response = self._append_personal_escalation(response)
                return response
            else:
                return self._generate_escalation_low_confidence()
        
        if decision.type == DecisionType.CLARIFY_REQUIRED:
            # Thử dùng LLM tổng hợp từ top contexts nếu có
            if all_contexts and len(all_contexts) >= 2:
                return self._generate_synthesized_answer(decision, all_contexts, user_question)
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
    
    def _generate_synthesized_answer(
        self, 
        decision: Decision, 
        contexts: List[RetrievedContext], 
        user_question: str
    ) -> FormattedResponse:
        """Dùng LLM tổng hợp câu trả lời từ nhiều contexts."""
        try:
            # Build context string từ top 3 contexts (reduced for speed)
            context_parts = []
            for i, ctx in enumerate(contexts[:3]):
                part = f"--- Nguồn {i+1}: {ctx.problem_title or 'N/A'} ---\n"
                if ctx.answer_content:
                    part += ctx.answer_content
                if ctx.answer_steps:
                    steps = "\n".join(f"  {j+1}. {s}" for j, s in enumerate(ctx.answer_steps))
                    part += f"\nCác bước:\n{steps}"
                if ctx.answer_notes:
                    part += f"\nLưu ý: {ctx.answer_notes}"
                context_parts.append(part)
            
            contexts_text = "\n\n".join(context_parts)
            
            prompt = self.SYNTHESIS_PROMPT.format(
                user_question=user_question,
                contexts=contexts_text
            )
            
            response_text = self._call_llm_synthesis(prompt)
            
            # Validate response
            if not response_text or len(response_text) < 20:
                logger.warning("LLM synthesis response too short, falling back")
                if contexts[0]:
                    return self._generate_direct_answer(decision, contexts[0], user_question)
                return self._generate_escalation_low_confidence()
            
            return FormattedResponse(
                message=response_text,
                source_citation="",
                decision_type=DecisionType.DIRECT_ANSWER
            )
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback to first context
            if contexts and contexts[0]:
                return self._generate_direct_answer(decision, contexts[0], user_question)
            return self._generate_escalation_low_confidence()
    
    def _call_llm_synthesis(self, prompt: str) -> str:
        """Call LLM for synthesis with specific settings."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                temperature=0.3,  # Lower temperature for more factual response
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM synthesis call failed: {e}")
            raise
    
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
    
    def _append_personal_escalation(self, response: FormattedResponse) -> FormattedResponse:
        """Thêm thông tin escalation khi cần tra soát giao dịch cá nhân."""
        escalation_info = """

---
📞 **Để kiểm tra thông tin giao dịch cụ thể của bạn**, mình cần chuyển yêu cầu đến bộ phận hỗ trợ.

**📱 Hotline:** 18001091 (nhánh 3)
**📍 Điểm giao dịch:** Các cửa hàng VinaPhone trên toàn quốc

Khi liên hệ, vui lòng cung cấp:
• Số điện thoại đăng ký VNPT Money
• Thời gian giao dịch  
• Mã giao dịch (nếu có)

Tổng đài viên sẽ hỗ trợ kiểm tra ngay cho bạn."""
        
        new_message = response.message + escalation_info
        return FormattedResponse(
            message=new_message,
            source_citation=response.source_citation,
            decision_type=response.decision_type
        )
    
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
    
    def generate(
        self, 
        decision: Decision, 
        context: Optional[RetrievedContext], 
        user_question: str,
        all_contexts: Optional[List[RetrievedContext]] = None
    ) -> FormattedResponse:
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
