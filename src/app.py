import os
import logging
from chainlit.cli import run_chainlit
import chainlit as cl
from dotenv import load_dotenv

from pipeline import create_pipeline, ChatbotPipeline
from schema import DecisionType

# Try to import advanced features
try:
    from rate_limiter import get_rate_limiter, RateLimitTier
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False

try:
    from monitoring import get_monitoring_dashboard
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

pipeline: ChatbotPipeline = None
last_responses = {}


def get_user_tier(session_id: str) -> str:
    """
    Get user's rate limit tier based on session or authentication.
    In a real app, this would check user authentication/subscription level.
    """
    # Default to STANDARD tier for all users
    # This can be extended to check user authentication and subscription
    return "STANDARD"


def get_pipeline() -> ChatbotPipeline:
    global pipeline
    
    if pipeline is None:
        logger.info("Khởi tạo pipeline...")
        
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        redis_url = os.getenv("REDIS_URL")
        use_llm = os.getenv("USE_LLM", "true").lower() == "true"
        
        # Advanced features flags - disabled by default for stability
        enable_rate_limiting = os.getenv("ENABLE_RATE_LIMITING", "false").lower() == "true"
        enable_monitoring = os.getenv("ENABLE_MONITORING", "false").lower() == "true"
        
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY chưa được cấu hình")
            use_llm = False
        
        pipeline = create_pipeline(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            openai_api_key=openai_api_key or "",
            redis_url=redis_url,
            use_llm=use_llm,
            enable_rate_limiting=enable_rate_limiting,
            enable_monitoring=enable_monitoring
        )
        
        logger.info("Pipeline đã sẵn sàng")
        if enable_rate_limiting:
            logger.info("Rate limiting: ENABLED")
        if enable_monitoring:
            logger.info("Monitoring: ENABLED")
    
    return pipeline


@cl.on_chat_start
async def on_chat_start():
    session_id = cl.user_session.get("id")
    cl.user_session.set("session_id", session_id)
    
    welcome_message = """Xin chào! Mình là trợ lý ảo của **VNPT Money**.

Mình có thể hỗ trợ bạn về:
* Nạp/rút tiền Mobile Money
* Chuyển tiền
* Thanh toán dịch vụ
* Liên kết ngân hàng
* Chính sách và điều khoản

**Lưu ý:** Để kiểm tra trạng thái giao dịch cụ thể, bạn vui lòng liên hệ hotline **18001091 (nhánh 3)**

Bạn cần hỗ trợ gì ạ?"""
    
    await cl.Message(content=welcome_message).send()
    logger.info(f"Phiên mới: {session_id}")


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    user_message = message.content
    user_tier = get_user_tier(session_id)
    
    logger.info(f"Tin nhắn từ {session_id} (tier={user_tier}): {user_message[:50]}...")
    
    response = None
    async with cl.Step(name="Đang xử lý...") as step:
        try:
            bot = get_pipeline()
            response = bot.process(user_message, session_id, user_tier)
            response_text = response.message
            
            last_responses[session_id] = {
                "question": user_message,
                "answer": response_text,
                "decision_type": response.decision_type
            }
            
            step.output = "Hoàn thành"
            
        except Exception as e:
            logger.error(f"Lỗi xử lý: {e}", exc_info=True)
            response_text = "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại hoặc liên hệ hotline **18001091 (nhánh 3)**."
            step.output = "Lỗi"
    
    answer_types = [DecisionType.DIRECT_ANSWER, DecisionType.ANSWER_WITH_CLARIFY]
    
    if response and response.decision_type in answer_types:
        actions = [
            cl.Action(
                name="feedback_helpful",
                payload={"action": "helpful"},
                label="Hữu ích"
            ),
            cl.Action(
                name="feedback_not_helpful",
                payload={"action": "not_helpful"},
                label="Chưa hữu ích"
            )
        ]
        await cl.Message(content=response_text, actions=actions).send()
    else:
        await cl.Message(content=response_text).send()


@cl.action_callback("feedback_helpful")
async def on_feedback_helpful(action: cl.Action):
    session_id = cl.user_session.get("session_id")
    logger.info(f"Phản hồi tích cực từ {session_id}")
    
    await action.remove()
    await cl.Message(content="Cảm ơn bạn đã phản hồi! Bạn có câu hỏi nào khác không?").send()


@cl.action_callback("feedback_not_helpful")
async def on_feedback_not_helpful(action: cl.Action):
    session_id = cl.user_session.get("session_id")
    logger.info(f"Phản hồi tiêu cực từ {session_id}")
    
    await action.remove()
    
    followup_actions = [
        cl.Action(
            name="option_rephrase",
            payload={"action": "rephrase"},
            label="Hỏi cách khác"
        ),
        cl.Action(
            name="option_hotline",
            payload={"action": "hotline"},
            label="Liên hệ tổng đài"
        ),
        cl.Action(
            name="option_continue",
            payload={"action": "continue"},
            label="Hỏi câu khác"
        )
    ]
    
    await cl.Message(
        content="Mình xin lỗi vì chưa giúp được bạn! Bạn muốn:",
        actions=followup_actions
    ).send()


@cl.action_callback("option_rephrase")
async def on_option_rephrase(action: cl.Action):
    await action.remove()
    
    session_id = cl.user_session.get("session_id")
    last_context = last_responses.get(session_id, {})
    original_question = last_context.get("question", "")
    
    if original_question:
        await cl.Message(
            content=f"Câu hỏi trước của bạn là: *\"{original_question}\"*\n\nBạn có thể diễn đạt lại theo cách khác, hoặc cung cấp thêm chi tiết để mình hiểu rõ hơn nhé!"
        ).send()
    else:
        await cl.Message(
            content="Bạn có thể đặt lại câu hỏi theo cách khác, hoặc cung cấp thêm chi tiết để mình hiểu rõ hơn nhé!"
        ).send()


@cl.action_callback("option_hotline")
async def on_option_hotline(action: cl.Action):
    await action.remove()
    
    await cl.Message(
        content="""Để được hỗ trợ trực tiếp, bạn vui lòng liên hệ:

**Hotline**: **18001091** (nhánh 3)
**Thời gian**: 24/7

Tổng đài viên sẽ lắng nghe và hỗ trợ bạn ngay!"""
    ).send()


@cl.action_callback("option_continue")
async def on_option_continue(action: cl.Action):
    await action.remove()
    await cl.Message(content="Không sao! Bạn cứ hỏi câu khác nhé, mình sẵn sàng hỗ trợ!").send()


@cl.on_chat_end
async def on_chat_end():
    session_id = cl.user_session.get("session_id")
    
    if session_id:
        try:
            bot = get_pipeline()
            bot.clear_session(session_id)
            logger.info(f"Kết thúc phiên: {session_id}")
        except Exception as e:
            logger.error(f"Lỗi xóa phiên: {e}")


@cl.on_settings_update
async def on_settings_update(settings):
    logger.info(f"Cấu hình thay đổi: {settings}")


@cl.action_callback("health")
async def health_check(action):
    try:
        bot = get_pipeline()
        health_data = {"status": "healthy"}
        
        # Add monitoring data if available
        if MONITORING_AVAILABLE and bot.monitoring:
            dashboard_stats = bot.monitoring.get_dashboard_stats()
            health_data["stats"] = {
                "requests_total": dashboard_stats.total_requests,
                "avg_latency_ms": dashboard_stats.avg_latency_ms,
                "active_sessions": dashboard_stats.active_sessions,
                "uptime_seconds": dashboard_stats.uptime_seconds
            }
        
        return health_data
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    run_chainlit(__file__)
