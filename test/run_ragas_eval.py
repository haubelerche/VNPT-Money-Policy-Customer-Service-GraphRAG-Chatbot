"""
RAGAS Evaluation Runner cho VNPT Money Chatbot.

Chạy đánh giá chất lượng RAG pipeline với 2 chế độ:
1. Full pipeline: Chạy câu hỏi qua toàn bộ pipeline → thu thập answer + contexts → đánh giá
2. Standalone: Đánh giá trực tiếp trên dữ liệu có sẵn (answer + contexts đã thu thập trước)

Usage:
    # Chế độ 1: Full pipeline evaluation (cần Neo4j + OpenAI)
    python test/run_ragas_eval.py --mode full

    # Chế độ 2: Standalone evaluation từ dataset (chỉ cần OpenAI)
    python test/run_ragas_eval.py --mode standalone

    # Chỉ đánh giá 1 số metrics
    python test/run_ragas_eval.py --mode standalone --metrics faithfulness answer_relevancy

    # Dùng RAGAS library (cần cài thêm ragas, langchain-openai)
    python test/run_ragas_eval.py --mode standalone --use-ragas-lib
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_standalone_eval(args):
    """
    Chế độ Standalone: Đánh giá chất lượng từ dataset sẵn có.
    Không cần chạy pipeline thực tế - mô phỏng answer từ ground_truth.
    Phù hợp để test module đánh giá nhanh.
    """
    from ragas_evaluation import (
        RAGASEvaluator,
        EvalSample,
        load_eval_dataset,
        save_eval_report,
        print_eval_report,
    )
    
    # Load dataset
    dataset_path = args.dataset or os.path.join(
        os.path.dirname(__file__), "eval_dataset.json"
    )
    test_data = load_eval_dataset(dataset_path)
    
    if args.limit:
        test_data = test_data[: args.limit]
    
    logger.info(f"Loaded {len(test_data)} samples from {dataset_path}")
    
    # Build samples (dùng ground_truth làm cả answer và context để test module)
    samples = []
    for item in test_data:
        sample = EvalSample(
            question=item["question"],
            contexts=[item["ground_truth"]],  # Context = ground truth (giả lập perfect retrieval)
            answer=item["ground_truth"],       # Answer = ground truth (giả lập perfect generation)
            ground_truth=item["ground_truth"],
            metadata=item.get("metadata", {}),
        )
        samples.append(sample)
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(
        llm_model=args.llm_model or "gpt-4o-mini",
        embedding_model=args.embedding_model or "text-embedding-3-small",
    )
    
    # Run evaluation
    if args.use_ragas_lib:
        report = evaluator.evaluate_with_ragas(samples, metrics=args.metrics)
    else:
        report = evaluator.evaluate_builtin(samples)
    
    # Output
    print_eval_report(report)
    
    # Save if requested
    if args.output:
        save_eval_report(report, args.output)
    else:
        default_output = os.path.join(
            os.path.dirname(__file__),
            f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        save_eval_report(report, default_output)
    
    return report


def run_full_pipeline_eval(args):
    """
    Chế độ Full: Chạy câu hỏi qua pipeline thực tế, 
    thu thập answer + contexts, rồi đánh giá.
    Cần Neo4j + OpenAI đang chạy.
    """
    from ragas_evaluation import (
        RAGASEvaluator,
        PipelineEvaluator,
        load_eval_dataset,
        save_eval_report,
        print_eval_report,
    )
    from pipeline import create_pipeline
    
    # Load dataset  
    dataset_path = args.dataset or os.path.join(
        os.path.dirname(__file__), "eval_dataset.json"
    )
    test_data = load_eval_dataset(dataset_path)
    
    if args.limit:
        test_data = test_data[: args.limit]
    
    logger.info(f"Loaded {len(test_data)} samples")
    
    # Initialize pipeline
    pipeline = create_pipeline(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        redis_url=os.getenv("REDIS_URL"),
        use_llm=True,
        enable_monitoring=False,  # Không cần monitoring khi eval
    )
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(
        llm_model=args.llm_model or "gpt-4o-mini",
        embedding_model=args.embedding_model or "text-embedding-3-small",
    )
    
    pipeline_evaluator = PipelineEvaluator(pipeline=pipeline, evaluator=evaluator)
    
    # Run evaluation
    report = pipeline_evaluator.run_evaluation(
        test_data=test_data,
        session_id="ragas_eval_session",
        use_ragas_lib=args.use_ragas_lib,
    )
    
    # Output
    print_eval_report(report)
    
    # Save
    if args.output:
        save_eval_report(report, args.output)
    else:
        default_output = os.path.join(
            os.path.dirname(__file__),
            f"eval_report_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        save_eval_report(report, default_output)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation Runner cho VNPT Money Chatbot"
    )
    parser.add_argument(
        "--mode",
        choices=["standalone", "full"],
        default="standalone",
        help="Chế độ đánh giá: standalone (không cần pipeline) hoặc full (chạy pipeline thực)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Đường dẫn tới file dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file output report JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số samples đánh giá",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Chỉ đánh giá các metrics cụ thể (faithfulness, answer_relevancy, context_precision, context_recall)",
    )
    parser.add_argument(
        "--use-ragas-lib",
        action="store_true",
        default=False,
        help="Sử dụng thư viện RAGAS chính thức (cần cài ragas, langchain-openai)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Model LLM dùng cho đánh giá (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Model embedding dùng cho similarity (default: text-embedding-3-small)",
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is required")
        print("Set it in .env file or export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    
    if args.mode == "standalone":
        run_standalone_eval(args)
    elif args.mode == "full":
        run_full_pipeline_eval(args)


if __name__ == "__main__":
    main()
