"""
RAG Quality Guard - VNPT Money Chatbot.

Script tự động:
1. Chạy RAGAS eval trên expanded dataset (50 samples)
2. So sánh kết quả với baseline (20 samples trước đó)
3. Phát hiện regression theo từng category
4. Xuất report chi tiết + khuyến nghị

Usage:
    # Chạy đánh giá mở rộng
    python test/quality_guard.py --mode eval

    # So sánh với baseline
    python test/quality_guard.py --mode compare --baseline test/eval_report_full_20260227_134035.json

    # Chạy quick check (chỉ 10 samples)
    python test/quality_guard.py --mode eval --limit 10
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# === Quality thresholds (minimum acceptable scores) ===
QUALITY_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.75,
    "answer_similarity": 0.70,
}

# === Regression tolerance (max % drop allowed vs baseline) ===
REGRESSION_TOLERANCE = 0.05  # 5% drop allowed


def load_report(filepath: str) -> Dict[str, Any]:
    """Load eval report JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_by_category(report: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Phân tích kết quả theo category."""
    categories = defaultdict(lambda: defaultdict(list))
    
    for detail in report.get("details", []):
        category = detail.get("metadata", {}).get("category", "unknown")
        difficulty = detail.get("metadata", {}).get("difficulty", "unknown")
        
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_similarity"]:
            score = detail.get(metric, 0)
            categories[category][metric].append(score)
            categories[f"difficulty_{difficulty}"][metric].append(score)
    
    # Average scores per category
    results = {}
    for cat, metrics in categories.items():
        results[cat] = {}
        for metric, scores in metrics.items():
            results[cat][metric] = sum(scores) / len(scores) if scores else 0
        results[cat]["count"] = len(next(iter(metrics.values())))
    
    return results


def find_weak_samples(report: Dict[str, Any], threshold: float = 0.80) -> List[Dict]:
    """Tìm các sample có điểm thấp."""
    weak = []
    for i, detail in enumerate(report.get("details", [])):
        low_metrics = []
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            score = detail.get(metric, 0)
            if score < threshold:
                low_metrics.append(f"{metric}={score:.2f}")
        
        if low_metrics:
            weak.append({
                "index": i + 1,
                "question": detail.get("question", "")[:60],
                "issues": low_metrics,
                "metadata": detail.get("metadata", {}),
            })
    
    return weak


def compare_reports(baseline: Dict, current: Dict) -> Dict[str, Any]:
    """So sánh 2 reports, phát hiện regression."""
    comparison = {
        "baseline_samples": baseline.get("num_samples", 0),
        "current_samples": current.get("num_samples", 0),
        "metrics": {},
        "regressions": [],
        "improvements": [],
    }
    
    for metric_key, agg_key in [
        ("faithfulness", "avg_faithfulness"),
        ("answer_relevancy", "avg_answer_relevancy"),
        ("context_precision", "avg_context_precision"),
        ("context_recall", "avg_context_recall"),
        ("answer_similarity", "avg_answer_similarity"),
    ]:
        base_score = baseline.get(agg_key, 0)
        curr_score = current.get(agg_key, 0)
        diff = curr_score - base_score
        
        comparison["metrics"][metric_key] = {
            "baseline": base_score,
            "current": curr_score,
            "diff": diff,
            "pct_change": (diff / base_score * 100) if base_score > 0 else 0,
        }
        
        if diff < -REGRESSION_TOLERANCE:
            comparison["regressions"].append({
                "metric": metric_key,
                "baseline": base_score,
                "current": curr_score,
                "drop": abs(diff),
            })
        elif diff > REGRESSION_TOLERANCE:
            comparison["improvements"].append({
                "metric": metric_key,
                "baseline": base_score,
                "current": curr_score,
                "gain": diff,
            })
    
    comparison["has_regression"] = len(comparison["regressions"]) > 0
    return comparison


def print_quality_report(report: Dict, category_analysis: Dict, weak_samples: List, comparison: Optional[Dict] = None):
    """In báo cáo chất lượng tổng hợp."""
    print("\n" + "=" * 80)
    print("  [*] RAG QUALITY GUARD REPORT - VNPT Money Chatbot")
    print("=" * 80)
    print(f"  Timestamp : {report.get('timestamp', 'N/A')}")
    print(f"  Samples   : {report.get('num_samples', 0)}")
    print(f"  Eval Time : {report.get('total_eval_time_seconds', 0):.1f}s")
    
    # === Overall Metrics ===
    print("\n" + "-" * 80)
    print("  OVERALL METRICS")
    print("-" * 80)
    print(f"  {'Metric':<25} {'Score':>8} {'Threshold':>10} {'Status':>8}")
    print("  " + "-" * 55)
    
    all_pass = True
    metrics = {
        "Faithfulness": report.get("avg_faithfulness", 0),
        "Answer Relevancy": report.get("avg_answer_relevancy", 0),
        "Context Precision": report.get("avg_context_precision", 0),
        "Context Recall": report.get("avg_context_recall", 0),
        "Answer Similarity": report.get("avg_answer_similarity", 0),
    }
    
    threshold_map = {
        "Faithfulness": "faithfulness",
        "Answer Relevancy": "answer_relevancy",
        "Context Precision": "context_precision",
        "Context Recall": "context_recall",
        "Answer Similarity": "answer_similarity",
    }
    
    for name, score in metrics.items():
        threshold = QUALITY_THRESHOLDS[threshold_map[name]]
        status = "PASS" if score >= threshold else "FAIL"
        if score < threshold:
            all_pass = False
        print(f"  {name:<25} {score:>7.2%} {threshold:>9.0%} {status:>8}")
    
    # === Category Breakdown ===
    if category_analysis:
        print("\n" + "-" * 80)
        print("  BREAKDOWN BY CATEGORY")
        print("-" * 80)
        
        # Filter out difficulty categories for this section
        cat_only = {k: v for k, v in category_analysis.items() if not k.startswith("difficulty_")}
        diff_only = {k: v for k, v in category_analysis.items() if k.startswith("difficulty_")}
        
        print(f"  {'Category':<25} {'N':>3} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Recall':>7} {'Sim':>7}")
        print("  " + "-" * 70)
        for cat, scores in sorted(cat_only.items()):
            n = scores.get("count", 0)
            f = scores.get("faithfulness", 0)
            r = scores.get("answer_relevancy", 0)
            p = scores.get("context_precision", 0)
            c = scores.get("context_recall", 0)
            s = scores.get("answer_similarity", 0)
            # Flag low categories
            flag = " (!)" if min(f, r, p, c) < 0.70 else ""
            print(f"  {cat:<25} {n:>3} {f:>6.0%} {r:>7.0%} {p:>7.0%} {c:>7.0%} {s:>7.0%}{flag}")
        
        print(f"\n  {'Difficulty':<25} {'N':>3} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Recall':>7} {'Sim':>7}")
        print("  " + "-" * 70)
        for cat, scores in sorted(diff_only.items()):
            n = scores.get("count", 0)
            f = scores.get("faithfulness", 0)
            r = scores.get("answer_relevancy", 0)
            p = scores.get("context_precision", 0)
            c = scores.get("context_recall", 0)
            s = scores.get("answer_similarity", 0)
            flag = " (!)" if min(f, r, p, c) < 0.70 else ""
            print(f"  {cat:<25} {n:>3} {f:>6.0%} {r:>7.0%} {p:>7.0%} {c:>7.0%} {s:>7.0%}{flag}")
    
    # === Weak Samples ===
    if weak_samples:
        print("\n" + "-" * 80)
        print(f"  WEAK SAMPLES ({len(weak_samples)} samples below 80%)")
        print("-" * 80)
        for ws in weak_samples:
            cat = ws.get("metadata", {}).get("category", "?")
            diff = ws.get("metadata", {}).get("difficulty", "?")
            q = ws['question'].encode('ascii', 'replace').decode('ascii')[:60]
            print(f"  #{ws['index']:<3} [{cat}/{diff}] {q}")
            issues = ', '.join(ws['issues']).encode('ascii', 'replace').decode('ascii')
            print(f"       Issues: {issues}")
    else:
        print("\n  No weak samples detected (all >= 80%)")
    
    # === Comparison with Baseline ===
    if comparison:
        print("\n" + "-" * 80)
        print("  COMPARISON WITH BASELINE")
        print("-" * 80)
        print(f"  Baseline: {comparison['baseline_samples']} samples → Current: {comparison['current_samples']} samples")
        print(f"\n  {'Metric':<25} {'Baseline':>9} {'Current':>9} {'Change':>9}")
        print("  " + "-" * 55)
        for metric, data in comparison["metrics"].items():
            diff_str = f"{data['pct_change']:+.1f}%"
            indicator = "[+]" if data["diff"] > 0 else "[-]" if data["diff"] < -0.02 else "[=]"
            print(f"  {metric:<25} {data['baseline']:>8.2%} {data['current']:>9.2%} {diff_str:>8} {indicator}")
        
        if comparison["regressions"]:
            print(f"\n  [!] REGRESSION DETECTED in {len(comparison['regressions'])} metrics:")
            for reg in comparison["regressions"]:
                print(f"     - {reg['metric']}: {reg['baseline']:.2%} → {reg['current']:.2%} (↓{reg['drop']:.2%})")
        else:
            print(f"\n  No regressions detected")
    
    # === Summary ===
    print("\n" + "=" * 80)
    if all_pass and (not comparison or not comparison.get("has_regression")):
        print("  QUALITY GUARD: PASSED")
    else:
        print("  QUALITY GUARD: NEEDS ATTENTION")
        if not all_pass:
            print("     - Some metrics below threshold")
        if comparison and comparison.get("has_regression"):
            print("     - Regression detected vs baseline")
    print("=" * 80)


def run_eval(args):
    """Chạy evaluation trên expanded dataset."""
    from ragas_evaluation import (
        RAGASEvaluator,
        PipelineEvaluator,
        load_eval_dataset,
        save_eval_report,
    )
    from pipeline import create_pipeline
    
    # Load expanded dataset
    dataset_path = args.dataset or os.path.join(
        os.path.dirname(__file__), "eval_dataset_expanded.json"
    )
    test_data = load_eval_dataset(dataset_path)
    
    if args.limit:
        test_data = test_data[:args.limit]
    
    logger.info(f"Loaded {len(test_data)} samples from {dataset_path}")
    
    # Initialize pipeline
    pipeline = create_pipeline(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        redis_url=os.getenv("REDIS_URL"),
        use_llm=True,
        enable_monitoring=False,
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
        session_id="quality_guard_eval",
    )
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or os.path.join(
        os.path.dirname(__file__),
        f"quality_guard_report_{timestamp}.json",
    )
    
    # Add metadata to report details
    from dataclasses import asdict
    report_dict = asdict(report)
    for i, detail in enumerate(report_dict.get("details", [])):
        if i < len(test_data):
            detail["metadata"] = test_data[i].get("metadata", {})
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Report saved to {output_path}")
    
    # Analyze
    category_analysis = analyze_by_category(report_dict)
    weak_samples = find_weak_samples(report_dict)
    
    # Compare with baseline if provided
    comparison = None
    if args.baseline and os.path.exists(args.baseline):
        baseline = load_report(args.baseline)
        comparison = compare_reports(baseline, report_dict)
    
    # Print report
    print_quality_report(report_dict, category_analysis, weak_samples, comparison)
    
    return report_dict


def run_compare(args):
    """So sánh 2 reports."""
    if not args.baseline:
        print("ERROR: --baseline is required for compare mode")
        sys.exit(1)
    
    if not args.current:
        # Find latest quality guard report
        import glob
        reports = glob.glob(os.path.join(os.path.dirname(__file__), "quality_guard_report_*.json"))
        if not reports:
            reports = glob.glob(os.path.join(os.path.dirname(__file__), "eval_report_full_*.json"))
        if not reports:
            print("ERROR: No report files found. Run eval first.")
            sys.exit(1)
        args.current = max(reports, key=os.path.getmtime)
        print(f"Using latest report: {args.current}")
    
    baseline = load_report(args.baseline)
    current = load_report(args.current)
    
    category_analysis = analyze_by_category(current)
    weak_samples = find_weak_samples(current)
    comparison = compare_reports(baseline, current)
    
    print_quality_report(current, category_analysis, weak_samples, comparison)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Quality Guard - VNPT Money Chatbot"
    )
    parser.add_argument(
        "--mode",
        choices=["eval", "compare"],
        default="eval",
        help="Mode: eval (run evaluation) or compare (compare reports)",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None, 
                       help="Path to baseline report for comparison")
    parser.add_argument("--current", type=str, default=None,
                       help="Path to current report (for compare mode)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY required")
        sys.exit(1)
    
    if args.mode == "eval":
        run_eval(args)
    elif args.mode == "compare":
        run_compare(args)


if __name__ == "__main__":
    main()
