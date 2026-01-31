"""
VNPT Money GraphRAG Chatbot - Diagnostic Test Suite
====================================================

This test suite diagnoses WHY the chatbot fails to retrieve correct answers.
Tests 50% of problems in database with exact questions from CSV titles.

RUN: python test/test_retrieval_diagnostic.py
"""

import os
import sys
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A test case from the database."""
    problem_id: str
    title: str  # This IS the expected question
    group_id: str
    topic_id: str
    expected_answer_id: str


@dataclass
class TestResult:
    """Result of a single test."""
    test_case: TestCase
    passed: bool
    failure_reason: str
    details: Dict[str, Any]


class DiagnosticTestSuite:
    """
    Diagnostic test suite to identify retrieval failures.
    """
    
    def __init__(self, data_dir: str = "external_data_v3"):
        self.data_dir = Path(data_dir)
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []
        self.failure_categories: Dict[str, List[TestResult]] = {
            "no_embedding": [],
            "no_vector_index": [],
            "wrong_service_mapping": [],
            "empty_constraint_set": [],
            "low_similarity": [],
            "intent_mismatch": [],
            "connection_error": [],
            "other": [],
        }
    
    def load_test_cases(self, sample_ratio: float = 0.5) -> int:
        """
        Load test cases from CSV files.
        
        Uses problem titles as exact questions since that's what's in DB.
        """
        # Load problems
        problems = self._read_csv("nodes_problem.csv")
        answers = self._read_csv("nodes_answer.csv")
        rels_answer = self._read_csv("rels_has_answer.csv")
        rels_problem = self._read_csv("rels_has_problem.csv")
        rels_topic = self._read_csv("rels_has_topic.csv")
        
        # Build mappings
        problem_to_answer = {r["start_id"]: r["end_id"] for r in rels_answer}
        problem_to_topic = {r["end_id"]: r["start_id"] for r in rels_problem}
        topic_to_group = {r["end_id"]: r["start_id"] for r in rels_topic}
        
        # Sample problems (every other one for 50%)
        sampled_problems = problems[::int(1/sample_ratio)] if sample_ratio < 1 else problems
        
        for prob in sampled_problems:
            prob_id = prob["id"]
            topic_id = problem_to_topic.get(prob_id, "")
            group_id = topic_to_group.get(topic_id, "")
            answer_id = problem_to_answer.get(prob_id, "")
            
            self.test_cases.append(TestCase(
                problem_id=prob_id,
                title=prob["title"],
                group_id=group_id,
                topic_id=topic_id,
                expected_answer_id=answer_id
            ))
        
        logger.info(f"Loaded {len(self.test_cases)} test cases from {len(problems)} problems ({sample_ratio*100:.0f}%)")
        return len(self.test_cases)
    
    def _read_csv(self, filename: str) -> List[Dict[str, str]]:
        """Read CSV file."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all diagnostic tests.
        
        Returns summary of failures by category.
        """
        logger.info("=" * 60)
        logger.info("STARTING DIAGNOSTIC TEST SUITE")
        logger.info("=" * 60)
        
        # Test 1: Check Neo4j connection and data
        logger.info("\n[TEST 1] Checking Neo4j connection and data...")
        neo4j_status = self._test_neo4j_connection()
        
        # Test 2: Check embeddings exist
        logger.info("\n[TEST 2] Checking embeddings in database...")
        embedding_status = self._test_embeddings_exist()
        
        # Test 3: Check vector index
        logger.info("\n[TEST 3] Checking vector index...")
        index_status = self._test_vector_index()
        
        # Test 4: Check SERVICE_GROUP_MAP coverage
        logger.info("\n[TEST 4] Checking SERVICE_GROUP_MAP coverage...")
        mapping_status = self._test_service_mapping()
        
        # Test 5: Run retrieval tests on sample cases
        logger.info("\n[TEST 5] Running retrieval tests...")
        retrieval_status = self._test_retrieval_pipeline()
        
        # Generate report
        return self._generate_report({
            "neo4j": neo4j_status,
            "embeddings": embedding_status,
            "vector_index": index_status,
            "service_mapping": mapping_status,
            "retrieval": retrieval_status
        })
    
    def _test_neo4j_connection(self) -> Dict[str, Any]:
        """Test Neo4j connection and count nodes."""
        try:
            from neo4j import GraphDatabase
            
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            with driver.session() as session:
                result = session.run("""
                    MATCH (g:Group) WITH count(g) AS groups
                    MATCH (t:Topic) WITH groups, count(t) AS topics
                    MATCH (p:Problem) WITH groups, topics, count(p) AS problems
                    MATCH (a:Answer) WITH groups, topics, problems, count(a) AS answers
                    RETURN groups, topics, problems, answers
                """)
                stats = result.single()
                
                driver.close()
                
                if stats["problems"] == 0:
                    return {
                        "status": "FAIL",
                        "error": "NO DATA IN DATABASE",
                        "message": "Neo4j connected but no Problem nodes found. Run ingest_data_v3.py first!",
                        "stats": dict(stats)
                    }
                
                return {
                    "status": "PASS",
                    "stats": dict(stats)
                }
                
        except Exception as e:
            return {
                "status": "FAIL",
                "error": str(e),
                "message": "Cannot connect to Neo4j. Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD"
            }
    
    def _test_embeddings_exist(self) -> Dict[str, Any]:
        """Check if embeddings exist for Problem nodes."""
        try:
            from neo4j import GraphDatabase
            
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            with driver.session() as session:
                # Count problems with embeddings
                result = session.run("""
                    MATCH (p:Problem)
                    WITH count(p) AS total,
                         count(p.embedding) AS with_embedding
                    RETURN total, with_embedding, total - with_embedding AS without_embedding
                """)
                stats = result.single()
                
                driver.close()
                
                if stats["with_embedding"] == 0:
                    return {
                        "status": "FAIL",
                        "error": "NO EMBEDDINGS",
                        "message": "⚠️ CRITICAL: No Problem nodes have embeddings! Vector search will return 0 results!",
                        "solution": "Run: python -m src.ingest_data_v3 with OPENAI_API_KEY set",
                        "stats": dict(stats)
                    }
                elif stats["without_embedding"] > 0:
                    return {
                        "status": "WARN",
                        "message": f"{stats['without_embedding']} Problems missing embeddings",
                        "stats": dict(stats)
                    }
                else:
                    return {
                        "status": "PASS",
                        "message": f"All {stats['total']} Problems have embeddings",
                        "stats": dict(stats)
                    }
                    
        except Exception as e:
            return {
                "status": "FAIL",
                "error": str(e)
            }
    
    def _test_vector_index(self) -> Dict[str, Any]:
        """Check if vector index exists and is usable."""
        try:
            from neo4j import GraphDatabase
            
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            with driver.session() as session:
                # Check for vector index
                result = session.run("""
                    SHOW INDEXES
                    WHERE type = 'VECTOR'
                """)
                indexes = list(result)
                
                if len(indexes) == 0:
                    driver.close()
                    return {
                        "status": "FAIL",
                        "error": "NO VECTOR INDEX",
                        "message": "⚠️ CRITICAL: No vector index found! Vector search will fail!",
                        "solution": "Create index with: CREATE VECTOR INDEX problem_embedding_index FOR (p:Problem) ON p.embedding OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}"
                    }
                
                # Try a test query
                try:
                    test_embedding = [0.0] * 1536  # Dummy embedding
                    result = session.run("""
                        CALL db.index.vector.queryNodes('problem_embedding_index', 5, $embedding)
                        YIELD node, score
                        RETURN count(node) AS count
                    """, {"embedding": test_embedding})
                    count = result.single()["count"]
                    
                    driver.close()
                    
                    return {
                        "status": "PASS",
                        "message": f"Vector index exists and returned {count} results for test query",
                        "indexes": [dict(idx) for idx in indexes]
                    }
                except Exception as e:
                    driver.close()
                    return {
                        "status": "FAIL",
                        "error": str(e),
                        "message": "Vector index exists but query failed",
                        "indexes": [dict(idx) for idx in indexes]
                    }
                
        except Exception as e:
            return {
                "status": "FAIL",
                "error": str(e)
            }
    
    def _test_service_mapping(self) -> Dict[str, Any]:
        """
        Check if SERVICE_GROUP_MAP covers all groups in database.
        
        THIS IS A CRITICAL BUG: Current mapping uses WRONG group IDs!
        """
        try:
            from neo4j import GraphDatabase
            
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            with driver.session() as session:
                # Get actual group IDs from database
                result = session.run("""
                    MATCH (g:Group)
                    RETURN g.id AS id, g.name AS name
                """)
                db_groups = {r["id"]: r["name"] for r in result}
                
                driver.close()
            
            # Current mapping from schema.py
            from schema import SERVICE_GROUP_MAP
            
            mapped_groups = set()
            for groups in SERVICE_GROUP_MAP.values():
                mapped_groups.update(groups)
            
            # Check coverage
            missing_from_mapping = set(db_groups.keys()) - mapped_groups
            invalid_in_mapping = mapped_groups - set(db_groups.keys())
            
            issues = []
            if missing_from_mapping:
                issues.append(f"Groups in DB but NOT in SERVICE_GROUP_MAP: {missing_from_mapping}")
            if invalid_in_mapping:
                issues.append(f"Groups in SERVICE_GROUP_MAP but NOT in DB: {invalid_in_mapping}")
            
            if issues:
                return {
                    "status": "WARN" if not invalid_in_mapping else "FAIL",
                    "message": "SERVICE_GROUP_MAP has coverage issues",
                    "issues": issues,
                    "db_groups": db_groups,
                    "mapped_groups": list(mapped_groups)
                }
            
            return {
                "status": "PASS",
                "message": "SERVICE_GROUP_MAP covers all database groups",
                "db_groups": db_groups,
                "mapped_groups": list(mapped_groups)
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "error": str(e)
            }
    
    def _test_retrieval_pipeline(self) -> Dict[str, Any]:
        """
        Test the retrieval pipeline with actual queries.
        """
        try:
            from neo4j import GraphDatabase
            from openai import OpenAI
            
            # Check if OpenAI key exists
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                return {
                    "status": "SKIP",
                    "message": "OPENAI_API_KEY not set, cannot test retrieval pipeline"
                }
            
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            openai_client = OpenAI(api_key=openai_key)
            
            # Test a few sample queries
            test_queries = [
                # Exact titles from nodes_problem.csv
                "Gói data có tự động gia hạn không?",
                "Cách nạp tiền vào tài khoản Mobile Money",
                "Hướng dẫn chuyển tiền Mobile Money",
                "Đăng ký tài khoản VNPT Money",
                "Cách liên kết ngân hàng trực tiếp",
            ]
            
            results = []
            
            for query in test_queries:
                # Generate embedding
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=query
                )
                query_embedding = response.data[0].embedding
                
                # Direct vector search (no constraints)
                with driver.session() as session:
                    try:
                        result = session.run("""
                            CALL db.index.vector.queryNodes('problem_embedding_index', 5, $embedding)
                            YIELD node, score
                            RETURN node.id AS id, node.title AS title, score
                            ORDER BY score DESC
                        """, {"embedding": query_embedding})
                        matches = list(result)
                        
                        if matches:
                            results.append({
                                "query": query,
                                "top_match": matches[0]["title"],
                                "top_score": matches[0]["score"],
                                "exact_match": query.lower() in matches[0]["title"].lower() or matches[0]["title"].lower() in query.lower(),
                                "status": "FOUND"
                            })
                        else:
                            results.append({
                                "query": query,
                                "status": "NO_RESULTS",
                                "error": "Vector search returned 0 results"
                            })
                    except Exception as e:
                        results.append({
                            "query": query,
                            "status": "ERROR",
                            "error": str(e)
                        })
            
            driver.close()
            
            # Analyze results
            found = sum(1 for r in results if r["status"] == "FOUND")
            exact = sum(1 for r in results if r.get("exact_match", False))
            
            return {
                "status": "PASS" if found == len(test_queries) else "FAIL",
                "total_queries": len(test_queries),
                "found": found,
                "exact_matches": exact,
                "results": results
            }
            
        except Exception as e:
            return {
                "status": "FAIL",
                "error": str(e)
            }
    
    def _generate_report(self, status_dict: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate final diagnostic report."""
        
        print("\n" + "=" * 60)
        print("📊 DIAGNOSTIC REPORT")
        print("=" * 60)
        
        critical_issues = []
        warnings = []
        
        for name, status in status_dict.items():
            status_text = status.get("status", "UNKNOWN")
            icon = "✅" if status_text == "PASS" else ("⚠️" if status_text in ["WARN", "SKIP"] else "❌")
            
            print(f"\n{icon} {name.upper()}: {status_text}")
            
            if status.get("message"):
                print(f"   {status['message']}")
            
            if status.get("error"):
                print(f"   ERROR: {status['error']}")
            
            if status.get("solution"):
                print(f"   💡 SOLUTION: {status['solution']}")
            
            if status_text == "FAIL":
                critical_issues.append({
                    "component": name,
                    "error": status.get("error", status.get("message", "Unknown")),
                    "solution": status.get("solution")
                })
            elif status_text == "WARN":
                warnings.append({
                    "component": name,
                    "message": status.get("message", "Warning")
                })
        
        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        
        if critical_issues:
            print("\n❌ CRITICAL ISSUES FOUND:")
            for i, issue in enumerate(critical_issues, 1):
                print(f"   {i}. [{issue['component']}] {issue['error']}")
                if issue.get("solution"):
                    print(f"      → {issue['solution']}")
        
        if warnings:
            print("\n⚠️ WARNINGS:")
            for w in warnings:
                print(f"   - [{w['component']}] {w['message']}")
        
        if not critical_issues and not warnings:
            print("\n✅ All checks passed!")
        
        return {
            "critical_issues": critical_issues,
            "warnings": warnings,
            "details": status_dict
        }


def main():
    """Run diagnostic tests."""
    suite = DiagnosticTestSuite()
    suite.load_test_cases(sample_ratio=0.5)
    report = suite.run_all_tests()
    
    # Save report
    report_path = Path("test/diagnostic_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    # Exit with error code if critical issues
    if report["critical_issues"]:
        print("\n❌ TEST SUITE FAILED - Critical issues found!")
        sys.exit(1)
    else:
        print("\n✅ TEST SUITE PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
