"""
Graph Schema:
    Nodes:
        - Solution (1-1 mapping with qa.jsonl)
        - Service (business actions: nap_tien, rut_tien, etc.)
        - Problem (problem types: huong_dan, that_bai, etc.)
        - Bank (from supported_banks.json)
        - Outcome (result states: money_deducted, not_received, etc.)
    
    Relationships:
        - (Service)-[:HAS_PROBLEM]->(Problem)
        - (Problem)-[:HAS_SOLUTION]->(Solution)
        - (Solution)-[:APPLIES_TO]->(Bank)  [optional, if bank-specific]
        - (Solution)-[:HAS_OUTCOME]->(Outcome)  [optional, if outcome-specific]

Retrieval Tiers (fallback logic):
    Tier 4: Service + Problem + Outcome + Bank
    Tier 3: Service + Problem + Outcome
    Tier 2: Service + Problem + Bank
    Tier 1: Service + Problem
    Tier 0: Service only (generic)
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Tuple
from neo4j import GraphDatabase, Driver
import logging

from normalize_content import (
    ServiceTaxonomy,
    ProblemTypeTaxonomy,
    StateTaxonomy,
    OutcomeTaxonomy,
    BankTaxonomy,
    UnifiedNormalizer,
)

logger = logging.getLogger(__name__)

class Neo4jConfig:
    """Neo4j connection configuration."""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7999",
        user: str = "neo4j",
        password: str = "YourStrongPassword123",
        database: str = "neo4j",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
    
    @classmethod
    def from_env(cls) -> Neo4jConfig:
        """Load configuration from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7999"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "YourStrongPassword123"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


# ============================================================================
# Connection Manager
# ============================================================================

class Neo4jConnection:
    """Manages Neo4j driver connection."""
    
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self._driver: Optional[Driver] = None
    
    def connect(self) -> Driver:
        """Establish connection to Neo4j."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
            )
            logger.info(f"Connected to Neo4j at {self.config.uri}")
        return self._driver
    
    def close(self):
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
    
    def verify_connectivity(self) -> bool:
        """Verify connection to Neo4j."""
        try:
            driver = self.connect()
            driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {e}")
            return False
    
    def get_session(self):
        """Get a Neo4j session."""
        driver = self.connect()
        return driver.session(database=self.config.database)


# ============================================================================
# Schema Manager
# ============================================================================

class SchemaManager:
    """Manages Neo4j schema (constraints and indexes)."""
    
    def __init__(self, connection: Neo4jConnection):
        self.connection = connection
    
    def drop_all_constraints(self):
        """Drop all existing constraints."""
        with self.connection.get_session() as session:
            # Get list of all constraints
            result = session.run("SHOW CONSTRAINTS")
            constraints = [record["name"] for record in result]
            
            # Drop each constraint
            for constraint_name in constraints:
                try:
                    session.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                    logger.info(f"Dropped constraint: {constraint_name}")
                except Exception as e:
                    logger.warning(f"Failed to drop constraint {constraint_name}: {e}")
    
    def create_constraints(self):
        """Create uniqueness constraints for all node types."""
        constraints = [
            "CREATE CONSTRAINT solution_id IF NOT EXISTS FOR (s:Solution) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT service_name IF NOT EXISTS FOR (s:Service) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT problem_name IF NOT EXISTS FOR (p:Problem) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT state_name IF NOT EXISTS FOR (st:State) REQUIRE st.name IS UNIQUE",
            "CREATE CONSTRAINT outcome_name IF NOT EXISTS FOR (o:Outcome) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT bank_id IF NOT EXISTS FOR (b:Bank) REQUIRE b.bank_id IS UNIQUE",
        ]
        
        with self.connection.get_session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")
    
    def create_indexes(self):
        """Create indexes for performance optimization."""
        indexes = [
            "CREATE INDEX solution_qnorm IF NOT EXISTS FOR (s:Solution) ON (s.question_norm)",
            "CREATE INDEX solution_bank IF NOT EXISTS FOR (s:Solution) ON (s.bank_id)",
            "CREATE INDEX solution_sheet IF NOT EXISTS FOR (s:Solution) ON (s.sheet)",
            "CREATE INDEX solution_section IF NOT EXISTS FOR (s:Solution) ON (s.section)",
        ]
        
        with self.connection.get_session() as session:
            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"Created index: {index}")
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")
    
    def setup_schema(self):
        """Setup complete schema (constraints + indexes)."""
        logger.info("Setting up Neo4j schema...")
        self.drop_all_constraints()  # Drop old constraints first
        self.create_constraints()
        self.create_indexes()
        logger.info("Schema setup complete")
    
    def clear_database(self):
        """Clear all nodes and relationships (USE WITH CAUTION)."""
        with self.connection.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Database cleared")


# ============================================================================
# Data Ingestion
# ============================================================================

class DataIngestor:
    """Ingests data from qa.jsonl and supported_banks.json into Neo4j."""
    
    def __init__(self, connection: Neo4jConnection):
        self.connection = connection
    
    def ingest_banks(self, banks_path: Optional[str] = None):
        """Ingest bank nodes from supported_banks.json."""
        if not BankTaxonomy._loaded:
            BankTaxonomy.load_from_json(banks_path)
        
        query = """
        UNWIND $banks AS bank
        MERGE (b:Bank {bank_id: bank.bank_id})
        SET b.name = bank.name,
            b.supported = bank.supported,
            b.aliases = bank.aliases
        """
        
        banks_data = [
            {
                "bank_id": bank_id,
                "name": data["display"],
                "supported": data.get("supported", True),
                "aliases": data.get("aliases", []),
            }
            for bank_id, data in BankTaxonomy.BANKS.items()
        ]
        
        with self.connection.get_session() as session:
            result = session.run(query, banks=banks_data)
            logger.info(f"Ingested {len(banks_data)} banks")
    
    def ingest_qa_jsonl(self, qa_path: str):
        """
        Ingest QA rules from qa.jsonl into Neo4j graph.
        
        For each record:
        1. Create/merge Service node
        2. Create/merge Problem node
        3. Create Solution node
        4. Create relationships: Service -> Problem -> Solution
        5. If bank_id exists, link Solution -> Bank
        6. If outcome exists (not 'unknown'), link Solution -> Outcome
        """
        if not os.path.exists(qa_path):
            raise FileNotFoundError(f"QA file not found: {qa_path}")
        
        records = []
        with open(qa_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
        
        logger.info(f"Loaded {len(records)} records from {qa_path}")
        
        # Process each record
        ingested_count = 0
        skipped_count = 0
        
        for record in records:
            try:
                if self._ingest_single_qa(record):
                    ingested_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                logger.error(f"Failed to ingest record {record.get('id')}: {e}")
                skipped_count += 1
        
        logger.info(f"Ingestion complete: {ingested_count} ingested, {skipped_count} skipped")
    
    def _ingest_single_qa(self, record: Dict) -> bool:
        """
        Ingest a single QA record.
        Returns True if ingested, False if skipped.
        """
        qa_id = record.get("id")
        if not qa_id:
            logger.warning("Record missing 'id', skipping")
            return False
        
        # Extract labels
        labels = record.get("labels", {})
        service_raw = labels.get("service")
        problem_raw = labels.get("problem")
        
        # Normalize service and problem using taxonomy
        service = UnifiedNormalizer.normalize_service(service_raw)
        problem = UnifiedNormalizer.normalize_problem(problem_raw)
        
        if not service:
            logger.warning(f"Record {qa_id}: invalid/missing service '{service_raw}', skipping")
            return False
        
        if not problem:
            logger.warning(f"Record {qa_id}: invalid/missing problem '{problem_raw}', skipping")
            return False
        
        # CRITICAL: Remove Vietnamese accents to ensure consistency with query params
        # Neo4j will store: "rut_tien" not "rút_tiền"
        from normalize_content import _norm_text
        service = _norm_text(service)  # "rút_tiền" -> "rut_tien"
        problem = _norm_text(problem)  # "bị_trừ_tiền" -> "bi_tru_tien"
        
        # Get display names for nodes (keep accents for display)
        service_display = ServiceTaxonomy.to_display(
            ServiceTaxonomy.from_slot_key(service)
        ) or service
        problem_display = ProblemTypeTaxonomy.to_display(
            ProblemTypeTaxonomy.from_slot_key(problem)
        ) or problem
        
        # Extract other fields
        question = record.get("text", "")
        question_norm = record.get("text_norm", question.lower())
        answer = record.get("answer", "")
        source = record.get("source", {})
        
        # Bank (optional)
        bank_id_raw = record.get("bank_id")
        bank_id = None
        if bank_id_raw and bank_id_raw.lower() not in ("null", "none", ""):
            bank_pair = UnifiedNormalizer.normalize_bank_pair(bank_id_raw)
            if bank_pair:
                bank_id = bank_pair[0]  # canonical bank_id
        
        # Outcome (optional)
        outcome_raw = labels.get("outcome")
        outcome = None
        if outcome_raw and outcome_raw.lower() not in ("unknown", "null", "none", ""):
            outcome_cid = OutcomeTaxonomy.from_slot_key(outcome_raw)
            if outcome_cid:
                outcome = OutcomeTaxonomy.to_slot_key(outcome_cid)
                # Normalize outcome to remove accents
                from normalize_content import _norm_text
                outcome = _norm_text(outcome)
        
        # State (for potential future use, not used in current schema)
        state_raw = labels.get("state")
        state = None
        if state_raw:
            state_cid = StateTaxonomy.from_slot_key(state_raw)
            if state_cid:
                state = StateTaxonomy.to_slot_key(state_cid)
                # Normalize state to remove accents
                from normalize_content import _norm_text
                state = _norm_text(state)
        
        # Build the cypher query
        self._create_solution_graph(
            qa_id=qa_id,
            service=service,
            service_display=service_display,
            problem=problem,
            problem_display=problem_display,
            question=question,
            question_norm=question_norm,
            answer=answer,
            source=source,
            bank_id=bank_id,
            outcome=outcome,
            state=state,
        )
        
        return True
    
    def _create_solution_graph(
        self,
        qa_id: str,
        service: str,
        service_display: str,
        problem: str,
        problem_display: str,
        question: str,
        question_norm: str,
        answer: str,
        source: Dict,
        bank_id: Optional[str],
        outcome: Optional[str],
        state: Optional[str],
    ):
        """Create graph: Direct Solution->Service/Problem/State/Outcome relationships.
        
        New architecture: Solution as central node with direct relationships to taxonomy nodes.
        This allows flexible tiered retrieval without rigid hierarchy constraints.
        
        Relationships:
            (sol:Solution)-[:OF_SERVICE]->(s:Service)
            (sol)-[:OF_PROBLEM]->(p:Problem)
            (sol)-[:OF_STATE]->(st:State)
            (sol)-[:OF_OUTCOME]->(o:Outcome)
        """
        
        query = """
        // Create Solution node
        MERGE (sol:Solution {id: $qa_id})
        SET
            sol.question = $question,
            sol.question_norm = $question_norm,
            sol.answer = $answer,
            sol.sheet = $sheet,
            sol.section = $section,
            sol.bank_id = $bank_id
        
        // Create taxonomy nodes
        WITH sol
        MERGE (s:Service {name: $service})
        ON CREATE SET s.display_name = $service_display
        
        WITH sol, s
        MERGE (p:Problem {name: $problem})
        ON CREATE SET p.display_name = $problem_display
        
        WITH sol, s, p
        MERGE (st:State {name: $state})
        
        WITH sol, s, p, st
        MERGE (o:Outcome {name: $outcome})
        
        // Create relationships from Solution to taxonomies
        WITH sol, s, p, st, o
        MERGE (sol)-[:OF_SERVICE]->(s)
        MERGE (sol)-[:OF_PROBLEM]->(p)
        MERGE (sol)-[:OF_STATE]->(st)
        MERGE (sol)-[:OF_OUTCOME]->(o)
        
        RETURN sol.id AS qa_id
        """
        
        params = {
            "qa_id": qa_id,
            "service": service,
            "service_display": service_display,
            "problem": problem,
            "problem_display": problem_display,
            "question": question,
            "question_norm": question_norm,
            "answer": answer,
            "sheet": source.get("sheet", ""),
            "section": source.get("section", ""),
            "bank_id": bank_id,
            "state": state or "unknown",
            "outcome": outcome or "unknown",
        }
        
        with self.connection.get_session() as session:
            result = session.run(query, **params)
            record = result.single()
            if record:
                logger.debug(f"Created solution graph for QA: {record['qa_id']}")


# ============================================================================
# Query Engine (Tiered Retrieval)
# ============================================================================

class QueryEngine:
    """
    Tiered retrieval engine with unified scoring.
    
    New architecture with 4 tiers:
        Tier 4: service + problem + state + outcome (+ bank)
        Tier 3: service + problem (+ optional state/outcome bonus) (+ bank)
        Tier 2: service only (+ bank)
        Tier 1: problem only (+ bank)
    
    All tiers are evaluated in a single query with UNION, returning unified scores.
    """
    
    def __init__(self, connection: Neo4jConnection):
        self.connection = connection
    
    def retrieve_solutions(
        self,
        service: Optional[str] = None,
        problem: Optional[str] = None,
        bank_id: Optional[str] = None,
        outcome: Optional[str] = None,
        state: Optional[str] = None,
        q_norm: Optional[str] = None,
        limit: int = 5,
        service_confidence: float = 0.0,
        problem_confidence: float = 0.0,
    ) -> List[Dict]:
        """Retrieve solutions using tiered retrieval with OPTIONAL filters.
        
        CRITICAL: Filters are optional to prevent "lọc chết" (filtering out correct answers).
        If confidence < 0.60, filter is disabled (NULL param) to allow rerank to decide.
        
        Args:
            service: Service slot (e.g., 'nap_tien') - NULL if confidence < 0.60
            problem: Problem slot (e.g., 'huong_dan') - NULL if confidence < 0.60
            state: State slot (e.g., 'failed', 'success')
            outcome: Outcome slot (e.g., 'money_deducted')
            bank_id: Bank ID (e.g., 'VCB')
            q_norm: Normalized question text for lexical matching bonus
            limit: Maximum number of solutions to return
            service_confidence: Confidence in service detection (for logging)
            problem_confidence: Confidence in problem detection (for logging)
        
        Returns:
            List of solution dictionaries with scores, tier, and slot information for reranking
        """
        # Allow retrieval even with no slots (will use semantic matching only)
        if not service and not problem:
            logger.warning("No service or problem provided - using semantic-only retrieval")
        
        query = """
        WITH
            $service  AS service,
            $problem  AS problem,
            $state    AS state,
            $outcome  AS outcome,
            $bank_id  AS bank_id,
            $q_norm   AS q_norm
        
        CALL {
            WITH service, problem, state, outcome, bank_id, q_norm
            
            // ========== OPTIONAL FILTERS: Use NULL to disable filter ==========
            // WHERE ($service IS NULL OR s.name = $service)
            // This allows broader retrieval when classification confidence is low
            
            // ---------- TIER 4: service + problem + bank (state/outcome as bonuses) ----------
            MATCH (sol:Solution)-[:OF_SERVICE]->(svc:Service)
            MATCH (sol)-[:OF_PROBLEM]->(prb:Problem)
            OPTIONAL MATCH (sol)-[:OF_STATE]->(st:State)
            OPTIONAL MATCH (sol)-[:OF_OUTCOME]->(out:Outcome)
            WHERE ($service IS NULL OR svc.name = $service)
              AND ($problem IS NULL OR prb.name = $problem)
              AND ($bank_id IS NULL OR sol.bank_id IS NULL OR sol.bank_id = $bank_id)
            WITH sol, svc.name AS matched_service, prb.name AS matched_problem,
                 st.name AS matched_state, out.name AS matched_outcome,
                 4 AS tier, 
                 100 + CASE WHEN st.name = $state AND $state IS NOT NULL AND $state <> 'unknown' THEN 5 ELSE 0 END 
                     + CASE WHEN out.name = $outcome AND $outcome IS NOT NULL AND $outcome <> 'unknown' THEN 5 ELSE 0 END AS base_score
            RETURN sol, matched_service, matched_problem, matched_state, matched_outcome, tier, base_score
            
            UNION
            
            // ---------- TIER 3: service + problem (no bank filter) ----------
            MATCH (sol:Solution)-[:OF_SERVICE]->(svc:Service)
            MATCH (sol)-[:OF_PROBLEM]->(prb:Problem)
            OPTIONAL MATCH (sol)-[:OF_STATE]->(st:State)
            OPTIONAL MATCH (sol)-[:OF_OUTCOME]->(out:Outcome)
            WHERE ($service IS NULL OR svc.name = $service)
              AND ($problem IS NULL OR prb.name = $problem)
            WITH sol, svc.name AS matched_service, prb.name AS matched_problem,
                 st.name AS matched_state, out.name AS matched_outcome,
                 3 AS tier, 
                 70 + CASE WHEN st.name = $state AND $state IS NOT NULL AND $state <> 'unknown' THEN 3 ELSE 0 END 
                    + CASE WHEN out.name = $outcome AND $outcome IS NOT NULL AND $outcome <> 'unknown' THEN 3 ELSE 0 END
                    + CASE WHEN sol.bank_id = $bank_id AND $bank_id IS NOT NULL THEN 5 ELSE 0 END AS base_score
            RETURN sol, matched_service, matched_problem, matched_state, matched_outcome, tier, base_score
            
            UNION
            
            // ---------- TIER 2: service only (+ bank bonus) ----------
            MATCH (sol:Solution)-[:OF_SERVICE]->(svc:Service)
            OPTIONAL MATCH (sol)-[:OF_PROBLEM]->(prb:Problem)
            OPTIONAL MATCH (sol)-[:OF_STATE]->(st:State)
            OPTIONAL MATCH (sol)-[:OF_OUTCOME]->(out:Outcome)
            WHERE ($service IS NULL OR svc.name = $service)
              AND ($bank_id IS NULL OR sol.bank_id IS NULL OR sol.bank_id = $bank_id)
            WITH sol, svc.name AS matched_service, prb.name AS matched_problem,
                 st.name AS matched_state, out.name AS matched_outcome,
                 2 AS tier, 40 AS base_score
            RETURN sol, matched_service, matched_problem, matched_state, matched_outcome, tier, base_score
            
            UNION
            
            // ---------- TIER 1: problem only (+ bank bonus) ----------
            MATCH (sol:Solution)-[:OF_PROBLEM]->(prb:Problem)
            OPTIONAL MATCH (sol)-[:OF_SERVICE]->(svc:Service)
            OPTIONAL MATCH (sol)-[:OF_STATE]->(st:State)
            OPTIONAL MATCH (sol)-[:OF_OUTCOME]->(out:Outcome)
            WHERE ($problem IS NULL OR prb.name = $problem)
              AND ($bank_id IS NULL OR sol.bank_id IS NULL OR sol.bank_id = $bank_id)
            WITH sol, svc.name AS matched_service, prb.name AS matched_problem,
                 st.name AS matched_state, out.name AS matched_outcome,
                 1 AS tier, 30 AS base_score
            RETURN sol, matched_service, matched_problem, matched_state, matched_outcome, tier, base_score
            
            UNION
            
            // ---------- TIER 0: bank only (ultimate fallback) ----------
            MATCH (sol:Solution)
            OPTIONAL MATCH (sol)-[:OF_SERVICE]->(svc:Service)
            OPTIONAL MATCH (sol)-[:OF_PROBLEM]->(prb:Problem)
            OPTIONAL MATCH (sol)-[:OF_STATE]->(st:State)
            OPTIONAL MATCH (sol)-[:OF_OUTCOME]->(out:Outcome)
            WHERE $bank_id IS NOT NULL AND sol.bank_id = $bank_id
            WITH sol, svc.name AS matched_service, prb.name AS matched_problem,
                 st.name AS matched_state, out.name AS matched_outcome,
                 0 AS tier, 20 AS base_score
            RETURN sol, matched_service, matched_problem, matched_state, matched_outcome, tier, base_score
        }
        
        WITH sol, matched_service, matched_problem, matched_state, matched_outcome, tier, base_score, q_norm
        WHERE sol IS NOT NULL
        
        // Lexical bonus for question similarity
        WITH sol, matched_service, matched_problem, matched_state, matched_outcome, tier,
            base_score
            + CASE
                WHEN q_norm IS NULL OR q_norm = '' THEN 0
                WHEN sol.question_norm = q_norm THEN 20
                WHEN sol.question_norm CONTAINS q_norm THEN 8
                WHEN q_norm CONTAINS sol.question_norm THEN 5
                ELSE 0
              END AS score
        
        ORDER BY score DESC, tier DESC, sol.id ASC
        
        // Return rich features for reranking
        RETURN
            sol.id            AS qa_id,
            sol.question      AS question,
            sol.question_norm AS question_norm,
            sol.answer        AS answer,
            sol.bank_id       AS bank_id,
            sol.sheet         AS sheet,
            sol.section       AS section,
            matched_service   AS service,
            matched_problem   AS problem,
            matched_state     AS state,
            matched_outcome   AS outcome,
            tier,
            score
        LIMIT $limit
        """
        
        params = {
            "service": service,
            "problem": problem,
            "state": state,
            "outcome": outcome,
            "bank_id": bank_id,
            "q_norm": q_norm,
            "limit": limit,
        }
        
        results = self._execute_query(query, **params)
        if results:
            logger.info(
                f"Retrieved {len(results)} solutions | "
                f"Filters: service={service} (conf={service_confidence:.2f}), "
                f"problem={problem} (conf={problem_confidence:.2f}), "
                f"bank={bank_id} | "
                f"Top: tier={results[0].get('tier', '?')}, score={results[0].get('score', 0):.1f}"
            )
        else:
            logger.warning(
                f"No solutions found | "
                f"Filters: service={service}, problem={problem}, bank={bank_id}"
            )
        
        return results
    
    def _execute_query(self, query: str, **params) -> List[Dict]:
        """Execute a query and return results as list of dictionaries."""
        try:
            with self.connection.get_session() as session:
                result = session.run(query, **params)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def get_solution_by_id(self, qa_id: str) -> Optional[Dict]:
        """Retrieve a specific solution by its QA ID."""
        query = """
        MATCH (sol:Solution {id: $qa_id})
        OPTIONAL MATCH (sol)-[:OF_SERVICE]->(service:Service)
        OPTIONAL MATCH (sol)-[:OF_PROBLEM]->(problem:Problem)
        OPTIONAL MATCH (sol)-[:OF_STATE]->(state:State)
        OPTIONAL MATCH (sol)-[:OF_OUTCOME]->(outcome:Outcome)
        RETURN sol.id AS qa_id,
               sol.question AS question,
               sol.question_norm AS question_norm,
               sol.answer AS answer,
               sol.bank_id AS bank_id,
               sol.sheet AS sheet,
               sol.section AS section,
               service.name AS service,
               problem.name AS problem,
               state.name AS state,
               outcome.name AS outcome
        """
        results = self._execute_query(query, qa_id=qa_id)
        return results[0] if results else None


# ============================================================================
# High-Level API
# ============================================================================

class Neo4jRuleEngine:
    """High-level API for Neo4j rule-based chatbot system."""
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        if config is None:
            config = Neo4jConfig.from_env()
        
        self.connection = Neo4jConnection(config)
        self.schema_manager = SchemaManager(self.connection)
        self.ingestor = DataIngestor(self.connection)
        self.query_engine = QueryEngine(self.connection)
    
    def initialize_database(self, clear_existing: bool = False):
        """Initialize database schema."""
        if clear_existing:
            logger.warning("Clearing existing database...")
            self.schema_manager.clear_database()
        
        self.schema_manager.setup_schema()
    
    def ingest_all_data(
        self,
        qa_path: str,
        banks_path: Optional[str] = None,
    ):
        """Ingest all data (banks + QA rules) into Neo4j."""
        logger.info("Starting data ingestion...")
        
        # Ingest banks first (referenced by QA rules)
        self.ingestor.ingest_banks(banks_path)
        
        # Ingest QA rules
        self.ingestor.ingest_qa_jsonl(qa_path)
        
        logger.info("Data ingestion complete")
    
    def retrieve(
        self,
        service: Optional[str] = None,
        problem: Optional[str] = None,
        bank_id: Optional[str] = None,
        outcome: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 10,
        service_confidence: float = 0.0,
        problem_confidence: float = 0.0,
    ) -> List[Dict]:
        """Retrieve solutions using tiered retrieval with optional filters."""
        return self.query_engine.retrieve_solutions(
            service=service,
            problem=problem,
            bank_id=bank_id,
            outcome=outcome,
            state=state,
            limit=limit,
            service_confidence=service_confidence,
            problem_confidence=problem_confidence,
        )
    
    def get_solution(self, qa_id: str) -> Optional[Dict]:
        """Get a specific solution by ID."""
        return self.query_engine.get_solution_by_id(qa_id)
    
    def close(self):
        """Close the Neo4j connection."""
        self.connection.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# Utility Functions
# ============================================================================

def get_default_qa_path() -> str:
    """Get default path to qa.jsonl."""
    here = os.path.dirname(__file__)
    parent = os.path.dirname(here)
    return os.path.join(parent, "external_data", "qa.jsonl")


def get_default_banks_path() -> str:
    """Get default path to supported_banks.json."""
    here = os.path.dirname(__file__)
    parent = os.path.dirname(here)
    return os.path.join(parent, "external_data", "supported_banks.json")


# ============================================================================
# CLI/Testing Interface
# ============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    # Initialize engine
    engine = Neo4jRuleEngine()
    
    try:
        # Check connectivity
        if not engine.connection.verify_connectivity():
            logger.error("Cannot connect to Neo4j. Please check your configuration.")
            exit(1)
        
        print("\n" + "="*80)
        print("Neo4j Rule Engine - VNPT Money Chatbot")
        print("="*80)
        
        # Check if we need to initialize (simple check: try to count solutions)
        with engine.connection.get_session() as session:
            result = session.run("MATCH (s:Solution) RETURN count(s) as count")
            solution_count = result.single()["count"]
        
        if solution_count == 0:
            print("\n⚠️  Database is empty. Initializing and ingesting data...")
            print("-" * 80)
            
            # Initialize schema
            engine.initialize_database(clear_existing=False)
            
            # Ingest data
            engine.ingest_all_data(
                qa_path=get_default_qa_path(),
                banks_path=get_default_banks_path(),
            )
            
            print("\n✅ Database initialized successfully!")
            print(f"📊 Total solutions ingested: ", end="")
            
            with engine.connection.get_session() as session:
                result = session.run("MATCH (s:Solution) RETURN count(s) as count")
                solution_count = result.single()["count"]
                print(solution_count)
        else:
            print(f"\n✅ Database already initialized with {solution_count} solutions")
        
        # Example retrieval
        print("\n" + "="*80)
        print("Example 1: Retrieve solutions for 'nạp tiền' + 'hướng dẫn'")
        print("="*80)
        
        solutions = engine.retrieve(
            service="nạp_tiền",
            problem="huong_dan",
            limit=3,
        )
        
        if solutions:
            for i, sol in enumerate(solutions, 1):
                print(f"\n--- Solution {i} (Tier: {sol.get('tier')}) ---")
                print(f"QA ID: {sol['qa_id']}")
                print(f"Question: {sol['question']}")
                print(f"Answer: {sol['answer'][:150]}...")
                print(f"Service: {sol['service']}, Problem: {sol['problem']}")
        else:
            print("\n❌ No solutions found")
        
        print("\n" + "="*80)
        print("Example 2: Retrieve bank-specific solution (CTG/VietinBank)")
        print("="*80)
        
        solutions = engine.retrieve(
            service="nạp_tiền",
            problem="khac",
            bank_id="CTG",
            limit=2,
        )
        
        if solutions:
            for i, sol in enumerate(solutions, 1):
                print(f"\n--- Solution {i} (Tier: {sol.get('tier')}) ---")
                print(f"QA ID: {sol['qa_id']}")
                print(f"Bank: {sol.get('bank_name', 'N/A')} ({sol.get('bank_id', 'N/A')})")
                print(f"Question: {sol['question']}")
                print(f"Answer: {sol['answer'][:150]}...")
        else:
            print("\n❌ No solutions found")
        
        print("\n" + "="*80)
        print("Example 3: Retrieve with outcome 'money_deducted'")
        print("="*80)
        
        solutions = engine.retrieve(
            service="nạp_tiền",
            problem="huong_dan",
            outcome="money_deducted",
            limit=2,
        )
        
        if solutions:
            for i, sol in enumerate(solutions, 1):
                print(f"\n--- Solution {i} (Tier: {sol.get('tier')}) ---")
                print(f"QA ID: {sol['qa_id']}")
                print(f"Outcome: {sol.get('outcome', 'N/A')}")
                print(f"Answer: {sol['answer'][:150]}...")
        else:
            print("\n❌ No solutions found")
        
        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        print("="*80 + "\n")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        engine.close()
