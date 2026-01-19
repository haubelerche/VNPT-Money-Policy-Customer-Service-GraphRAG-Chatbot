"""Clear database and re-ingest with clean slate"""
from neo4j_config import Neo4jRuleEngine, get_default_qa_path, get_default_banks_path

engine = Neo4jRuleEngine()

print("="*80)
print("CLEARING DATABASE AND RE-INGESTING")
print("="*80)

# Step 1: Clear everything
print("\n1. Clearing database...")
engine.schema_manager.clear_database()
print("✅ Database cleared")

# Step 2: Setup schema
print("\n2. Setting up schema...")
engine.schema_manager.setup_schema()
print("✅ Schema created")

# Step 3: Ingest data
print("\n3. Ingesting data...")
engine.ingest_all_data(
    qa_path=get_default_qa_path(),
    banks_path=get_default_banks_path()
)

# Step 4: Verify
print("\n4. Verifying ingestion...")
with engine.connection.get_session() as session:
    # Count solutions
    result = session.run("MATCH (s:Solution) RETURN count(s) as cnt")
    sol_count = result.single()['cnt']
    print(f"   Total solutions: {sol_count}")
    
    # Check taxonomy nodes
    result = session.run("MATCH (s:Service) RETURN count(s) as cnt")
    service_count = result.single()['cnt']
    print(f"   Total services: {service_count}")
    
    result = session.run("MATCH (p:Problem) RETURN count(p) as cnt")
    problem_count = result.single()['cnt']
    print(f"   Total problems: {problem_count}")
    
    result = session.run("MATCH (st:State) RETURN count(st) as cnt")
    state_count = result.single()['cnt']
    print(f"   Total states: {state_count}")
    
    result = session.run("MATCH (o:Outcome) RETURN count(o) as cnt")
    outcome_count = result.single()['cnt']
    print(f"   Total outcomes: {outcome_count}")
    
    # Check a sample solution with relationships
    result = session.run("""
        MATCH (sol:Solution {id: '3592ec6d2348'})
        OPTIONAL MATCH (sol)-[:OF_SERVICE]->(service:Service)
        OPTIONAL MATCH (sol)-[:OF_PROBLEM]->(problem:Problem)
        OPTIONAL MATCH (sol)-[:OF_STATE]->(state:State)
        OPTIONAL MATCH (sol)-[:OF_OUTCOME]->(outcome:Outcome)
        RETURN sol.question as question,
               service.name as service,
               problem.name as problem,
               state.name as state,
               outcome.name as outcome,
               sol.bank_id as bank_id
    """)
    record = result.single()
    if record:
        print(f"\n   Sample Solution (3592ec6d2348):")
        print(f"   ✅ Question: {record['question'][:50]}...")
        print(f"   ✅ Service: {record['service']}")
        print(f"   ✅ Problem: {record['problem']}")
        print(f"   ✅ State: {record['state']}")
        print(f"   ✅ Outcome: {record['outcome']}")
        print(f"   ✅ Bank: {record['bank_id']}")
    
    # Check service 'nap_tien' solutions
    result = session.run("""
        MATCH (sol:Solution)-[:OF_SERVICE]->(s:Service {name: 'nap_tien'})
        RETURN count(sol) as cnt
    """)
    cnt = result.single()['cnt']
    print(f"\n   Service 'nap_tien': {cnt} solutions")

print("\n" + "="*80)
print("✅ RE-INGESTION COMPLETE!")
print("="*80)

engine.close()
