"""
Test retrieval accuracy for difficult questions
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from openai import OpenAI

# Test queries that chatbot answered incorrectly
test_queries = [
    "Tôi thanh toán nhầm hóa đơn hoặc nhầm số tiền thì có hủy được không?",
    "Vì sao tôi không thể nạp số tiền lớn dù trong tài khoản ngân hàng vẫn còn đủ?",
    "Tôi nhập sai OTP nhiều lần thì tài khoản có bị khóa không?",
    "Tôi không nhận được mã OTP khi nạp tiền dù đã chờ khá lâu",
]

def main():
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Get embedding
        response = client.embeddings.create(model='text-embedding-3-small', input=query)
        query_embedding = response.data[0].embedding
        
        with driver.session() as session:
            # Vector search
            result = session.run("""
                CALL db.index.vector.queryNodes("problem_embedding_index", 5, $embedding)
                YIELD node, score
                RETURN node.id AS id, node.title AS title, node.description AS desc, score
                ORDER BY score DESC
                LIMIT 5
            """, {'embedding': query_embedding})
            
            print("\nTop 5 Retrieved Problems:")
            for i, record in enumerate(result, 1):
                print(f"  {i}. [{record['score']:.4f}] {record['id']}")
                print(f"     Title: {record['title']}")
                if record['desc']:
                    print(f"     Desc: {record['desc'][:100]}...")
    
    driver.close()
    print("\n" + "="*60)
    print("Analysis: Check if the top results are relevant to the queries")

if __name__ == "__main__":
    main()
