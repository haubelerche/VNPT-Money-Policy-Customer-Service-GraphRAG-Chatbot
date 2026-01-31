# -*- coding: utf-8 -*-
"""
VNPT-MONEY CHATBOT - Dataset Rebuilder V3
Properly parses ALL raw CSV files and connects problems to topics.

Structure: GROUP → HAS_TOPIC → TOPIC → HAS_PROBLEM → PROBLEM → HAS_ANSWER → ANSWER

Key changes from V2:
- Parses ALL 4 raw CSV files properly
- Links problems directly to topics based on Children nodes column
- No placeholders - only real data from raw files
"""
import csv
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# Paths
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "external_data_v3"

OUTPUT_DIR.mkdir(exist_ok=True)


def normalize_id(text: str) -> str:
    """Convert Vietnamese text to snake_case ID."""
    if not text:
        return ""
    vn_chars = {
        'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ắ': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ấ': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'đ': 'd',
        'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'í': 'i', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ó': 'o', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ố': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ớ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ú': 'u', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ứ': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ý': 'y', 'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    }
    
    text = text.lower().strip()
    for vn, en in vn_chars.items():
        text = text.replace(vn, en)
    
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_')[:60]


# =============================================================================
# GROUP DEFINITIONS
# =============================================================================

GROUPS = {
    "quyen_rieng_tu": {
        "name": "Quyền riêng tư",
        "description": "Chính sách bảo mật và xử lý thông tin khách hàng",
        "order": 1
    },
    "ho_tro_khach_hang": {
        "name": "Hỗ trợ khách hàng", 
        "description": "Hướng dẫn sử dụng dịch vụ VNPT Money",
        "order": 2
    },
    "dieu_khoan": {
        "name": "Điều khoản",
        "description": "Điều khoản và điều kiện sử dụng dịch vụ",
        "order": 3
    },
    "dich_vu": {
        "name": "Dịch vụ",
        "description": "Danh mục các dịch vụ thanh toán",
        "order": 4
    }
}

# Mapping: group_id -> raw CSV filename
RAW_FILES = {
    "quyen_rieng_tu": "CHATBOT RESOURCES - quyen_rieng_tu.csv",
    "ho_tro_khach_hang": "CHATBOT RESOURCES - ho_tro_khach_hang.csv",
    "dieu_khoan": "CHATBOT RESOURCES - dieu_khoan.csv",
    "dich_vu": "CHATBOT RESOURCES - dich_vu.csv"
}


def detect_csv_format(filepath: Path) -> Tuple[dict, List[str]]:
    """Detect column names in CSV file. Returns (col_map, original_headers)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        original_headers = next(reader)
        
    col_map = {}
    for i, h in enumerate(original_headers):
        h_lower = h.lower().strip()
        if 'parent' in h_lower:
            col_map['parent'] = i  # Use index instead of name
        elif 'children' in h_lower or 'child' in h_lower:
            col_map['children'] = i
        elif 'problem' in h_lower:
            col_map['problem'] = i
        elif 'answer' in h_lower:
            col_map['answer'] = i
    
    return col_map, original_headers


def parse_raw_csv(group_id: str, filename: str) -> Tuple[Set[str], List[dict]]:
    """
    Parse raw CSV file to extract:
    1. All unique topic names (from Children nodes column)
    2. All problem-answer pairs with their topic association
    
    Returns: (set of topic names, list of problem-answer dicts)
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        print(f"  ⚠ File not found: {filename}")
        return set(), []
    
    col_map, headers = detect_csv_format(filepath)
    print(f"  Columns detected: {col_map} -> {[headers[i] for i in col_map.values()]}")
    
    topics_found = set()
    qa_pairs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)  # Use csv.reader instead of DictReader
        next(reader)  # Skip header row
        
        current_parent = None
        current_child = None
        
        for row in reader:
            if not row or len(row) < max(col_map.values()) + 1:
                continue
            
            # Get values using column indices
            parent = row[col_map.get('parent', 0)].strip() if col_map.get('parent') is not None else ''
            children = row[col_map.get('children', 1)].strip() if col_map.get('children') is not None else ''
            problem = row[col_map.get('problem', 2)].strip() if col_map.get('problem') is not None else ''
            answer = row[col_map.get('answer', 3)].strip() if col_map.get('answer') is not None else ''
            
            # Update context - children column is the TOPIC
            if children:
                current_child = children
            if parent:
                current_parent = parent
            
            # The topic is the most recent non-empty children field
            # If no children, use the parent (which is the "category")
            topic_name = current_child if current_child else current_parent
            
            if topic_name:
                topics_found.add(topic_name)
            
            # Skip rows without Q&A
            if not problem or not answer:
                continue
            
            if not topic_name:
                print(f"    ⚠ No topic for problem: {problem[:50]}...")
                continue
            
            qa_pairs.append({
                'topic_name': topic_name,
                'problem': problem,
                'answer': answer
            })
    
    print(f"  Found {len(topics_found)} topics, {len(qa_pairs)} Q&A pairs")
    return topics_found, qa_pairs


def build_dataset():
    """Build complete dataset by parsing all raw files."""
    
    all_topics = []  # List of topic dicts
    all_problems = []  # List of problem dicts
    all_answers = []  # List of answer dicts
    rels_has_topic = []  # group -> topic
    rels_has_problem = []  # topic -> problem
    rels_has_answer = []  # problem -> answer
    
    topic_id_map = {}  # (group_id, topic_name) -> topic_id
    
    global_topic_order = 1
    global_problem_counter = 1
    global_answer_counter = 1
    
    print("=" * 70)
    print("PARSING RAW CSV FILES")
    print("=" * 70)
    
    for group_id, filename in RAW_FILES.items():
        print(f"\n[{group_id}] {filename}")
        
        topics_found, qa_pairs = parse_raw_csv(group_id, filename)
        
        # Create topic nodes for this group
        for topic_name in sorted(topics_found):
            topic_id = f"{group_id}__{normalize_id(topic_name)}"
            
            # Avoid duplicates
            if topic_id in [t['id'] for t in all_topics]:
                print(f"    ⚠ Duplicate topic ID: {topic_id}")
                continue
            
            all_topics.append({
                'id': topic_id,
                'name': topic_name,
                'group_id': group_id,
                'keywords': '',
                'order': global_topic_order
            })
            
            rels_has_topic.append({
                'start_id': group_id,
                'end_id': topic_id
            })
            
            topic_id_map[(group_id, topic_name)] = topic_id
            global_topic_order += 1
        
        # Create problem and answer nodes
        problem_per_topic = defaultdict(int)
        
        for qa in qa_pairs:
            topic_name = qa['topic_name']
            topic_id = topic_id_map.get((group_id, topic_name))
            
            if not topic_id:
                # Try fuzzy match
                norm_name = normalize_id(topic_name)
                for (gid, tname), tid in topic_id_map.items():
                    if gid == group_id and normalize_id(tname) == norm_name:
                        topic_id = tid
                        break
            
            if not topic_id:
                print(f"    ⚠ No topic match for: {topic_name[:40]}")
                continue
            
            problem_per_topic[topic_id] += 1
            problem_num = problem_per_topic[topic_id]
            
            problem_id = f"{topic_id}__prob_{problem_num}"
            answer_id = f"{group_id}__ans_{global_answer_counter}"
            
            all_problems.append({
                'id': problem_id,
                'title': qa['problem'][:500],
                'description': '',
                'intent': normalize_id(qa['problem'][:50]),
                'keywords': '',
                'sample_questions': '',
                'status': 'active'
            })
            
            all_answers.append({
                'id': answer_id,
                'summary': qa['problem'][:100],
                'content': qa['answer'],
                'steps': '',
                'notes': '',
                'status': 'active'
            })
            
            rels_has_problem.append({
                'start_id': topic_id,
                'end_id': problem_id
            })
            
            rels_has_answer.append({
                'start_id': problem_id,
                'end_id': answer_id
            })
            
            global_problem_counter += 1
            global_answer_counter += 1
    
    return (
        all_topics, all_problems, all_answers,
        rels_has_topic, rels_has_problem, rels_has_answer
    )


def write_csv(filepath: Path, fieldnames: List[str], data: List[dict]):
    """Write data to CSV with UTF-8 BOM for proper encoding."""
    with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"  ✓ {filepath.name}: {len(data)} rows")


def main():
    print("=" * 70)
    print("VNPT-MONEY CHATBOT - Dataset Rebuilder V3")
    print("Schema: GROUP → TOPIC → PROBLEM → ANSWER")
    print("=" * 70)
    
    # Build dataset
    (
        topics, problems, answers,
        rels_has_topic, rels_has_problem, rels_has_answer
    ) = build_dataset()
    
    print("\n" + "=" * 70)
    print("WRITING OUTPUT FILES")
    print("=" * 70 + "\n")
    
    # Write groups
    groups_data = [
        {'id': gid, 'name': info['name'], 'description': info['description'], 'order': info['order']}
        for gid, info in GROUPS.items()
    ]
    write_csv(
        OUTPUT_DIR / "nodes_group.csv",
        ['id', 'name', 'description', 'order'],
        groups_data
    )
    
    # Write topics
    write_csv(
        OUTPUT_DIR / "nodes_topic.csv",
        ['id', 'name', 'group_id', 'keywords', 'order'],
        topics
    )
    
    # Write problems
    write_csv(
        OUTPUT_DIR / "nodes_problem.csv",
        ['id', 'title', 'description', 'intent', 'keywords', 'sample_questions', 'status'],
        problems
    )
    
    # Write answers
    write_csv(
        OUTPUT_DIR / "nodes_answer.csv",
        ['id', 'summary', 'content', 'steps', 'notes', 'status'],
        answers
    )
    
    # Write relationships
    write_csv(
        OUTPUT_DIR / "rels_has_topic.csv",
        ['start_id', 'end_id'],
        rels_has_topic
    )
    
    write_csv(
        OUTPUT_DIR / "rels_has_problem.csv",
        ['start_id', 'end_id'],
        rels_has_problem
    )
    
    write_csv(
        OUTPUT_DIR / "rels_has_answer.csv",
        ['start_id', 'end_id'],
        rels_has_answer
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Groups:        {len(groups_data)}")
    print(f"  Topics:        {len(topics)}")
    print(f"  Problems:      {len(problems)}")
    print(f"  Answers:       {len(answers)}")
    print(f"  HAS_TOPIC:     {len(rels_has_topic)}")
    print(f"  HAS_PROBLEM:   {len(rels_has_problem)}")
    print(f"  HAS_ANSWER:    {len(rels_has_answer)}")
    
    # Per-group stats
    print("\n  Per-group breakdown:")
    for gid in GROUPS:
        topic_count = len([t for t in topics if t['group_id'] == gid])
        problem_count = len([r for r in rels_has_problem if r['start_id'].startswith(gid)])
        print(f"    {gid}: {topic_count} topics, {problem_count} problems")
    
    print("\n" + "=" * 70)
    print(f"✓ Output written to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
