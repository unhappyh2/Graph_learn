import pandas as pd
from collections import defaultdict

def load_knowledge_graph(file_path):
    # 读取知识图谱文件
    triples = []
    relations = set()
    entities = set()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            # 转换为整数ID
            h = int(h)
            t = int(t)
            triples.append((h, r, t))
            relations.add(r)
            entities.add(h)
            entities.add(t)
    
    return triples, relations, entities

def map_relations_to_ids(relations):
    # 为每种关系分配一个从1开始的整数ID
    relation_to_id = {relation: idx + 1 for idx, relation in enumerate(relations)}
    return relation_to_id

def replace_relations_with_ids(triples, relation_to_id):
    # 用数字ID替换关系
    triples_with_ids = [(h, relation_to_id[r], t) for h, r, t in triples]
    return triples_with_ids

def analyze_knowledge_graph(triples_with_ids):
    # 统计每种关系的频次
    relation_counts = defaultdict(int)
    # 统计每个头实体的出度
    head_degrees = defaultdict(int)
    # 统计每个尾实体的入度
    tail_degrees = defaultdict(int)
    
    for h, r, t in triples_with_ids:
        relation_counts[r] += 1
        head_degrees[h] += 1
        tail_degrees[t] += 1
    
    # 转换为DataFrame以便于展示
    relation_stats = pd.DataFrame({
        'Relation ID': list(relation_counts.keys()),
        'Count': list(relation_counts.values())
    }).sort_values('Count', ascending=False)
    
    return relation_stats, head_degrees, tail_degrees

def print_statistics(triples_with_ids, relations, entities, relation_stats, relation_to_id):
    print(f"知识图谱统计信息:")
    print(f"三元组总数: {len(triples_with_ids)}")
    print(f"关系类型数: {len(relations)}")
    print(f"实体总数: {len(entities)}")
    print("\n关系分布:")
    print(relation_stats)
    print("\n关系映射表:")
    for relation, id in relation_to_id.items():
        print(f"关系: {relation}, ID: {id}")

def Get_kg(file_path):
    # 文件路径
    
    # 加载知识图谱
    triples, relations, entities = load_knowledge_graph(file_path)
    
    # 将关系映射为ID
    relation_to_id = map_relations_to_ids(relations)
    
    # 用ID替换关系
    triples_with_ids = replace_relations_with_ids(triples, relation_to_id)
    
    # 分析知识图谱
    relation_stats, head_degrees, tail_degrees = analyze_knowledge_graph(triples_with_ids)
    
    # 打印统计信息
    print_statistics(triples_with_ids, relations, entities, relation_stats, relation_to_id)
    
    # 输出一些示例三元组
    print("\n示例三元组:")
    for triple in triples_with_ids[:5]:
        print(f"头实体: {triple[0]}, 关系ID: {triple[1]}, 尾实体: {triple[2]}")
    
    # 计算一些额外的统计信息
    avg_head_degree = sum(head_degrees.values()) / len(head_degrees)
    avg_tail_degree = sum(tail_degrees.values()) / len(tail_degrees)
    
    print(f"\n平均出度: {avg_head_degree:.2f}")
    print(f"平均入度: {avg_tail_degree:.2f}")

    return triples_with_ids, relation_to_id, len(entities)

