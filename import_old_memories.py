#!/usr/bin/env python3
"""导入旧hermes plugin memory_store.db到NAS HMEM Server API"""
import sqlite3
import requests
import json
import time

NAS_API = "http://192.168.1.10:8090/api/v1"
API_KEY = "change-me"
OLD_DB = "/root/.hermes/memory_store.db"

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def get_old_data():
    """读取旧memory_store.db"""
    conn = sqlite3.connect(OLD_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # 列出所有表
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    print(f"找到表: {tables}")
    
    # 读取facts表（主要记忆数据）
    c.execute("SELECT * FROM facts")
    facts = [dict(row) for row in c.fetchall()]
    print(f"facts表: {len(facts)} 条")
    
    # 读取entities表
    c.execute("SELECT * FROM entities")
    entities = [dict(row) for row in c.fetchall()]
    print(f"entities表: {len(entities)} 条")
    
    conn.close()
    return facts, entities

def import_to_nas(hmem_type, items, namespace="default"):
    """导入数据到NAS HMEM"""
    count = 0
    for item in items:
        try:
            # 构建content字段
            if hmem_type == "observation":
                content = item.get("content", item.get("text", ""))
                mem_action = item.get("action", "")
            elif hmem_type == "experience":
                content = item.get("content", item.get("text", ""))
                mem_action = item.get("action", "")
            elif hmem_type == "insight":
                content = item.get("content", item.get("summary", ""))
                mem_action = "insight"
            elif hmem_type == "mental_model":
                content = item.get("content", item.get("model", ""))
                mem_action = "mental_model"
            else:
                content = json.dumps(item, ensure_ascii=False)
                mem_action = ""
            
            if not content or len(content.strip()) == 0:
                continue
            
            payload = {
                "content": content[:500],  # 限制长度
                "namespace": namespace,
                "memory_type": hmem_type,
            }
            if mem_action:
                payload["mem_action"] = mem_action
            
            resp = requests.post(
                f"{NAS_API}/memories",
                headers=HEADERS,
                json=payload,
                timeout=10
            )
            if resp.status_code == 200:
                count += 1
            else:
                print(f"  导入失败 {hmem_type}: {resp.status_code} {resp.text[:200]}")
                time.sleep(1)
                
        except Exception as e:
            print(f"  异常 {hmem_type}: {e}")
            time.sleep(1)
    
    return count

def main():
    print("=" * 50)
    print("开始导入旧hermes plugin记忆到NAS HMEM")
    print("=" * 50)
    
    # 检查NAS HMEM连接
    try:
        resp = requests.get(f"{NAS_API}/health", headers=HEADERS, timeout=5)
        print(f"NAS HMEM状态: {resp.json()}")
    except Exception as e:
        print(f"无法连接NAS HMEM: {e}")
        return
    
    # 读取旧数据
    facts, entities = get_old_data()
    
    if not facts:
        print("旧数据为空，无需导入")
        return
    
    # 分类导入
    import_counts = {
        "observation": 0,
        "experience": 0,
        "insight": 0,
        "mental_model": 0,
    }
    
    print("\n导入facts...")
    for fact in facts:
        # 根据memory_type分类
        mtype = fact.get("memory_type", "observation")
        if mtype not in import_counts:
            mtype = "observation"
        
        count = import_to_nas(mtype, [fact])
        if count:
            import_counts[mtype] += 1
    
    print(f"\n导入entities...")
    for entity in entities:
        count = import_to_nas("observation", [entity])
        if count:
            import_counts["observation"] += 1
    
    print("\n" + "=" * 50)
    print("导入完成:")
    for mtype, cnt in import_counts.items():
        print(f"  {mtype}: {cnt} 条")
    print("=" * 50)

if __name__ == "__main__":
    main()
