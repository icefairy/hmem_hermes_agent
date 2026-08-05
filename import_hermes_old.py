#!/usr/bin/env python3
import sqlite3
import requests
import time

NAS_API = "http://192.168.1.10:8090/api/v1"
API_KEY = "m4WBKIF921073940D64b849029B698A595Df65"
OLD_DB = "/root/.hermes/memory_store.db"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_old_data():
    conn = sqlite3.connect(OLD_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM facts")
    facts = [dict(r) for r in c.fetchall()]
    
    c.execute("SELECT * FROM entities")
    entities = [dict(r) for r in c.fetchall()]
    
    c.execute("SELECT * FROM memory_banks")
    banks = [dict(r) for r in c.fetchall()]
    
    conn.close()
    return facts, entities, banks

def import_to_nas(content, memory_type="observation", namespace="default"):
    payload = {
        "content": content[:1000],
        "namespace": namespace,
        "memory_type": memory_type,
    }
    
    try:
        resp = requests.post(
            f"{NAS_API}/memories",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        if resp.status_code == 200:
            return True
        else:
            print(f"  ❌ [FAIL] status={resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ [EXCEPT] {e}")
        return False

def main():
    print("=" * 60)
    print("导入旧hermes plugin记忆到NAS HMEM")
    print("=" * 60)
    
    # 检查NAS HMEM连接
    try:
        resp = requests.get("http://192.168.1.10:8090/health", timeout=5)
        data = resp.json()
        if data.get("status") != "ok":
            print(f"❌ NAS HMEM状态异常: {data}")
            return
        print(f"✅ NAS HMEM状态正常")
    except Exception as e:
        print(f"❌ 无法连接NAS HMEM: {e}")
        return
    
    # 读取旧数据
    facts, entities, banks = get_old_data()
    print(f"📚 读取旧数据: {len(facts)} facts, {len(entities)} entities, {len(banks)} banks")
    
    # 导入facts
    print("\n📤 导入facts...")
    success_count = 0
    fail_count = 0
    for fact in facts:
        content = fact["content"]
        if not content or not content.strip():
            continue
        
        # 根据category映射memory_type
        category = fact.get("category", "")
        if category == "tool":
            memory_type = "observation"
        elif category == "user_pref":
            memory_type = "experience"
        elif category == "insight":
            memory_type = "insight"
        else:
            memory_type = "observation"
        
        if import_to_nas(content, memory_type, "default"):
            success_count += 1
        else:
            fail_count += 1
        
        if success_count % 10 == 0:
            print(f"   进度: {success_count}/{len(facts)}")
    
    print(f"   ✅ 成功: {success_count}, ❌ 失败: {fail_count}")
    
    # 导入entities
    print("\n📤 导入entities...")
    entity_count = 0
    for entity in entities:
        content = f"实体: {entity.get('name', '')} (类型: {entity.get('entity_type', 'unknown')})"
        if import_to_nas(content, "observation", "default"):
            entity_count += 1
    print(f"   ✅ 成功: {entity_count}/{len(entities)}")
    
    # 导入memory_banks
    print("\n📤 导入memory_banks...")
    bank_count = 0
    for bank in banks:
        content = f"记忆银行: {bank.get('bank_name', '')} ({bank.get('dim', 0)}维, {bank.get('fact_count', 0)}条记忆)"
        if import_to_nas(content, "observation", "default"):
            bank_count += 1
    print(f"   ✅ 成功: {bank_count}/{len(banks)}")
    
    # 查看结果
    print("\n" + "=" * 60)
    print("查看NAS HMEM统计...")
    resp = requests.get(f"{NAS_API}/stats", headers=HEADERS, timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        print(f"总记忆数: {data.get('total_memories', 0)}")
        by_type = data.get("by_type", {})
        for t, c in by_type.items():
            print(f"  {t}: {c}")
    print("=" * 60)

if __name__ == "__main__":
    main()
