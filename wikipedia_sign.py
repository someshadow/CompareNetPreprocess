import requests
from urllib.parse import quote

def get_wikidata_description(entity_name):
    proxy = {
        'http': 'http://127.0.0.1:7890',  
        'https': 'http://127.0.0.1:7890'
    }

    # 设置请求参数
    search_params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",
        "format": "json"
    }
    
    # 设置代理（如果提供）
    proxies = proxy if proxy else None
    
    try:
        # 步骤1: 搜索实体获取 QID
        search_url = "https://www.wikidata.org/w/api.php"
        search_response = requests.get(search_url, params=search_params, proxies=proxies, timeout=30)
        search_response.raise_for_status()  # 检查HTTP错误
        
        search_data = search_response.json()
        
        # 检查是否找到结果
        if not search_data.get("search"):
            return None
        
        # 获取最佳匹配结果（通常第一个）
        best_match = search_data["search"][0]
        qid = best_match["id"]
        
        # 步骤2: 获取实体详细描述
        entity_params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "descriptions|labels",
            "languages": "en",
            "format": "json"
        }
        
        entity_response = requests.post(search_url, params=entity_params, proxies=proxies, timeout=20)
        entity_response.raise_for_status()
        
        entity_data = entity_response.json()
        entity_info = entity_data["entities"][qid]
        
        # 返回结构化结果
        # return {
        #     "qid": qid,
        #     "description": entity_info["descriptions"]["en"]["value"] if "en" in entity_info["descriptions"] else "",
        #     "label": entity_info["labels"]["en"]["value"] if "en" in entity_info["labels"] else ""
        # }
        return entity_info["descriptions"]["en"]["value"] if "en" in entity_info["descriptions"] else ""
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {str(e)}")
        return None
    except KeyError as e:
        print(f"数据解析错误: {str(e)}")
        return None

# # 使用示例（设置你的代理信息）
# PROXY_SETTINGS = {
#     'http': 'http://127.0.0.1:7890',  
#     'https': 'http://127.0.0.1:7890'
# }

# # 不需要代理时设为 None
# result = get_wikidata_description("Neil Little")

# if result:
#     print(result)
# else:
#     print("未找到实体或请求失败")