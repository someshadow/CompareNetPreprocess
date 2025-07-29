import ast

def safe_parse(s):
    """安全解析包含数字的字符串"""
    try:
        # 尝试解析为列表
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        # 如果失败，使用字符串处理方法
        s_clean = s.strip('[]').strip()
        parts = [x for x in s_clean.split() if x]
        try:
            # 尝试转换为数字（整数或浮点数）
            return [int(x) if '.' not in x else float(x) for x in parts]
        except ValueError:
            # 最后尝试转换为浮点数
            return [float(x) for x in parts]

def convert_row_robust(row):
    topics = safe_parse(row['assigned_topics'])
    weights = safe_parse(row['topic_weight'])
    return topics + weights

def transform_list(input_list):
    result = []
    for item in input_list:
        # 提取每个子列表的四个元素
        a, b, c, d = item
        # 按格式重组为新元素：[[a, c], [b, d]]
        new_item = [[a, c], [b, d]]
        result.append(new_item)
    return result