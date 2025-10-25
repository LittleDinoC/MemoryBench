import json,re
import chinese2digits as c2d
from tqdm import tqdm
import sys
from .xingshi import DataSegmentXingshi

def get_reason(doc): # 截取doc的“判决”部分
    parser = DataSegmentXingshi(punctuation_replace=True)
    result = parser.parse(doc)
    return result['reason']

def get_penalcode_index_from_text(full_doc):
    doc = get_reason(full_doc)
    patterns = [
        r"《中华人民共和国刑法》第.*?[。《判附规]",  # 匹配《中华人民共和国刑法》第xx条到特定关键词
        r"《刑法》第.*?[。《判附规]",             # 匹配简称《刑法》第xx条到特定关键词
        r"《中华人民共和国刑法》\s?[零一二三四五六七八九十第]+.*?[。《判附规]"  # 匹配《中华人民共和国刑法》后跟空格和数字到特定关键词
    ]
    matches = set()  # 用 set 来去重
    for pattern in patterns:
        matches.update(re.findall(pattern, doc))
    
    if len(matches) == 0:
        doc = full_doc
        for pattern in patterns:
            matches.update(re.findall(pattern, doc))
    # print(matches)
    # 从上述《刑法》第xx条中，获取具体条目编号
    ret = set()
    for match in matches:
        nums = get_num_from_text(match)
        for num in nums:
            ret.add(num) 
    
    ret = list(ret)
    return ret

def get_num_from_text(doc): # 这里是一个简化了的处理方法，匹配所有“第xxx条”并转换成阿拉伯数字
    # pattern = r"[》、]第[一二三四五六七八九零十百]条"
    pattern = r"第[一二三四五六七八九零十百]+条"
    matches = re.findall(pattern, doc)
    ret = []
    for match in matches:
        try:
            # 尝试转换中文数字为阿拉伯数字
            converted_list = c2d.takeNumberFromString(match)['digitsStringList']
            assert len(converted_list) == 1
            ret.append(converted_list[0])
        except Exception as e:
            print(f"跳过不符合格式的匹配: {match}, 错误: {e}")
            pass 
            # 如果转换失败，自动跳过
            
    ret = list(set(ret))
    return ret