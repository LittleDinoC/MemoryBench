import json,os,re
from tqdm import tqdm
import chinese2digits as c2d
import sys
from .xingshi import DataSegmentXingshi
judge_list = ["管制", "拘役", "有期徒刑", "罚金", "无期徒刑", "死刑", "无罪", "免予刑事处罚", "免于刑事处罚", "免予刑事处分"]

def get_judgment(doc): # 截取doc的“判决”部分
    parser = DataSegmentXingshi(punctuation_replace=True)
    result = parser.parse(doc)
    return result['judgment']

def get_time_string_from_text(doc): # 提取(包含刑期)的完整字符串
    # print('\n', doc)
    # 修正后的正则表达式模式
    ret = []
    for judge in judge_list[:3]:
        pattern = re.compile(rf'{judge}.{{1,7}}[年月]') # judge一直匹配到‘年’字或者‘月’字
        matches = re.findall(pattern, doc)
        ch_punct_pattern = re.compile(r'[,;，。！？、；：（以缓至]') # 截取到标点符号/“缓刑”之前
        for i in range(len(matches)):
            match = matches[i]
            ch_punct_pos = ch_punct_pattern.search(match)
            if ch_punct_pos:
                matches[i] = match[:ch_punct_pos.start()]
        
        ret += matches

    for judge in judge_list[4:]:
        pattern = re.compile(rf'{judge}')
        matches = re.findall(pattern, doc)
        ret += matches

    return ret

def get_amt_string_from_text(doc): # 提取包含罚金金额的完整字符串
    pattern = re.compile(rf'罚金.{{1,15}}元') # 一直匹配到‘元’字
    matches = re.findall(pattern, doc)
    ch_punct_pattern = re.compile(r'[，。！？、；：以已至（]') # 截取到标点符号之前
    for i in range(len(matches)):
        match = matches[i]
        ch_punct_pos = ch_punct_pattern.search(match)
        if ch_punct_pos:
            matches[i] = match[:ch_punct_pos.start()]
    for match in matches:
        if not '元' in match:
            matches.remove(match)
    for i in range(len(matches)):
        matches[i] = matches[i].replace(",", "")
    return matches

def get_time_from_text(doc):
    # print('-' * 80)
    full_doc = doc
    doc = get_judgment(doc)
    ret = get_time_string_from_text(doc)
    if len(ret) == 0:
        ret = get_time_string_from_text(full_doc)
    
    ret = list(set(ret))
    # print(ret)
    return ret

def get_amt_from_text(doc):
    full_doc = doc
    doc = get_judgment(doc)
    ret = get_amt_string_from_text(doc)
    if len(ret) == 0:
        ret = get_amt_string_from_text(full_doc)
    
    ret = list(set(ret))
    # print(ret)
    return ret

def calc_time_sum(doc):
    all_judge_time_str = get_time_from_text(doc)
    if len(all_judge_time_str) == 0: # 如果没有提取到刑期长度
        return -1
    
    time_sum = 0
    for judge_time_str in all_judge_time_str:
        num_list = c2d.takeNumberFromString(judge_time_str)['digitsStringList']
        num = 0
        if len(num_list) == 2:
            if '年' in judge_time_str and '月' in judge_time_str: # 如果是x年x月的格式
                num = int(num_list[0]) * 12 + int(num_list[1])
            else:
                print('发生错误：', judge_time_str)
                num = int(num_list[0]) # 取第一个
        elif len(num_list) == 1:
            if '年' in judge_time_str:
                num = int(num_list[0]) * 12
            elif '月' in judge_time_str:
                num = int(num_list[0])
        elif len(num_list) == 0:
            if '无期徒刑' in judge_time_str:
                num = 240
            elif '死刑' in judge_time_str:
                num = 10001 # 一会儿只需检查是否返回的数额大于10000，即可知道是否出现死刑了
            else:
                num = 0
        else:
            print('有不合规范的刑期长度：', judge_time_str)
        
        time_sum += num
        
    return time_sum

def calc_amt_sum(doc):
    all_amt_str = get_amt_from_text(doc)
    if len(all_amt_str) == 0: # 如果没有提取到罚金金额
        return -1
    amt_sum = 0
    for amt_str in all_amt_str:
        num_list = c2d.takeNumberFromString(amt_str)['digitsStringList']
        if len(num_list) == 1:
            
            if "." in amt_str:
                amt_sum += int(num_list[0].split('.')[0])
            else:
                amt_sum += int(num_list[0])
        else:
            print('金额格式不对', amt_str)
    
    return amt_sum

def get_time_sum_chinese(doc):
    """返回刑期的中文表述"""
    all_judge_time_str = get_time_from_text(doc)
    if len(all_judge_time_str) == 0:
        return "提取不到刑期"
    
    # 检查是否包含死刑或无期徒刑
    for judge_time_str in all_judge_time_str:
        if '死刑' in judge_time_str:
            return "死刑"
        if '无期徒刑' in judge_time_str:
            return "无期徒刑"
        if '无罪' in judge_time_str:
            return "无罪"
        if '免予刑事处罚' in judge_time_str or '免于刑事处罚' in judge_time_str or '免予刑事处分' in judge_time_str:
            return "免予刑事处罚"
    
    # 计算总刑期
    time_sum = 0
    for judge_time_str in all_judge_time_str:
        num_list = c2d.takeNumberFromString(judge_time_str)['digitsStringList']
        num = 0
        if len(num_list) == 2:
            if '年' in judge_time_str and '月' in judge_time_str:
                num = int(num_list[0]) * 12 + int(num_list[1])
            else:
                num = int(num_list[0])
        elif len(num_list) == 1:
            if '年' in judge_time_str:
                num = int(num_list[0]) * 12
            elif '月' in judge_time_str:
                num = int(num_list[0])
        
        time_sum += num
    
    if time_sum == 0:
        return "提取不到刑期"
    
    # 转换为中文表述
    if time_sum >= 12:
        years = time_sum // 12
        months = time_sum % 12
        if months == 0:
            return f"{years}年"
        else:
            return f"{years}年{months}个月"
    else:
        return f"{time_sum}个月"

def get_amt_sum_chinese(doc):
    """返回罚金的中文表述"""
    all_amt_str = get_amt_from_text(doc)
    if len(all_amt_str) == 0:
        return "提取不到金额"
    
    amt_sum = 0
    for amt_str in all_amt_str:
        num_list = c2d.takeNumberFromString(amt_str)['digitsStringList']
        if len(num_list) == 1:
            amt_sum += int(num_list[0])
        else:
            print('金额格式不对', amt_str)
    
    if amt_sum == 0:
        return "提取不到金额"
    
    # 转换为中文表述
    if amt_sum >= 10000:
        wan = amt_sum // 10000
        remainder = amt_sum % 10000
        if remainder == 0:
            return f"{wan}万元"
        else:
            return f"{wan}万{remainder}元"
    else:
        return f"{amt_sum}元"
