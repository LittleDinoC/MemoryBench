import os
from glob import glob
import logging
import regex as re
import warnings
import pkg_resources
import regex as re
import traceback
import cn2an

class DataSegmentXingshiBase():
    ALL_FIELDS = ["heading", "fact", "reason", "judgment", "appendix"]

    def __init__(self, punctuation_replace=True):
        self.punctuation_replace = punctuation_replace
        self.jiezhibiaodian = ["。", "）", ")", "”", "\n", "？"] # 截止到标点

    def punctuation_replace_fun(self, data): # 将英文标点全部替换成中文标点
        from bs4 import BeautifulSoup
        data = data.replace(':', '：')
        data = data.replace(' ', '')
        data = data.replace('(', '（')
        data = data.replace(')', '）')
        data = data.replace('\u3000', ' ')
        data = data.replace('\xa0', ' ')
        data = data.replace('<p>', '')
        data = data.replace('</p>', '')
        data = data.replace('<br>', '')
        data = data.replace('&#xD;', '')
        data = data.replace("\uFFFD", "")
        data = re.sub(r'(?<!\d),(?!\d)', '，', data) # 除了数字之间的,不改，其它的改成中文逗号

        soup = BeautifulSoup(data, "html.parser") # 去除所有html结构
        for data in soup(['style', 'script']):
            # Remove tags
            data.decompose()
        data = ' '.join(soup.stripped_strings)

        return data

    def parse(self, wenshu):
        if self.punctuation_replace:
            wenshu = self.punctuation_replace_fun(wenshu)
        wenshu = {"content": wenshu}
        # current content用于存储逐段删除后的剩余内容
        wenshu["current_content"] = wenshu["content"]

        for field in self.ALL_FIELDS:
            eval(f"self._set_{field}(wenshu)")

        del wenshu['current_content']
        return wenshu

    def del_fun(self, wenshu, field): # 如果 wenshu[field] 非空，则从 wenshu["current_content"] 中删除该字段内容。
        if wenshu[field].strip():
            wenshu["current_content"] = wenshu["current_content"].replace(wenshu[field], '')

    def text_end_itertools_min(self, end_list, content, end_supplement='?='): # 在 content 里查找 end_list 里所有可能的结束词，并返回最短匹配文本。
        return_text = ''
        min_len = 100000
        for pe in end_list:
            pattern_text = fr'.*?({end_supplement}{pe})'
            text_search = re.search(pattern_text, content, re.DOTALL)
            # print(text_search)
            if text_search:
                current_text = text_search.group()
                if current_text == "":
                    return ""
                tem_text = re.sub(r'\s+', '', current_text, re.DOTALL)
                if tem_text != '' and len(current_text) < min_len:
                    return_text = current_text
                    min_len = len(return_text)
        return return_text

    def text_end_itertools(self, end_list, content, end_supplement='?='): # 从 content 里查找 end_list 里的词，并返回第一个匹配项（而非最短匹配项）。
        return_text = ''
        for pe in end_list:
            pattern_text = fr'.*?({end_supplement}{pe})'
            text_search = re.search(pattern_text, content, re.DOTALL)
            if text_search:
                return_text = text_search.group()
                break
        return return_text
    
    
class DataSegmentXingshiGongsuYishenPanjue(DataSegmentXingshiBase):

    def _set_heading(self, wenshu): # 头部的基础信息等
        wenshu["heading"] = ""
        pattern_list = [r'审理终结[\u4e00-\u9fa5]{0,10}。', r'公开开庭审理[\u4e00-\u9fa5]{0,10}。']
        wenshu["heading"] = self.text_end_itertools(pattern_list, wenshu["current_content"], '')
        if re.sub(r'\s+', '', wenshu["heading"]) == '':
            pattern_list = [r'[\u4e00-\u9fa5]{0,10}(公诉|抗诉)机关[\u4e00-\u9fa5，、]{0,30}(认为|指控)']
            wenshu["heading"] = self.text_end_itertools(pattern_list, wenshu["current_content"])
        self.del_fun(wenshu, "heading")

    def _set_fact(self, wenshu): # 经法庭审理查明的事实和据以定案的证据，到“本院认为”为止
        ###fact: 经法庭审理查明的事实和据以定案的证据
        wenshu["fact"] = ""
        pattern = [r'本院认为']
        wenshu["fact"] = self.text_end_itertools(pattern, wenshu["current_content"])

        self.del_fun(wenshu, "fact")

    def _set_reason(self, wenshu): # 裁判理由：包含本院认为、引用法条
        wenshu["reason"] = ''
        pattern = [
            r'判决如下[:：,，\n]',
            r'判决如下',
            r'裁定(如下)?[:：\n]',

        ]
        wenshu["reason"] = self.text_end_itertools(pattern, wenshu["current_content"], '')
        if not len(wenshu["reason"]): # 改成了如果找不到“判决如下”，就把后面所有的都设置成reason
            wenshu["reason"] = wenshu["current_content"]
        self.del_fun(wenshu, "reason")

    def _set_judgment(self, wenshu): # 判决结果
        ###panjuejieguo:判决结果
        pattern_list = [
            r'如[\u4e00-\u9fa5]{0,3}不服本判决',
            r'\n\s*本判决为终审判决',
            r'如[\u4e00-\u9fa5]{0,5}未按本判决指定的期间[\u4e00-\u9fa5]{0,5}给付金钱义务',
            r'\n\s*(代理)?审[\s]{0,3}判[\s]{0,3}长',
            r'\n\s*(代理)?审[\s]{0,3}判[\s]{0,3}员',
            '附录',
            '附：',
            r'附[\u4e00-\u9fa5]{0,10}法律[\u4e00-\u9fa5]{0,10}',
            r'附[\u4e00-\u9fa5]{0,10}：',
            r"\n\s*本案[^\n]{0,10}法律",
            r"\n《",
            '$',
        ]
        wenshu["judgment"] = self.text_end_itertools(pattern_list, wenshu["current_content"])
        if not len(wenshu['judgment']):
            wenshu["judgment"] = wenshu["current_content"]
        self.del_fun(wenshu, "judgment")

    def _set_appendix(self, wenshu): # 尾巴上那些内容
        wenshu["appendix"] = wenshu['current_content']
        self.del_fun(wenshu, "appendix")


class DataSegmentXingshi():
    
    def __init__(self, punctuation_replace=False):
        self.yishengongsu = DataSegmentXingshiGongsuYishenPanjue(punctuation_replace)

    def parse(self, wenshu):
        wenshu = self.yishengongsu.parse(wenshu)
        return wenshu
