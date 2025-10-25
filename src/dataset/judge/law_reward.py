import re
from law_extraction import get_penalcode_index_from_text
from judge_extraction import calc_time_sum, calc_amt_sum
from crime_extraction import get_crime_from_text
from nltk.translate.meteor_score import meteor_score
import jieba

def get_reward(exp_text, gen_text):
    # 1. judge time 差异占比
    time_exp = calc_time_sum(exp_text)
    time_gen = calc_time_sum(gen_text)
    if time_exp == -1:
        time_diff = 0
    elif time_gen == -1:
        time_diff = 1
    # 如果两个时间都为0，则认为没有差异
    elif time_exp == 0 and time_gen == 0:
        time_diff = 0
    # 如果一个时间为0，另一个不为0，则认为差异为1
    elif time_exp == 0 or time_gen == 0:
        time_diff = 1
    else:
        denom = max(abs(time_exp), abs(time_gen))
        time_diff = abs(time_exp - time_gen) / denom

    # 2. judge 金额 差异占比
    amt_exp = calc_amt_sum(exp_text)
    amt_gen = calc_amt_sum(gen_text)
    if amt_exp == -1:
        amt_diff = 0
    elif amt_gen == -1:
        amt_diff = 1
    # 如果两个金额都为0，则认为没有差异
    elif amt_exp == 0 and amt_gen == 0:
        amt_diff = 0
    # 如果一个金额为0，另一个不为0，则认为差异为1
    elif amt_exp == 0 or amt_gen == 0:
        amt_diff = 1
    else:
        denom = max(abs(amt_exp), abs(amt_gen))
        amt_diff = abs(amt_exp - amt_gen) / denom

    # 3. 引用法条 差异占比
    law_exp = set(get_penalcode_index_from_text(exp_text))
    law_gen = set(get_penalcode_index_from_text(gen_text))
    law_union = law_exp | law_gen
    if len(law_union) == 0:
        law_diff = 0
    else:
        law_diff = len(law_union - (law_exp & law_gen)) / len(law_union)

    # 4. 罪名 差异占比
    crime_exp = set(get_crime_from_text(exp_text))
    crime_gen = set(get_crime_from_text(gen_text))
    crime_union = crime_exp | crime_gen
    if len(crime_union) == 0:
        crime_diff = 0
    else:
        crime_diff = len(crime_union - (crime_exp & crime_gen)) / len(crime_union)

    # 5. meteor分数
    reference_tokens = list(jieba.cut(exp_text))
    hypothesis_tokens = list(jieba.cut(gen_text))
    meteor = meteor_score([reference_tokens], hypothesis_tokens)

    # reward = 1 - (四项差异占比平均) + meteor分数
    diff_score = (time_diff + amt_diff + law_diff + crime_diff) / 4
    reward = 1 - diff_score + meteor
    return {
        'reward': reward,
        'time_diff': time_diff,
        'amt_diff': amt_diff,
        'law_diff': law_diff,
        'crime_diff': crime_diff,
        'meteor': meteor
    }
