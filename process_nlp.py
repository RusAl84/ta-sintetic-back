import os
import json
import nltk
import pymorphy2


db_fileName = "./data_cl.json"

def add_data(text):
    import pathlib
    path = pathlib.Path(db_fileName)
    content = []
    data = get_pattern(text)
    data = add_print_text(data)
    if path.exists():
        with open(db_fileName, "r", encoding="UTF8") as file:
            jsoncontent = file.read()
        content = json.loads(jsoncontent)
        content.append(data)
        jsonstring = json.dumps(content, ensure_ascii=False)
        with open(db_fileName, "w", encoding="UTF8") as file:
            file.write(jsonstring)
    else:
        content.append(data)
        jsonstring = json.dumps(content, ensure_ascii=False)
        with open(db_fileName, "w", encoding="UTF8") as file:
            file.write(jsonstring)
    return content


def load_db():
    import pathlib
    path = pathlib.Path(db_fileName)
    if path.exists():
        with open(db_fileName, "r", encoding="UTF8") as file:
            jsoncontent = file.read()
        content = json.loads(jsoncontent)
        return content
    else:
        return [{}]
    

def clear_db():
    import pathlib
    path = pathlib.Path(db_fileName)
    if path.exists():
        os.remove(db_fileName)


def data_proc(filename, save_filename, threshold=0):
    # with open("./uploads/"+filename+".json", "r", encoding="UTF8") as file:
    with open(filename, "r", encoding="UTF8") as file:
        content = file.read()
    messages = json.loads(content)
    text = ""
    count_messages = len(messages)
    print(count_messages)
    num = 0
    proc_messages = []  
    for m in messages:
        text = m["text"]
        print(f"{num / count_messages * 100}     {count_messages-num}     {num} / {count_messages}")
        num += 1
        if len(text) < threshold:
            continue
        line = get_pattern(text)
        line["date"] = m["date"]
        line["message_id"] = m["message_id"]
        line["user_id"] = m["user_id"]
        line["reply_message_id"] = m["reply_message_id"]
        proc_messages.append(line)
    jsonstring = json.dumps(proc_messages, ensure_ascii=False)
    # print(jsonstring)
    # name = filename.split(".")[0]
    # with open(f"./uploads/{name}_proc.json", "w", encoding="UTF8") as file:
    with open(save_filename, "w", encoding="UTF8") as file:
        file.write(jsonstring)
    # return proc_messages

def load_data_proc(filename):
    with open(filename, "r", encoding="UTF8") as file:
        content = file.read()
    messages = json.loads(content)
    return messages


def remove_digit(data):
    str2 = ''
    for c in data:
        if c not in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '«', '»', '–', "\""):
            str2 = str2 + c
    data = str2
    return data


def remove_punctuation(data):
    str2 = ''
    import string
    pattern = string.punctuation
    for c in data:
        if c not in pattern:
            str2 = str2 + c
        else:
            str2 = str2 + ""
    data = str2
    return data


def remove_stopwords(data):
    str2 = ''
    from nltk.corpus import stopwords
    russian_stopwords = stopwords.words("russian")
    for word in data.split():
        if word not in (russian_stopwords):
            str2 = str2 + " " + word
    data = str2
    return data


def remove_short_words(data, length=1):
    str2 = ''
    for line in data.split("\n"):
        str3 = ""
        for word in line.split():
            if len(word) > length:
                str3 += " " + word
        str2 = str2 + "\n" + str3
    data = str2
    return data


def remove_paragraf_to_lower(data):
    data = data.lower()
    data = data.replace('\n', ' ')
    return data


def remove_all(data):
    data = remove_digit(data)
    data = remove_punctuation(data)
    data = remove_stopwords(data)
    data = remove_short_words(data, length=3)
    data = remove_paragraf_to_lower(data)
    return data


def get_RAKE(text):
    from rake_nltk import Metric, Rake
    r = Rake(language="russian")
    r.extract_keywords_from_text(text)
    numOfKeywords = 20
    keywords = r.get_ranked_phrases()[:numOfKeywords]
    return keywords


def get_YAKE(text):
    import yake
    language = "ru"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    numOfKeywords = 20
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    l=[]
    for item in keywords:
        l.append(list(item))
    return l


def get_KeyBERT(text):
    from keybert import KeyBERT
    kw_model = KeyBERT()
    # keywords = kw_model.extract_keywords(doc)
    numOfKeywords = 20
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english',
                            use_maxsum=True, nr_candidates=20, top_n=numOfKeywords)
    l=[]
    for item in keywords:
        l.append(list(item))
    return l


def set_scores(l):
    count = len(l)
    new_l=[]
    import random
    random.uniform(0, 1)
    for item in l:
        new_l.append([item, random.uniform(0, 1)])
    return new_l

def get_pattern(text):
    line = {}
    line['text'] = text.strip()
    line['remove_all'] = remove_all(text).strip()
    line['normal_form'] = get_normal_form(remove_all(text).strip())
    line['RAKE'] = set_scores(get_RAKE(text))
    line['YAKE'] = get_YAKE(text)
    line['BERT'] = get_KeyBERT(text)
    return line

def get_sintetic(text):
    line = {}
    line['text'] = text.strip()
    line['remove_all'] = remove_all(text).strip()
    line['normal_form'] = get_normal_form(remove_all(text).strip())
    line['RAKE'] = set_scores(get_RAKE(text))
    line['YAKE'] = get_YAKE(text)
    line['BERT'] = get_KeyBERT(text)
    str1="Возможные фразы сгенерированные нейросетевыми методами: \n"
    l= 26
    num_all = 0
    num_sin = 0
    for item in line['RAKE']:
        num_all +=1
        if len(item[0])> l:
            str1+=f"Фрагмент {num_sin}: {item[0]} \n"
            num_sin+=1
    for item in line['BERT']:
        num_all +=1
        if len(item[0])> l:
            str1+=f"Фрагмент {num_sin}: {item[0]} \n"
            num_sin+=1            
    for item in line['YAKE']:
        num_all +=1
        if len(item[0])> l:
            str1+=f"Фрагмент {num_sin}: {item[0]} \n"
            num_sin+=1
            
    str1+=f"Общее количество фраз сгенерированные нейросетевыми методами: {num_sin/num_all*100} %\n"                   
    line['sintetic'] = str1
    return line


def add_print_text(data):
    RAKE_text =[]
    for item in data['RAKE']:
        RAKE_text.append(item[0])
    YAKE_text =[]
    for item in data['YAKE']:
        YAKE_text.append(item[0])    
    BERT_text =[]
    for item in data['BERT']:
        BERT_text.append(item[0])
    
    str1 = str(f"Исходный текст: {data['text']} \n\n"
            f" Нормальная форма: {data['normal_form']} \n\n"
            f" RAKE: {RAKE_text} \n\n"
            f" YAKE: {YAKE_text} \n\n"
            f" BERT: {BERT_text} \n\n")
    data['print_text'] = str1
    # print(str1)
    return data


def get_normal_form_mas(words):
    morph = pymorphy2.MorphAnalyzer()
    result = []
    for word in words.split():
        p = morph.parse(word)[0]
        result.append(p.normal_form)
    return result


def get_normal_form(words):
    morph = pymorphy2.MorphAnalyzer()
    p = morph.parse(words)[0]
    return p.normal_form


def load_data(filename='data.txt'):
    with open(filename, "r", encoding='utf-8') as file:
        data = file.read()
    return data

def remove_from_patterns(text, pattern):
    str2 = ''
    for c in text:
        if c not in pattern:
            str2 = str2 + c
    return str2

def display(text):
    print(text) 
    print("--------------------------------")

def remove_paragraf_and_toLower(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = ' '.join([k for k in text.split(" ") if k])
    return text


def nltk_download():
    nltk.download('stopwords')
    nltk.download('punkt')
    

def calc_intersection_list(list1, list2):
    count = 0
    for item1 in list1:
        for item2 in list2:
            count += calc_intersection_text(item1, item2)
    return count

def calc_intersection_text(text1, text2):
    count = 0
    text1 = str(text1)
    text2 = str(text2)
    for item1 in text1.split():
        for item2 in text2.split():
            if item1 == item2:
                count += 1
    return count

def calc_score(data1, data2):
    pass


def find_cl(filename):
    messages = load_data_proc(filename)
    data_cl = load_db()
    cl_messages = []
    # def calc_intersection_all(text1, l2):
    #     max_counts = 0
    #     for item in l2:
    #         current_counts = calc_intersection_one(text1, item['normal_form'])
    #         if current_counts > max_counts:
    #             max_counts = current_counts
    #     return max_counts
    # counts = []
    find_data = []
    for m in messages:
        item = m
        num = 0
        item["RAKE_COUNT"] = 0
        item["RAKE_NUM"] = 0
        item["YAKE_COUNT"] = 0
        item["YAKE_NUM"] = 0
        item["BERT_COUNT"] = 0
        item["BERT_NUM"] = 0
        for cl in data_cl:
            intersect_RAKE = calc_intersection_list(m['RAKE'], cl['RAKE'])
            if intersect_RAKE>item["RAKE_COUNT"]:
                item["RAKE_COUNT"] = intersect_RAKE
                item["RAKE_NUM"] = num
            intersect_YAKE = calc_intersection_list(m['YAKE'], cl['YAKE'])
            if intersect_YAKE>item["YAKE_COUNT"]:
                item["YAKE_COUNT"] = intersect_YAKE
                item["YAKE_NUM"] = num
            intersect_BERT = calc_intersection_list(m['BERT'], cl['BERT'])
            if intersect_BERT>item["BERT_COUNT"]:
                item["BERT_COUNT"] = intersect_BERT
                item["BERT_NUM"] = num
            num += 1
        find_data.append(item)
    jsonstring = json.dumps(find_data, ensure_ascii=False)
    with open("./find_data.json", "w", encoding="UTF8") as file:
        file.write(jsonstring)


def find_type(filename, type='RAKE'):
    messages = load_data_proc(filename)
    find_data = []
    RAKE_set=set() 
    YAKE_set=set() 
    BERT_set=set() 
    for m in messages:
        RAKE_set.add(m['RAKE_COUNT'])
        YAKE_set.add(m['YAKE_COUNT'])
        BERT_set.add(m['BERT_COUNT'])
    RAKE_s = max(RAKE_set)
    YAKE_s = max(YAKE_set)
    BERT_s = max(BERT_set)
    # RAKE_set=sorted(RAKE_set, reverse=True)
    # YAKE_set=sorted(YAKE_set, reverse=True)
    # BERT_set=sorted(BERT_set, reverse=True)
    counts=3
    if type == 'RAKE':
        for m in messages:
            if m['RAKE_COUNT'] >= RAKE_s-counts:
                m = add_print_text(m)
                find_data.append(m)
    if type == 'YAKE':
        for m in messages:
            if m['YAKE_COUNT'] >= YAKE_s-counts:
                m = add_print_text(m)
                find_data.append(m)                    
    if type == 'BERT':
        for m in messages:
            if m['BERT_COUNT'] >= BERT_s-counts:
                m = add_print_text(m)
                find_data.append(m)                     
    jsonstring = json.dumps(find_data, ensure_ascii=False)
    with open("./find_d.json", "w", encoding="UTF8") as file:
        file.write(jsonstring)
    return jsonstring    


def convertMs2String(milliseconds):
    import datetime
    dt = datetime.datetime.fromtimestamp(milliseconds )
    return dt


def convertJsonMessages2text(filename):
    with open(filename, "r", encoding="UTF8") as file:
        content = file.read()
    messages = json.loads(content)
    text = ""
    for m in messages:
        text += f"{convertMs2String(m['date'])} {m['message_id']}  {m['user_id']} {m['reply_message_id']}  {m['text']}  <br>\n"
        #text += f"{m['text']}\n"
    return text


if __name__ == '__main__':
    # nltk_download()
    #s1 = """
    # Дарим 1000 бонусов за 1-ю авторизацию в мобильном приложении до 22.03.2023. Используйте бонусы на онлайн покупки. Clck.ru/33gyhM
    #"""
    #add_data(s1)
    # t = get_pattern(data)
    # print(t)

    filename="d:/ml/chat/andromedica1.json"
    save_filename="./data_proc.json"
    
    data_proc(filename, save_filename, 32)
    find_cl(save_filename)
    find_type("./find_data.json", 'RAKE')
    
