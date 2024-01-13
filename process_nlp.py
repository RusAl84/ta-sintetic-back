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
    l= 42
    num_all = 1
    num_sin = 1
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
            
    str1+=f"\n \n Общее количество фраз сгенерированные нейросетевыми методами: {num_sin/num_all*100} %\n"                   
    if num_sin >2:
        line['sintetic'] = str1
    else:
        line['sintetic'] = f"Фраз сгенерированные нейросетевыми методами не обнаружено\n"
    if num_sin == 21:
        line['sintetic'] = f"Фраз сгенерированные нейросетевыми методами не обнаружено\n"

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
    RAKE_set=sorted(RAKE_set, reverse=True)
    YAKE_set=sorted(YAKE_set, reverse=True)
    BERT_set=sorted(BERT_set, reverse=True)
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
    s1 = """
     У любой компании, независимо от её размера, имеется внутренняя инфраструктура, состоящая как из сервисов обслуживающих сам бизнес, так и из систем информационной безопасности (ИБ). С ростом их количества становится трудно осуществлять мониторинг происходящих событий и выявлять инциденты. Более того, нет единого стандарта журналирования, что осложняет задачу анализа журналов, влечёт за собой пропуск части событий или требует увеличения штата сотрудников для их мониторинга. Такие методы несовместимы с прогрессивным подходом к ИБ. Решить этот вопрос поможет внедрение системы мониторинга и анализа событий по информационной безопасности (SIEM), которая позволит упростить этот процесс и автоматизировать выявление сложных инцидентов по событиям из нескольких источников. Необходимость использования SIEM С развитием бизнеса компании обрастают как бизнес-сервисами, так и системами информационной безопасности. В определённый момент наступает этап зрелости, когда не только процессы требуют соответствия требованиям регуляторов, но и сами работники ИБ и ИТ начинают нуждаться в инструменте мониторинга технических данных и поиска признаков вредоносной активности. В итоге компания приходит к выводу о необходимости SOC (Security Operations Center) — команды специалистов по мониторингу, анализу и реагированию на события и инциденты. Для реализации потребностей таких специалистов требуется инструмент сбора, корреляции и анализа логов, то есть SIEM. Для начала следует определить, нужна ли компании SIEM. В целом можно выделить следующие предпосылки к её внедрению: Большое количество разнообразных систем в инфраструктуре. Требуется единая точка входа для сбора и анализа журналов. Большое количество «белых пятен» в инфраструктуре компании, особенно при быстром росте и развитии бизнес-систем. Каждый отдел или команда ведёт и сопровождает только свои системы и процессы, не имея представления о смежных системах и интеграциях. У части систем может вообще не оказаться ответственного администратора. Потребность в своевременном реагировании на инциденты. Любой простой и любые неполадки наносят материальный ущерб, так что разрешать такие ситуации нужно оперативно. Большое количество систем защиты информации (СЗИ) имеют собственные журналы, события из которых необходимо соотносить между собой для выявления сложных инцидентов. Есть достаточно большой штат ИБ-специалистов, который нуждается в единой консоли для решения задач по инцидентам и событиям. Необходимо обеспечивать соответствие требованиям нормативных правовых актов регуляторов (законов № 152-ФЗ, 161-ФЗ, 187-ФЗ, приказов ФСТЭК России № 21, 17 и 31, СТО БР ИББС, РС БР ИББС-2.5-2014, ГОСТ Р 57580.1-2017, международного стандарта PCI DSS). Необходимость импортозамещения имеющейся SIEM от иностранного вендора. Требуется независимая оценка состояния инфраструктуры. Необходимо охватить средством мониторинга не только центральный офис, но и все филиалы. Например, SIEM-систему Kaspersky Unified Monitoring and Analysis Platform (KUMA) можно использовать как для централизованного управления событиями во всех филиалах, так и для локального мониторинга событий в каждом из них. Подобная функциональность позволяет ответственным за ИБ в филиалах и подразделениях или внешним поставщикам ИБ-услуг обнаруживать и приоритизировать угрозы в рамках своих зон ответственности. Одним из показателей высокой зрелости компании в области информационной безопасности является наличие собственного SOC, т. е. различных ИБ-инструментов, ИТ- и ИБ-специалистов, а также процессов для реагирования на инциденты и их расследования. SIEM на этом пути является первой ступенью, которая помогает значительно увеличить скорость реакции. По факту SIEM — это ядро, но не первостепенный инструмент для выстраивания информационной безопасности. Компании задумываются о SIEM-системе и в принципе смотрят в сторону создания собственного SOC или отдела мониторинга ИБ тогда, когда уже есть достаточное количество СЗИ и необходимо выстроить или скорректировать внутренние ИБ-процессы. При этом SIEM может быть интересна не только специалистам по ИБ, но и ИТ-специалистам, пусть и в ограниченном доступе. Например, если нам важна работа некой бизнес-значимой системы в режиме 24×7, мы можем отслеживать её доступность путём сбора логов в SIEM; если получить логи не удаётся, это становится сигналом о недоступности системы. Ещё одним нестандартным примером может служить контроль невмешательства в конфигурационные файлы ИТ- или бизнес-систем, для того чтобы любые изменения в их настройках были контролируемыми. Когда в компании приходят к выводу о необходимости внедрения SIEM, встаёт выбор: пойти к облачному провайдеру и заказать у него услугу SOC, куда будет уже добавлена стоимость их SIEM, либо внедрить и поддерживать собственное локальное (on-premise) решение. Кстати, у облачного SOC есть различные тарифы и возможности, с которыми важно внимательно ознакомиться перед приобретением, так как в конечном счёте можно остаться с очень ограниченной функциональностью — например, просто получать сигналы об обнаружении вирусной активности на хостах. С таким же успехом можно настроить оповещения и получать эту информацию от Kaspersky Security Center, без приобретения услуги SOC. Также важно отметить, что пока не будет обеспечена базовая защита инфраструктуры, а в штате не будет достаточного количества специалистов по ИБ, SIEM-система не принесёт пользы. В такой ситуации она будет просто нерентабельна и неинформативна.  При наличии же большой инфраструктуры, штата ИБ-специалистов, парка из сотен или тысяч бизнес-серверов и большого количества различных СЗИ встаёт дилемма: использовать программное обеспечение как услугу (SaaS) или внедрить SIEM в собственной инфраструктуре. Рассмотрим «плюсы» внедрения локальной системы (вместо SaaS-сервиса). Основным и главным «плюсом» являются полный контроль и понимание собственной инфраструктуры ИБ- и ИТ-специалистами. Облачный провайдер будет знать о вашей инфраструктуре и её процессах только то, что вы предоставите в виде анкет или логов подключённых систем, а инфраструктура и процессы имеют свойство динамически изменяться. Также нужно иметь в виду, что легитимная по версии облачного провайдера активность конкретно для вашей компании может быть аномальной. Сроки исполнения задач по подключению новых систем, написанию коннекторов, новых правил корреляции и расследованию инцидентов стоят на контроле у внутренней команды SIEM. Срочные задачи будут выполняться в приоритетном для неё порядке. Возможность неограниченного подключения новых источников, написания своих правил нормализации, обогащения, составления сложных правил корреляции. В случае локального развёртывания нет необходимости изменять рамки договора услуг или выделять дополнительные средства под такие задачи. Локальная SIEM-система не позволит упустить, забыть или проигнорировать случившийся инцидент. На него всё равно нужно будет отреагировать, и у руководства будет карт-бланш на проверку исполнительности сотрудников в части расследования инцидента. Возможность подключения внешней команды для расследования сложного инцидента в случае нехватки компетенций. Ей можно продемонстрировать все данные в локальной SIEM для установления причин возникшего инцидента и выдачи рекомендаций по недопущению подобных инцидентов в будущем. Пилотные проекты SIEM: как проводить и на что обратить внимание В этой части статьи коснёмся того, каким образом может проводиться выбор производителя системы. Источниками информации и критериев могут быть: Руководства, обзоры рынка, презентации систем. Рекомендации от других компаний в вашем или похожем сегменте бизнеса. Собственные требования и опыт, удобство работы для аналитиков, простота управления, написания новых правил корреляции событий, разработки коннекторов. Цена продукта и стоимость владения (на дистанции в три года и более). Разнообразие и достаточность набора коннекторов «из коробки», удобство сбора логов и формата их представления. Производительность системы — число обрабатываемых событий в секунду (EPS). Кстати, KUMA от «Лаборатории Касперского» отличается на сегодня одним из наиболее высоких показателей производительности, подтверждённым в тестах на реальной инфраструктуре — более 300 тысяч EPS. 
    """
    t = get_sintetic(s1)
    print(t['sintetic'])

