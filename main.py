import argparse
import logging
from pathlib import Path

import enchant
import nltk
import numpy as np
from enchant.checker import SpellChecker
from fuzzywuzzy import process
from nltk.stem.snowball import RussianStemmer

# настроим  логер

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

COST_LEVENSHTEIN  = 1
COST_JAR= 0.199

MIN_SIGLE=65
MAX_SIGLE=85

#  для раздения на  предложения, а потом на слова
nltk.download('punkt')


def get_list_word():
    """
    Получить словарь исходных слов
    :return: list()
    """
    list_word = list()
    with open("spell/ru_RU.dic", "r", encoding='utf8') as file:
        lines = file.readlines()
        for i in lines:
            #  убрем  лишние символы из словаря
            i = i.strip()
            index = i.find('/')
            if index == -1:
                index = len(i) + 1
            list_word.append(i[0:index])

    return list_word


def jakkara(s1, s2):
    """
    Алгоритм сравнения строк на основе коэффициента Жаккара
    :param s1:
    :param s2:
    :return:
    """
    c = 0
    s1 = set(s1)
    s2 = set(s2)
    a = len(s1)
    b = len(s2)
    for ch in s1:
        if ch in s2:
            c += 1
    return 1 - c / (a + b - c)


def levenshtein(seq1, seq2):
    """
    Алгоритм Левенштейна
    :param seq1:
    :param seq2:
    :return:
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def suggest(word_s):
    """
    Поиск похожих слов
    :param word_s: искомое млово
    :return: dict()
    """
    list_word = get_list_word()
    results = dict()
    # поиск похожего слова в нашем словаре
    for word in list_word:
        if word[0] != word_s[0]: # исключаем ненужные,   начинаются на другую  букуву
            continue
        if word_s == word: # если искомое  слово сразу совпало с  исходным
            results[1] = word
            break
        cost = levenshtein(word, word_s) # расстояние  по Ливенштейну
        cost1 = jakkara(word, word_s) # и коеффициент
        # если подходит под  наши условия, вероятно слвоа похожи
        if cost <= COST_LEVENSHTEIN and cost1 <= COST_JAR: 
            print("Lev:" + str(cost) + " " + word)
            print("Jak: " + str(cost1) + " " + word)
            results[cost] = word
    return results


def find_error_word(text):
    """
    Поиск ошибок в предоложении
    """
    text = text.lower()
    # для   хранения ошибочных слов
    error_words = list()
    # разбиваем на слова
    words = nltk.word_tokenize(text)
    
    # для нахождения однокоренных слов
    stemmer = RussianStemmer()

    result = list()
    # по всем словам нашего предложения
    for a in words:        
        if len(a) <= 3: # ну союзы и тд можно сразу  отсеить
            result.append(a)
            continue
        # получить список похожих слов
        sim = suggest(a)
        if len(sim) >= 1:
            # могло совпасть исходное и  то что в словаре
            if a in sim.values():
                result.append(a)
                continue
            # наиболее приближенное к  нашему слову  - предполагаемое  правильное слово
            index_sim = sim[max(sim.keys())]
            # формируем  список однокоренных слов из  списка похожих слов
            stem_l = [stemmer.stem(x) for x in sim.values()]
            # пробуем определить однокоренное, то есть убираем  помехи
            procent = process.extractOne(a, stem_l)
            # граничные  рамки, если маньше  75 -  ну точно не то слово (интуитивно)
            logging.info("Процент схожести" + procent)
            if procent[1] >= MAX_SIGLE and procent[1] < MIN_SIGLE:
                result.append(a)
            else:
                # нашли ошибочное 
                error_words.append(a)
                result.append(index_sim)
        else:
            result.append(a)
    return " ".join(result), error_words


def parser_file(path_file=None):
    """
      Функция поиска ошибок в тексте 
    """
    f = open(path_file, 'r', encoding='utf8')
    try:
        # разбиваем на предложения
        tokens = nltk.sent_tokenize(f.read())
        # по всем прделожениям
        for i in tokens:
            # анализируем наше предложение
            correct_sentense, error_words = find_error_word(i)
            logging.info("Исходное  предложение: " + i)
            if error_words:
                logging.info("Найденные ошибки: " + str(error_words))
                logging.info("Возможно имелось ввиду: " + correct_sentense)
    except Exception as e:
        logging.error(e)
    finally:
        f.close()


if __name__ == '__main__':
    # поиск офрограяических ошибок
    # на вход принимает путь до файла с текстом
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="file_path", type=Path)
    p = parser.parse_args()
    if not p.file_path or not p.file_path.exists():
        logging.error("Не сущуствует  файла")
    else:
        logging.info("Поиск  ошибок ...")
    parser_file(p.file_path)
