import argparse
import logging
from pathlib import Path

import enchant
import nltk
import numpy as np
from enchant.checker import SpellChecker
from fuzzywuzzy import process
from nltk.stem.snowball import RussianStemmer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# if 'ru_RU' not in enchant._broker.list_languages():
#     logging.info(
#         "Нужно скопировать файлы  из директории  spell  в  директрию site-packages/enchant/data/mingw64/share/enchant/hunspell")
#     exit()

# неплохое решение поиска  ошибочных слова, ну лан
# spell_checker = SpellChecker("ru_RU")
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
            i = i.strip()
            index = i.find('/')
            if index == -1:
                index = len(i) + 1
            list_word.append(i[0:index])

    return list_word


def jakkara(s1, s2):
    """
    алгоритм сравнения строк на основе коэффициента Жаккара
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
    алгоритм Левенштейна
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
     Поиск походих слов
    :param word_s: искомое млово
    :return: dict()
    """
    list_word = get_list_word()
    results = dict()
    for word in list_word:
        if word[0] != word_s[0]:
            continue
        if word_s == word:
            results[1] = word
            continue
        cost = levenshtein(word, word_s)
        cost1 = jakkara(word, word_s)
        if cost <= 1 and cost1 <= 0.199:
            print("Lev:" + str(cost) + " " + word)
            print("Jak: " + str(cost1) + " " + word)
            results[cost] = word
    return results


def find_error_word(text):
    text = text.lower()
    error_words = list()
    # разбиваем на слова
    words = nltk.word_tokenize(text)
    stemmer = RussianStemmer()

    result = list()
    for a in words:
        a = a.replace(".", "").replace(",", "").strip()
        if len(a) <= 3:
            result.append(a)
            continue
        # предплагаемые
        sim = suggest(a)
        if len(sim) >= 1:
            if a in sim.values():
                result.append(a)
                continue
            index_sim = sim[max(sim.keys())]
            # найдем  наиболее однокоренне
            stem_l = [stemmer.stem(x) for x in sim.values()]
            procent = process.extractOne(a, stem_l)
            # граничне  рамки, если маньше  75 -  ну точно не то слово
            if procent[1] >= 85 or procent[1] < 65:
                result.append(a)
            else:
                error_words.append(a)
                result.append(index_sim)
        else:
            result.append(a)
    return " ".join(result), error_words


def parser_file(path_file=None):
    f = open(path_file, 'r', encoding='utf8')
    try:
        tokens = nltk.sent_tokenize(f.read())
        for i in tokens:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="file_path", type=Path)
    p = parser.parse_args()
    if not p.file_path or not p.file_path.exists():
        logging.error("Не сущуствует  файла")
    else:
        logging.info("Поиск  ошибок ...")
    parser_file(p.file_path)
