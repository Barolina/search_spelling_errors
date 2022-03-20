import argparse
import logging
from pathlib import Path

import enchant
import nltk
from enchant.checker import SpellChecker
from nltk.stem.snowball import RussianStemmer
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

if 'ru_RU' not in enchant._broker.list_languages():
    logging.info(
        "Нужно скопировать файлы  из директории  spell  в  директрию site-packages/enchant/data/mingw64/share/enchant/hunspell")
    exit()

spell_checker = SpellChecker("ru_RU")
nltk.download('punkt')


def get_list_word():
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


def levenshtein(word1, word2):
    columns = len(word1) + 1
    rows = len(word2) + 1

    # build first row
    current_row = [0]
    for column in range(1, columns):
        current_row.append(current_row[column - 1] + 1)

    for row in range(1, rows):
        previous_row = current_row
        current_row = [previous_row[0] + 1]

        for column in range(1, columns):

            insert_cost = current_row[column - 1] + 1
            delete_cost = previous_row[column] + 1

            if word1[column - 1] != word2[row - 1]:
                replace_cost = previous_row[column - 1] + 1
            else:
                replace_cost = previous_row[column - 1]

            current_row.append(min(insert_cost, delete_cost, replace_cost))

    return current_row[-1]


def lev(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current_row = range(n + 1)
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]


def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + 1)  # transposition

    return d[lenstr1 - 1, lenstr2 - 1]


def suggest(word_s):
    error = set()
    list_word = get_list_word()
    results = dict()
    import numpy as np
    array = np.array(list_word)
    for word in list_word:
        if word[0] != word_s[0]:
            continue
        # cost = damerau_levenshtein_distance(word, word_s)
        # cost_jar = tanimoto(word, word_s)
        cost =  damerau_levenshtein_distance(word, word_s)
        if cost <= 1:
            print(str(cost) + " " + word)
            results[cost] = word
    # logging.info("Левенштейн " + results )
    # result = list(zip(list_word, list(normalized_damerau_levenshtein_distance(word_s, array))))
    # number = damerau_levenshtein_distance(word_1, word_2)
    # print('\n' + text, sorted(result, key=lambda x: x[1]))

    # command, rate = min(result, key=lambda x: x[1])
    # if rate > 0.25:
    #     results[rate]= command
    return results


def find_error_word(text):
    text = text.lower()
    error_words = list()
    words = text.replace("  ", " ").split(" ")
    result = list()
    for a in words:
        a = a.replace(".","").replace(",","").strip()
        sim = set()
        if not a:
            continue
        if len(a) <= 3:
            result.append(a)
            continue
        stemmer = RussianStemmer()
        # поиск однокоренного слова
        stemmer_review = stemmer.stem(a)
        sim = suggest(a)
        if len(sim) >= 1:
            index_sim = sim[min(sim.keys())]
            # если  совпали исходные слова или  однокоренные
            if a == index_sim:
                result.append(a)
            else:
                result.append(index_sim)
                error_words.append(a)
                logging.info("Ошибочное слова " + a)
        else:
            result.append(a)
        # print(sim)
    # print(result)
    return " ".join(result), error_words


def parser_file(path_file=None):
    f = open(path_file, 'r', encoding='utf8')
    try:
        tokens = nltk.sent_tokenize(f.read())
        for i in tokens:
            correct_sentense, error_words = find_error_word(i)
            if error_words:
                logging.info("Исходное  предложение: " + i)
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
