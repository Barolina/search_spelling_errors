import argparse
import difflib
import logging
from pathlib import Path

import enchant
import nltk
from enchant.checker import SpellChecker

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


def find_error_word(text):
    text = text.lower()
    spell_checker.set_text(text)

    error_words = list()

    for i in spell_checker:
        print(i.word)
        error_words.append(str(i.word))
        suggestions = set(spell_checker.suggest(i.word.lower()))
        sim = dict()
        for j in suggestions:
            measure = difflib.SequenceMatcher(None, i.word.lower(), j.lower()).ratio()
            sim[measure] = j
        correct_word = sim[max(sim.keys())]
        i.replace(correct_word)
        # print("Coorect  sentense " + spell_checker.get_text())
    return spell_checker.get_text(), error_words


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
