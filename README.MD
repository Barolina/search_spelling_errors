####  Используемые  либы

- nltk - для   статической обрабоки  текста ( разбиение на  предложения  и слова)
- словарь от  libreoffice

###### Лирика: существуют либы для поиска ошибок, к примеру 
- pyenchant - для поиска  ошибок 
- difflib - для  вычисления  различий  между словами

###### Но мы будем  использовать для  поиска ошибок
- алгоритм Левенштейна для сравнения 2 строк
- и алгоритм сравнения строк на основе коэффициента Жаккара
- затем  для  однознаности  -  поиск  однокоренных слов

##### Предположения,  границы схожести,  которые можно  регулирвоать

- по Ливенштейну, если  расстоние между словами < 1, то скорее  всего они очень похожи
- коэффициент Жаккара, если коэффициент  < 0.199, то скорее всего слова  так же очень схожи
- Если слова однокоренные  то есть  процент схожести < 85 , то сокрее всего исходное слова  написано неверно

##### Архитектура алгоритма 

```mermaid
flowchart TB
	A[Загрузить справочник  слов]
	B[Разбить исходный текст на слова]
	C[Пройти по  каждому слову]		
	M[Определить наименьшее расстояние  \n по Левенштейну и коэффициент Жаккара \n каждого слова по словарю]
	D[Если расстояние \n по Ливенштейну < 1 \n и коэффициент Жаккара <=0.199]
	H[То получаем список похожих слов с нашим]	
	K[Определяем  процент схожести \n исходного слова  с нашим найденным списком, \n если оно меньше 85, \n то слово  скорее всего  написано \n  неверно \n, помогла функция  из nltk]
	L[Заменяем ошибочное  слово \n в исходном тексте]
	A --> B
	B --> C	 	
	C --> M	
	M --> D
	D --> H	
	H --> K	
	K --> L
	

```

#### Настройка  ```Ubuntu```

1. Создать виртуальное окружение  

  ```python3 -m venv venv ```
  
2. Активируем  окружение 

``` source venv/bin/activate```

3. Установить зависимости 
 
  ```pip install -r requirements.txt```
 
3. И запуск)

  ```python3 main.py -f sample.txt ```


#### Пример запуска ```Windows```

```python
python  main.py -f C:\Users\user\Documents\PROJECTS\orfo_error\sample.txt
```

**Результат работы скрипта**

```python
2022-03-20 16:58:35,246 [INFO] Поиск  ошибок ...
сонце
2022-03-20 16:58:35,519 [INFO] Исходное  предложение:           Выплыла из-за леса сонце.
2022-03-20 16:58:35,521 [INFO] Найденные ошибки: ['сонце']
2022-03-20 16:58:35,522 [INFO] Возможно имелось ввиду:           выплыла из-за леса солнце.
....
```
