# Проект по развитию GraphRAG

👉 [GraphRAG Arxiv](https://arxiv.org/pdf/2404.16130)

👉 [Origin Microsoft Library](https://github.com/microsoft/graphrag)

## Обзор

В этом репозитории представлен проект по модификации GraphRAG-подхода.

### 1. Адаптация GraphRAG под русскоязычную модель YandexGPT Experimental и тестирование опенсор-моделей внутри GraphRAG
Мы переписали библиотеку GraphRAG так, чтобы сделать доступной возможность построения графа с помощью модели YandexGPT (код [здесь](https://github.com/alinaavanesyan/GraphRAG_for_YandexGPT)). Дело в том, что сейчас существуют библиотеки, которые позволяют тестировать внутри GraphRAG модели, отличные от OpenAI-моделей, однако только те, что представлены в Ollama. Мы же решили эту проблему, сделав код фрейморвка более гибким.

С помощью переписанного кода мы построили графы на следующих данных:

*data/gazeta.txt* - часть датасета IlyaGusev/gazeta с саммаризированными новостными статьями, собранными из различных источников (отрывок из ≈250k токенов).

*data/podcast.txt* - транскрипты подкастов с YouTube-канала "Уютный ФКНчик" (15 выпусков, ≈150k токенов).

Мы также протестировали опенсорс-модель DeepSeek-R1 с помощью библиотеки [nano-graphrag](https://github.com/gusye1234/nano-graphrag).
Для того чтобы сделать это самостоятельно, необходимо выполнить следующие команды:

1.
```
git clone https://github.com/gusye1234/nano-graphrag.git
cd nano-graphrag
pip install -e .
```
2. Далее в папке nano-graphrag находим файл *prompt.py* и заменяем его на наш файл *graphrag_modifications/nano-graphrag/prompt.py* (мы переписали пропмпты под нашу задачу: перевели на русский язык и добавили уточнения, чтобы DeepSeek генерировал ответ в нужном нам формате). Затем создаем папку *tests* и добавляем туда данные, на которых мы хотим построить граф (мы использовали каждый из двух файлов в папке *graphrag_modifications/data*).

3. Выполняем следующую команду, чтобы скачать модель, на основе которой мы будем строить граф:
```
ollama pull deepseek-r1:14b
```
4. Наконец, запускаем построение графа (скачиваем тетрадку *graphrag_modifications/nano-graphrag/deep14b_gazeta_launch.py* и перемещаем её в нашу папку с библиотекой *nano-graphrag/examples*):
```
python3 deep14b_gazeta_launch.py
```

### 2. Оптимизация алгоритма выделения сообществ посредством встраивания алгоритма Лейдена, BFS и Баяна (код [здесь](https://github.com/mashagodunova/graphrag))



3аа
