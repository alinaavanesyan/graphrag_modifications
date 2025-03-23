# Проект по развитию GraphRAG

👉 [GraphRAG Arxiv](https://arxiv.org/pdf/2404.16130)

👉 [Origin Microsoft Library](https://github.com/microsoft/graphrag)

## Обзор

В этом репозитории представлен проект, направленный на:
1. Адаптацию GraphRAG под русскоязычную модель YandexGPT Experimental (код [здесь](https://github.com/alinaavanesyan/GraphRAG_for_YandexGPT))

2. Оптимизацию алгоритма выделения сообществ посредством встраивания алгоритма Лейдена, BFS и Баяна (код [здесь](https://github.com/mashagodunova/graphrag))

Мы также протестировали опенсорс-модели внутри GraphRAG с помощью библиотеки [nano-graphrag](https://github.com/gusye1234/nano-graphrag). В рамках проекта мы тестировали модель DeepSeek-R1, для этого нужно выполнить следующие команды:

```
git clone https://github.com/gusye1234/nano-graphrag.git
cd nano-graphrag
pip install -e .
```
Далее в папке nano-graphrag находим файл *prompt.py* и заменяем его на наш файл *graphrag_modifications/nano-graphrag/prompt.py* (мы переписали пропмпты под нашу задачу: перевели на русский язык и добавили уточнения, чтобы DeepSeek генерировал ответ в нужном нам формате). Затем создаем папку *tests* и добавляем туда данные, на которых мы хотим построить граф (мы использовали каждый из двух файлов в папке *graphrag_modifications/data*).

Выполняем следующую команду, чтобы скачать модель, на основе которой мы будем строить граф:
```
ollama pull deepseek-r1:14b
```

Наконец, запускаем построение графа (скачиваем тетрадку *graphrag_modifications/nano-graphrag/deep14b_gazeta_launch.py* и перемещаем её в нашу папку с библиотекой *nano-graphrag/examples*):
```
python3 deep14b_gazeta_launch.py
```
