import os
import sys

sys.path.append("..")
import logging

import re
import json
import asyncio
import tiktoken
from typing import Union
from collections import Counter, defaultdict
from my_splitter import SeparatorSplitter
from my_utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)

from dataclasses import asdict
from prompt import GRAPH_FIELD_SEP, PROMPTS
from op import _map_global_communities
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
import json
import asyncio

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

MODEL = "deepseek-r1:14b"

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL_DIM = 768
EMBEDDING_MODEL_MAX_TOKENS = 8192
SYSTEM_PROMPT = "You are an intelligent assistant and will follow the instructions given to you to fulfill the goal. The answer should be in the format as in the given example."

async def ollama_model_if_cache(
    prompt, system_prompt=SYSTEM_PROMPT, history_messages=[], **kwargs
) -> str:
    # remove kwargs that are not supported by ollama
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    response = await ollama_client.chat(model=MODEL, messages=messages,**kwargs)

    result = response["message"]["content"]
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    # -----------------------------------------------------
    return result


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


WORKING_DIR = "./deepseek14b_gazeta"

def my_json_convert(my_str):
    answers = {}
    answers['key'] = my_str
    return answers

async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    
    }
    logger.info(f"Loaded {len(community_schema)} communities: {community_schema}")
    
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]

    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    logger.info(f"Retrieved {len(community_datas)} communities after filtering")
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")
    
    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config
    )
    print('map_commuinties')
    print(map_communities_points) 
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    print(final_support_points)
    print('final_support')

    if not len(final_support_points):
        return PROMPTS["fail_response"]
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )
    logger.info(f"Final support points before filtering: {final_support_points}")
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
            Importance Score: {dp['score']}
            {dp['answer']}
            """
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    return response


async def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=ollama_embedding,
        convert_response_to_json_func=my_json_convert
    )

    results = {}

    prompts = [
    'Какие места упоминаются в данном тексте (в данных текстах)?',
    'Какие люди упоминаются в данном тексте? (в данных текстах?)',
    'Чем занимается каждый человек, упомянутый в тексте? (выведи пары “человек-его область деятельность”)',
    'Какие организации упоминаются в предложенном тексте? Выведи организацию и конкретные задачи, которыми занимаются в данной организации',
    'Какие организации/люди/группы людей взаимодействуют в этом тексте? Выведи триплеты: объект1—действие—объект2'
     ]
    param = QueryParam()
    for p in prompts:
        resp = await global_query(
                p,
                rag.chunk_entity_relation_graph,
                rag.entities_vdb,
                rag.community_reports,
                rag.text_chunks,
                param,
                asdict(rag)
                )
        results[p] = resp
    
    with open('deep14b_gazeta_results.json', 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

def metrics(res_path):
    with open(res_path, 'r') as file:
        prompts_results = json.load(file)

    final = {}
    for i,(ques,ans) in enumerate(prompts_results.items()):
        compr = f"""
        На вопрос по корпусу новостных текстов {ques} модель ответила следующим образом: {ans}
        Оцени по шкале от 0 до 10 ответ модели по метрике Comprehensiveness и объясни свое решение.
        (метрика Comprehensiveness, или Полнота, показывает, насколько подробно ответ покрывает все аспекты и детали, связанные с заданным вопросом. Эта метрика измеряет способность ответа охватить максимальное количество деталей, связанных с поставленным вопросом. Чем более полный и детализированный ответ, тем выше показатель полноты).
        """
        div = f"""
        На вопрос по корпусу новостных текстов {ques} модель ответила следующим образом: {ans}
        Оцени по шкале от 0 до 10 ответ модели по метрике Diversity  и объясни свое решение.
        (метрика Diversity, или Разнообразие, показывает, насколько ответ разнообразен и богат в предоставлении различных перспектив и идей. Метрика оценивает способность ответа включать разные точки зрения и контексты, что позволяет раскрыть сложность и многосторонность темы)
        """
        emp = f"""
        На вопрос по корпусу новостных текстов {ques} модель ответила следующим образом:{ans}
        Оцени по шкале от 0 до 10 ответ модели по метрике Empowerment  и объясни свое решение.
        (метрика Empowerment, или Помощь в принятии решений, показывает, насколько ответ помогает пользователю лучше понять тему и сделать обоснованные выводы. Эта метрика отражает, насколько ответ способен предоставить контекст, обоснование и объяснение, которые помогут читателю глубже разобраться в вопросе и сделать правильные выводы)
        """
        direct = f"""
        На вопрос по корпусу новостных текстов {ques} модель ответила следующим образом: {ans}
        Оцени по шкале от 0 до 10 ответ модели по метрике Directness  и объясни свое решение.
        (метрика Directness, или Прямота, показывает, насколько конкретно и чётко ответ адресует поставленный вопрос. Прямота измеряет, насколько прямо ответ отвечает на сам вопрос, без лишней информации или отклонений от темы).
        """
        c = ''
        dvr = ''
        e = ''
        dr = ''
    
        c = rag.query(compr, param=QueryParam(mode="global"))
        dvr = rag.query(div, param=QueryParam(mode="global"))
        ce = rag.query(emp, param=QueryParam(mode="global"))
        dr = rag.query(direct, param=QueryParam(mode="global"))
    
        result_metrics = {'Полнота':c,'Разнообразие':dvr, 'Помощь в принятии решений':e,'Прямота':dr}
        final[i] = {}
        final[i]['Промпт'] = ques
        final[i]['Ответ'] = ans
        final[i]['Оценки'] = result_metrics

    with open('/home/user/avanesyan/nano-graphrag/qwen_podcast_metrics.json', 'w', encoding='utf-8') as json_file:
        json.dump(final, json_file, ensure_ascii=False, indent=4)

def insert():
    from time import time

    with open("./tests_alina/podcast.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/kv_graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=ollama_embedding,
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)

# We're using Ollama to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)
async def ollama_embedding(texts: list[str]) -> np.ndarray:
    embed_text = []
    for text in texts:
        data = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embed_text.append(data["embedding"])

    return embed_text

# Раскомментируйте для построения графа
if __name__ == "__main__":
    insert()
    # Раскомментируйте для получения метрик
    # metrics()


# Раскомментируйте для глобального поиска
# async def main():
#     await query()

# if __name__ == "__main__":
#     asyncio.run(main())