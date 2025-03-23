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

from prompt import GRAPH_FIELD_SEP, PROMPTS

async def _map_global_communities(
    query: str,
    communities_data: list[CommunitySchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    use_model_func = global_config["best_model_func"]
    community_groups = []
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group) :]
    print('комьюнити групс')
    print(community_groups)

    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"],
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)
        #print('комьюнити контекст')
        #print(community_context)
        #print('комьюнити сэкшн лист')
        #print(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
        #print('промпт')
        #print(sys_prompt)
        res_prompt = query + sys_prompt

        response = await use_model_func(
            res_prompt,
            system_prompt=sys_prompt,
            **query_param.global_special_community_map_llm_kwargs,
        )
        data = use_string_json_convert_func(response)
        #print('response')
        #print(response)
        #print(type(response))
        #print('points')
        #print(type(data))
        #print(data.keys())
        #print(data)
        #print(type(data['key']))
        #print(data['key'].keys())
        print('answer')
        print(data)
        if i == 1:
            with open('op_output.txt', 'w') as f:
                f.write(str(data))
        if '<think>' in str(data['key']) and 'points' in str(data['key']):
            data['key'] = str(data['key']).replace('\n', '')
            data['key'] = data['key'].replace('```', '')
            print('data[key]')
            print(data['key'])
            if 'json' in data['key']:
                json_part = data['key'].split("json")[1]
            else:
                json_part = data['key'].split('</think>')[1]
            print('json_part')
            print(json_part)
            if "'points'" not in json_part and '"points"' not in json_part:
                json_part = json_part.replace('points', '"points"')
            json_part = json_part.replace("{points: ", '{"points":')
            parsed_data = json.loads(json_part)
            return parsed_data['points']
        if data['key'].startswith('"points"') or data['key'].startswith("'points'"):
            data['key'] = str(data['key']).replace('«points»', '"points"')
            data['key'] = data['key'].replace('Points', 'points')
            data['key'] = data['key'].replace('POINTS', 'points')
            json_part = '{' + data['key'] + '}'
            return json.loads(json_part)       
        try:
            parsed_data = json.loads(data['key'])
            #print('parsed')
            #print(parsed_data)
            return parsed_data['points']
        except:
            data['key'] = str(data['key'])
            data['key'] = data['key'].replace('`', '')
            data['key'] = data['key'].replace('\n', '')
            #data['key'] = data['key'].replace('json', '')
            data['key'] = data['key'].replace('JSON:', 'json')
            data['key'] = data['key'].replace('JSON', 'json')
            if "'points'" not in data['key'] and '"points"' not in data['key']:
                data['key'] = data['key'].replace('points', '"points"')
            data['key'] = data['key'].replace("{points:", '{"points":')            
            if 'json' in data['key']:
                json_part = data['key'].split('json')[1]
                return json.loads(json_part)['points']
            elif 'score:' in data['key']:
                parsed_data = json.loads(data['key'])
                return parsed_data['points']
            else:
                parsed_data = data['key']
            print('parsed')
            print(parsed_data)
            return parsed_data
        #return data['key'].get("points", [])
        #return response

    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    return responses


