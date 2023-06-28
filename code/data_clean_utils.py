# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import logging
from tqdm import tqdm
import gc
import random
import time
import multiprocessing
import concurrent.futures
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    BertTokenizer,
    BertModel,
)
import fire
import itertools
import faiss
import math
import pandas as pd
from collections import Counter
sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
logger = logging.getLogger(__name__)
from data_utils import df_reader, load_jsonl, save_json, load_json, df_saver
from sentence_transformers import SentenceTransformer, LoggingHandler
try:
    from text2vec import SentenceModel
except:
    pass
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)


def faiss_search_gpu(dimension, vectors, query_vectors, k, metrics="cosine"):
    ngpus = faiss.get_num_gpus()
    logger.info(f"Start faiss search, number of GPUs:{ngpus}")
    if metrics == "cosine":
        faiss.normalize_L2(vectors)
        faiss.normalize_L2(query_vectors)
        flat_index = faiss.IndexFlatIP(dimension)
    elif metrics == "l2":
        flat_index = faiss.IndexFlatL2(dimension)
    else:
        raise TypeError(f"not supportted type: {metrics}")
    if ngpus > 0:
        flat_index = faiss.index_cpu_to_all_gpus(flat_index)
    logger.info(f"Fininsh init index")
    flat_index.add(vectors)  # add vectors to the index
    logger.info(f"Fininsh add vectors")
    logger.info(flat_index.ntotal)

    distances, indices = flat_index.search(query_vectors, k)  # actual search
    del flat_index
    return distances, indices


def get_llama_embedding(
    texts,
    model_name_or_path,
    batch_size=32,
    max_length=128,
    emb_dim=4096,
    emb_type="AVG",
):
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    logger.info(tokenizer.all_special_ids, tokenizer.all_special_tokens)
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True,
    )
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to("cuda")
    all_hidden_states = np.zeros(shape=(len(texts), emb_dim), dtype=np.float32)
    # get hidden state from model
    with torch.no_grad():
        for i_batch, batch in tqdm(
            enumerate(torch.split(inputs.input_ids, batch_size))
        ):
            hidden_state = model.forward(
                input_ids=batch, output_hidden_states=True
            ).hidden_states[0]
            logger.info(hidden_state.shape)
            logger.info(hidden_state)
            if emb_type == "AVG":
                hidden_state = torch.mean(hidden_state, dim=1)
            elif emb_type == "MAX":
                hidden_state = torch.max(hidden_state, dim=1)[0]
            else:
                raise TypeError(f"not supportted type: {emb_type}")
            logger.info(hidden_state.shape)
            all_hidden_states[i_batch * batch_size : (i_batch + 1) * batch_size] = (
                hidden_state.detach().cpu().numpy().astype(np.float32)
            )
    return all_hidden_states





def get_openai_embedding(data_path):
    data = load_jsonl(data_path)
    logger.info("load suc")
    data = [(item["embedding"], item["id"]) for item in data]
    none_empty_embeddings = []
    none_empty_ids = []
    logger.info(len(data))
    for embedding, idx in data:
        if len(embedding) == 1536:
            none_empty_embeddings.append(embedding)
            none_empty_ids.append(idx)
    logger.info(len(none_empty_embeddings), len(none_empty_ids))
    none_empty_embeddings = np.array(
        [np.array(embedding, dtype=np.float32) for embedding in none_empty_embeddings]
    )
    del data
    gc.collect()
    return none_empty_embeddings, none_empty_ids



def text2vec_zh(texts, model_name_or_path="", batch_size=128, max_length=256, gpu=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    model = SentenceModel(model_name_or_path, device=f'cuda:{gpu}')
    model.max_seq_length = max_length
    sentence_embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )
    logger.info(f"GPU {gpu} 处理了 {len(texts)} 条数据")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return sentence_embeddings

def async_text2vec_zh(texts, model_name_or_path, batch_size, max_length, gpus):
    # 将数据拆分为与 GPU 数量相等的块
    num_gpus = len(gpus)
    chunk_size =  math.ceil(len(texts) / num_gpus)
    data_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    assert len(data_chunks) == num_gpus
    results = []
    with concurrent.futures.ProcessPoolExecutor(num_gpus) as executor:
        # 使用 map 函数将每个 GPU ID 与相应的数据块一起传递给 data_processing_function
        results = list(executor.map(text2vec_zh, data_chunks,  itertools.repeat(model_name_or_path), itertools.repeat(batch_size), itertools.repeat(max_length), gpus))

    # 将处理后的向量数据汇总
    embeddings = []
    for result in results:
        embeddings.extend(result)
    torch.cuda.empty_cache()
    return np.asarray(embeddings)


def async_text2vec(texts, model_name_or_path, batch_size, max_length, gpus):
    # 将数据拆分为与 GPU 数量相等的块
    model = SentenceTransformer(model_name_or_path)
    model.max_seq_length = max_length
    pool = model.start_multi_process_pool({f'cuda:{gpu}' for gpu in gpus})
    num_gpus = len(gpus)
    chunk_size =  math.ceil(len(texts) / num_gpus)
    #Compute the embeddings using the multi-process pool
    embeddings = model.encode_multi_process(texts, pool, batch_size, chunk_size)
    logger.info(f"Embeddings computed. Shape:{embeddings.shape}")

    #Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings

    
def drop_dup(
    data_paths,
    output_path,
    data_columns,
    model_name_or_path="/data/zhangchong/llm_models/text2vec-large-chinese",
    gpus=[0, 1, 2, 3],
    topk=300,
    threshold=0.9,
    batch_size=350,
    max_length=256,
    verbose=False,
):
    pretty_print = lambda x: '\n'.join(x) if isinstance(x, list) else x
    logger.info(f"load data from:\n{pretty_print(data_paths)}")
    def read_add_data_set_name(data_path):
        assert os.path.isfile(data_path), f"{data_path} is not a file"
        data_frame = df_reader(data_path).fillna('')
        if "dataset_name" not in data_frame.columns:
            data_frame["dataset_name"] = os.path.basename(data_path)
        return data_frame
    data = pd.concat([read_add_data_set_name(data_path) for data_path in data_paths])
    logger.info(f"load data suc, total shape: {data.shape}")
    data["rerange_id"] = range(len(data))
    # in case of origin index is not continuous
    texts = data[data_columns].astype(str).agg("".join, axis=1).values.tolist()
    if "score" in data.columns:
        id_vs_score = dict(zip(data["id"], data["score"]))
    ids = data["rerange_id"].values
    id_vs_text = dict(zip(ids, texts))
    logger.info(f"\n{pretty_print([str(idx) + ' ' + text for idx, text in random.sample(id_vs_text.items(), 10)])}")
    # text2vec_func = async_text2vec_zh if "chinese" in model_name_or_path else async_text2vec
    embeddings = async_text2vec(
        texts,
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
        max_length=max_length,
        gpus=gpus
    )
    logger.info(f"get embeddings suc, shape: {embeddings.shape}")
    topk_cosine_sim, closest_topk_index = faiss_search_gpu(
        embeddings.shape[-1], embeddings, embeddings, topk, metrics="cosine"
    )
    keep_info = {}
    drop_or_not = {}
    for i_sample, i_topk_index in enumerate(tqdm(closest_topk_index)):
        # 已经是排过序
        i_topk_cosine_sim = topk_cosine_sim[i_sample]
        i_topk_sim_ids = ids[i_topk_index]
        is_bigger_than_threshold = i_topk_cosine_sim > threshold
        i_topk_cosine_sim = i_topk_cosine_sim[is_bigger_than_threshold]
        i_topk_sim_ids = i_topk_sim_ids[is_bigger_than_threshold]
        closest_num = len(i_topk_cosine_sim)
        if closest_num < 2:
            continue
        # keep the largest scores
        if "score" in data.columns:
            topk_scores = np.array(
                list(map(lambda x: id_vs_score.get(x, 0), i_topk_sim_ids))
            )
            i_topk_sim_ids = i_topk_sim_ids[topk_scores.argsort()[::-1]]
            i_topk_cosine_sim = i_topk_cosine_sim[topk_scores.argsort()[::-1]]
        i_topk_sim_texts = [id_vs_text.get(sim_id, "") for sim_id in i_topk_sim_ids]
        if verbose:
            logger.info(f"ID: {ids[i_sample]}, TEXT: {texts[i_sample]}")
            logger.info(i_topk_sim_texts)
            logger.info(i_topk_cosine_sim)
        # keep the first one
        for sim_id in i_topk_sim_ids:
            if keep_info.get(sim_id, None) is None:
                keep_info[sim_id] = i_topk_sim_ids[0]
                drop_or_not[sim_id] = sim_id != i_topk_sim_ids[0]

    logger.info(f"origin data len: {len(data)}")
    logger.info(f"drop len:{len([k for k, v in drop_or_not.items() if v])}")
    data["center_id"] = data["rerange_id"].map(lambda x: keep_info.get(x, x))
    data['dropped'] = data['rerange_id'].map(lambda x: drop_or_not.get(x, False))
    clusters_num_count = Counter(list(keep_info.values()))
    data['cluster_counts'] = data['center_id'].map(lambda x: clusters_num_count.get(x, 1))
    if output_path:
        df_saver(data, output_path)
        logger.info(f"save suc to {output_path}")



if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    fire.Fire(drop_dup)

