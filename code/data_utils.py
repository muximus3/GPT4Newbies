# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import logging
sys.path.append(os.path.normpath(f'{os.path.dirname(os.path.abspath(__file__))}/..'))
logger = logging.getLogger(__name__)

def df_reader(data_path, header: int | None = 0, usecols: list[str | int] | None = None ,sep='\t', sheet_name=0) -> pd.DataFrame:
    extention = data_path.split('.')[-1]
    match extention:
        case 'jsonl':
            df_data = pd.read_json(data_path, lines=True)
        case 'json':
            df_data = pd.read_json(data_path)
        case 'xlsx':
            df_data = pd.read_excel(data_path,header=header, usecols=usecols, sheet_name=sheet_name)
        case 'csv' | 'tsv':
            df_data = pd.read_csv(data_path, header=header, usecols=usecols, sep=sep)
        case 'pkl':
            df_data = pd.read_pickle(data_path)
        case 'parquet':
            df_data = pd.read_parquet(data_path)
        case _:
            raise AssertionError(f'not supported file type:{data_path}, suport types: json, jsonl, xlsx, csv, parquet, pkl')
    return df_data

    
def df_saver(df: pd.DataFrame, data_path):
    if data_path.endswith('json'):
        df.to_json(df, orient='records', force_ascii=False)
    if data_path.endswith('jsonl'):
        df.to_json(df, orient='records', force_ascii=False, lines=True)
    elif data_path.endswith('xlsx'):
        df2xlsx(df, data_path, index=False)
    elif data_path.endswith('csv'):
        df.to_csv(data_path)
    else:
        raise AssertionError(f'not supported file type:{data_path}, suport types: json, jsonl, xlsx, csv')

def df2xlsx(df: pd.DataFrame, save_path: str, sheet_name='Sheet1', mode='w', index=False):
    if mode not in ['w', 'a']:
        raise AssertionError('mode not in [\'w\', \'a\']')
    if mode == 'a' and not os.path.isfile(save_path):
        mode = 'w'
    if mode == 'a':
        engine = 'openpyxl'
    else:
        engine = 'xlsxwriter'

    with pd.ExcelWriter(save_path, engine=engine, mode=mode) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=index)

        
def get_left_data(total_msgs : list[dict], processed_msgs: list[dict]):
    processed_msgs_idx = [int(msg['id']) for msg in processed_msgs]
    left_msgs =  [msg for msg in total_msgs if msg['id'] not in processed_msgs_idx]
    print(f'total msg: {len(total_msgs)}')
    print(f'processed msg: {len(processed_msgs_idx)}')
    print(f'left msg: {len(left_msgs)}')
    return left_msgs