
import re
import json

import pandas as pd

from datasets import Dataset
from torch.utils.data import DataLoader


def Table2TextDataLoader(fname, tokenizer, batch_size, mode="train"):
    """
    Build Data Loader

    """
    def preprocess_function(examples):
        tokenizer_input = tokenizer([tokenizer.bos_token+s+tokenizer.eos_token for s in examples["table"]],
                                    padding="max_length", max_length=512, truncation=True, return_tensors="pt", return_token_type_ids=False)
        encoder_input_ids = tokenizer_input["input_ids"]
        encoder_attention_mask = tokenizer_input["attention_mask"]

        if mode=="train":
            tokenizer_output = tokenizer([tokenizer.bos_token+s+tokenizer.eos_token for s in examples["text"]],
                                        padding="max_length", max_length=512, truncation=True, return_tensors="pt", return_token_type_ids=False)
            decoder_input_ids = tokenizer_output["input_ids"]
            decoder_attention_mask = tokenizer_output["attention_mask"]

            return {
                "input_ids": encoder_input_ids,
                "attention_mask": encoder_attention_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
            }

        return {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
        }

    dataset = load_dataset(fname, mode)
    dataset = dataset.map(
        preprocess_function, batched=True, num_proc=8, remove_columns=dataset.column_names
    ).with_format("torch")

    dataloader = DataLoader(dataset, shuffle=(True if mode=="train" else False), batch_size=batch_size)

    return dataloader


def jsonlload(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        j_list = [json.loads(line) for line in lines]

    return j_list


def jsonldump(j_list, fname):
    with open(fname, "w", encoding='utf-8') as f:
        for json_data in j_list:
            f.write(json.dumps(json_data, ensure_ascii=False)+'\n')

def make_table(table_data):
    table = {}
    col_len = 0
    row_len = 0

    for data in table_data['input']['table']:
        col = data['col']
        row = data['row']
        colspan = data['colspan']
        rowspan = data['rowspan']
        value = data['value']
        if data['is_header']:
            value = "# " + value

        col_len = col+colspan if col+colspan > col_len else col_len
        row_len = row+rowspan if row+rowspan > row_len else row_len


        if row not in table:
            table[row] = {}


        for row_span in range(0, rowspan):
            if row + row_span not in table:
                table[row+row_span] = {}

            for col_span in range(0, colspan):
                table[row+row_span][col+col_span] = "[SPAN]"

        table[row][col] = value


    text = []
    for row in range(row_len):
        prev_span = 0
        text_row = table[row][0]
        for col in range(1, col_len):
            if table[row][col] == "[SPAN]":
                text_row += " [SPAN] "
                prev_span = 1
            else:
                if prev_span:
                    text_row += table[row][col]
                    prev_span = 0
                else:
                    text_row += " [TAB] " + table[row][col]
        text.append(text_row)

    text = " [NL] ".join(text)
    text = re.sub('[\s]+', ' ', text)

    return text
            
            
def jsonl2df(j_list, mode):
    data_dict = {"table": []}
    if mode == "train":
        data_dict["text"] = []
    
    for j in j_list:
        try:
            table = make_table(j)
        except:
            continue
        if mode == "train":
            for text in j['output']:
                data_dict["table"].append(table)
                data_dict["text"].append(text)
        else:
            data_dict["table"].append(table)

    df = pd.DataFrame(data_dict)
    return df



def load_dataset(fname, mode):
    j_list = jsonlload(fname)
    df = jsonl2df(j_list, mode)
    dataset = Dataset.from_pandas(df)

    return dataset
