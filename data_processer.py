import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from datasets import Dataset, DatasetDict, load_metric

class TS_Processor:
    """
    TS_Processor 是一个用于处理时间序列分类数据的类，将原始的csv文件转化成HuggingFace的DatasetDict格式来喂给模型。

    核心思路：
    1. 读取指定路径下的所有CSV文件，将其作为时间序列数据。
    2. 可选：根据IQR方法剔除异常值。
    3. 将每个CSV文件中的数据按照设定比例划分为训练、验证和测试集。
    4. 对数值型数据进行标准化（均值为0，方差为1）。
    5. 使用滑动窗口方法将时间序列分割成固定长度的片段，并为每个片段分配标签。
    6. 返回Huggingface DatasetDict 格式的数据集，包含 train、validation、test 三个部分。

    参数说明：
    sequence_length (int): 每个时间序列样本的长度。
    stride (int): 滑动窗口的步长。
    train_ratio (float): 训练集比例。
    val_ratio (float): 验证集比例。
    remove_extreme_values (bool): 是否去除异常值。
    iqr_threshold (float): IQR阈值，控制异常值检测的灵敏度。

    这个类我可能还会后续扩展，比如添加施加白噪声的功能来生成含有噪声的数据喂给模型，进而提升鲁棒性；或添加可以处理.ts文件的数据功能。
    """
    def __init__(self, sequence_length=512, stride=32, train_ratio=0.7, val_ratio=0.1, remove_extreme_values=False, iqr_threshold=7.0):
        self.sequence_length = sequence_length
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        self.enable_remove_extreme_values = remove_extreme_values
        self.iqr_threshold = iqr_threshold
        self.numeric_cols = None
        self.scaler = StandardScaler()
        self.removed_outliers = []
    
    def __call__(self, data_path):
        self.data_path = data_path
        return self.process_dataset()

    def __repr__(self):
        return f"TS_Processor(sequence_length={self.sequence_length}, stride={self.stride}, remove_extreme_values={self.enable_remove_extreme_values}, iqr_threshold={self.iqr_threshold})"
    
    def remove_extreme_values(self, df: pd.DataFrame, file_name: str) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            raise ValueError("There are no numerical values in Dataframe")
        
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.iqr_threshold * IQR
        upper_bound = Q3 + self.iqr_threshold * IQR
        
        outliers = df[(df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)].dropna(how='all')
        
        if not outliers.empty:
            for index in outliers.index:
                self.removed_outliers.append((file_name, index, df.loc[index].to_dict()))
        
        mask = ~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)
        return df[mask].reset_index(drop=True)
    
    def load_and_split_data(self):
        train_list, val_list, test_list = [], [], []
        all_files = os.listdir(self.data_path)
        
        for index, file in enumerate(all_files):
            file_path = os.path.join(self.data_path, file)
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.replace(" ", "")
            numeric_col = df.columns.difference(["time"])
            
            if self.enable_remove_extreme_values:
                df = self.remove_extreme_values(df, file)
            
            train_size = int(self.train_ratio * len(df))
            val_size = int((self.train_ratio + self.val_ratio) * len(df))
            
            train_list.append(df.iloc[:train_size])
            val_list.append(df.iloc[train_size:val_size])
            test_list.append(df.iloc[val_size:])
        
        self.numeric_cols = numeric_col
        return train_list, val_list, test_list

    def normalize_data(self, train_list, val_list, test_list):
        train_data_cat = pd.concat(train_list, ignore_index=True)
        self.scaler.fit(train_data_cat[self.numeric_cols])
        
        for lst in [train_list, val_list, test_list]:
            for df in lst:
                df.loc[:, self.numeric_cols] = self.scaler.transform(df.loc[:, self.numeric_cols])
        
        return train_list, val_list, test_list

    def sliding_window_split(self, df, label):
        sequences, labels = [], []
        data_array = df[self.numeric_cols].values
        n_samples = len(data_array)
        
        for idx in range(0, n_samples - self.sequence_length + 1, self.stride):
            sequences.append(data_array[idx : idx + self.sequence_length])
            labels.append(label)
        
        return sequences, labels

    def process_dataset(self):
        train_list, val_list, test_list = self.load_and_split_data()
        train_list, val_list, test_list = self.normalize_data(train_list, val_list, test_list)
        
        train_sequences_all, train_labels_all = [], []
        val_sequences_all, val_labels_all = [], []
        test_sequences_all, test_labels_all = [], []
        
        for i, (df_train, df_val, df_test) in enumerate(zip(train_list, val_list, test_list)):
            train_seqs, train_labs = self.sliding_window_split(df_train, i)
            train_sequences_all.extend(train_seqs)
            train_labels_all.extend(train_labs)
            
            val_seqs, val_labs = self.sliding_window_split(df_val, i)
            val_sequences_all.extend(val_seqs)
            val_labels_all.extend(val_labs)
            
            test_seqs, test_labs = self.sliding_window_split(df_test, i)
            test_sequences_all.extend(test_seqs)
            test_labels_all.extend(test_labs)
        
        if self.enable_remove_extreme_values and self.removed_outliers:
            print("Detected and removed outliers:")
            for file_name, index, data in self.removed_outliers:
                print(f"File: {file_name}, Index: {index}, Data: {data}")
            
        return DatasetDict({
            "train": Dataset.from_dict({"sequence": train_sequences_all, "label": train_labels_all}),
            "validation": Dataset.from_dict({"sequence": val_sequences_all, "label": val_labels_all}),
            "test": Dataset.from_dict({"sequence": test_sequences_all, "label": test_labels_all})
        })

def collate_fn(batch):
    past_values = torch.tensor([item["sequence"] for item in batch], dtype=torch.float32)
    target_values = torch.tensor([item["label"] for item in batch], dtype=torch.long) 

    return {"past_values": past_values, "target_values": target_values} 

metric = load_metric("accuracy")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    return metric.compute(predictions=predictions, references=labels)
