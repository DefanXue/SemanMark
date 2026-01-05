"""
统一的数据集加载器
"""
import json

class DatasetLoader:
    def __init__(self, config):
        self.dataset_name = config['dataset']['name']
        self.config = config
        self._resolve_data_path()

    def _resolve_data_path(self):
        """根据数据集名称解析数据路径"""
        paths = self.config['dataset']['paths']
        dataset_path = paths[self.dataset_name]

        # 如果是字符串（简单路径），直接使用
        if isinstance(dataset_path, str):
            self.data_path = dataset_path
        # 如果是字典（train/test分割），默认使用test
        elif isinstance(dataset_path, dict):
            self.data_path = dataset_path.get('test', list(dataset_path.values())[0])
        else:
            raise ValueError(f"Invalid dataset path format: {dataset_path}")

    def load_codes(self, num_codes, start_index=0, split='test'):
        """
        加载代码

        Args:
            num_codes: 要加载的代码数量
            start_index: 起始索引
            split: 数据集分割 ('train' 或 'test')，仅当配置中有该分割时生效
        """
        # 解析数据路径
        paths = self.config['dataset']['paths']
        dataset_path = paths[self.dataset_name]

        # 如果是字典（train/test分割），使用指定的分割
        if isinstance(dataset_path, dict):
            data_path = dataset_path.get(split)
            if data_path is None:
                raise ValueError(f"Split '{split}' not available for dataset {self.dataset_name}")
        else:
            # 如果是字符串（简单路径），忽略split参数，直接使用路径
            data_path = dataset_path

        codes = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start_index:
                    continue
                if len(codes) >= num_codes:
                    break
                data = json.loads(line)
                codes.append(data['code'])
        return codes
