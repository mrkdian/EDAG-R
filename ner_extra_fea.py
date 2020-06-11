import os
import json
from pyltp import Segmentor, Postagger
from tqdm import tqdm
import jieba

LTP_DATA_DIR = './ltp_model'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')

segmentor = Segmentor()
segmentor.load(cws_model_path)
postagger = Postagger()
postagger.load(pos_model_path)
files = [
    'Data/sample_train.json',
    'Data/dev.json',
    'Data/test.json'
]

for f in files:
    data = json.load(open(f, mode='r', encoding='utf-8'))
    pbar = tqdm(total=len(data))
    newf = list(f)
    newf.insert(newf.index('/') + 1, 'extra_')
    newf = ''.join(newf)
    for recguid, _data in data:
        _data['cut_word_label'] = []
        for sent in _data['sentences']:
            words = list(segmentor.segment(sent))
            cut_word_label = []
            for word in words:
                cut_word_label.append(1)
                for _ in word[1:]:
                    cut_word_label.append(0)
            _data['cut_word_label'].append(cut_word_label)
        pbar.update(1)
    json.dump(data, open(newf, mode='w', encoding='utf-8'))
print(1)