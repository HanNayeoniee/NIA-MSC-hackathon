# LuxiaBot Decoder part
# 생성 기반 대화 모델

### Code Structure
```bash
et5
├── assets
├── data
├── code
│   ├── script
│   │   ├── train.sh
│   │   └── infer.sh
│   ├── seq2seq/
│   ├── arguments.py
│   ├── convert_result.py
│   ├── eval.py
│   ├── infer.py
│   ├── seq2seq_data_processor.py
│   ├── seq2seq_data_utils.py
│   ├── seq2seq_dataset_et5.py
│   └── train.py
├── dataset
│   ├── 1207_dataset/
│   └── 버전별 데이터셋
├── model
│   ├── raw
│   │   └── etri_t5/
│   └── 버전별 모델
├── README.md
└── requirements.txt
```