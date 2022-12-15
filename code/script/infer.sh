export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID 

###### 경로 수정
DATASET_DIR=../dataset/1207_dataset/
TEST_PATH=$DATASET_DIR/test.jsonl

# TRAIN_OUTPUT_PATH=/workspace/chitchat/et5/model/1012/ ## 출력 디렉토리 변경 시 수정 필요
# INFER_OUTPUT_PATH=$TRAIN_OUTPUT_PATH/infer.jsonl
TRAIN_OUTPUT_PATH=../model/test
INFER_OUTPUT_PATH=$TRAIN_OUTPUT_PATH/infer.jsonl


# mkdir $TRAIN_OUTPUT_PATH

pushd ../


# INPUT : 학습완료모델, 테스트데이터
# OUTPUT : 추론결과
python infer.py \
--trained_model $TRAIN_OUTPUT_PATH \
--test_fpath $TEST_PATH \
--output_fpath $INFER_OUTPUT_PATH \
\
--process_model seq2seq_jsonl \
--model_name_or_path '' \
--per_device_eval_batch_size 12 \
--predict_with_generate
# popd