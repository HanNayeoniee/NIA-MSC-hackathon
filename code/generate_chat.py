import os
import json
import glob
import pandas as pd
from tqdm import tqdm
import argparse
from submit import *

def json_load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def preprocessing_dialog(dialog):
    new_dialog = list()
    prev_speaker = None
    prev_utterance = ""
    prev_summary = ""
    for i, _dialog in enumerate(dialog):
        if not prev_speaker:
            prev_speaker = _dialog["speaker"]
            prev_utterance = _dialog["utterance"]
            prev_summary = _dialog["summary"]
            continue

        if prev_speaker == _dialog["speaker"]:
            prev_utterance += " " + _dialog["utterance"]
            prev_summary += " " + _dialog["summary"]

        elif prev_speaker != _dialog["speaker"]:
            new_dialog.append({
                "speaker": prev_speaker,
                "utterance": prev_utterance.strip(),
                "summary": prev_summary.strip()
            })
            prev_speaker = _dialog["speaker"]
            prev_utterance = _dialog["utterance"]
            prev_summary = _dialog["summary"]
            
    new_dialog.append({
        "speaker": prev_speaker,
        "utterance": prev_utterance.strip(),
        "summary": prev_summary.strip()
    })
    
    return new_dialog


def preprocessing_input(utterance):
    utt1 = {"speaker": "speaker1",
            "utterance": "<empty>",
            "summary": ""}
    utt2 = {"speaker": "speaker2",
            "utterance": "<empty>",
            "summary": ""}
    
    if len(utterance) == 1:
        total_utt = [utt1, utt2, utt1, utt2, utt1, utt2]
    elif len(utterance) == 2:
        total_utt = [utt1, utt2, utt1, utt2] + utterance
    elif len(utterance) == 4:
        total_utt = [utt1, utt2] + utterance
    else:
        total_utt = utterance
        
        
    # 인퍼런스 입력 포매팅
    data = total_utt
    user_memory = ""
    agent_memory = ""
    
    for i in range(len(data)):
        if data[i]["summary"]:
            if data[i]["speaker"] == "speaker1":
                agent_memory += data[i]["summary"].strip()
            else:
                user_memory += data[i]["summary"].strip()
        
        if user_memory:
            in_str = f"<user_memory>{user_memory}</user_memory>"
        else:
            in_str = "<user_memory><empty></user_memory>"
            
        if agent_memory:
            in_str += f"<agent_memory>{agent_memory}</agent_memory><dialogue>입력:"
        else:
            in_str += "<agent_memory><empty></agent_memory><dialogue>입력:"
        

        in_str += "<user>"
        in_str += data[5]["utterance"]
        in_str += "<agent>"
        in_str += data[4]["utterance"]
        in_str += "<user>"
        in_str += data[3]["utterance"]
        in_str += "<agent>"
        in_str += data[2]["utterance"]
        in_str += "<user>"
        in_str += data[1]["utterance"]
        in_str += "<agent>"
        in_str += data[0]["utterance"]
        in_str += "</dialogue>"
        
    return total_utt, in_str


def main(args):
    out_folder = args.output_folder
    createFolder(out_folder)

    target = args.input_folder + "/*.json"
    # print(target)
    file_list = glob.glob(target)


    for file_path in file_list:
        file_name = file_path.split("/")[-1][:-5]
        data = json_load(file_path)
        
        sessionInfo = data["sessionInfo"]
        for session in sessionInfo:
            
            # 엑셀 만들기
            sent_type = ["speaker1_summary", "speaker2_summary"]
            sent = []
            speaker1_summary = "\n".join(session["prevAggregatedpersonaSummary"]["speaker1"])
            speaker2_summary = "\n".join(session["prevAggregatedpersonaSummary"]["speaker2"])
            sent.append(speaker1_summary)
            sent.append(speaker2_summary)
            
            dialog = session["dialog"]
            for dia in dialog:
                sent_type.append(dia["speaker"])
                sent.append(dia["utterance"])
            
            # 인퍼런스 결과
            new_dialog = preprocessing_dialog(dialog)
            utterance, in_str = preprocessing_input(new_dialog[-6:])

            MODEL_NAME = "../model/1216_shuffle"
            pred = predict_utterace(MODEL_NAME, in_str)

            sent_type.append("speaker1_generated")
            sent.append(pred)
            

        # 데이터프레임으로 저장
        res_df = pd.DataFrame()    
        res_df["sent_type"] = sent_type
        res_df["sent"] = sent
        res_df["이전 세션 정보 사용"] = None
        res_df["적절한 발화 여부"] = None
        save_path = os.path.join(out_folder, file_name + ".xlsx")
        res_df.to_excel(save_path, index=False)
        # print("saved:", save_path)
    print("="*20 + "SAVED ALL PREDICTIONS" + "="*20)


if __name__ == "__main__":
    # export GIT_PYTHON_REFRESH=quiet
    parser = argparse.ArgumentParser(description='Make a Submission')
    parser.add_argument("--input_folder", type=str, default="./mydata/ver2", help="입력 데이터 경로")
    parser.add_argument("--output_folder", type=str, default="klue/roberta-base", help="출력 데이터 저장 경로")
    args = parser.parse_args()

    main(args)
