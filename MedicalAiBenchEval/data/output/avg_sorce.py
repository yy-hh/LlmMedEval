import pandas as pd
import json
from pathlib import Path
 
file_path = "./quick_check_20260113_074316_u3sqij.xlsx "


def _read_table(input_path: str) -> pd.DataFrame:
    p = str(input_path or "").strip().strip('"').strip("'")
    suffix = Path(p).suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(p)

    encodings_to_try = ["utf-8", "utf-8-sig", "gb18030", "cp936", "latin1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return pd.read_csv(p)


def get_NS(input_file):
    df = _read_table(input_file)
    
    def extract_NS(json_str):
        if pd.isna(json_str):
            return 0.0
        result = json.loads(json_str)
        return float(result.get("normalized", 0)),result.get()

    if 'ant_answer_judged_json_scores' in df.columns:
        df['NS'] = df['ant_answer_judged_json_scores'].apply(extract_NS)
        df['']
        avg_NS = df['NS'].mean()
        print(f"Average NS: {avg_NS:.4f}")
        count_len=len(df)
        return avg_NS,count_len
    else:
        print("未找到ant_answer_judged_json_scores列")

avg_sorce,count_len=get_NS(file_path)

print(f"平均得分：{avg_sorce}")
print(f"样本数量：{count_len}")
        




    
