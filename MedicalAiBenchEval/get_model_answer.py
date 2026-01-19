import os

import pandas as pd
from openai import OpenAI


BASE_URL = os.getenv("ARK_BASE_URL")
API_KEY = os.getenv("ARK_API_KEY") 
MODEL = "Baichuan-M2-32B"

INPUT_XLSX = "data/input/GAPS-NSCLC-preview.xlsx"
OUTPUT_XLSX = "data/input/GAPS-NSCLC-Baichuan-M2-32B.xlsx"
ANSWER_COLUMN = "Baichuan-M2-32B-answer"

TEMPERATURE = 1
MAX_TOKENS = 10240
OVERWRITE = False


def strip_think(text: str) -> str:
    s = "" if text is None else str(text)
    if "<think/>" in s:
        s = s.split("<think/>")[-1]
    elif "</think>" in s:
        s = s.split("</think>")[-1]
    return s.strip()


def preview(text: str, limit: int = 200) -> str:
    s = "" if text is None else str(text).replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


def main() -> int:
    if not BASE_URL:
        raise SystemExit("请在脚本里填好 BASE_URL")
    if not API_KEY:
        raise SystemExit("请设置环境变量 ARK_API_KEY（或 OPENAI_API_KEY）")
    if not MODEL:
        raise SystemExit("请在脚本里填好 MODEL（通常是 ep-xxxx 推理接入点）")

    df = pd.read_excel(INPUT_XLSX)
    if "question" not in df.columns:
        raise SystemExit("Excel 缺少 question 列")
    if ANSWER_COLUMN not in df.columns:
        df[ANSWER_COLUMN] = ""

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"Base URL: {BASE_URL}")
    print(f"Model: {MODEL}")
    print(f"Rows: {len(df)}")

    for i in range(len(df)):
        q = df.at[i, "question"]
        question = "" if pd.isna(q) else str(q).strip()
        if not question:
            continue
        existing = df.at[i, ANSWER_COLUMN]
        existing_text = "" if pd.isna(existing) else str(existing)
        if (not OVERWRITE) and existing_text.strip():
            print(f"[{i+1}/{len(df)}] skip ({ANSWER_COLUMN} 非空)")
            continue

        print(f"[{i+1}/{len(df)}] question: {preview(question, 160)}")
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": question}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = r.choices[0].message.content or ""
        cleaned = strip_think(raw)
        print(f"[{i+1}/{len(df)}] raw: {preview(raw)}")
        print(f"[{i+1}/{len(df)}] cleaned: {preview(cleaned)}")
        df.at[i, ANSWER_COLUMN] = cleaned

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    print(f"已写入 {ANSWER_COLUMN} -> {OUTPUT_XLSX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

