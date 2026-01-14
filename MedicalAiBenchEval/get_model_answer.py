import argparse
import os

import pandas as pd
from openai import OpenAI


def strip_think(text: str) -> str:
    s = "" if text is None else str(text)
    if "<think/>" in s:
        s = s.split("<think/>")[-1]
    elif "</think>" in s:
        s = s.split("</think>")[-1]
    else:
        s = s
    return s.strip()


def normalize_base_url(value: str) -> str:
    s = (value or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            s = s.rstrip("/")
    if not s.endswith("/v1"):
        s = s + "/v1"
    return s


def preview(text: str, limit: int = 200) -> str:
    s = "" if text is None else str(text).replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/input/GAPS-NSCLC-preview.xlsx")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_url = normalize_base_url(os.getenv("OPENAI_BASE_URL") ) 
    model = (os.getenv("SGLANG_MODEL") or "o3").strip()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    temperature = float(os.getenv("SGLANG_TEMPERATURE") or "1.0")
    max_tokens_env = os.getenv("SGLANG_MAX_TOKENS")
    max_tokens = int(max_tokens_env) if max_tokens_env else None

    if base_url == "/v1":
        raise SystemExit("请设置环境变量 SGLANG_BASE_URL，例如：http://127.0.0.1:30000/v1")
    if not model:
        raise SystemExit("请设置环境变量 SGLANG_MODEL，例如：your-model-name")

    input_path = args.input
    output_path = args.output or input_path

    df = pd.read_excel(input_path)
    if "question" not in df.columns:
        raise SystemExit("Excel 缺少 question 列")
    if "o3" not in df.columns:
        df["o3"] = ""
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"Rows: {len(df)}")

    for i in range(len(df)):
        q = df.at[i, "question"]
        question = "" if pd.isna(q) else str(q).strip()
        if not question:
            continue
        existing = df.at[i, "o3"]
        existing_text = "" if pd.isna(existing) else str(existing)
        if not args.overwrite and existing_text.strip():
            print(f"[{i+1}/{len(df)}] skip (o3 非空)")
            continue

        print(f"[{i+1}/{len(df)}] question: {preview(question, 160)}")
        raw = ""
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=temperature,
                #max_tokens=max_tokens,
            )
            raw = r.choices[0].message.content or ""
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            if status_code == 404:
                r = client.completions.create(
                    model=model,
                    prompt=question,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw = r.choices[0].text or ""
            else:
                print(f"[{i+1}/{len(df)}] error: {e}")
                raise
        cleaned = strip_think(raw)
        print(f"[{i+1}/{len(df)}] raw: {preview(raw)}")
        print(f"[{i+1}/{len(df)}] cleaned: {preview(cleaned)}")
        df.at[i, "o3"] = cleaned

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    print(f"已写入 o3 -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

