import os
import sys
import json
import time
import csv
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from langchain_upstage import ChatUpstage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a professional question-generation agent specializing in the Korean K-12 national curriculum.

Task:
- Convert learning objectives + achievement standards into ONE natural Korean casual question (반말).
- Must be open-ended, like a real user's curiosity.
- Avoid formal/academic/LLM-like structure.
- Use only casual Korean (“~이야?”, “~해줘”, “~알려줘”).

Output exactly ONE Korean casual-speech query.
"""),
    ("human", """
Learning Objective (Korean): {learning_objectives}
Achievement Standards (Korean): {achievements_standards}

Create one natural Korean user query in casual Korean (반말).
""")
])

VARIATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
Generate a NEW Korean casual-speech question (반말) related to the same learning objective and achievement standards.

Rules:
- MUST NOT be a paraphrase of the base query.
- Must ask about a DIFFERENT ANGLE or DIFFERENT ASPECT.
- Slight inaccuracy is acceptable if it increases diversity.
- Keep it open-ended and casual.
- Should reflect thought patterns common among Koreans educated in the national curriculum.

Return ONE Korean casual-speech question.
"""),
    ("human", """
Learning Objective: {learning_objectives}
Achievement Standards: {achievements_standards}
Base Query: {base_query}

Generate a NEW Korean casual-speech question asking/talking to AI assistant.
""")
])

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert evaluator for the Korean national curriculum.

Evaluate if the query:
1) Reflects the learning objective.
2) Incorporates key ideas from the achievement standards.
3) Sounds natural as Korean casual speech.
4) Matches typical Korean user's tone.

Respond ONLY in JSON:
{"score": 1-5, "suggestion": "brief improvement idea"}
"""),
    ("human", """
Learning Objective: {learning_objectives}
Achievement Standards: {achievements_standards}
Question: {query}

Respond in JSON:
{"score": 1-5, "suggestion": "brief improvement idea"}
""")
])

REVISE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
Revise the query in natural Korean casual-speech (반말).
Avoid teacher-like tone and LLM-like structure.
"""),
    ("human", """
Learning Objective: {learning_objectives}
Achievement Standards: {achievements_standards}
Original: {query}
Feedback: {feedback}

Revise the query.
""")
])

CULTURAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert in Korean culture and shared societal perspectives.

Evaluate whether the question:
- Aligns with Korean cultural and social context,
- Reflects typical Korean ways of thinking,
- Is not a generic universal question,
- Is meaningfully grounded in Korean context.

Respond ONLY in JSON:
{"is_relevant": true/false, "reason": "brief explanation"}
"""),
    ("human", """
Question: {query}

Respond in JSON:
{"is_relevant": true/false, "reason": "brief explanation"}
""")
])

NATURAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Polish the query to sound like natural Korean casual-speech (반말)."),
    ("human", "Question: {query}\n\nPolish this query.")
])

VARIANT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert Korean query rewriter.
Given a base Korean casual-speech query, rewrite it into three Korean variants with different country references.

1) ko_no_country:
  - No explicit/implicit country mention
  - Do NOT use "우리나라", "우리 사회", "대한민국", "한국", etc.

2) ko_implicit_country:
  - No explicit country names
  - Use implicit expressions like "우리나라", "우리 사회" when appropriate

3) ko_explicit_country:
  - Explicitly mention "대한민국" or "한국"

All:
- Natural Korean casual speech (반말)
- Close to original meaning
- Respond with strict JSON only.
"""),
    ("human", """
Base Korean casual query: {query}

Respond in JSON:
{ "ko_no_country": "...", "ko_implicit_country": "...", "ko_explicit_country": "..." }
""")
])

MULTILINGUAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
Translate the query into English, Simplified Chinese, and Japanese.
All translations must be natural conversational language.

Respond ONLY in JSON:
{ "en": "...", "zh": "...", "ja": "..." }
"""),
    ("human", """
Korean query: {query}

Respond in JSON:
{ "en": "...", "zh": "...", "ja": "..." }
""")
])


DETAIL_COLUMNS = [
    "code",
    "query_id",
    "learning_objectives",
    "achievements_standards",
    "final_query",
    "query_ko_no_country",
    "query_ko_implicit_country",
    "query_ko_explicit_country",
    "query_en",
    "query_zh",
    "query_ja",
    "eval_score",
    "cultural_ok",
    "cultural_reason",
    "iterations",
    "status",
    "error_reason",
]

SIMPLE_COLUMNS = [
    "code",
    "query_id",
    "learning_objectives",
    "achievements_standards",
    "final_query",
    "query_ko_no_country",
    "query_ko_implicit_country",
    "query_ko_explicit_country",
    "query_en",
    "query_zh",
    "query_ja",
]


def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def safe_json_parse(text: str, default=None):
    if not isinstance(text, str):
        return default
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        text = text[s:e + 1]
    try:
        return json.loads(text)
    except Exception:
        return default


def invoke_with_retries(chain, payload: Dict[str, Any], retries: int, backoff: float) -> str:
    last_err = None
    for i in range(retries + 1):
        try:
            return chain.invoke(payload).content.strip()
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(backoff * (i + 1))
            else:
                raise last_err


def open_csv_append(path: str, columns: List[str], write_header_if_new: bool = True):
    ensure_parent_dir(path)
    is_new = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    f = open(path, "a", encoding="utf-8-sig", newline="")
    w = csv.DictWriter(f, fieldnames=columns)
    if write_header_if_new and is_new:
        w.writeheader()
        f.flush()
    return f, w


def load_resume_index(detail_csv: str) -> Tuple[set, Dict[str, str], Dict[str, str]]:
    """
    Returns:
      processed_keys: set of "code_queryid"
      base_q1_by_code: code -> q1 final_query (only if status==success and query_id==1)
      cultural_reason_by_code: code -> cultural_reason (for q1 success)
    """
    if not os.path.exists(detail_csv) or os.path.getsize(detail_csv) == 0:
        return set(), {}, {}

    df = pd.read_csv(detail_csv, encoding="utf-8-sig")
    processed = set(df["code"].astype(str) + "_" + df["query_id"].astype(str))

    q1 = df[(df["query_id"] == 1) & (df["status"] == "success")]
    base_q = {str(r["code"]): str(r["final_query"]) for _, r in q1.iterrows()}
    base_reason = {str(r["code"]): str(r.get("cultural_reason", "") or "") for _, r in q1.iterrows()}

    return processed, base_q, base_reason


def generate_refine(
    lo: str,
    ach: str,
    gen_chain,
    eval_chain,
    revise_chain,
    natural_chain,
    max_revise: int,
    enable_eval: bool,
    retries: int,
    backoff: float,
) -> Tuple[str, Optional[int], int]:
    q = invoke_with_retries(gen_chain, {
        "learning_objectives": lo,
        "achievements_standards": ach
    }, retries, backoff)

    score = None
    suggestion = ""
    iters = 0

    if enable_eval:
        raw = invoke_with_retries(eval_chain, {
            "learning_objectives": lo,
            "achievements_standards": ach,
            "query": q
        }, retries, backoff)
        obj = safe_json_parse(raw, {}) or {}
        score = int(obj.get("score", 1))
        suggestion = str(obj.get("suggestion", "") or "")

        while score < 5 and iters < max_revise:
            q = invoke_with_retries(revise_chain, {
                "learning_objectives": lo,
                "achievements_standards": ach,
                "query": q,
                "feedback": suggestion
            }, retries, backoff)

            raw = invoke_with_retries(eval_chain, {
                "learning_objectives": lo,
                "achievements_standards": ach,
                "query": q
            }, retries, backoff)
            obj = safe_json_parse(raw, {}) or {}
            score = int(obj.get("score", 1))
            suggestion = str(obj.get("suggestion", "") or "")
            iters += 1

    q = invoke_with_retries(natural_chain, {"query": q}, retries, backoff)
    return q, score, iters


def cultural_check(
    q: str,
    cultural_chain,
    enable_cultural: bool,
    retries: int,
    backoff: float,
) -> Tuple[bool, str]:
    if not enable_cultural:
        return True, "cultural_check_disabled"

    raw = invoke_with_retries(cultural_chain, {"query": q}, retries, backoff)
    obj = safe_json_parse(raw, {}) or {}
    ok = bool(obj.get("is_relevant", False))
    reason = str(obj.get("reason", "") or "")
    return ok, reason


def make_variants(
    q: str,
    variant_chain,
    retries: int,
    backoff: float,
) -> Dict[str, str]:
    raw = invoke_with_retries(variant_chain, {"query": q}, retries, backoff)
    v = safe_json_parse(raw, {}) or {}
    return {
        "ko_no_country": v.get("ko_no_country", "") or "",
        "ko_implicit_country": v.get("ko_implicit_country", "") or "",
        "ko_explicit_country": v.get("ko_explicit_country", "") or "",
    }


def make_augmentations(
    lo: str,
    ach: str,
    base_query: str,
    variation_chain,
    natural_chain,
    k: int,
    retries: int,
    backoff: float,
) -> List[str]:
    out = []
    for _ in range(k):
        q = invoke_with_retries(variation_chain, {
            "learning_objectives": lo,
            "achievements_standards": ach,
            "base_query": base_query
        }, retries, backoff)
        q = invoke_with_retries(natural_chain, {"query": q}, retries, backoff)
        out.append(q)
    return out


def translate_unique_explicit_queries(
    detail_csv: str,
    simple_csv: str,
    hq_csv: str,
    multilingual_chain,
    retries: int,
    backoff: float,
    sleep_per_call: float,
):
    """
    FINAL step:
    - Load detail_csv
    - For all success rows, use ko_explicit_country (fallback final_query) as translation source
    - Translate UNIQUE sources once (cache)
    - Overwrite query_en/zh/ja for ALL success rows
    - Regenerate simple_csv and hq_csv from updated detail
    """
    df = pd.read_csv(detail_csv, encoding="utf-8-sig")

    success_mask = (df["status"] == "success")
    src_series = df.loc[success_mask, "query_ko_explicit_country"].fillna("").astype(str).str.strip()
    fallback_series = df.loc[success_mask, "final_query"].fillna("").astype(str).str.strip()
    src = src_series.where(src_series != "", fallback_series)

    unique_src = sorted(set([s for s in src.tolist() if s]))
    cache: Dict[str, Dict[str, str]] = {}

    def translate_one(kq: str) -> Dict[str, str]:
        raw = invoke_with_retries(multilingual_chain, {"query": kq}, retries, backoff)
        obj = safe_json_parse(raw, {}) or {}
        return {
            "en": obj.get("en", "") or "",
            "zh": obj.get("zh", "") or "",
            "ja": obj.get("ja", "") or "",
        }

    for kq in tqdm(unique_src, desc="🌐 Translating unique ko_explicit queries"):
        cache[kq] = translate_one(kq)
        if sleep_per_call > 0:
            time.sleep(sleep_per_call)

    translated_en = []
    translated_zh = []
    translated_ja = []

    for s in src.tolist():
        m = cache.get(s, {"en": "", "zh": "", "ja": ""})
        translated_en.append(m["en"])
        translated_zh.append(m["zh"])
        translated_ja.append(m["ja"])

    df.loc[success_mask, "query_en"] = translated_en
    df.loc[success_mask, "query_zh"] = translated_zh
    df.loc[success_mask, "query_ja"] = translated_ja

    df.to_csv(detail_csv, index=False, encoding="utf-8-sig")

    ensure_parent_dir(simple_csv)
    df[SIMPLE_COLUMNS].to_csv(simple_csv, index=False, encoding="utf-8-sig")

    ensure_parent_dir(hq_csv)
    hq_df = df[(df["status"] == "success") & (df["cultural_ok"] == True)]
    HQ_COLUMNS = [
        "code",
        "learning_objectives",
        "achievements_standards",
        "query_id",
        "query_ko_no_country",
        "query_ko_implicit_country",
        "query_ko_explicit_country",
        "query_en",
        "query_zh",
        "query_ja",
    ]
    hq_df[HQ_COLUMNS].to_csv(hq_csv, index=False, encoding="utf-8-sig")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--detail_csv", required=True)
    ap.add_argument("--simple_csv", required=True)
    ap.add_argument("--hq_csv", required=True)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--solar_model", default="solar-pro2")
    ap.add_argument("--openai_model", default="gpt-4o")

    ap.add_argument("--max_revise", type=int, default=5)
    ap.add_argument("--augment_k", type=int, default=2)

    ap.add_argument("--disable_eval", action="store_true")
    ap.add_argument("--disable_cultural", action="store_true")

    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=float, default=1.0)

    ap.add_argument("--sleep_per_row", type=float, default=0.2, help="small cushion between rows")
    ap.add_argument("--sleep_per_translate_call", type=float, default=0.0, help="cushion between translation calls (unique only)")
    ap.add_argument("--flush_every", type=int, default=50, help="flush CSV every N appended rows")

    return ap.parse_args()


def main():
    args = parse_args()

    upstage_key = os.getenv("UPSTAGE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not upstage_key:
        print("❌ UPSTAGE_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    need_openai = (not args.disable_eval) or (not args.disable_cultural)
    if need_openai and not openai_key:
        print("❌ OPENAI_API_KEY not set but eval/cultural enabled.", file=sys.stderr)
        sys.exit(1)

    df_in = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    required = {"code", "learning_objectives", "achievements_standards"}
    missing = required - set(df_in.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(list(missing))}")

    solar = ChatUpstage(api_key=upstage_key, model=args.solar_model, temperature=0.0)
    gpt = ChatOpenAI(model=args.openai_model, temperature=0.0, openai_api_key=openai_key) if openai_key else None

    gen_chain = GEN_PROMPT | solar
    variation_chain = VARIATION_PROMPT | solar
    revise_chain = REVISE_PROMPT | solar
    natural_chain = NATURAL_PROMPT | solar
    variant_chain = VARIANT_PROMPT | solar
    multilingual_chain = MULTILINGUAL_PROMPT | solar

    eval_chain = (EVAL_PROMPT | gpt) if (gpt and not args.disable_eval) else None
    cultural_chain = (CULTURAL_PROMPT | gpt) if (gpt and not args.disable_cultural) else None

    processed, base_q1_by_code, cultural_reason_by_code = (set(), {}, {})
    if args.resume:
        processed, base_q1_by_code, cultural_reason_by_code = load_resume_index(args.detail_csv)

    detail_f, detail_w = open_csv_append(args.detail_csv, DETAIL_COLUMNS, write_header_if_new=True)
    simple_f, simple_w = open_csv_append(args.simple_csv, SIMPLE_COLUMNS, write_header_if_new=True)

    appended = 0

    try:
        for _, r in tqdm(df_in.iterrows(), total=len(df_in), desc="🧩 Generating queries"):
            code = str(r["code"])
            lo = str(r["learning_objectives"])
            ach = str(r["achievements_standards"])

            target_qids = [1] + list(range(2, 2 + args.augment_k))
            if args.resume and all((f"{code}_{qid}" in processed) for qid in target_qids):
                continue

            key1 = f"{code}_1"
            if (not args.resume) or (key1 not in processed):
                try:
                    q1, score, iters = generate_refine(
                        lo, ach,
                        gen_chain=gen_chain,
                        eval_chain=eval_chain,
                        revise_chain=revise_chain,
                        natural_chain=natural_chain,
                        max_revise=args.max_revise,
                        enable_eval=(eval_chain is not None),
                        retries=args.retries,
                        backoff=args.backoff,
                    )

                    ok, reason = cultural_check(
                        q1,
                        cultural_chain=cultural_chain,
                        enable_cultural=(cultural_chain is not None),
                        retries=args.retries,
                        backoff=args.backoff,
                    )

                    if not ok:
                        row_detail = {
                            "code": code,
                            "query_id": 1,
                            "learning_objectives": lo,
                            "achievements_standards": ach,
                            "final_query": q1,
                            "query_ko_no_country": "",
                            "query_ko_implicit_country": "",
                            "query_ko_explicit_country": "",
                            "query_en": "",
                            "query_zh": "",
                            "query_ja": "",
                            "eval_score": score if score is not None else "",
                            "cultural_ok": False,
                            "cultural_reason": reason,
                            "iterations": iters,
                            "status": "filtered",
                            "error_reason": "cultural_filter_failed",
                        }
                        detail_w.writerow(row_detail)
                        detail_f.flush()
                        processed.add(key1)
                        time.sleep(args.sleep_per_row)
                        continue

                    v1 = make_variants(q1, variant_chain, args.retries, args.backoff)

                    row_detail = {
                        "code": code,
                        "query_id": 1,
                        "learning_objectives": lo,
                        "achievements_standards": ach,
                        "final_query": q1,
                        "query_ko_no_country": v1["ko_no_country"],
                        "query_ko_implicit_country": v1["ko_implicit_country"],
                        "query_ko_explicit_country": v1["ko_explicit_country"],
                        "query_en": "",
                        "query_zh": "",
                        "query_ja": "",
                        "eval_score": score if score is not None else "",
                        "cultural_ok": True,
                        "cultural_reason": reason,
                        "iterations": iters,
                        "status": "success",
                        "error_reason": "",
                    }
                    detail_w.writerow(row_detail)
                    simple_w.writerow({k: row_detail[k] for k in SIMPLE_COLUMNS})

                    processed.add(key1)
                    base_q1_by_code[code] = q1
                    cultural_reason_by_code[code] = reason

                    appended += 2
                except Exception as e:
                    traceback.print_exc()
                    err_msg = f"{type(e).__name__}: {e}"
                    row_fail = {
                        "code": code,
                        "query_id": 1,
                        "learning_objectives": lo,
                        "achievements_standards": ach,
                        "final_query": "",
                        "query_ko_no_country": "",
                        "query_ko_implicit_country": "",
                        "query_ko_explicit_country": "",
                        "query_en": "",
                        "query_zh": "",
                        "query_ja": "",
                        "eval_score": "",
                        "cultural_ok": "",
                        "cultural_reason": "",
                        "iterations": "",
                        "status": "failed",
                        "error_reason": err_msg,
                    }
                    detail_w.writerow(row_fail)
                    detail_f.flush()
                    processed.add(key1)
                    time.sleep(args.sleep_per_row)
                    continue

            base_query = base_q1_by_code.get(code)
            base_reason = cultural_reason_by_code.get(code, "")

            if not base_query:
                time.sleep(args.sleep_per_row)
                continue

            missing_qids = [qid for qid in range(2, 2 + args.augment_k) if f"{code}_{qid}" not in processed]
            if missing_qids:
                try:
                    aug = make_augmentations(
                        lo, ach, base_query,
                        variation_chain=variation_chain,
                        natural_chain=natural_chain,
                        k=len(missing_qids),
                        retries=args.retries,
                        backoff=args.backoff,
                    )

                    for qid, q in zip(missing_qids, aug):
                        v = make_variants(q, variant_chain, args.retries, args.backoff)
                        row_detail = {
                            "code": code,
                            "query_id": qid,
                            "learning_objectives": lo,
                            "achievements_standards": ach,
                            "final_query": q,
                            "query_ko_no_country": v["ko_no_country"],
                            "query_ko_implicit_country": v["ko_implicit_country"],
                            "query_ko_explicit_country": v["ko_explicit_country"],
                            "query_en": "",
                            "query_zh": "",
                            "query_ja": "",
                            "eval_score": "",
                            "cultural_ok": True,
                            "cultural_reason": base_reason,
                            "iterations": "",
                            "status": "success",
                            "error_reason": "",
                        }
                        detail_w.writerow(row_detail)
                        simple_w.writerow({k: row_detail[k] for k in SIMPLE_COLUMNS})
                        processed.add(f"{code}_{qid}")
                        appended += 2

                except Exception as e:
                    traceback.print_exc()
                    err_msg = f"{type(e).__name__}: {e}"
                    row_fail = {
                        "code": code,
                        "query_id": "",
                        "learning_objectives": lo,
                        "achievements_standards": ach,
                        "final_query": "",
                        "query_ko_no_country": "",
                        "query_ko_implicit_country": "",
                        "query_ko_explicit_country": "",
                        "query_en": "",
                        "query_zh": "",
                        "query_ja": "",
                        "eval_score": "",
                        "cultural_ok": "",
                        "cultural_reason": "",
                        "iterations": "",
                        "status": "failed",
                        "error_reason": f"augmentation_error: {err_msg}",
                    }
                    detail_w.writerow(row_fail)
                    detail_f.flush()

            # periodic flush
            if appended >= args.flush_every:
                detail_f.flush()
                simple_f.flush()
                appended = 0

            time.sleep(args.sleep_per_row)

    finally:
        detail_f.flush()
        simple_f.flush()
        detail_f.close()
        simple_f.close()

    translate_unique_explicit_queries(
        detail_csv=args.detail_csv,
        simple_csv=args.simple_csv,
        hq_csv=args.hq_csv,
        multilingual_chain=multilingual_chain,
        retries=args.retries,
        backoff=args.backoff,
        sleep_per_call=args.sleep_per_translate_call,
    )

    print("✅ Done.")
    print(f"- Detail (updated): {args.detail_csv}")
    print(f"- Simple (updated): {args.simple_csv}")
    print(f"- HQ (updated): {args.hq_csv}")


if __name__ == "__main__":
    main()
