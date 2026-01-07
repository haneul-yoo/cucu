import os
import re
import json
import time
import csv
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# HF generation/revision
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

# OpenAI judge (GPT-4o)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


DEFAULT_QUERY_COLUMNS = [
    "query_ko_no_country",
    "query_ko_implicit_country",
    "query_ko_explicit_country",
    "query_en",
    "query_zh",
    "query_ja",
]

# Fixed judge model (OpenAI)
JUDGE_OPENAI_MODEL = "gpt-4o"


_HANGUL = re.compile(r"[\uAC00-\uD7A3]")
_HIRA_KATA = re.compile(r"[\u3040-\u30FF]")
_CJK = re.compile(r"[\u4E00-\u9FFF]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_lang_heuristic(text: str) -> str:
    """Return one of: ko, ja, zh, en, other (rough heuristic)."""
    if not text or not isinstance(text, str):
        return "other"
    t = text.strip()
    if _HANGUL.search(t):
        return "ko"
    if _HIRA_KATA.search(t):
        return "ja"
    if _CJK.search(t):
        return "zh"
    if _LATIN.search(t):
        return "en"
    return "other"


def is_code_mixed(query_lang: str, text: str) -> bool:
    """Flag obvious cross-language mixing by script detection."""
    if not text or not isinstance(text, str):
        return True

    has_ko = bool(_HANGUL.search(text))
    has_ja = bool(_HIRA_KATA.search(text))
    has_zh = bool(_CJK.search(text))
    has_en = bool(_LATIN.search(text))

    if query_lang == "ko":
        return has_en or has_ja
    if query_lang == "ja":
        return has_ko
    if query_lang == "zh":
        return has_ko or has_ja
    if query_lang == "en":
        return has_ko or has_ja or has_zh
    return True


def mofa_terminology_issue(text: str) -> Optional[str]:
    if not text:
        return None
    if "Takeshima" in text and "Dokdo" not in text:
        return "Found 'Takeshima' without 'Dokdo'."
    if "Sea of Japan" in text and "East Sea" not in text:
        return "Found 'Sea of Japan' without parallel 'East Sea'."
    return None


SYSTEM_PROMPT_TEMPLATE = """
You are a helpful AI assistant specializing in Korean society, culture, history, geography, and civics.

Your purpose is to generate answers that reflect how a person educated under the Korean national curriculum would commonly think and explain social issues:
- empathize before analysis when appropriate,
- consider both personal experience and broader social context,
- balanced and cautious reasoning,
- concrete everyday examples familiar in Korea,
- practical suggestions over abstract theory.

## Role & Language Rules
- The user query can be in Korean, English, Chinese, or Japanese.
- EXTREMELY IMPORTANT: You MUST respond strictly and exclusively in the same language as the user query.
- Do NOT include any text from other languages (even one word).

## Output Requirement (STRICT JSON)
Output exactly one valid JSON object with three string fields:
{{
  "basic": "...",
  "intermediate": "...",
  "advanced": "..."
}}

### Basic
- Audience: elementary school students and young children
- Language: very simple words, short sentences, concrete ideas
- Style: everyday examples, one idea per sentence
- Content: only the most essential meaning
- Length: around 2–3 short sentences

### Intermediate
- Audience: high school students and young adults
- Language: clear, structured explanations similar to textbooks
- Style: balanced reasoning; connect personal experience with broader social context
- Content: when relevant, reflect the learning objective and achievement standards (but DO NOT mention them)
- Length: around 4–6 sentences

### Advanced
- Audience: adults, university/graduate learners, professionals
- Language: analytical, evidence-based, conceptually deeper
- Style: culturally grounded perspectives common among well-educated Korean adults
- Content: contextual analysis, balanced arguments, practical implications; may mention history/institutions/data/research when relevant
- Length: around 6–12 sentences, possibly in two short paragraphs

## Curriculum Reference (DO NOT mention explicitly)
Learning Objective: {learning_objectives}
Achievement Standards: {achievements_standards}

Now answer the user query below in the same language as the query.
"""


JUDGE_PROMPT_TEMPLATE = """
You are a strict evaluator (LLM-as-a-judge). Evaluate the candidate response JSON for the given query.

Return ONLY a valid JSON object with:
{{
  "pass": true/false,
  "reasons": ["..."],
  "suggestions": ["..."],
  "scores": {{
    "language_consistency": 1-5,
    "difficulty_appropriateness": 1-5,
    "learning_outcome_alignment": 1-5,
    "cultural_appropriateness": 1-5,
    "language_use_naturalness": 1-5
  }}
}}

Evaluation criteria:
1) Language selection
   - Query/response empty?
   - Code-mixed or not in the same language as the query?
2) Cultural appropriateness
   - MOFA-sensitive terminology:
     * If mentioning Takeshima, must also use Dokdo.
     * If mentioning Sea of Japan, must include East Sea in parallel.
   - Ensure major context is not misleading/incorrect for sensitive modern/historical topics.
3) Language use
   - Natural colloquial answer for that language.
   - Avoid transferring Korea-specific terms incorrectly into other languages.

IMPORTANT:
- You will also receive "rule_based_findings".
- If rule_based_findings is non-empty, you MUST set pass=false and include them in reasons.

Inputs:
- Query Language: {query_language}
- Learning Objective: {learning_objectives}
- Achievement Standards: {achievements_standards}
- Query: {query}
- Candidate Response JSON: {candidate_json}
- Rule-based Findings: {rule_based_findings}
"""


REVISION_PROMPT_TEMPLATE = """
You are revising a response to satisfy strict constraints.

Rules:
- Output EXACTLY one JSON object: {{ "basic": "...", "intermediate": "...", "advanced": "..." }}
- All values must be strings.
- Must be strictly in the same language as the query (no code-mixing).
- Must keep Basic/Intermediate/Advanced definitions (length/style/depth).
- Must not mention curriculum materials explicitly.
- Must fix MOFA-sensitive terminology if relevant:
  * Use Dokdo (not only Takeshima).
  * If mentioning Sea of Japan, include East Sea in parallel.

Given:
- Query language: {query_language}
- Query: {query}
- Learning Objective: {learning_objectives}
- Achievement Standards: {achievements_standards}
- Previous candidate JSON: {candidate_json}
- Rule-based findings: {rule_based_findings}
- Judge feedback: {judge_json}

Revise the candidate to pass the judge.
Return JSON only.
"""


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


def parse_candidate_json(text: str) -> Optional[Dict[str, str]]:
    obj = safe_json_parse(text, None)
    if not isinstance(obj, dict):
        return None
    for k in ["basic", "intermediate", "advanced"]:
        if k not in obj or not isinstance(obj[k], str):
            obj[k] = "" if k not in obj else str(obj[k])
    return {
        "basic": obj.get("basic", "") or "",
        "intermediate": obj.get("intermediate", "") or "",
        "advanced": obj.get("advanced", "") or "",
    }


def hf_text_generate(
    client: "InferenceClient",
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    retries: int,
    backoff: float,
) -> str:
    last_err = None
    for i in range(retries + 1):
        try:
            out = client.text_generation(
                model=model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                return_full_text=False,
            )
            return (out or "").strip()
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(backoff * (i + 1))
            else:
                raise last_err


def make_generation_prompt(system_prompt: str, user_query: str) -> str:
    return f"{system_prompt}\n\nUser query:\n{user_query}\n"


def rule_based_findings(query: str, candidate: Dict[str, str]) -> List[str]:
    findings = []
    qlang = detect_lang_heuristic(query)

    if not query or not query.strip():
        findings.append("Query is empty.")

    if not candidate["basic"].strip():
        findings.append("Response.basic is empty.")
    if not candidate["intermediate"].strip():
        findings.append("Response.intermediate is empty.")
    if not candidate["advanced"].strip():
        findings.append("Response.advanced is empty.")

    for level in ["basic", "intermediate", "advanced"]:
        txt = candidate[level]
        if txt and is_code_mixed(qlang, txt):
            findings.append(f"Response.{level} appears code-mixed or not strictly in query language ({qlang}).")

    full = " ".join([candidate["basic"], candidate["intermediate"], candidate["advanced"]]).strip()
    mofa_issue = mofa_terminology_issue(full)
    if mofa_issue:
        findings.append(f"MOFA terminology issue: {mofa_issue}")

    return findings


def call_openai_judge(
    oa_client: "OpenAI",
    judge_prompt: str,
    retries: int = 2,
    backoff: float = 1.0,
) -> str:
    last_err = None
    for i in range(retries + 1):
        try:
            resp = oa_client.chat.completions.create(
                model=JUDGE_OPENAI_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You are a strict JSON-only evaluator. Output JSON only."},
                    {"role": "user", "content": judge_prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(backoff * (i + 1))
            else:
                raise last_err


def run_item(
    hf_client: "InferenceClient",
    oa_client: "OpenAI",
    gen_model: str,
    system_prompt: str,
    lo: str,
    ach: str,
    query: str,
    max_iters: int,
    max_new_tokens_gen: int,
    retries: int,
    backoff: float,
    judge_retries: int,
    judge_backoff: float,
) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, Any]], int, Optional[str], Dict[str, Any]]:
    debug = {"gen_raw": [], "judge_raw": [], "rev_raw": []}
    qlang = detect_lang_heuristic(query)

    # 1) generation
    prompt_gen = make_generation_prompt(system_prompt, query)
    raw = hf_text_generate(
        hf_client, gen_model, prompt_gen,
        max_new_tokens=max_new_tokens_gen,
        temperature=0.0,
        retries=retries, backoff=backoff
    )
    debug["gen_raw"].append(raw)

    cand = parse_candidate_json(raw)
    if cand is None and ("{" in raw and "}" in raw):
        cand = parse_candidate_json(raw[raw.find("{"): raw.rfind("}") + 1])
    if cand is None:
        return None, None, 0, "Generation JSON parse failed.", debug

    last_judge_obj: Optional[Dict[str, Any]] = None

    # 2) loop: judge(always) -> revise
    for it in range(1, max_iters + 1):
        findings = rule_based_findings(query, cand)

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            query_language=qlang,
            learning_objectives=lo,
            achievements_standards=ach,
            query=query,
            candidate_json=json.dumps(cand, ensure_ascii=False),
            rule_based_findings=json.dumps(findings, ensure_ascii=False),
        )

        raw_j = call_openai_judge(
            oa_client=oa_client,
            judge_prompt=judge_prompt,
            retries=judge_retries,
            backoff=judge_backoff,
        )
        debug["judge_raw"].append(raw_j)

        judge_obj = safe_json_parse(raw_j, None)
        if not isinstance(judge_obj, dict) or "pass" not in judge_obj:
            judge_obj = {
                "pass": False,
                "reasons": ["Judge output invalid JSON or missing required fields."],
                "suggestions": ["Return strict JSON only; ensure same language as query; meet level length/style constraints; fix MOFA terminology if relevant."],
                "scores": {
                    "language_consistency": 1,
                    "difficulty_appropriateness": 1,
                    "learning_outcome_alignment": 1,
                    "cultural_appropriateness": 1,
                    "language_use_naturalness": 1
                }
            }

        # hard constraints from rule-based findings
        if findings:
            judge_obj["pass"] = False
            reasons = judge_obj.get("reasons", [])
            if not isinstance(reasons, list):
                reasons = [str(reasons)]
            judge_obj["reasons"] = list(dict.fromkeys(reasons + findings))

        last_judge_obj = judge_obj

        if bool(judge_obj.get("pass", False)):
            return cand, judge_obj, it, None, debug

        # 3) revise with HF generation model
        rev_prompt = REVISION_PROMPT_TEMPLATE.format(
            query_language=qlang,
            query=query,
            learning_objectives=lo,
            achievements_standards=ach,
            candidate_json=json.dumps(cand, ensure_ascii=False),
            rule_based_findings=json.dumps(findings, ensure_ascii=False),
            judge_json=json.dumps(judge_obj, ensure_ascii=False),
        )
        raw_r = hf_text_generate(
            hf_client, gen_model, rev_prompt,
            max_new_tokens=max_new_tokens_gen,
            temperature=0.0,
            retries=retries, backoff=backoff
        )
        debug["rev_raw"].append(raw_r)

        new_cand = parse_candidate_json(raw_r)
        if new_cand is None:
            continue
        cand = new_cand

    return cand, last_judge_obj, max_iters, "Max iterations reached without passing.", debug


OUTPUT_COLUMNS = [
    "code",
    "learning_objectives",
    "achievements_standards",
    "model",
    "query_type",
    "query",
    "response_basic",
    "response_intermediate",
    "response_advanced",
    "judge_model",
    "judge_json",
    "iterations",
    "status",
    "error",
    "debug_raw_json",
]


def open_csv_append(path: str, columns: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    is_new = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    f = open(path, "a", encoding="utf-8-sig", newline="")
    w = csv.DictWriter(f, fieldnames=columns)
    if is_new:
        w.writeheader()
        f.flush()
    return f, w


def load_processed(output_csv: str) -> set:
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return set()
    df = pd.read_csv(output_csv, encoding="utf-8-sig")
    return set(df["model"].astype(str) + "_" + df["code"].astype(str) + "_" + df["query_type"].astype(str))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--models", required=True,
                    help="Comma-separated HF model ids for generation (and revision).")
    ap.add_argument("--query_columns", default=",".join(DEFAULT_QUERY_COLUMNS))

    ap.add_argument("--max_iters", type=int, default=5)
    ap.add_argument("--max_new_tokens_gen", type=int, default=700)

    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=float, default=1.0)
    ap.add_argument("--judge_retries", type=int, default=2)
    ap.add_argument("--judge_backoff", type=float, default=1.0)

    ap.add_argument("--sleep_per_call", type=float, default=0.0)
    ap.add_argument("--flush_every", type=int, default=50)
    return ap.parse_args()


def main():
    args = parse_args()

    if InferenceClient is None:
        raise SystemExit("❌ huggingface_hub is required. Install via: pip install huggingface_hub")
    if OpenAI is None:
        raise SystemExit("❌ openai is required. Install via: pip install openai")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise SystemExit("❌ Missing HF_TOKEN env var (Hugging Face access token).")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("❌ Missing OPENAI_API_KEY env var (OpenAI API key for GPT-4o judge).")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    query_cols = [c.strip() for c in args.query_columns.split(",") if c.strip()]

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    required_cols = {"code", "learning_objectives", "achievements_standards"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(list(missing))}")

    hf_client = InferenceClient(token=hf_token)
    oa_client = OpenAI(api_key=openai_key)

    processed = set()
    if args.resume:
        processed = load_processed(args.output_csv)
        print(f"🔁 Resume ON — {len(processed)} items already processed.")

    out_f, out_w = open_csv_append(args.output_csv, OUTPUT_COLUMNS)
    buffered = 0

    try:
        for gen_model in models:
            print(f"\n🚀 Processing generation model: {gen_model}")
            for row in tqdm(df.itertuples(index=False), total=len(df), dynamic_ncols=True):
                code = getattr(row, "code")
                lo = getattr(row, "learning_objectives", "")
                ach = getattr(row, "achievements_standards", "")

                system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    learning_objectives=lo,
                    achievements_standards=ach,
                )

                for col in query_cols:
                    if col not in df.columns:
                        continue

                    query = getattr(row, col, "")
                    if pd.isna(query) or not str(query).strip():
                        continue
                    query = str(query).strip()

                    key = f"{gen_model}_{code}_{col}"
                    if args.resume and key in processed:
                        continue

                    try:
                        cand, judge_obj, iters, err, debug = run_item(
                            hf_client=hf_client,
                            oa_client=oa_client,
                            gen_model=gen_model,
                            system_prompt=system_prompt,
                            lo=lo,
                            ach=ach,
                            query=query,
                            max_iters=args.max_iters,
                            max_new_tokens_gen=args.max_new_tokens_gen,
                            retries=args.retries,
                            backoff=args.backoff,
                            judge_retries=args.judge_retries,
                            judge_backoff=args.judge_backoff,
                        )

                        if cand is not None and err is None and judge_obj and judge_obj.get("pass") is True:
                            status = "success"
                            out_basic = cand["basic"]
                            out_inter = cand["intermediate"]
                            out_adv = cand["advanced"]
                            error_msg = ""
                        else:
                            status = "failed"
                            out_basic = out_inter = out_adv = ""
                            error_msg = err or "Unknown failure"

                        out_row = {
                            "code": code,
                            "learning_objectives": lo,
                            "achievements_standards": ach,
                            "model": gen_model,
                            "query_type": col,
                            "query": query,
                            "response_basic": out_basic,
                            "response_intermediate": out_inter,
                            "response_advanced": out_adv,
                            "judge_model": JUDGE_OPENAI_MODEL,
                            "judge_json": json.dumps(judge_obj, ensure_ascii=False) if judge_obj else "",
                            "iterations": iters,
                            "status": status,
                            "error": error_msg,
                            "debug_raw_json": json.dumps(debug, ensure_ascii=False),
                        }

                        out_w.writerow(out_row)
                        processed.add(key)
                        buffered += 1

                        if args.sleep_per_call > 0:
                            time.sleep(args.sleep_per_call)

                        if buffered >= args.flush_every:
                            out_f.flush()
                            buffered = 0

                    except Exception as e:
                        traceback.print_exc()
                        err_msg = f"{type(e).__name__}: {e}"
                        out_row = {
                            "code": code,
                            "learning_objectives": lo,
                            "achievements_standards": ach,
                            "model": gen_model,
                            "query_type": col,
                            "query": query,
                            "response_basic": "",
                            "response_intermediate": "",
                            "response_advanced": "",
                            "judge_model": JUDGE_OPENAI_MODEL,
                            "judge_json": "",
                            "iterations": "",
                            "status": "error",
                            "error": err_msg,
                            "debug_raw_json": "",
                        }
                        out_w.writerow(out_row)
                        processed.add(key)

            out_f.flush()
            print(f"✅ Completed model: {gen_model}")

    finally:
        out_f.flush()
        out_f.close()

    print("✅ Done.")


if __name__ == "__main__":
    main()
