import os
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, create_model
from langchain_core.prompts import ChatPromptTemplate

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemma-4-31b-it")


# ── Base model ────────────────────────────────────────────────────────────────

class _ContractBase(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)


# ── Dynamic schema builder ────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    key = name.lower().strip()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_") or "field"


def _make_schema(sanitized_keys: list[str]):
    defs: dict = {}
    for key in sanitized_keys:
        label = key.replace("_", " ")
        defs[key] = (
            Optional[str],
            Field(None, description=f"Final policy-enforced value for '{label}'"),
        )
        defs[f"{key}_reasoning"] = (
            Optional[str],
            Field(None, description=f"Reasoning (≤15 words) for '{label}'"),
        )
    return create_model("DynamicContract", __base__=_ContractBase, **defs)


# ── Universal single prompt ───────────────────────────────────────────────────

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a universal contract analysis and policy enforcement engine.

You will receive two documents and a list of fields to extract.
You have NO prior assumptions about what either document contains.
Your job is to read both documents, understand their structure and meaning on your own,
and then extract each requested field with full policy enforcement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — UNDERSTAND THE POLICY DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read the POLICY RULES thoroughly. Ask yourself:
  • What is this policy about? What does it govern?
  • Does it define tiers, levels, bands, categories, or plans?
  • If yes — what columns does the tier table have? Read every header.
  • What column or combination of columns uniquely determines which tier applies?
    (It could be a date, a month, an uptime %, a fee, a region, a size — anything.)
  • Are there rules that override, cap, floor, or default specific values?
  • Are there penalty or credit clauses? Under what conditions?
Remember the entire policy structure. You will use it in Phase 3.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — UNDERSTAND THE CONTRACT DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read the CONTRACT DOCUMENT thoroughly. Identify:
  • What kind of agreement is this? What is its subject matter?
  • What are the key facts stated: parties, dates, amounts, durations, percentages, terms?
  • Which of those facts could be used to match a policy tier or rule?
    (Look for anything that corresponds to the policy's matching column from Phase 1.)
Remember the full contract content. You will use it in Phase 3 and 4.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — APPLY THE POLICY TO THE CONTRACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Using what you learned in Phases 1 and 2:

  If the policy has tiers/levels:
    → Find the value(s) in the contract that match the policy's matching column.
    → Select the ONE tier/level that applies to this contract.
    → Record all values from that tier row: every field the policy table defines for it.
    → These policy values are AUTHORITATIVE — they override whatever the contract says
      for those same fields.

  If the policy has rules/overrides (not a tier table):
    → Identify which rules apply to this contract.
    → Note which fields those rules affect and what values they mandate.

  If the policy has no rules relevant to this contract:
    → No overrides. Extract directly from the contract.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 4 — EXTRACT EVERY REQUESTED FIELD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each field in the list, determine its value using this priority order:

  1. POLICY AUTHORITY (highest)
     Does the policy tier or a policy rule mandate a specific value for this field?
     → Use the policy value. Do not use the contract's stated value.

  2. CONTRACT CONTENT
     Search the contract for the field's meaning, not just its exact name.
     Understand what the field label means conceptually and find the corresponding
     information in the contract text, even if it is worded differently.
     → Extract the value exactly as written (strip currency symbols and commas from numbers).

  3. POLICY DEFAULT
     Does the policy provide a default or fallback value for this field?
     → Use it.

  4. NOT FOUND
     → Return null.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING — required for every field (≤ 15 words)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use exactly one of:
  "Extracted from contract"                — value came directly from the agreement
  "Policy override — <brief reason>"       — policy replaced or corrected the contract value
  "Inferred from policy — <brief reason>"  — absent in contract, filled from policy
  "Not found"                              — absent from both documents

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Never invent or assume values not present in the provided documents.
• Strip all currency symbols and formatting characters from numeric values.
• Copy text values (names, dates) exactly as they appear in the source.
• Output ONLY the JSON object — nothing else.
"""),
    ("user",
     "Fields to extract:\n{field_definitions}\n\n"
     "CONTRACT DOCUMENT:\n{contract_text}\n\n"
     "POLICY RULES:\n{conditions_context}")
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _select_text(text: str) -> str:
    if len(text) <= 12000:
        return text
    return text[:10000] + "\n\n[... document continues ...]\n\n" + text[-2000:]


# ── Programmatic tier override ────────────────────────────────────────────────

def _nearest_tier(fee: int, tiers: list[dict]) -> dict | None:
    if not tiers:
        return None
    return min(tiers, key=lambda t: abs(t["monthly_fee"] - fee))


def _apply_tier_overrides(output: dict, tiers: list[dict]) -> dict:
    if not tiers:
        return output

    # Find the monthly fee field and parse its value
    fee_value: int | None = None
    for original_name, data in output.items():
        key = _sanitize(original_name)
        if ("monthly" in key and "fee" in key) or "retainer" in key:
            raw = data.get("value")
            if raw:
                try:
                    fee_value = int(re.sub(r"[^0-9]", "", str(raw)))
                    break
                except (ValueError, TypeError):
                    pass

    if not fee_value:
        return output

    tier = _nearest_tier(fee_value, tiers)
    if not tier:
        return output

    print(f"Tier override: fee={fee_value} → nearest tier '{tier['name']}' (₹{tier['monthly_fee']})")

    for original_name in list(output.keys()):
        key = _sanitize(original_name)
        if ("monthly" in key and "fee" in key) or "retainer" in key:
            output[original_name] = {
                "value": str(tier["monthly_fee"]),
                "reasoning": f"Policy override — nearest tier '{tier['name']}' (₹{tier['monthly_fee']:,})",
            }
        elif "service" in key and "tier" in key:
            output[original_name] = {
                "value": tier["name"],
                "reasoning": "Policy override — matched by monthly fee proximity",
            }
        elif "minimum_uptime" in key or (key == "uptime_percent"):
            output[original_name] = {
                "value": str(tier["uptime_percent"]),
                "reasoning": f"Policy override — {tier['name']} requires {tier['uptime_percent']}% minimum uptime",
            }

    return output


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_contract_data(text: str, fields: list[str], policy_tiers: list[dict] | None = None) -> dict:
    """
    Extract and policy-enforce each field in `fields` from contract `text`.
    Returns: {original_field_name: {"value": str|None, "reasoning": str}}
    """
    from rag_client import query_conditions
    from langchain_google_genai import ChatGoogleGenerativeAI

    key_map: dict[str, str] = {}
    for name in fields:
        key = _sanitize(name)
        if key not in key_map:
            key_map[key] = name

    sanitized_keys = list(key_map.keys())
    print(f"Extracting {len(sanitized_keys)} fields: {sanitized_keys}")

    contract_text = _select_text(text)
    conditions_context = query_conditions(sanitized_keys)
    print(f"RAG returned {len(conditions_context)} chars of policy context.")

    field_definitions_str = "\n".join(
        f"• {original} (key: {key})"
        for key, original in key_map.items()
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
    )

    DynamicSchema = _make_schema(sanitized_keys)
    structured_llm = llm.with_structured_output(DynamicSchema)
    chain = EXTRACTION_PROMPT | structured_llm

    for attempt in range(3):
        try:
            raw_result = chain.invoke({
                "contract_text": contract_text,
                "conditions_context": conditions_context or "No policy context available.",
                "field_definitions": field_definitions_str,
            })
            if raw_result is not None and hasattr(raw_result, "model_dump"):
                raw: dict = raw_result.model_dump()  # type: ignore[union-attr]
                output: dict = {}
                for key, original in key_map.items():
                    output[original] = {
                        "value": raw.get(key),
                        "reasoning": raw.get(f"{key}_reasoning") or "Not found",
                    }
                if policy_tiers:
                    output = _apply_tier_overrides(output, policy_tiers)
                return output
        except Exception as e:
            print(f"[Attempt {attempt + 1}] Extraction failed: {e}")

    return {"error": "Extraction failed after 3 attempts"}
