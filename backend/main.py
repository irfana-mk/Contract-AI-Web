import json
import os
import shutil
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from extractor import extract_text_from_pdf
from llm_client import extract_contract_data
from rag_client import get_conditions_count, get_embedder, index_conditions_pdf, parse_policy_tiers

MEDIA_DIR = os.path.join(os.path.dirname(__file__), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Pre-loading sentence transformer...")
        get_embedder()
    except Exception as e:
        print(f"Startup warning: {e}")
    yield


app = FastAPI(title="Contract AI API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/check-conditions/")
async def check_conditions():
    try:
        count = get_conditions_count()
        return {"is_loaded": count > 0, "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/")
async def analyze_with_policy(
    contract_file: UploadFile = File(...),
    policy_file: UploadFile = File(...),
    fields: str = Form(...),
):
    """
    Main analysis endpoint:
      - policy_file  : PDF that is chunked and indexed into Qdrant (required)
      - contract_file: Agreement PDF to extract & enforce policy on (required)
      - fields       : JSON array of field-name strings to extract (required)

    Returns: { field_name: { value: str|null, reasoning: str } }
    """
    # ── Validate files ────────────────────────────────────────
    if not contract_file.filename or not contract_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Contract file must be a valid PDF.")
    if not policy_file.filename or not policy_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Policy file must be a valid PDF.")

    # ── Validate fields ───────────────────────────────────────
    try:
        field_list: list[str] = json.loads(fields)
        if not isinstance(field_list, list) or not field_list:
            raise ValueError
        field_list = [str(f).strip() for f in field_list if str(f).strip()]
        if not field_list:
            raise ValueError
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="'fields' must be a non-empty JSON array of field-name strings.",
        )

    policy_path = os.path.join(MEDIA_DIR, "policy_" + policy_file.filename)
    contract_path = os.path.join(MEDIA_DIR, "contract_" + contract_file.filename)

    try:
        # ── Step 1: Index policy PDF into Qdrant ─────────────
        with open(policy_path, "wb") as f:
            shutil.copyfileobj(policy_file.file, f)

        print(f"Indexing policy: {policy_file.filename}...")
        success, policy_text = index_conditions_pdf(policy_path)
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract text from the policy PDF. It may be empty or image-based.",
            )
        policy_tiers = parse_policy_tiers(policy_text)
        print(f"Policy indexed successfully. Parsed {len(policy_tiers)} tiers: {[t['name'] for t in policy_tiers]}")

        # ── Step 2: Extract text from agreement PDF ───────────
        with open(contract_path, "wb") as f:
            shutil.copyfileobj(contract_file.file, f)

        print(f"Extracting text from agreement: {contract_file.filename}...")
        contract_text = extract_text_from_pdf(contract_path)

        if not contract_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the agreement PDF. It may be empty or image-based.",
            )

        # ── Step 3: LLM extraction + policy enforcement ───────
        print(f"Running extraction for {len(field_list)} fields...")
        extracted_data = extract_contract_data(contract_text, field_list, policy_tiers=policy_tiers)
        return JSONResponse(content=extracted_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in (policy_path, contract_path):
            if os.path.exists(p):
                os.remove(p)
