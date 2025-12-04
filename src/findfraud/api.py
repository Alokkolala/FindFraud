from __future__ import annotations

import os
from io import BytesIO, StringIO
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .features import FeatureConfig
from .model import ModelConfig
from .rules import RuleConfig
from .scorer import ScoringPipeline


class TransactionInput(BaseModel):

    model_config = ConfigDict(extra="allow")

    transaction_id: str | None = Field(
        default=None, description="Unique identifier; defaults to `<nameOrig>_<step>` if omitted."
    )
    step: int = Field(..., description="Time step of the transaction")
    type: str = Field(..., description="Payment type (e.g., CASH_OUT, TRANSFER)")
    amount: float = Field(..., description="Transaction amount")
    nameOrig: str = Field(..., description="Originating account ID")
    oldbalanceOrg: float = Field(..., description="Starting balance of origin account")
    newbalanceOrig: float = Field(..., description="Ending balance of origin account")
    nameDest: str = Field(..., description="Destination account ID")
    oldbalanceDest: float = Field(..., description="Starting balance of destination account")
    newbalanceDest: float = Field(..., description="Ending balance of destination account")
    isFraud: float | None = Field(0.0, description="Optional label; ignored for scoring")
    isFlaggedFraud: float | None = Field(0.0, description="Optional label; ignored for scoring")


class FraudModelService:

    def __init__(self, model_path: str, model_choice: str = "tabular") -> None:
        self.model_path = Path(model_path)
        self.model_choice = model_choice
        self.pipeline = ScoringPipeline(
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            rule_config=RuleConfig(),
            model_choice=model_choice,
        )
        self.model_bundle: dict | None = None
        self.reload()

    def reload(self) -> None:

        if self.model_choice == "gnn":
            if self.pipeline.graph_model_trainer is None:
                raise ValueError("Graph components are not initialized.")
            self.model_bundle = self.pipeline.graph_model_trainer.load(self.model_path)
        else:
            self.model_bundle = self.pipeline.model_trainer.load(self.model_path)
            self.pipeline._restore_feature_engineer(self.model_bundle)

    def score_records(self, records: List[dict]) -> pd.DataFrame:

        if not records:
            raise ValueError("No transactions provided.")
        raw = pd.DataFrame(records)
        raw = self.pipeline.loader.validate_frame(raw)
        output, _, _, _ = self.pipeline.score_frame(
            raw, str(self.model_path), model_bundle=self.model_bundle
        )
        return output


app = FastAPI(title="FindFraud API", version="1.0.0")

cors_setting = os.getenv("FINDFRAUD_CORS_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in cors_setting.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_service() -> None:
    model_path = os.getenv("FINDFRAUD_MODEL_PATH", "models/synth.joblib")
    model_choice = os.getenv("FINDFRAUD_MODEL_TYPE", "tabular")
    try:
        app.state.service = FraudModelService(model_path=model_path, model_choice=model_choice)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model at startup: {exc}") from exc


def _service() -> FraudModelService:
    service: FraudModelService | None = getattr(app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="Model service is not initialized")
    return service


@app.post("/detect_fraud")
async def detect_fraud(payload: TransactionInput) -> JSONResponse:

    try:
        result = _service().score_records([payload.model_dump()])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    row = result.iloc[0]
    response = {
        "transaction_id": row["transaction_id"],
        "fraud_score": float(row["fraud_score"]),
        "is_suspicious": bool(row["is_suspicious"]),
        "explanation": row["explanation"],
    }
    return JSONResponse(content=response)


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)) -> StreamingResponse:

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported")

    try:
        content = await file.read()
        frame = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    try:
        result = _service().score_records(frame.to_dict(orient="records"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    buffer = StringIO()
    result.to_csv(buffer, index=False)
    buffer.seek(0)
    headers = {"Content-Disposition": f"attachment; filename={Path(file.filename).stem}_scored.csv"}
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv", headers=headers)


try:
    from mangum import Mangum

    handler = Mangum(app)
except Exception:
    handler = None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "findfraud.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("FINDFRAUD_API_RELOAD", "")),
    )
