from fastapi import APIRouter, HTTPException
from backend.app.schemas.evaluation import EvaluateRequest, EvaluateResponse
from backend.app.services.evaluation import calculate_rouge

router = APIRouter()

@router.post("", response_model=EvaluateResponse)
def evaluate_summary(payload: EvaluateRequest):
    try:
        if not payload.generated_summary.strip():
            raise HTTPException(status_code=400, detail="Generated summary cannot be empty")
        if not payload.reference_summary.strip():
            raise HTTPException(status_code=400, detail="Reference summary cannot be empty")
            
        result = calculate_rouge(payload.generated_summary, payload.reference_summary)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
