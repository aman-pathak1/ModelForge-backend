from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel

from database import get_db
from models import User, Dataset, EDASummary
from auth import get_current_active_user

router = APIRouter(tags=["data_tracking"])

# Pydantic Schemas
class DatasetCreate(BaseModel):
    file_name: str
    number_of_rows: int
    number_of_columns: int
    column_names: List[str]
    column_types: Dict[str, str]

class EDASummaryCreate(BaseModel):
    dataset_id: str
    missing_summary: List[dict]
    distributions: List[dict]
    correlations: dict
    key_insights: List[dict]

@router.post("/api/datasets")
def create_dataset_metadata(dataset: DatasetCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    db_dataset = Dataset(
        user_id=current_user.id,
        file_name=dataset.file_name,
        number_of_rows=dataset.number_of_rows,
        number_of_columns=dataset.number_of_columns,
        column_names=dataset.column_names,
        column_types=dataset.column_types
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

@router.get("/api/datasets")
def list_datasets(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    datasets = db.query(Dataset).filter(Dataset.user_id == current_user.id).order_by(Dataset.upload_time.desc()).all()
    return datasets

@router.post("/api/eda")
def save_eda_summary(eda: EDASummaryCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    dataset = db.query(Dataset).filter(Dataset.id == eda.dataset_id, Dataset.user_id == current_user.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or unauthorized")

    db_eda = EDASummary(
        user_id=current_user.id,
        dataset_id=eda.dataset_id,
        missing_summary=eda.missing_summary,
        distributions=eda.distributions,
        correlations=eda.correlations,
        key_insights=eda.key_insights
    )
    db.add(db_eda)
    db.commit()
    db.refresh(db_eda)
    return db_eda

@router.get("/api/eda/{dataset_id}")
def get_eda_summary(dataset_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    eda = db.query(EDASummary).filter(EDASummary.dataset_id == dataset_id, EDASummary.user_id == current_user.id).first()
    if not eda:
        raise HTTPException(status_code=404, detail="EDA Summary not found")
    return eda
