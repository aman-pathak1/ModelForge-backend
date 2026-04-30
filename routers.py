from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel

from database import get_db
from models import User, Dataset, EDASummary, TrainedModel
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
    health_score: int = 0
    preprocessing_details: Dict[str, Any] = {}

class ModelSave(BaseModel):
    dataset_id: str
    model_name: str
    accuracy: str
    task_type: str

@router.post("/api/datasets")
def create_dataset_metadata(dataset: DatasetCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    # 1. Duplicate Check
    existing = db.query(Dataset).filter(
        Dataset.file_name == dataset.file_name, 
        Dataset.user_id == current_user.id
    ).first()
    
    if existing:
        return existing # Return existing if name matches

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
    # Join with EDA for card details
    results = []
    for d in datasets:
        eda = db.query(EDASummary).filter(EDASummary.dataset_id == d.id).first()
        results.append({
            "id": d.id,
            "file_name": d.file_name,
            "upload_time": d.upload_time,
            "number_of_rows": d.number_of_rows,
            "number_of_columns": d.number_of_columns,
            "health_score": eda.health_score if eda else None,
            "key_insights": eda.key_insights if eda else []
        })
    return results

@router.post("/api/eda")
def save_eda_summary(eda: EDASummaryCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    dataset = db.query(Dataset).filter(Dataset.id == eda.dataset_id, Dataset.user_id == current_user.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or unauthorized")

    # Update existing or create new
    db_eda = db.query(EDASummary).filter(EDASummary.dataset_id == eda.dataset_id).first()
    if not db_eda:
        db_eda = EDASummary(user_id=current_user.id, dataset_id=eda.dataset_id)
        db.add(db_eda)

    db_eda.missing_summary = eda.missing_summary
    db_eda.distributions = eda.distributions
    db_eda.correlations = eda.correlations
    db_eda.key_insights = eda.key_insights
    db_eda.health_score = eda.health_score
    db_eda.preprocessing_details = eda.preprocessing_details
    
    db.commit()
    db.refresh(db_eda)
    return db_eda

@router.get("/api/eda/{dataset_id}")
def get_eda_summary(dataset_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    eda = db.query(EDASummary).filter(EDASummary.dataset_id == dataset_id, EDASummary.user_id == current_user.id).first()
    if not eda:
        raise HTTPException(status_code=404, detail="EDA Summary not found")
    return eda

@router.post("/api/models")
def save_trained_model(model: ModelSave, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    db_model = TrainedModel(
        user_id=current_user.id,
        dataset_id=model.dataset_id,
        model_name=model.model_name,
        accuracy=model.accuracy,
        task_type=model.task_type
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

@router.get("/api/models")
def list_models(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    return db.query(TrainedModel).filter(TrainedModel.user_id == current_user.id).order_by(TrainedModel.created_at.desc()).all()
