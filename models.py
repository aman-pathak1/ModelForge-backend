import uuid
from sqlalchemy import Column, String, Boolean, DateTime, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    datasets = relationship("Dataset", back_populates="user")
    eda_summaries = relationship("EDASummary", back_populates="user")

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    file_name = Column(String, nullable=False)
    upload_time = Column(DateTime(timezone=True), server_default=func.now())
    number_of_rows = Column(Integer)
    number_of_columns = Column(Integer)
    column_names = Column(JSON)
    column_types = Column(JSON)

    user = relationship("User", back_populates="datasets")
    eda_summaries = relationship("EDASummary", back_populates="dataset")

class EDASummary(Base):
    __tablename__ = "eda_summaries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    dataset_id = Column(String, ForeignKey("datasets.id"))
    missing_summary = Column(JSON)
    distributions = Column(JSON)
    correlations = Column(JSON)
    key_insights = Column(JSON)

    user = relationship("User", back_populates="eda_summaries")
    dataset = relationship("Dataset", back_populates="eda_summaries")
