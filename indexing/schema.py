from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    TABLE_SUMMARY = "table_summary"


class SourceType(str, Enum):
    FINQA = "finqa"
    EDGAR = "edgar"


class Document(BaseModel):
    doc_id: str
    chunk_id: str
    content: str
    doc_type: DocumentType
    source_type: SourceType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class TableDocument(Document):
    doc_type: DocumentType = DocumentType.TABLE
    table_data: Optional[List[List[str]]] = None
    headers: Optional[List[str]] = None
    summary: Optional[str] = None


class FinQADocument(Document):
    source_type: SourceType = SourceType.FINQA
    question_id: Optional[str] = None
    pre_text: Optional[List[str]] = None
    post_text: Optional[List[str]] = None
    table: Optional[List[List[Any]]] = None


class EDGARDocument(Document):
    source_type: SourceType = SourceType.EDGAR
    ticker: str
    cik: str
    filing_type: str
    filing_date: datetime
    section: Optional[str] = None
    accession_number: str
    

class RetrievalResult(BaseModel):
    document: Document
    score: float
    rank: int


class ProgramStep(BaseModel):
    operation: str
    arguments: List[Any]
    result: Optional[float] = None


class Program(BaseModel):
    steps: List[ProgramStep]
    final_answer: Optional[float] = None
    variables: Dict[str, float] = Field(default_factory=dict)


class FinQAExample(BaseModel):
    id: str
    question: str
    pre_text: List[str]
    post_text: List[str]
    table: List[List[Any]]
    answer: str
    program: Optional[List[str]] = None
    gold_inds: Optional[Dict[str, Any]] = None


class FinQAPrediction(BaseModel):
    question_id: str
    question: str
    predicted_answer: float
    gold_answer: str
    program: Program
    retrieved_evidence: List[RetrievalResult]
    execution_success: bool
    error_message: Optional[str] = None
    intermediate_values: Dict[str, float] = Field(default_factory=dict)


class EDGARQuery(BaseModel):
    query: str
    ticker: Optional[str] = None
    filing_type: Optional[str] = None
    date_range: Optional[tuple[datetime, datetime]] = None


class EDGARAnswer(BaseModel):
    query: str
    answer: str
    as_of_date: str
    citations: List[Dict[str, str]]
    retrieved_chunks: List[RetrievalResult]
    verified_numbers: Dict[str, bool] = Field(default_factory=dict)
    confidence: float = 1.0
