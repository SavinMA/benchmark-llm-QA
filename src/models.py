from pydantic import BaseModel, Field
from typing import Dict, Any, List
from datetime import datetime

class Query(BaseModel):
    """Модель запроса
    query: str - запрос
    expected_answer: str - ожидаемый ответ
    """
    query: str
    expected_answer: str


class QueryAnswerWithActualAnswer(Query):
    """Модель запроса с фактическим ответом
    query: str - запрос
    expected_answer: str - ожидаемый ответ
    actual_answer: str - фактический ответ
    """
    actual_answer: str



class BenchmarkResult(BaseModel):
    """Результат выполнения бенчмарка"""
    test_name: str = Field(default="")
    query: str = Field(default="")
    expected_answer: str = Field(default="")
    actual_answer: str = Field(default="")
    score: float = Field(default=0.0)
    evaluation_details: Dict[str, Any] = Field(default_factory=Dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_seconds: float = Field(default=0.0)


class BenchmarkSummary(BaseModel):
    """Сводка результатов бенчмарка"""
    total_tests: int = Field(default=0)
    average_score: float = Field(default=0.0)
    min_score: float = Field(default=0.0)
    max_score: float = Field(default=0.0)
    passed_tests: int = Field(default=0)
    failed_tests: int = Field(default=0)
    total_execution_time: float = Field(default=0.0)
    results: List[BenchmarkResult] = Field(default_factory=list)