"""
Базовый класс для LLM бенчмарков
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from .models import QueryAnswerWithActualAnswer, Query, BenchmarkResult, BenchmarkSummary


class LLMBenchmark(ABC):
    """Абстрактный базовый класс для LLM бенчмарков"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация бенчмарка
        
        Args:
            config: Конфигурация бенчмарка
        """
        self.config = config
        self.results: List[BenchmarkResult] = []
    
    @abstractmethod
    async def evaluate_single(
        self, 
        query_answer_with_expected_answer: QueryAnswerWithActualAnswer
    ) -> BenchmarkResult:
        """
        Оценивает один ответ
        
        Args:
            query_answer_with_expected_answer: Вопрос, ожидаемый ответ и фактический ответ
            
        Returns:
            Результат оценки
        """
        pass
    
    @abstractmethod
    async def run_benchmark(
        self, 
        test_cases: List[Query]
    ) -> BenchmarkSummary:
        """
        Запускает полный бенчмарк
        
        Args:
            test_cases: Список тестовых случаев с полями query, expected_answer, actual_answer
            
        Returns:
            Сводка результатов
        """
        pass
    
    def get_results(self) -> List[BenchmarkResult]:
        """Возвращает все результаты"""
        return self.results
    
    def clear_results(self):
        """Очищает результаты"""
        self.results.clear()

    