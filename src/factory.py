"""
Фабрика для создания различных типов бенчмарков
"""
from typing import Dict, Any, List
from enum import Enum
from .benchmark import LLMBenchmark
from .mistral_benchmark import MistralBenchmark
from .ollama_benchmark import OllamaBenchmark

class BenchmarkType(Enum):
    """Типы доступных бенчмарков"""
    OLLAMA_ANSWER_EVALUATION = "ollama_answer_evaluation"
    MISTRAL_ANSWER_EVALUATION = "mistral_answer_evaluation"
    LOCAL_MODEL_ANSWER_EVALUATION = "local_model_answer_evaluation"

class BenchmarkFactory:
    """Фабрика для создания бенчмарков"""
    
    @staticmethod
    def create_benchmark(
        benchmark_type: BenchmarkType,
        config: Dict[str, Any]
    ) -> LLMBenchmark:
        """
        Создает бенчмарк указанного типа
        
        Args:
            benchmark_type: Тип бенчмарка
            config: Конфигурация бенчмарка
            
        Returns:
            Экземпляр бенчмарка
        """
        if benchmark_type == BenchmarkType.LLM_ANSWER_EVALUATION:
            return OllamaBenchmark(config)
        elif benchmark_type == BenchmarkType.MISTRAL_ANSWER_EVALUATION:
            return MistralBenchmark(config)
        
        raise ValueError(f"Неподдерживаемый тип бенчмарка: {benchmark_type}")
    
    @staticmethod
    def get_available_benchmarks() -> List[str]:
        """Возвращает список доступных типов бенчмарков"""
        return [bt.value for bt in BenchmarkType]
