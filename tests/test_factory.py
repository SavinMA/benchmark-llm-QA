"""
Тесты для фабрики бенчмарков
"""
import pytest
from unittest.mock import patch, MagicMock

from src.factory import BenchmarkFactory, BenchmarkType


class TestBenchmarkFactory:
    """Тесты для фабрики бенчмарков"""
    
    def test_benchmark_type_enum(self):
        """Тест перечисления типов бенчмарков"""
        assert hasattr(BenchmarkType, 'OLLAMA_LLM_EVALUATION')
        assert hasattr(BenchmarkType, 'MISTRAL_LLM_EVALUATION')
        assert hasattr(BenchmarkType, 'LLM_ANSWER_EVALUATION')  # Алиас для совместимости
    
    @patch('src.factory.OllamaBenchmark')
    def test_create_ollama_benchmark(self, mock_ollama):
        """Тест создания Ollama бенчмарка"""
        config = {
            'model': 'llama2',
            'host': 'localhost:11434',
            'passing_score': 0.7
        }
        
        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance
        
        benchmark = BenchmarkFactory.create_benchmark(
            BenchmarkType.OLLAMA_LLM_EVALUATION,
            config
        )
        
        mock_ollama.assert_called_once_with(config)
        assert benchmark == mock_instance
    
    @patch('src.factory.MistralBenchmark')
    def test_create_mistral_benchmark(self, mock_mistral):
        """Тест создания Mistral бенчмарка"""
        config = {
            'model': 'mistral-large-latest',
            'api_key': 'test-key',
            'passing_score': 0.7
        }
        
        mock_instance = MagicMock()
        mock_mistral.return_value = mock_instance
        
        benchmark = BenchmarkFactory.create_benchmark(
            BenchmarkType.MISTRAL_LLM_EVALUATION,
            config
        )
        
        mock_mistral.assert_called_once_with(config)
        assert benchmark == mock_instance
    
    @patch('src.factory.OllamaBenchmark')
    def test_create_benchmark_with_alias(self, mock_ollama):
        """Тест создания бенчмарка с использованием алиаса"""
        config = {
            'model': 'llama2',
            'host': 'localhost:11434'
        }
        
        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance
        
        # Используем алиас LLM_ANSWER_EVALUATION
        benchmark = BenchmarkFactory.create_benchmark(
            BenchmarkType.LLM_ANSWER_EVALUATION,
            config
        )
        
        mock_ollama.assert_called_once_with(config)
        assert benchmark == mock_instance
    
    def test_create_benchmark_invalid_type(self):
        """Тест создания бенчмарка с неверным типом"""
        config = {'model': 'test'}
        
        with pytest.raises(ValueError, match="Неподдерживаемый тип бенчмарка"):
            BenchmarkFactory.create_benchmark("INVALID_TYPE", config)
    
    def test_create_benchmark_none_config(self):
        """Тест создания бенчмарка с пустой конфигурацией"""
        with patch('src.factory.OllamaBenchmark') as mock_ollama:
            mock_instance = MagicMock()
            mock_ollama.return_value = mock_instance
            
            benchmark = BenchmarkFactory.create_benchmark(
                BenchmarkType.OLLAMA_LLM_EVALUATION,
                None
            )
            
            # Должен передать None как конфигурацию
            mock_ollama.assert_called_once_with(None)
    
    def test_get_available_types(self):
        """Тест получения доступных типов бенчмарков"""
        # Проверяем, что фабрика знает о всех типах
        available_types = [
            BenchmarkType.OLLAMA_LLM_EVALUATION,
            BenchmarkType.MISTRAL_LLM_EVALUATION,
            BenchmarkType.LLM_ANSWER_EVALUATION
        ]
        
        for benchmark_type in available_types:
            # Не должно выбрасывать исключение для валидных типов
            try:
                with patch('src.factory.OllamaBenchmark'), patch('src.factory.MistralBenchmark'):
                    BenchmarkFactory.create_benchmark(benchmark_type, {})
            except ValueError as e:
                if "Неподдерживаемый тип бенчмарка" in str(e):
                    pytest.fail(f"Тип {benchmark_type} должен поддерживаться")
    
    def test_benchmark_config_passing(self):
        """Тест передачи конфигурации в бенчмарк"""
        config = {
            'model': 'test-model',
            'temperature': 0.5,
            'max_tokens': 1000,
            'custom_param': 'custom_value'
        }
        
        with patch('src.factory.OllamaBenchmark') as mock_ollama:
            mock_instance = MagicMock()
            mock_ollama.return_value = mock_instance
            
            BenchmarkFactory.create_benchmark(
                BenchmarkType.OLLAMA_LLM_EVALUATION,
                config
            )
            
            # Проверяем, что конфигурация передана полностью
            mock_ollama.assert_called_once_with(config) 