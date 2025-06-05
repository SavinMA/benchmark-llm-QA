"""
Общие фикстуры для тестов
"""
import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List

from src.models import Query, QueryAnswerWithActualAnswer


@pytest.fixture
def sample_query():
    """Фикстура с примером Query"""
    return Query(
        query="Что такое Python?",
        expected_answer="Python - это язык программирования высокого уровня"
    )


@pytest.fixture
def sample_query_answer():
    """Фикстура с примером QueryAnswerWithActualAnswer"""
    return QueryAnswerWithActualAnswer(
        query="Что такое машинное обучение?",
        expected_answer="Машинное обучение - это раздел искусственного интеллекта",
        actual_answer="ML - это подраздел AI, который позволяет компьютерам учиться"
    )


@pytest.fixture
def sample_test_cases():
    """Фикстура с примерами тестовых случаев"""
    return [
        {
            "query": "Что такое Python?",
            "expected_answer": "Python - язык программирования",
            "actual_answer": "Python это язык программирования высокого уровня"
        },
        {
            "query": "Что такое AI?",
            "expected_answer": "AI - искусственный интеллект",
            "actual_answer": "Искусственный интеллект - это технология"
        },
        {
            "query": "Как работает машинное обучение?",
            "expected_answer": "ML использует алгоритмы для анализа данных",
            "actual_answer": "Машинное обучение анализирует данные с помощью алгоритмов"
        }
    ]


@pytest.fixture
def ollama_config():
    """Фикстура с конфигурацией для Ollama"""
    return {
        'model': 'llama2',
        'host': 'localhost:11434',
        'passing_score': 0.7,
        'evaluation_criteria': {
            'accuracy': 0.4,
            'completeness': 0.3,
            'relevance': 0.3
        }
    }


@pytest.fixture
def mistral_config():
    """Фикстура с конфигурацией для Mistral"""
    return {
        'model': 'mistral-large-latest',
        'api_key': 'test-api-key',
        'passing_score': 0.7,
        'timeout': 60,
        'evaluation_criteria': {
            'accuracy': 0.4,
            'completeness': 0.3,
            'relevance': 0.3
        }
    }


@pytest.fixture
def mock_evaluation_response():
    """Фикстура с примером ответа от LLM для оценки"""
    return {
        "scores": {
            "accuracy": 0.9,
            "completeness": 0.8,
            "relevance": 0.85
        },
        "overall_score": 0.85,
        "explanation": "Ответ точно отражает суть вопроса и предоставляет полную информацию",
        "suggestions": "Можно добавить больше примеров для лучшего понимания"
    }


@pytest.fixture(scope="session")
def event_loop():
    """Создание event loop для асинхронных тестов"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Маркеры для классификации тестов
pytest_plugins = ["pytest_asyncio"] 