"""
Тесты для модулей данных
"""
import pytest
from pydantic import ValidationError
from src.models import Query, QueryAnswerWithActualAnswer


class TestQuery:
    """Тесты для модели Query"""
    
    def test_query_creation_valid(self):
        """Тест создания валидного объекта Query"""
        query = Query(
            query="Что такое Python?",
            expected_answer="Python - это язык программирования"
        )
        
        assert query.query == "Что такое Python?"
        assert query.expected_answer == "Python - это язык программирования"
    
    def test_query_creation_empty_query(self):
        """Тест создания Query с пустым запросом"""
        query = Query(query="", expected_answer="Ответ")
        assert query.query == ""
        assert query.expected_answer == "Ответ"
    
    def test_query_creation_empty_expected_answer(self):
        """Тест создания Query с пустым ожидаемым ответом"""
        query = Query(query="Вопрос", expected_answer="")
        assert query.query == "Вопрос"
        assert query.expected_answer == ""
    
    def test_query_creation_missing_fields(self):
        """Тест создания Query без обязательных полей"""
        with pytest.raises(ValidationError):
            Query(query="Только вопрос")
        
        with pytest.raises(ValidationError):
            Query(expected_answer="Только ответ")


class TestQueryAnswerWithActualAnswer:
    """Тесты для модели QueryAnswerWithActualAnswer"""
    
    def test_query_answer_creation_valid(self):
        """Тест создания валидного объекта QueryAnswerWithActualAnswer"""
        qa = QueryAnswerWithActualAnswer(
            query="Что такое AI?",
            expected_answer="AI - это искусственный интеллект",
            actual_answer="Искусственный интеллект - это технология"
        )
        
        assert qa.query == "Что такое AI?"
        assert qa.expected_answer == "AI - это искусственный интеллект"
        assert qa.actual_answer == "Искусственный интеллект - это технология"
    
    def test_query_answer_inheritance(self):
        """Тест наследования от Query"""
        qa = QueryAnswerWithActualAnswer(
            query="Тест",
            expected_answer="Ожидаемый",
            actual_answer="Фактический"
        )
        
        # Проверяем, что объект является экземпляром Query
        assert isinstance(qa, Query)
    
    def test_query_answer_creation_missing_actual_answer(self):
        """Тест создания QueryAnswerWithActualAnswer без фактического ответа"""
        with pytest.raises(ValidationError):
            QueryAnswerWithActualAnswer(
                query="Вопрос",
                expected_answer="Ожидаемый ответ"
            )
    
    def test_query_answer_creation_empty_actual_answer(self):
        """Тест создания QueryAnswerWithActualAnswer с пустым фактическим ответом"""
        qa = QueryAnswerWithActualAnswer(
            query="Вопрос",
            expected_answer="Ожидаемый ответ",
            actual_answer=""
        )
        
        assert qa.actual_answer == ""
    
    def test_query_answer_unicode_handling(self):
        """Тест обработки unicode символов"""
        qa = QueryAnswerWithActualAnswer(
            query="Что такое 🤖?",
            expected_answer="Это робот 🤖",
            actual_answer="Робот - это машина 🤖"
        )
        
        assert "🤖" in qa.query
        assert "🤖" in qa.expected_answer
        assert "🤖" in qa.actual_answer 