"""
Тесты для утилит бенчмарка
"""
import pytest
import json
import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path

from src.benchmark_utils import BenchmarkReporter, load_test_cases_from_json, load_test_cases_from_csv
from src.models import QueryAnswerWithActualAnswer


@pytest.fixture
def sample_benchmark_result():
    """Фикстура с примером результата бенчмарка"""
    from src.benchmark import BenchmarkResult, BenchmarkSummary
    
    results = [
        BenchmarkResult(
            test_name="test_1",
            query="Что такое Python?",
            expected_answer="Python - это язык программирования",
            actual_answer="Python это язык программирования высокого уровня",
            score=0.85,
            evaluation_details={
                "scores": {"accuracy": 0.9, "completeness": 0.8, "relevance": 0.85},
                "explanation": "Хорошее соответствие",
                "suggestions": "Добавить больше деталей"
            },
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            execution_time_seconds=2.5
        ),
        BenchmarkResult(
            test_name="test_2",
            query="Что такое AI?",
            expected_answer="AI - искусственный интеллект",
            actual_answer="Искусственный интеллект - это технология",
            score=0.75,
            evaluation_details={
                "scores": {"accuracy": 0.8, "completeness": 0.7, "relevance": 0.75},
                "explanation": "Неплохое соответствие",
                "suggestions": "Быть более точным"
            },
            timestamp=datetime(2024, 1, 1, 12, 1, 0),
            execution_time_seconds=3.0
        )
    ]
    
    return BenchmarkSummary(
        total_tests=2,
        passed_tests=2,
        failed_tests=0,
        average_score=0.8,
        min_score=0.75,
        max_score=0.85,
        total_execution_time=5.5,
        total_time_fill_db=2.0,
        total_time_analyze=3.5,
        results=results,
        pipeline_params={"model": "test-model", "temperature": 0.1}
    )


class TestBenchmarkReporter:
    """Тесты для BenchmarkReporter"""
    
    def test_export_to_json(self, sample_benchmark_result):
        """Тест экспорта в JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_results.json")
            
            BenchmarkReporter.export_to_json(
                sample_benchmark_result,
                filepath,
                include_details=True
            )
            
            # Проверяем, что файл создан
            assert os.path.exists(filepath)
            
            # Проверяем содержимое
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data['metadata']['total_tests'] == 2
            assert data['metadata']['average_score'] == 0.8
            assert len(data['results']) == 2
            assert data['results'][0]['test_name'] == 'test_1'
    
    def test_export_to_json_without_details(self, sample_benchmark_result):
        """Тест экспорта в JSON без деталей"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_results.json")
            
            BenchmarkReporter.export_to_json(
                sample_benchmark_result,
                filepath,
                include_details=False
            )
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'metadata' in data
            assert 'results' not in data
    
    def test_export_to_csv(self, sample_benchmark_result):
        """Тест экспорта в CSV"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_results.csv")
            
            BenchmarkReporter.export_to_csv(sample_benchmark_result, filepath)
            
            # Проверяем, что файл создан
            assert os.path.exists(filepath)
            
            # Проверяем содержимое
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # Проверяем заголовки
            assert len(rows) >= 3  # заголовок + 2 строки данных
            headers = rows[0]
            assert 'test_name' in headers
            assert 'query' in headers
            assert 'overall_score' in headers
            
            # Проверяем данные
            assert rows[1][0] == 'test_1'  # test_name
            assert rows[2][0] == 'test_2'  # test_name
    
    def test_generate_html_report(self, sample_benchmark_result):
        """Тест генерации HTML отчета"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_report.html")
            
            BenchmarkReporter.generate_html_report(sample_benchmark_result, filepath)
            
            # Проверяем, что файл создан
            assert os.path.exists(filepath)
            
            # Проверяем содержимое
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert '<html' in content
            assert 'Отчет по бенчмарку LLM' in content
            assert 'test_1' in content
            assert 'test_2' in content
    
    def test_print_summary(self, sample_benchmark_result, capsys):
        """Тест вывода сводки в консоль"""
        BenchmarkReporter.print_summary(sample_benchmark_result)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Результаты бенчмарка" in output
        assert "Всего тестов: 2" in output
        assert "Средний балл: 0.800" in output


class TestDataLoaders:
    """Тесты для загрузчиков данных"""
    
    def test_load_test_cases_from_json(self):
        """Тест загрузки тестовых случаев из JSON"""
        test_data = [
            {
                "query": "Что такое Python?",
                "expected_answer": "Python - язык программирования",
                "actual_answer": "Python это язык программирования"
            },
            {
                "query": "Что такое AI?",
                "expected_answer": "AI - искусственный интеллект",
                "actual_answer": "Искусственный интеллект"
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_cases.json")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False)
            
            loaded_data = load_test_cases_from_json(filepath)
            
            assert len(loaded_data) == 2
            assert loaded_data[0]['query'] == "Что такое Python?"
            assert loaded_data[1]['query'] == "Что такое AI?"
    
    def test_load_test_cases_from_json_nonexistent_file(self):
        """Тест загрузки из несуществующего JSON файла"""
        with pytest.raises(FileNotFoundError):
            load_test_cases_from_json("nonexistent_file.json")
    
    def test_load_test_cases_from_csv(self):
        """Тест загрузки тестовых случаев из CSV"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_cases.csv")
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['query', 'expected_answer', 'actual_answer'])
                writer.writerow(['Что такое Python?', 'Python - язык программирования', 'Python это язык'])
                writer.writerow(['Что такое AI?', 'AI - искусственный интеллект', 'Искусственный интеллект'])
            
            loaded_data = load_test_cases_from_csv(filepath)
            
            assert len(loaded_data) == 2
            assert loaded_data[0]['query'] == "Что такое Python?"
            assert loaded_data[1]['query'] == "Что такое AI?"
    
    def test_load_test_cases_from_csv_nonexistent_file(self):
        """Тест загрузки из несуществующего CSV файла"""
        with pytest.raises(FileNotFoundError):
            load_test_cases_from_csv("nonexistent_file.csv") 