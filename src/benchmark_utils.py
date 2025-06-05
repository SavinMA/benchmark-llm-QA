"""
Утилиты для работы с результатами бенчмарков
"""
import json
import csv
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from .models import BenchmarkResult, BenchmarkSummary


class BenchmarkReporter:
    """Класс для создания отчетов по результатам бенчмарков"""
    
    @staticmethod
    def export_to_json(
        summary: BenchmarkSummary, 
        filepath: str,
        include_details: bool = True
    ):
        """
        Экспортирует результаты в JSON файл
        
        Args:
            summary: Сводка результатов бенчмарка
            filepath: Путь к файлу для сохранения
            include_details: Включать ли детальную информацию о каждом тесте
        """
        data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_tests': summary.total_tests,
                'average_score': summary.average_score,
                'min_score': summary.min_score,
                'max_score': summary.max_score,
                'passed_tests': summary.passed_tests,
                'failed_tests': summary.failed_tests,
                'total_execution_time': summary.total_execution_time
            }
        }
        
        if include_details:
            data['results'] = []
            for result in summary.results:
                result_dict = {
                    'test_name': result.test_name,
                    'query': result.query,
                    'expected_answer': result.expected_answer,
                    'actual_answer': result.actual_answer,
                    'score': result.score,
                    'evaluation_details': result.evaluation_details,
                    'timestamp': result.timestamp.isoformat(),
                    'execution_time_seconds': result.execution_time_seconds
                }
                data['results'].append(result_dict)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def export_to_csv(summary: BenchmarkSummary, filepath: str):
        """
        Экспортирует результаты в CSV файл
        
        Args:
            summary: Сводка результатов бенчмарка
            filepath: Путь к файлу для сохранения
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Заголовки
            headers = [
                'test_name', 'query', 'expected_answer', 'actual_answer', 
                'overall_score', 'accuracy_score', 'completeness_score', 
                'relevance_score', 'explanation', 'suggestions', 
                'timestamp', 'execution_time_seconds'
            ]
            writer.writerow(headers)
            
            # Данные
            for result in summary.results:
                scores = result.evaluation_details.get('scores', {})
                row = [
                    result.test_name,
                    result.query,
                    result.expected_answer,
                    result.actual_answer,
                    result.score,
                    scores.get('accuracy', 0),
                    scores.get('completeness', 0),
                    scores.get('relevance', 0),
                    result.evaluation_details.get('explanation', ''),
                    result.evaluation_details.get('suggestions', ''),
                    result.timestamp.isoformat(),
                    result.execution_time_seconds
                ]
                writer.writerow(row)
    
    @staticmethod
    def generate_html_report(summary: BenchmarkSummary, filepath: str):
        """
        Генерирует HTML отчет
        
        Args:
            summary: Сводка результатов бенчмарка
            filepath: Путь к файлу для сохранения
        """
        html_template = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчет по бенчмарку LLM</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; flex-wrap: wrap; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 10px; background-color: #e8f4fd; border-radius: 5px; margin: 5px; }}
        .metric h3 {{ margin: 0; color: #2c3e50; }}
        .metric p {{ margin: 5px 0; font-size: 24px; font-weight: bold; color: #3498db; }}
        .results {{ margin-top: 20px; }}
        .result-item {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .result-item.passed {{ border-left: 12px solid #27ae60; }}
        .result-item.failed {{ border-left: 12px solid #e74c3c; }}
        .query {{ font-weight: bold; margin-bottom: 10px; }}
        .answers {{ display: flex; gap: 20px; margin: 10px 0; }}
        .answer {{ flex: 1; padding: 10px; border-radius: 3px; }}
        .expected {{ background-color: #d4edda; }}
        .actual {{ background-color: #fff3cd; }}
        .scores {{ margin: 10px 0; }}
        .score-item {{ display: inline-block; margin-right: 15px; padding: 5px 10px; background-color: #f8f9fa; border-radius: 3px; }}
        .pipeline-params {{ margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
        .pipeline-params h2 {{ margin-top: 0; color: #2c3e50; }}
        .param-item {{ margin: 5px 0; }}
        .param-name {{ font-weight: bold; color: #2c3e50; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Отчет по бенчмарку LLM</h1>
        <p>Дата создания: {timestamp}</p>
    </div>
    
    <div class="pipeline-params">
        <h2>Параметры пайплайна</h2>
        {pipeline_params_html}
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>Всего тестов</h3>
            <p>{total_tests}</p>
        </div>
        <div class="metric">
            <h3>Средний балл</h3>
            <p>{average_score:.3f}</p>
        </div>
        <div class="metric">
            <h3>Пройдено</h3>
            <p>{passed_tests}</p>
        </div>
        <div class="metric">
            <h3>Провалено</h3>
            <p>{failed_tests}</p>
        </div>
        <div class="metric">
            <h3>Время бенчмарка</h3>
            <p>{total_execution_time:.2f}с</p>
        </div>
        <div class="metric">
            <h3>Время заполнения базы данных</h3>
            <p>{total_time_fill_db:.2f}с</p>
        </div>
        <div class="metric">
            <h3>Время анализа</h3>
            <p>{total_time_analyze:.2f}с</p>
        </div>
    </div>
    
    <div class="results">
        <h2>Детальные результаты</h2>
        {results_html}
    </div>
</body>
</html>
        """
        
        # Generate pipeline parameters HTML
        pipeline_params_html = ""
        for param_name, param_value in summary.pipeline_params.items():
            pipeline_params_html += f"""
            <div class="param-item">
                <span class="param-name">{param_name}:</span> {param_value}
            </div>
            """
        
        results_html = ""
        for i, result in enumerate(summary.results, 1):
            status_class = "passed" if result.score >= 0.7 else "failed"
            scores = result.evaluation_details.get('scores', {})
            
            scores_html = ""
            for criterion, score in scores.items():
                scores_html += f'<span class="score-item">{criterion}: {score:.3f}</span>'
            
            result_html = f"""
            <div class="result-item {status_class}">
                <div class="query">Вопрос {i}: {result.query}</div>
                <div class="answers">
                    <div class="answer expected">
                        <strong>Ожидаемый ответ:</strong><br>
                        {result.expected_answer}
                    </div>
                    <div class="answer actual">
                        <strong>Фактический ответ:</strong><br>
                        {result.actual_answer}
                    </div>
                </div>
                <div class="scores">
                    <strong>Общий балл:</strong> {result.score:.3f}<br>
                    {scores_html}
                </div>
                <div style="margin-top: 10px;">
                    <strong>Объяснение:</strong> {result.evaluation_details.get('explanation', '')}
                </div>
                <div style="margin-top: 5px; font-style: italic;">
                    <strong>Рекомендации:</strong> {result.evaluation_details.get('suggestions', '')}
                </div>
            </div>
            """
            results_html += result_html
        
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_tests=summary.total_tests,
            average_score=summary.average_score,
            passed_tests=summary.passed_tests,
            failed_tests=summary.failed_tests,
            total_execution_time=summary.total_execution_time,
            total_time_fill_db=summary.total_time_fill_db,
            total_time_analyze=summary.total_time_analyze,
            pipeline_params_html=pipeline_params_html,
            results_html=results_html
        )
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    @staticmethod
    def print_summary(summary: BenchmarkSummary):
        """
        Выводит краткую сводку в консоль
        
        Args:
            summary: Сводка результатов бенчмарка
        """
        print("=" * 60)
        print("СВОДКА РЕЗУЛЬТАТОВ БЕНЧМАРКА")
        print("=" * 60)
        print(f"Всего тестов: {summary.total_tests}")
        print(f"Пройдено: {summary.passed_tests}")
        print(f"Провалено: {summary.failed_tests}")
        print(f"Процент успеха: {(summary.passed_tests / summary.total_tests * 100):.1f}%")
        print(f"Средний балл: {summary.average_score:.3f}")
        print(f"Минимальный балл: {summary.min_score:.3f}")
        print(f"Максимальный балл: {summary.max_score:.3f}")
        print(f"Время работы бенчмарка: {summary.total_execution_time:.2f} секунд")
        print(f"Время заполнения базы данных: {summary.total_time_fill_db:.2f} секунд")
        print(f"Время анализа: {summary.total_time_analyze:.2f} секунд")
        print("=" * 60)
        
        # Показываем несколько лучших и худших результатов
        sorted_results = sorted(summary.results, key=lambda x: x.score, reverse=True)
        
        print("\nЛУЧШИЕ РЕЗУЛЬТАТЫ:")
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"{i}. Балл: {result.score:.3f} - {result.query[:50]}...")
        
        print("\nХУДШИЕ РЕЗУЛЬТАТЫ:")
        for i, result in enumerate(sorted_results[-3:], 1):
            print(f"{i}. Балл: {result.score:.3f} - {result.query[:50]}...")


def load_test_cases_from_json(filepath: str) -> List[Dict[str, str]]:
    """
    Загружает тестовые случаи из JSON файла
    
    Args:
        filepath: Путь к JSON файлу
        
    Returns:
        Список тестовых случаев
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ожидаем формат: [{"query": "...", "expected_answer": "...", "actual_answer": "..."}]
    return data


def load_test_cases_from_csv(filepath: str) -> List[Dict[str, str]]:
    """
    Загружает тестовые случаи из CSV файла
    
    Args:
        filepath: Путь к CSV файлу
        
    Returns:
        Список тестовых случаев
    """
    df = pd.read_csv(filepath)
    required_columns = ['query', 'expected_answer', 'actual_answer']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Отсутствует обязательная колонка: {col}")
    
    return df[required_columns].to_dict('records') 