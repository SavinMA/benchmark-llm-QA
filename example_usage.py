"""
Пример использования системы бенчмарков для оценки ответов LLM
"""
import asyncio
import os
from typing import List, Dict

from src import BenchmarkFactory, BenchmarkType, BenchmarkReporter


# Примеры тестовых данных
SAMPLE_TEST_CASES = [
    {
        "query": "Кто является покупателем в данном тендере?",
        "expected_answer": "ПАО Метафракс",
        "actual_answer": "Покупателем является ПАО Метафракс"
    },
    {
        "query": "Какой срок поставки указан в документе?",
        "expected_answer": "10 рабочих дней",
        "actual_answer": "Срок поставки составляет 10 дней"
    },
    {
        "query": "Какая цена указана в тендере?",
        "expected_answer": "1 000 000 рублей",
        "actual_answer": "Цена составляет 1000000 рублей"
    },
    {
        "query": "Какой способ оплаты предусмотрен?",
        "expected_answer": "Безналичный расчет",
        "actual_answer": "Оплата безналичным расчетом"
    },
    {
        "query": "Как будет осуществляться доставка?",
        "expected_answer": "Самовывоз",
        "actual_answer": "Доставка будет осуществляться силами поставщика"
    }
]


async def run_benchmark_example():
    """
    Пример запуска бенчмарка
    """
    # Конфигурация бенчмарка
    config = {
        'model': 'llama2',  # Убедитесь, что эта модель установлена в Ollama
        'host': 'localhost:11434',
        'passing_score': 0.7,
        'evaluation_criteria': {
            'accuracy': 0.4,      # Точность ответа - 40%
            'completeness': 0.3,  # Полнота ответа - 30%
            'relevance': 0.3      # Релевантность - 30%
        }
    }
    
    # Создаем бенчмарк
    benchmark = BenchmarkFactory.create_benchmark(
        BenchmarkType.LLM_ANSWER_EVALUATION,
        config
    )
    
    print("Проверяем подключение к Ollama...")
    connection_info = benchmark.check_ollama_connection()
    print(f"Статус подключения: {connection_info}")
    
    if not connection_info['connected']:
        print("❌ Ошибка: Не удалось подключиться к Ollama")
        print("Убедитесь, что Ollama запущена и доступна по адресу localhost:11434")
        return
    
    if not connection_info['model_available']:
        print(f"❌ Ошибка: Модель {config['model']} недоступна")
        print(f"Доступные модели: {connection_info['available_models']}")
        return
    
    print("✅ Подключение к Ollama успешно!")
    print(f"Используемая модель: {config['model']}")
    
    # Запускаем бенчмарк
    print("\nЗапускаем бенчмарк...")
    summary = await benchmark.run_benchmark(SAMPLE_TEST_CASES)
    
    # Выводим результаты в консоль
    BenchmarkReporter.print_summary(summary)
    
    # Создаем папку для отчетов
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Экспортируем результаты в различные форматы
    print(f"\nСохраняем отчеты в папку {reports_dir}/...")
    
    # JSON отчет
    BenchmarkReporter.export_to_json(
        summary,
        f"{reports_dir}/benchmark_results.json"
    )
    print("✅ JSON отчет сохранен")
    
    # CSV отчет
    BenchmarkReporter.export_to_csv(
        summary,
        f"{reports_dir}/benchmark_results.csv"
    )
    print("✅ CSV отчет сохранен")
    
    # HTML отчет
    BenchmarkReporter.generate_html_report(
        summary,
        f"{reports_dir}/benchmark_report.html"
    )
    print("✅ HTML отчет сохранен")
    
    print(f"\nВсе отчеты сохранены в папке '{reports_dir}/'")
    print("Откройте benchmark_report.html в браузере для просмотра детального отчета")


async def run_custom_benchmark(test_cases: List[Dict[str, str]], config: Dict = None):
    """
    Запускает бенчмарк с пользовательскими данными
    
    Args:
        test_cases: Список тестовых случаев
        config: Пользовательская конфигурация (опционально)
    """
    # Используем конфигурацию по умолчанию, если не передана пользовательская
    if config is None:
        config = {
            'model': 'llama2',
            'host': 'localhost:11434',
            'passing_score': 0.7,
            'evaluation_criteria': {
                'accuracy': 0.4,
                'completeness': 0.3,
                'relevance': 0.3
            }
        }
    
    # Создаем и запускаем бенчмарк
    benchmark = BenchmarkFactory.create_benchmark(
        BenchmarkType.LLM_ANSWER_EVALUATION,
        config
    )
    
    # Проверяем подключение
    connection_info = benchmark.check_ollama_connection()
    if not connection_info['connected']:
        raise ConnectionError(f"Не удалось подключиться к Ollama: {connection_info.get('error', 'Неизвестная ошибка')}")
    
    # Запускаем бенчмарк
    summary = await benchmark.run_benchmark(test_cases)
    return summary


def create_sample_test_file():
    """
    Создает файл с примерами тестовых данных
    """
    import json
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    sample_file = f"{data_dir}/sample_test_cases.json"
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(SAMPLE_TEST_CASES, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Файл с примерами тестовых данных создан: {sample_file}")
    return sample_file


if __name__ == "__main__":
    print("=== Пример использования системы бенчмарков ===")
    
    # Создаем файл с примерами
    sample_file = create_sample_test_file()
    
    # Запускаем бенчмарк
    asyncio.run(run_benchmark_example()) 