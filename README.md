# LLM Benchmark

Библиотека для бенчмаркинга и оценки качества ответов больших языковых моделей (LLM).

## Описание

LLM Benchmark - это Python библиотека, предназначенная для автоматизированной оценки качества ответов языковых моделей. Библиотека поддерживает различные типы оценщиков и может работать с локальными моделями через Ollama, а также с облачными сервисами, такими как Mistral AI.

## Особенности

- 🎯 **Гибкая система оценки** - настраиваемые критерии оценки (точность, полнота, релевантность)
- 🔧 **Множественные оценщики** - поддержка Ollama и Mistral AI
- 📊 **Подробная отчетность** - экспорт в JSON, CSV и HTML форматы
- ⚡ **Асинхронная обработка** - быстрая обработка больших объемов данных
- 🏭 **Фабричный паттерн** - легкое создание различных типов бенчмарков

## Установка

### Из исходного кода

```bash
git clone https://github.com/yourusername/llm-benchmark.git
cd llm-benchmark
pip install -r requirements.txt
```

### Установка зависимостей

```bash
pip install -r requirements.txt
```

## Быстрый старт

### Основное использование

```python
import asyncio
from src import BenchmarkFactory, BenchmarkType

# Тестовые данные
test_cases = [
    {
        "query": "Что такое машинное обучение?",
        "expected_answer": "Машинное обучение - это раздел искусственного интеллекта",
        "actual_answer": "Machine Learning это подраздел ИИ, который позволяет компьютерам учиться"
    }
]

# Конфигурация
config = {
    'model': 'llama2',
    'host': 'localhost:11434',
    'passing_score': 0.7
}

async def run_benchmark():
    # Создание бенчмарка
    benchmark = BenchmarkFactory.create_benchmark(
        BenchmarkType.LLM_ANSWER_EVALUATION,
        config
    )
    
    # Запуск оценки
    summary = await benchmark.run_benchmark(test_cases)
    
    # Вывод результатов
    print(f"Средний балл: {summary.average_score:.3f}")
    print(f"Пройдено тестов: {summary.passed_tests}/{summary.total_tests}")

# Запуск
asyncio.run(run_benchmark())
```

## Поддерживаемые оценщики

### 1. Ollama (локальные модели)

```python
config = {
    'model': 'llama2',  # или другая модель
    'host': 'localhost:11434',
    'passing_score': 0.7,
    'evaluation_criteria': {
        'accuracy': 0.4,      # Точность - 40%
        'completeness': 0.3,  # Полнота - 30%
        'relevance': 0.3      # Релевантность - 30%
    }
}
```

**Требования:**
- Установленная и запущенная Ollama
- Загруженная модель (например, `ollama pull llama2`)

### 2. Mistral AI

```python
config = {
    'model': 'mistral-large-latest',
    'api_key': 'your-mistral-api-key',
    'passing_score': 0.7,
    'timeout': 60
}
```

**Требования:**
- API ключ от Mistral AI
- Интернет соединение

## Структура данных

### Входные данные

```python
test_case = {
    "query": "Вопрос для оценки",
    "expected_answer": "Ожидаемый правильный ответ",
    "actual_answer": "Фактический ответ от LLM"
}
```

### Результат оценки

```python
{
    "test_name": "eval_123456",
    "query": "Исходный вопрос",
    "expected_answer": "Ожидаемый ответ",
    "actual_answer": "Фактический ответ",
    "score": 0.85,
    "evaluation_details": {
        "scores": {
            "accuracy": 0.9,
            "completeness": 0.8,
            "relevance": 0.85
        },
        "explanation": "Подробное объяснение оценки",
        "suggestions": "Рекомендации по улучшению"
    },
    "timestamp": "2024-01-01T12:00:00",
    "execution_time_seconds": 2.5
}
```

## Экспорт результатов

### JSON отчет

```python
from src import BenchmarkReporter

BenchmarkReporter.export_to_json(
    summary, 
    "reports/results.json"
)
```

### CSV отчет

```python
BenchmarkReporter.export_to_csv(
    summary, 
    "reports/results.csv"  
)
```

### HTML отчет

```python
BenchmarkReporter.generate_html_report(
    summary,
    "reports/report.html"
)
```

## Примеры использования

Подробные примеры использования смотрите в файле `example_usage.py`.

## Разработка

### Структура проекта

```
llm-benchmark/
├── src/
│   ├── __init__.py
│   ├── models.py              # Модели данных
│   ├── benchmark.py           # Базовые классы
│   ├── ollama_benchmark.py    # Оценщик через Ollama
│   ├── mistral_benchmark.py   # Оценщик через Mistral AI
│   ├── benchmark_utils.py     # Утилиты и репортер
│   └── factory.py             # Фабрика бенчмарков
├── tests/                     # Тесты
├── requirements.txt           # Зависимости
├── pyproject.toml            # Конфигурация пакета
└── README.md                 # Документация
```

### Запуск тестов

```bash
python -m pytest tests/
```

### Установка в режиме разработки

```bash
pip install -e .
```

## Требования

- Python 3.8+
- Pydantic 2.0+
- Pandas 1.3+
- Ollama (для локальных моделей)
- aiohttp 3.8+ (для Mistral AI)

## Лицензия

MIT License. См. файл `LICENSE` для подробностей.

## Поддержка

Если у вас есть вопросы или предложения, создайте issue в репозитории.

---

**Важно:** Убедитесь, что у вас установлена и запущена Ollama, если вы планируете использовать локальные модели, или получен API ключ от Mistral AI для облачных оценок. 