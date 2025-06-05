"""
Оценщик ответов с использованием Mistral AI API
"""
import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime
import json
import aiohttp
import random

from .models import QueryAnswerWithActualAnswer
from .benchmark import LLMBenchmark, BenchmarkResult, BenchmarkSummary


class MistralBenchmark(LLMBenchmark):
    """Оценщик ответов на основе Mistral AI API"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация оценщика
        
        Args:
            config: Конфигурация с параметрами:
                - model: название модели Mistral (по умолчанию 'mistral-large-latest')
                - api_key: API ключ для Mistral AI
                - api_url: URL API (по умолчанию 'https://api.mistral.ai/v1/chat/completions')
                - passing_score: минимальный балл для прохождения (по умолчанию 0.7)
                - evaluation_criteria: критерии оценки
                - timeout: таймаут запроса в секундах (по умолчанию 60)
        """
        super().__init__(config)
        self.model = config.get('model', 'mistral-large-latest')
        self.api_key = config.get('api_key')
        self.api_url = config.get('api_url', 'https://api.mistral.ai/v1/chat/completions')
        self.passing_score = config.get('passing_score', 0.7)
        self.timeout = config.get('timeout', 60)
        self.evaluation_criteria = config.get('evaluation_criteria', {
            'accuracy': 0.4,      # Точность ответа
            'completeness': 0.3,  # Полнота ответа
            'relevance': 0.3      # Релевантность
        })
        
        if not self.api_key:
            raise ValueError("API ключ для Mistral AI обязателен. Укажите 'api_key' в конфигурации.")
    
    def _create_evaluation_prompt(
        self, 
        query_answer_with_expected_answer: QueryAnswerWithActualAnswer
    ) -> str:
        """
        Создает промпт для оценки ответа
        
        Args:
            query_answer_with_expected_answer: Объект с вопросом и ответами
            
        Returns:
            Промпт для LLM
        """
        criteria_text = "\n".join([
            f"- {criterion}: {weight*100}% от общей оценки" 
            for criterion, weight in self.evaluation_criteria.items()
        ])
        
        prompt = f"""
Ты - эксперт по оценке качества ответов нейронных сетей. 
Твоя задача - сравнить фактический ответ с ожидаемым и дать объективную оценку.

ВОПРОС: {query_answer_with_expected_answer.query}

ОЖИДАЕМЫЙ ОТВЕТ: {query_answer_with_expected_answer.expected_answer}

ФАКТИЧЕСКИЙ ОТВЕТ: {query_answer_with_expected_answer.actual_answer}

КРИТЕРИИ ОЦЕНКИ:
{criteria_text}

ИНСТРУКЦИЯ:
1. Проанализируй фактический ответ по каждому критерию
2. Дай оценку от 0 до 1 для каждого критерия
3. Рассчитай общую оценку как взвешенную сумму
4. Объясни свою оценку

ФОРМАТ ОТВЕТА (JSON):
{{
    "scores": {{
        "accuracy": 0.0-1.0,
        "completeness": 0.0-1.0,
        "relevance": 0.0-1.0
    }},
    "overall_score": 0.0-1.0,
    "explanation": "подробное объяснение оценки",
    "suggestions": "рекомендации по улучшению ответа"
}}

Отвечай только JSON, никакого дополнительного текста. Проверь JSON на корректность.
"""
        return prompt
    
    async def _make_api_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Выполняет запрос к Mistral AI API
        
        Args:
            messages: Список сообщений для отправки
            
        Returns:
            Ответ от API
            
        Raises:
            Exception: При ошибке запроса
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': 0.1,  # Низкая температура для стабильных оценок
            'max_tokens': 1000,
            'top_p': 0.9
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Mistral API error {response.status}: {error_text}")
                
                return await response.json()
    
    def _extract_json_from_response(self, response: str) -> str:
        """
        Извлекает JSON из ответа LLM, убирая markdown блоки
        
        Args:
            response: Ответ от LLM
            
        Returns:
            Очищенный JSON текст
        """
        response = response.strip()
        
        # Если ответ содержит markdown код блоки, извлекаем содержимое
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            response = response[start:end].strip()
        elif '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            response = response[start:end].strip()
        
        return response
    
    def _validate_evaluation_structure(self, evaluation_data: Dict[str, Any]) -> None:
        """
        Проверяет наличие обязательных полей в структуре оценки
        
        Args:
            evaluation_data: Словарь с данными оценки
            
        Raises:
            ValueError: Если отсутствуют обязательные поля
        """
        required_fields = ['scores', 'overall_score', 'explanation']
        for field in required_fields:
            if field not in evaluation_data:
                raise ValueError(f"Отсутствует обязательное поле: {field}")
    
    def _normalize_scores(self, evaluation_data: Dict[str, Any]) -> None:
        """
        Нормализует оценки критериев и пересчитывает общую оценку
        
        Args:
            evaluation_data: Словарь с данными оценки (модифицируется in-place)
        """
        scores = evaluation_data['scores']
        
        # Проверяем корректность оценок
        for criterion in self.evaluation_criteria.keys():
            if criterion not in scores:
                scores[criterion] = 0.0
            else:
                scores[criterion] = max(0.0, min(1.0, float(scores[criterion])))
        
        # Пересчитываем общую оценку на основе весов
        overall_score = sum(
            scores[criterion] * weight 
            for criterion, weight in self.evaluation_criteria.items()
        )
        evaluation_data['overall_score'] = overall_score
    
    def _create_fallback_evaluation(self, error_message: str) -> Dict[str, Any]:
        """
        Создает базовую структуру оценки в случае ошибки
        
        Args:
            error_message: Сообщение об ошибке
            
        Returns:
            Базовая структура оценки
        """
        return {
            'scores': {k: 0.0 for k in self.evaluation_criteria.keys()},
            'overall_score': 0.0,
            'explanation': error_message,
            'suggestions': 'Проверьте API ключ, настройки модели и повторите попытку'
        }
    
    def _parse_and_validate_json(self, json_text: str) -> Dict[str, Any]:
        """
        Парсит JSON и валидирует его структуру
        
        Args:
            json_text: JSON текст для парсинга
            
        Returns:
            Валидированные данные оценки
            
        Raises:
            json.JSONDecodeError: Если JSON некорректен
            ValueError: Если структура JSON некорректна
        """
        # Извлекаем чистый JSON
        clean_json = self._extract_json_from_response(json_text)
        
        # Парсим JSON
        evaluation_data = json.loads(clean_json)
        
        # Валидируем структуру
        self._validate_evaluation_structure(evaluation_data)
        
        # Нормализуем оценки
        self._normalize_scores(evaluation_data)
        
        return evaluation_data
    
    async def _fix_json_recursively(self, broken_json: str, error_str: str, max_attempts: int = 3, current_attempt: int = 1) -> Dict[str, Any]:
        """
        Рекурсивно исправляет некорректный JSON с помощью Mistral AI
        
        Args:
            broken_json: Некорректный JSON текст
            error_str: Описание ошибки
            max_attempts: Максимальное количество попыток исправления
            current_attempt: Текущая попытка
            
        Returns:
            Словарь с данными оценки или базовая структура при неудаче
        """
        if current_attempt > max_attempts:
            print(f"❌ Исчерпано максимальное количество попыток исправления JSON ({max_attempts})")
            return self._create_fallback_evaluation(f'Не удалось исправить JSON за {max_attempts} попыток')
        
        print(f"🔧 Попытка {current_attempt}/{max_attempts} исправления JSON...")
        
        # Формируем промпт для исправления JSON
        fix_prompt = f"""
Исправь следующий JSON, чтобы он был валидным. 
Сохрани оригинальную структуру и данные.
Ответ должен содержать ТОЛЬКО исправленный JSON без дополнительного текста, комментариев или объяснений.
Ошибка: '{error_str}'

Требуемая структура:
{{
    "scores": {{
        "accuracy": число от 0.0 до 1.0,
        "completeness": число от 0.0 до 1.0,
        "relevance": число от 0.0 до 1.0
    }},
    "overall_score": число от 0.0 до 1.0,
    "explanation": "текст объяснения",
    "suggestions": "текст рекомендаций"
}}

Исходный JSON для исправления:
{broken_json}
"""
        
        try:
            await asyncio.sleep(random.uniform(1, 2))
            # Делаем запрос к Mistral AI для исправления JSON
            messages = [{'role': 'user', 'content': fix_prompt}]
            response = await self._make_api_request(messages)
            
            # Извлекаем исправленный JSON из ответа
            fixed_json = response['choices'][0]['message']['content']
            
            # Пытаемся распарсить исправленный JSON
            try:
                evaluation_data = self._parse_and_validate_json(fixed_json)
                print(f"✅ JSON успешно исправлен на попытке {current_attempt}")
                return evaluation_data
                
            except json.JSONDecodeError as nested_error:
                print(f"❌ JSON всё ещё некорректен после попытки {current_attempt}: {str(nested_error)}")
                error_str = str(nested_error)
                # Рекурсивно пытаемся исправить снова
                return await self._fix_json_recursively(fixed_json, error_str, max_attempts, current_attempt + 1)
                
            except (ValueError, KeyError) as validation_error:
                print(f"❌ JSON корректен, но не хватает полей на попытке {current_attempt}: {str(validation_error)}")
                # Рекурсивно пытаемся исправить снова
                return await self._fix_json_recursively(fixed_json, error_str, max_attempts, current_attempt + 1)
                
        except Exception as api_error:
            print(f"❌ Ошибка при обращении к Mistral AI на попытке {current_attempt}: {str(api_error)}")
            # Рекурсивно пытаемся исправить снова
            return await self._fix_json_recursively(broken_json, error_str, max_attempts, current_attempt + 1)
    
    async def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """
        Парсит ответ Mistral AI и извлекает JSON с оценкой
        
        Args:
            response: Ответ от Mistral AI
            
        Returns:
            Словарь с данными оценки
        """
        try:
            return self._parse_and_validate_json(response)

        except json.JSONDecodeError as jsonError:
            print(f"❌ Ошибка парсинга JSON: {str(jsonError)}")
            print("🔄 Пытаюсь исправить JSON через Mistral AI...")
            error_str = str(jsonError)
            
            # Используем рекурсивный метод для исправления JSON
            return await self._fix_json_recursively(response, error_str, max_attempts=3, current_attempt=1)

        except (ValueError, KeyError) as e:
            # Если не удалось распарсить, возвращаем базовую структуру
            return self._create_fallback_evaluation(f'Не удалось распарсить ответ Mistral AI: {str(e)}')
    
    async def evaluate_single(
        self, 
        query_answer_with_expected_answer: QueryAnswerWithActualAnswer
    ) -> BenchmarkResult:
        """
        Оценивает один ответ с помощью Mistral AI
        
        Args:
            query_answer_with_expected_answer: Объект с вопросом и ответами
            
        Returns:
            Результат оценки
        """
        start_time = time.time()
        
        try:
            # Создаем промпт для оценки
            prompt = self._create_evaluation_prompt(query_answer_with_expected_answer)
            
            # Отправляем запрос к Mistral AI
            messages = [{'role': 'user', 'content': prompt}]
            response = await self._make_api_request(messages)
            
            # Парсим ответ
            llm_response = response['choices'][0]['message']['content']
            
            # Если ответ пустой или слишком короткий, возвращаем базовую структуру
            if not llm_response or len(llm_response) < 20:
                evaluation_data = {
                    'scores': {k: 0.0 for k in self.evaluation_criteria.keys()},
                    'overall_score': 0.0,
                    'explanation': 'Получен пустой или слишком короткий ответ от Mistral AI',
                    'suggestions': 'Проверьте настройки модели и повторите попытку'
                }
            else:
                # Пытаемся извлечь JSON из ответа
                evaluation_data = await self._parse_evaluation_response(llm_response)
            
            execution_time = time.time() - start_time

            query = query_answer_with_expected_answer.query
            expected_answer = query_answer_with_expected_answer.expected_answer
            actual_answer = query_answer_with_expected_answer.actual_answer
            
            return BenchmarkResult(
                test_name=f"eval_{hash(query)}",
                query=query,
                expected_answer=expected_answer,
                actual_answer=actual_answer,
                score=evaluation_data['overall_score'],
                evaluation_details=evaluation_data,
                timestamp=datetime.now(),
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            query = query_answer_with_expected_answer.query
            expected_answer = query_answer_with_expected_answer.expected_answer
            actual_answer = query_answer_with_expected_answer.actual_answer
            
            # В случае ошибки возвращаем результат с нулевой оценкой
            return BenchmarkResult(
                test_name=f"eval_{hash(query)}",
                query=query,
                expected_answer=expected_answer,
                actual_answer=actual_answer,
                score=0.0,
                evaluation_details={
                    'error': str(e),
                    'scores': {k: 0.0 for k in self.evaluation_criteria.keys()},
                    'overall_score': 0.0,
                    'explanation': f'Ошибка при оценке: {str(e)}',
                    'suggestions': 'Проверьте API ключ и настройки Mistral AI'
                },
                timestamp=datetime.now(),
                execution_time_seconds=execution_time
            )
    
    async def run_benchmark(
        self, 
        test_cases: List[QueryAnswerWithActualAnswer]
    ) -> BenchmarkSummary:
        """
        Запускает полный бенчмарк для всех тестовых случаев
        
        Args:
            test_cases: Список тестовых случаев
                
        Returns:
            Сводка результатов бенчмарка
        """
        start_time = time.time()
        self.clear_results()
        
        # Проверяем API ключ
        if not self.api_key:
            raise ValueError("API ключ для Mistral AI не настроен")
        
        results = []
        # Выполняем оценку для каждого тестового случая
        for test_case in test_cases:
            # Добавляем случайную задержку от 1 до 3 секунд между задачами
            await asyncio.sleep(random.uniform(1, 3))
            benchmark_result = await self.evaluate_single(test_case)
            results.append(benchmark_result)
               
        # Обрабатываем результаты
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                # Создаем результат с ошибкой
                error_result = BenchmarkResult(
                    test_name="error",
                    query="",
                    expected_answer="",
                    actual_answer="",
                    score=0.0,
                    evaluation_details={'error': str(result)},
                    timestamp=datetime.now(),
                    execution_time_seconds=0.0
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)
        
        self.results = valid_results
        
        # Подсчитываем статистику
        total_time = time.time() - start_time
        scores = [r.score for r in valid_results]
        
        summary = BenchmarkSummary(
            total_tests=len(valid_results),
            average_score=sum(scores) / len(scores) if scores else 0.0,
            min_score=min(scores) if scores else 0.0,
            max_score=max(scores) if scores else 0.0,
            passed_tests=sum(1 for score in scores if score >= self.passing_score),
            failed_tests=sum(1 for score in scores if score < self.passing_score),
            total_execution_time=total_time,
            results=valid_results
        )
        
        return summary
    
    def check_mistral_connection(self) -> Dict[str, Any]:
        """
        Проверяет подключение к Mistral AI и возвращает информацию о настройках
        
        Returns:
            Информация о подключении и настройках
        """
        try:
            if not self.api_key:
                return {
                    'connected': False,
                    'api_url': self.api_url,
                    'error': 'API ключ не настроен',
                    'selected_model': self.model,
                    'model_available': False
                }
            
            # Здесь можно добавить тестовый запрос к API для проверки подключения
            # Но это требует дополнительного API вызова, поэтому пока просто проверяем ключ
            
            return {
                'connected': True,
                'api_url': self.api_url,
                'selected_model': self.model,
                'model_available': True,
                'api_key_length': len(self.api_key) if self.api_key else 0
            }
            
        except Exception as e:
            return {
                'connected': False,
                'api_url': self.api_url,
                'error': str(e),
                'selected_model': self.model,
                'model_available': False
            } 