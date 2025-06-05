from sentence_transformers import SentenceTransformer, util
import torch
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from src.benchmark import LLMBenchmark
from src.models import Query, QueryAnswerWithActualAnswer, BenchmarkResult, BenchmarkSummary


class RelevanceModel(LLMBenchmark):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = SentenceTransformer(config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'))
        self.threshold = config.get('threshold', 0.5)

    def calculate_relevance(self, text1: str, text2: str) -> float:
        """
        Calculates the relevance score (cosine similarity) between two texts.
        """
        embedding1 = self.model.encode(text1, convert_to_tensor=True)
        embedding2 = self.model.encode(text2, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
        return cosine_similarity.item()

    def is_suitable(self, relevance_score: float) -> bool:
        """
        Determines if a response is suitable based on a relevance threshold.
        """
        return relevance_score >= self.threshold

    async def evaluate_single(
        self, 
        query_answer_with_expected_answer: QueryAnswerWithActualAnswer
    ) -> BenchmarkResult:
        query_text = query_answer_with_expected_answer.query
        actual_answer_text = query_answer_with_expected_answer.actual_answer
        expected_answer_text = query_answer_with_expected_answer.expected_answer

        # For relevance, we compare the actual answer to the expected answer or query
        # Let's compare actual_answer to expected_answer for direct relevance assessment
        # Or, if expected_answer is not always available or sufficient, compare actual_answer to query
        # For this model, I'll calculate relevance based on actual_answer vs expected_answer
        # and suitability based on actual_answer vs query.

        # Calculate relevance of actual_answer to expected_answer
        relevance_score_to_expected = self.calculate_relevance(actual_answer_text, expected_answer_text)

        # Calculate suitability of actual_answer to query
        suitability_score_to_query = self.calculate_relevance(actual_answer_text, query_text)
        suitable = self.is_suitable(suitability_score_to_query)

        # You can define your own scoring logic here.
        # For simplicity, let's use relevance_score_to_expected as the primary score.
        # The 'evaluation_details' can contain more granular information.
        score = relevance_score_to_expected # Using relevance to expected answer as the main score

        return BenchmarkResult(
            test_name="Relevance and Suitability Test",
            query=query_text,
            expected_answer=expected_answer_text,
            actual_answer=actual_answer_text,
            score=score,
            evaluation_details={
                "relevance_to_expected_answer": relevance_score_to_expected,
                "suitability_to_query": suitability_score_to_query,
                "is_suitable": suitable
            },
            execution_time_seconds=0.0 # Placeholder, as this model is fast
        )

    async def run_benchmark(
        self, 
        test_cases: List[QueryAnswerWithActualAnswer]
    ) -> BenchmarkSummary:
        self.clear_results() # Clear previous results
        
        total_score = 0.0
        passed_tests = 0
        
        for test_case in test_cases:
            # The `actual_answer` is expected to be provided within the test_case
            # as this model evaluates pre-generated LLM responses.
            
            query_answer_with_actual = QueryAnswerWithActualAnswer(
                query=test_case.query,
                expected_answer=test_case.expected_answer,
                actual_answer=test_case.actual_answer # Use the actual LLM response
            )
            
            result = await self.evaluate_single(query_answer_with_actual)
            self.results.append(result)
            total_score += result.score
            if result.evaluation_details.get("is_suitable", False):
                passed_tests += 1
        
        total_tests = len(self.results)
        average_score = total_score / total_tests if total_tests > 0 else 0.0
        min_score = min([r.score for r in self.results]) if total_tests > 0 else 0.0
        max_score = max([r.score for r in self.results]) if total_tests > 0 else 0.0

        return BenchmarkSummary(
            total_tests=total_tests,
            average_score=average_score,
            min_score=min_score,
            max_score=max_score,
            passed_tests=passed_tests,
            failed_tests=total_tests - passed_tests,
            total_execution_time=0.0, # Placeholder
            results=self.results
        )


# Example of how this would be used with BenchmarkFactory
# (This part is for illustration and not part of the class file itself)
# You would typically register this model with the BenchmarkFactory
# and then use the factory to create and run the benchmark.

# from src.factory import BenchmarkFactory, BenchmarkType

# if __name__ == "__main__":
#     # You would register your custom benchmark type first
#     # For example, in src/factory.py or a dedicated registration file
#     # BenchmarkFactory.register_benchmark_type("relevance_benchmark", RelevanceModel)

#     config = {"model_name": "sentence-transformers/all-MiniLM-L6-v2", "threshold": 0.7}
#     relevance_benchmark_instance = RelevanceModel(config)

#     test_cases_for_relevance = [
#         Query(query="What is the capital of France?", expected_answer="Paris"),
#         Query(query="Tell me about Python programming.", expected_answer="Python is a high-level, interpreted programming language..."),
#         # Add more test cases
#     ]

#     # In a real scenario, test_cases_for_relevance would be QueryAnswerWithActualAnswer
#     # where actual_answer is the response from an LLM.

#     async def main():
#         # This section mocks the actual_answer for demonstration purposes.
#         # In a real benchmark, `run_benchmark` would be called with `List[Query]`
#         # and the LLM would generate `actual_answer` internally or via a separate step.
#         # For this specific model, `evaluate_single` takes `QueryAnswerWithActualAnswer`.
#         # So, `run_benchmark`'s test_cases would effectively need to be enhanced with `actual_answer`
#         # before calling `evaluate_single`.
#         # The current implementation assumes actual_answer is mocked or provided.
#         summary = await relevance_benchmark_instance.run_benchmark(test_cases_for_relevance)
#         print("\nBenchmark Summary:")
#         print(f"Total Tests: {summary.total_tests}")
#         print(f"Average Score: {summary.average_score:.4f}")
#         print(f"Passed Tests (suitable): {summary.passed_tests}")
#         print(f"Failed Tests (not suitable): {summary.failed_tests}")
#         print("\nDetailed Results:")
#         for res in summary.results:
#             print(f"  Query: {res.query[:50]}...")
#             print(f"  Actual Answer: {res.actual_answer[:50]}...")
#             print(f"  Score: {res.score:.4f}, Suitable: {res.evaluation_details.get("is_suitable")}")
#             print(f"  Relevance to Expected: {res.evaluation_details.get("relevance_to_expected_answer"):.4f}")
#             print(f"  Suitability to Query: {res.evaluation_details.get("suitability_to_query"):.4f}\n")

#     # asyncio.run(main())