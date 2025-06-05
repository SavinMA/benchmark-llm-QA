from sentence_transformers import SentenceTransformer, util
import torch

class RelevanceModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def calculate_relevance(self, query: str, response: str) -> float:
        """
        Calculates the relevance score between a query and a response.
        The score is a cosine similarity between their embeddings, ranging from -1 to 1.
        Higher values indicate higher relevance.
        """
        # Encode the query and response into embeddings
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        response_embedding = self.model.encode(response, convert_to_tensor=True)

        # Calculate cosine similarity between the embeddings
        cosine_similarity = util.pytorch_cos_sim(query_embedding, response_embedding)

        # Return the similarity score as a float
        return cosine_similarity.item()

    def is_suitable(self, query: str, response: str, threshold: float = 0.5) -> bool:
        """
        Determines if a response is suitable based on a relevance threshold.
        """
        relevance_score = self.calculate_relevance(query, response)
        return relevance_score >= threshold

if __name__ == "__main__":
    # Example Usage
    model = RelevanceModel()

    query1 = "What is the capital of France?"
    response1 = "The capital of France is Paris."
    response2 = "The weather is nice today."

    print(f"Query: {query1}")
    print(f"Response: {response1}")
    relevance_score1 = model.calculate_relevance(query1, response1)
    print(f"Relevance Score: {relevance_score1:.4f}")
    print(f"Is Suitable (threshold=0.5): {model.is_suitable(query1, response1, threshold=0.5)}\n")

    print(f"Query: {query1}")
    print(f"Response: {response2}")
    relevance_score2 = model.calculate_relevance(query1, response2)
    print(f"Relevance Score: {relevance_score2:.4f}")
    print(f"Is Suitable (threshold=0.5): {model.is_suitable(query1, response2, threshold=0.5)}\n")

    query3 = "Tell me about machine learning."
    response3 = "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to 'learn' from data."
    response4 = "I like to eat apples."

    print(f"Query: {query3}")
    print(f"Response: {response3}")
    relevance_score3 = model.calculate_relevance(query3, response3)
    print(f"Relevance Score: {relevance_score3:.4f}")
    print(f"Is Suitable (threshold=0.5): {model.is_suitable(query3, response3, threshold=0.5)}\n")

    print(f"Query: {query3}")
    print(f"Response: {response4}")
    relevance_score4 = model.calculate_relevance(query3, response4)
    print(f"Relevance Score: {relevance_score4:.4f}")
    print(f"Is Suitable (threshold=0.5): {model.is_suitable(query3, response4, threshold=0.5)}\n")

    print("To run this model, make sure you have the 'sentence-transformers' library installed:")
    print("pip install sentence-transformers")