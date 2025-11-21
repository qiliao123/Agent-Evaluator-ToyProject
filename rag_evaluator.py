"""
Objective RAG Agent Evaluator with Accuracy-Based Metrics
This module provides objective, measurable metrics for evaluating RAG agent responses.
"""

try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS
from openai import OpenAI
import os
import re
from typing import Dict, List, Any
import json


def search_web(query: str, max_results: int = 10) -> list[dict]:
    """Conducts web searches using DuckDuckGo and returns a list of webpages."""
    results = []
    
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            results.append({
                'title': result.get('title', ''),
                'href': result.get('href', ''),
                'body': result.get('body', '')
            })
    
    return results


def rag_agent(query: str, api_key: str = None, max_results: int = 5) -> tuple[str, list[dict]]:
    """
    RAG Agent that searches the web and generates an answer using retrieved information.
    Returns both the answer and the search results for evaluation.
    """
    print(f"\n🔍 Searching for: {query}")
    search_results = search_web(query, max_results=max_results)
    
    if not search_results:
        return "No search results found for your query.", []
    
    # Prepare context from search results
    context = "Here are relevant web search results:\n\n"
    for i, result in enumerate(search_results, 1):
        context += f"{i}. {result['title']}\n"
        context += f"   URL: {result['href']}\n"
        context += f"   Content: {result['body']}\n\n"
    
    # Generate answer using LLM
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("\n⚠️  No OpenAI API key found. Returning search results only.\n")
        return context, search_results
    
    try:
        print("🤖 Generating answer...")
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided web search results. Always cite sources by mentioning the source number (e.g., [1], [2]) when making claims."},
                {"role": "user", "content": f"Based on the following search results, answer this question: {query}\n\n{context}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        return answer, search_results
        
    except Exception as e:
        print(f"\n❌ Error calling OpenAI API: {e}")
        print("Returning search results only.\n")
        return context, search_results


class ObjectiveEvaluator:
    """Objective evaluator using measurable, accuracy-based metrics."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
    
    def calculate_citation_coverage(self, response: str, num_sources: int) -> float:
        """
        Measures what percentage of available sources are cited in the response.
        
        Returns:
            Citation coverage score (0-1)
        """
        if num_sources == 0:
            return 0.0
        
        # Find all citation patterns like [1], [2], etc.
        citations = re.findall(r'\[(\d+)\]', response)
        unique_citations = set(int(c) for c in citations if int(c) <= num_sources)
        
        coverage = len(unique_citations) / num_sources
        return coverage
    
    def calculate_response_length_score(self, response: str, query: str) -> float:
        """
        Evaluates if response length is appropriate (not too short, not too long).
        
        Returns:
            Length appropriateness score (0-1)
        """
        word_count = len(response.split())
        
        # Optimal range: 50-300 words for most queries
        if 50 <= word_count <= 300:
            return 1.0
        elif word_count < 50:
            return word_count / 50  # Penalty for too short
        else:
            # Gentle penalty for being too long
            return max(0.5, 1.0 - (word_count - 300) / 1000)
    
    def calculate_query_term_coverage(self, response: str, query: str) -> float:
        """
        Measures how many important query terms appear in the response.
        
        Returns:
            Query term coverage score (0-1)
        """
        # Extract important words from query (remove common stopwords)
        stopwords = {'what', 'when', 'where', 'who', 'how', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
        query_terms = [word.lower() for word in query.split() if word.lower() not in stopwords and len(word) > 2]
        
        if not query_terms:
            return 1.0
        
        response_lower = response.lower()
        matched_terms = sum(1 for term in query_terms if term in response_lower)
        
        return matched_terms / len(query_terms)
    
    def calculate_source_diversity(self, search_results: List[dict]) -> float:
        """
        Measures diversity of sources (different domains).
        
        Returns:
            Source diversity score (0-1)
        """
        if not search_results:
            return 0.0
        
        domains = set()
        for result in search_results:
            url = result.get('href', '')
            # Extract domain
            match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if match:
                domains.add(match.group(1))
        
        # More unique domains = better diversity
        diversity = len(domains) / len(search_results)
        return diversity
    
    def verify_factual_grounding(self, response: str, search_results: List[dict], api_key: str = None) -> Dict[str, Any]:
        """
        Uses LLM to verify if claims in response are grounded in search results.
        
        Returns:
            Dictionary with grounding score and details
        """
        if not api_key:
            api_key = self.api_key
        
        if not api_key:
            return {
                "grounding_score": None,
                "error": "No API key available for factual verification"
            }
        
        # Prepare search context
        context = "\n\n".join([
            f"Source {i+1}:\nTitle: {r['title']}\nContent: {r['body']}"
            for i, r in enumerate(search_results)
        ])
        
        verification_prompt = f"""
You are a fact-checker. Your task is to verify if the claims made in the Response are supported by the Search Results.

Search Results:
{context}

Response to verify:
{response}

For each major claim in the Response, determine if it is:
1. SUPPORTED: Directly stated or clearly implied in the search results
2. UNSUPPORTED: Not found in the search results
3. CONTRADICTED: Conflicts with information in the search results

Provide your analysis in this format:
SUPPORTED_CLAIMS: [count]
UNSUPPORTED_CLAIMS: [count]
CONTRADICTED_CLAIMS: [count]
TOTAL_CLAIMS: [count]
GROUNDING_SCORE: [supported_claims / total_claims as decimal]

DETAILS: [Brief explanation of any unsupported or contradicted claims]
"""
        
        try:
            client = OpenAI(api_key=api_key)
            
            verification_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise fact-checker that verifies claims against source material."},
                    {"role": "user", "content": verification_prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            verification_text = verification_response.choices[0].message.content
            
            # Parse results
            supported = unsupported = contradicted = total = 0
            grounding_score = 0.0
            details = ""
            
            for line in verification_text.split('\n'):
                if 'SUPPORTED_CLAIMS:' in line:
                    match = re.search(r'\d+', line)
                    if match:
                        supported = int(match.group())
                elif 'UNSUPPORTED_CLAIMS:' in line:
                    match = re.search(r'\d+', line)
                    if match:
                        unsupported = int(match.group())
                elif 'CONTRADICTED_CLAIMS:' in line:
                    match = re.search(r'\d+', line)
                    if match:
                        contradicted = int(match.group())
                elif 'TOTAL_CLAIMS:' in line:
                    match = re.search(r'\d+', line)
                    if match:
                        total = int(match.group())
                elif 'GROUNDING_SCORE:' in line:
                    score_match = re.search(r'[\d.]+', line)
                    if score_match:
                        grounding_score = float(score_match.group())
                elif 'DETAILS:' in line:
                    details = line.split('DETAILS:')[1].strip()
            
            return {
                "grounding_score": grounding_score,
                "supported_claims": supported,
                "unsupported_claims": unsupported,
                "contradicted_claims": contradicted,
                "total_claims": total,
                "details": details,
                "full_verification": verification_text
            }
            
        except Exception as e:
            return {
                "grounding_score": None,
                "error": f"Verification failed: {e}"
            }
    
    def evaluate(self, query: str, response: str, search_results: List[dict]) -> Dict[str, Any]:
        """
        Comprehensive objective evaluation of RAG agent response.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print("\n📊 Running objective evaluation...")
        
        # Calculate objective metrics
        citation_coverage = self.calculate_citation_coverage(response, len(search_results))
        length_score = self.calculate_response_length_score(response, query)
        query_coverage = self.calculate_query_term_coverage(response, query)
        source_diversity = self.calculate_source_diversity(search_results)
        
        # Factual grounding verification (requires API)
        grounding_result = self.verify_factual_grounding(response, search_results, self.api_key)
        
        # Calculate overall score (weighted average)
        weights = {
            'citation_coverage': 0.15,
            'length_score': 0.10,
            'query_coverage': 0.20,
            'source_diversity': 0.10,
            'grounding_score': 0.45
        }
        
        grounding_score = grounding_result.get('grounding_score', 0.5)  # Default to 0.5 if unavailable
        if grounding_score is None:
            grounding_score = 0.5
        
        overall_score = (
            weights['citation_coverage'] * citation_coverage +
            weights['length_score'] * length_score +
            weights['query_coverage'] * query_coverage +
            weights['source_diversity'] * source_diversity +
            weights['grounding_score'] * grounding_score
        )
        
        return {
            "objective_metrics": {
                "citation_coverage": round(citation_coverage, 3),
                "length_score": round(length_score, 3),
                "query_term_coverage": round(query_coverage, 3),
                "source_diversity": round(source_diversity, 3),
                "factual_grounding": round(grounding_score, 3)
            },
            "overall_score": round(overall_score, 3),
            "grounding_details": grounding_result,
            "response_stats": {
                "word_count": len(response.split()),
                "num_sources_used": len(search_results),
                "num_citations": len(re.findall(r'\[\d+\]', response)),
                "unique_citations": len(set(re.findall(r'\[(\d+)\]', response)))
            }
        }


def print_evaluation(evaluation: Dict[str, Any]):
    """Pretty print the objective evaluation results."""
    print("\n" + "="*70)
    print("📊 OBJECTIVE RESPONSE EVALUATION")
    print("="*70)
    
    metrics = evaluation.get("objective_metrics", {})
    stats = evaluation.get("response_stats", {})
    
    print("\n📈 Objective Metrics (0.000-1.000 scale):")
    print(f"   Citation Coverage:    {metrics.get('citation_coverage', 'N/A'):.3f}  - Uses available sources")
    print(f"   Length Appropriateness: {metrics.get('length_score', 'N/A'):.3f}  - Response length quality")
    print(f"   Query Term Coverage:  {metrics.get('query_term_coverage', 'N/A'):.3f}  - Addresses query terms")
    print(f"   Source Diversity:     {metrics.get('source_diversity', 'N/A'):.3f}  - Variety of sources")
    print(f"   Factual Grounding:    {metrics.get('factual_grounding', 'N/A'):.3f}  - Claims backed by sources")
    
    print(f"\n🎯 OVERALL SCORE: {evaluation.get('overall_score', 'N/A'):.3f}")
    
    print("\n📝 Response Statistics:")
    print(f"   Word Count:        {stats.get('word_count', 'N/A')}")
    print(f"   Sources Retrieved: {stats.get('num_sources_used', 'N/A')}")
    print(f"   Total Citations:   {stats.get('num_citations', 'N/A')}")
    print(f"   Unique Sources Cited: {stats.get('unique_citations', 'N/A')}")
    
    grounding = evaluation.get("grounding_details", {})
    if grounding.get("total_claims"):
        print("\n🔍 Factual Verification:")
        print(f"   Total Claims Analyzed: {grounding.get('total_claims', 'N/A')}")
        print(f"   Supported:   {grounding.get('supported_claims', 'N/A')}")
        print(f"   Unsupported: {grounding.get('unsupported_claims', 'N/A')}")
        print(f"   Contradicted: {grounding.get('contradicted_claims', 'N/A')}")
        
        if grounding.get("details"):
            print(f"\n   Details: {grounding['details']}")
    
    # Provide interpretation
    print("\n💡 Interpretation:")
    overall = evaluation.get('overall_score', 0)
    if overall >= 0.8:
        print("   ✅ Excellent - High quality, well-grounded response")
    elif overall >= 0.6:
        print("   ✓  Good - Satisfactory response with minor issues")
    elif overall >= 0.4:
        print("   ⚠️  Fair - Response has notable weaknesses")
    else:
        print("   ❌ Poor - Response needs significant improvement")
    
    print("="*70)


def main():
    """Main CLI function with objective evaluation."""
    print("\n" + "="*70)
    print("🤖 RAG Agent with Objective Accuracy-Based Evaluation")
    print("="*70)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n⚠️  OpenAI API key not found in environment variables.")
        user_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
        if user_key:
            api_key = user_key
    
    evaluator = ObjectiveEvaluator(api_key=api_key)
    
    # Main loop
    while True:
        print("\n" + "-"*70)
        query = input("\n💭 Enter your query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            print("❌ Please enter a valid query.")
            continue
        
        # Get response from RAG agent
        response, search_results = rag_agent(query, api_key=api_key, max_results=5)
        
        print("\n" + "="*70)
        print("🤖 AGENT RESPONSE")
        print("="*70)
        print(response)
        print("="*70)
        
        # Ask if user wants evaluation
        evaluate_choice = input("\n📊 Evaluate this response? (y/n): ").strip().lower()
        if evaluate_choice == 'y':
            evaluation = evaluator.evaluate(query, response, search_results)
            print_evaluation(evaluation)
        
        # Ask if user wants to continue
        continue_choice = input("\n🔄 Ask another question? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
