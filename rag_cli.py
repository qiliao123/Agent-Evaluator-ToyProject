try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS
from openai import OpenAI
import os
import sys

def search_web(query: str, max_results: int = 10) -> list[dict]:
    """
    Conducts web searches using DuckDuckGo and returns a list of webpages.
    """
    results = []
    
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            results.append({
                'title': result.get('title', ''),
                'href': result.get('href', ''),
                'body': result.get('body', '')
            })
    
    return results


def rag_agent(query: str, api_key: str = None, max_results: int = 5) -> str:
    """
    RAG Agent that searches the web and generates an answer using retrieved information.
    """
    print(f"\n🔍 Searching for: {query}")
    search_results = search_web(query, max_results=max_results)
    
    if not search_results:
        return "No search results found for your query."
    
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
        return context
    
    try:
        print("🤖 Generating answer...")
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided web search results. Cite sources when possible."},
                {"role": "user", "content": f"Based on the following search results, answer this question: {query}\n\n{context}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        return answer
        
    except Exception as e:
        print(f"\n❌ Error calling OpenAI API: {e}")
        print("Returning search results only.\n")
        return context


def calculate_objective_metrics(query: str, response: str, search_results: list[dict]) -> dict:
    """
    Calculate objective metrics for response quality.
    
    Args:
        query: Original user query
        response: Agent's response
        search_results: Original search results used for RAG
    
    Returns:
        Dictionary with objective metrics
    """
    metrics = {}
    
    # 1. Source Citation Rate - Are sources cited?
    cited_sources = 0
    for result in search_results:
        if result['href'] in response or result['title'] in response:
            cited_sources += 1
    metrics['citation_rate'] = cited_sources / len(search_results) if search_results else 0
    
    # 2. Response Length (indicator of completeness)
    metrics['response_length'] = len(response.split())
    
    # 3. Query Term Coverage - Does response include query terms?
    query_terms = set(query.lower().split())
    response_lower = response.lower()
    covered_terms = sum(1 for term in query_terms if term in response_lower and len(term) > 3)
    metrics['query_coverage'] = covered_terms / len(query_terms) if query_terms else 0
    
    # 4. Information Density - Ratio of unique words to total words
    words = response.split()
    unique_words = set(words)
    metrics['information_density'] = len(unique_words) / len(words) if words else 0
    
    # 5. Source Content Overlap - How much content from sources is used?
    source_content = " ".join([r['body'] for r in search_results])
    common_words = set(source_content.lower().split()) & set(response.lower().split())
    metrics['source_overlap'] = len(common_words) / len(response.split()) if response.split() else 0
    
    return metrics


def evaluate_response(query: str, response: str, api_key: str = None, search_results: list[dict] = None) -> dict:
    """
    Enhanced evaluator with both objective metrics and LLM-based assessment.
    
    Args:
        query: Original user query
        response: Agent's response to evaluate
        api_key: OpenAI API key
        search_results: Original search results (for objective metrics)
    
    Returns:
        Dictionary with evaluation scores, objective metrics, and feedback
    """
    # Calculate objective metrics first
    objective_metrics = {}
    if search_results:
        objective_metrics = calculate_objective_metrics(query, response, search_results)
    
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        return {
            "error": "No OpenAI API key found for LLM evaluation.",
            "objective_metrics": objective_metrics,
            "scores": None,
            "feedback": "LLM evaluation requires OpenAI API key, but objective metrics are available."
        }
    
    try:
        print("\n📊 Evaluating response quality...")
        
        evaluation_prompt = f"""
You are an expert fact-checker and evaluator. Assess the quality of the following response objectively.

User Query: {query}

Agent Response: {response}

Evaluate based on OBJECTIVE CRITERIA (score 1-10):

1. FACTUAL ACCURACY: Can you verify the facts stated? Look for:
   - Verifiable claims vs unverifiable statements
   - Internal consistency
   - Absence of contradictions
   
2. QUERY ALIGNMENT: Does it directly answer what was asked?
   - Addresses the specific question
   - No irrelevant information

3. EVIDENCE QUALITY: Is the response backed by sources?
   - Citations or references included
   - Claims are supported, not just stated

4. SPECIFICITY: Are concrete details provided?
   - Specific facts, numbers, names, dates
   - Not just generic statements

5. COMPLETENESS: Does it cover key aspects?
   - Main points addressed
   - Important context included

Provide evaluation in this format:
FACTUAL_ACCURACY: [score]/10 - [brief justification]
QUERY_ALIGNMENT: [score]/10 - [brief justification]
EVIDENCE_QUALITY: [score]/10 - [brief justification]
SPECIFICITY: [score]/10 - [brief justification]
COMPLETENESS: [score]/10 - [brief justification]
OVERALL: [average]/10

OBJECTIVE_ASSESSMENT: [Identify specific factual claims that can be verified, potential inaccuracies, and missing information]
"""
        
        client = OpenAI(api_key=api_key)
        
        eval_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a rigorous fact-checker and evaluator. Focus on objective, verifiable criteria rather than subjective preferences."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent evaluation
            max_tokens=500
        )
        
        evaluation_text = eval_response.choices[0].message.content
        
        # Parse evaluation scores
        lines = evaluation_text.split('\n')
        scores = {}
        assessment = ""
        
        for line in lines:
            if 'FACTUAL_ACCURACY:' in line:
                scores['factual_accuracy'] = line.split(':')[1].strip()
            elif 'QUERY_ALIGNMENT:' in line:
                scores['query_alignment'] = line.split(':')[1].strip()
            elif 'EVIDENCE_QUALITY:' in line:
                scores['evidence_quality'] = line.split(':')[1].strip()
            elif 'SPECIFICITY:' in line:
                scores['specificity'] = line.split(':')[1].strip()
            elif 'COMPLETENESS:' in line:
                scores['completeness'] = line.split(':')[1].strip()
            elif 'OVERALL:' in line:
                scores['overall'] = line.split(':')[1].strip()
            elif 'OBJECTIVE_ASSESSMENT:' in line:
                assessment = line.split('OBJECTIVE_ASSESSMENT:')[1].strip()
        
        return {
            "objective_metrics": objective_metrics,
            "llm_scores": scores,
            "assessment": assessment,
            "full_evaluation": evaluation_text
        }
        
    except Exception as e:
        return {
            "error": f"Error during evaluation: {e}",
            "objective_metrics": objective_metrics,
            "llm_scores": None,
            "assessment": None
        }


def print_evaluation(evaluation: dict):
    """Pretty print the evaluation results."""
    print("\n" + "="*60)
    print("📊 RESPONSE EVALUATION")
    print("="*60)
    
    if "error" in evaluation and evaluation["error"]:
        print(f"❌ {evaluation['error']}")
        return
    
    # Check for both 'llm_scores' and 'scores' for compatibility
    scores = evaluation.get("llm_scores") or evaluation.get("scores")
    if scores:
        print("\n📈 LLM Evaluation Scores:")
        print(f"   Factual Accuracy:  {scores.get('factual_accuracy', 'N/A')}")
        print(f"   Query Alignment:   {scores.get('query_alignment', 'N/A')}")
        print(f"   Evidence Quality:  {scores.get('evidence_quality', 'N/A')}")
        print(f"   Specificity:       {scores.get('specificity', 'N/A')}")
        print(f"   Completeness:      {scores.get('completeness', 'N/A')}")
        print(f"   Overall:           {scores.get('overall', 'N/A')}")
    
    # Display objective metrics if available
    objective = evaluation.get("objective_metrics")
    if objective:
        print("\n📊 Objective Metrics:")
        print(f"   Citation Rate:        {objective.get('citation_rate', 0):.2%}")
        print(f"   Query Coverage:       {objective.get('query_coverage', 0):.2%}")
        print(f"   Information Density:  {objective.get('information_density', 0):.2%}")
        print(f"   Response Length:      {objective.get('response_length', 'N/A')} words")
    
    # Display assessment
    if evaluation.get("assessment"):
        print(f"\n💬 Assessment: {evaluation['assessment']}")
    
    if evaluation.get("feedback"):
        print(f"\n💬 Feedback: {evaluation['feedback']}")
    
    print("="*60)


def main():
    """Main CLI function."""
    print("\n" + "="*60)
    print("🤖 RAG Agent with Automatic Evaluation")
    print("="*60)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n⚠️  OpenAI API key not found in environment variables.")
        user_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
        if user_key:
            api_key = user_key
    
    # Main loop
    while True:
        print("\n" + "-"*60)
        query = input("\n💭 Enter your query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            print("❌ Please enter a valid query.")
            continue
        
        # Get response from RAG agent
        response = rag_agent(query, api_key=api_key, max_results=5)
        
        print("\n" + "="*60)
        print("🤖 AGENT RESPONSE")
        print("="*60)
        print(response)
        print("="*60)
        
        # Ask if user wants evaluation
        if api_key:
            evaluate = input("\n📊 Evaluate this response? (y/n): ").strip().lower()
            if evaluate == 'y':
                evaluation = evaluate_response(query, response, api_key=api_key)
                print_evaluation(evaluation)
        else:
            print("\n⚠️  Evaluation requires OpenAI API key.")
        
        # Ask if user wants to continue
        continue_choice = input("\n🔄 Ask another question? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
