try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS
from openai import OpenAI
import os
import sys
import re
import json

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
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided web search results. Always cite sources and be factual."},
                {"role": "user", "content": f"Based on the following search results, answer this question: {query}\n\n{context}"}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        return answer, search_results
        
    except Exception as e:
        print(f"\n❌ Error calling OpenAI API: {e}")
        print("Returning search results only.\n")
        return context, search_results


def evaluate_response_objective(query: str, response: str, search_results: list[dict], api_key: str = None) -> dict:
    """
    Objective evaluator that assesses response quality using accuracy-based metrics.
    
    Metrics:
    1. Factual Grounding: How well the response is grounded in the search results
    2. Information Coverage: What percentage of key information from sources is included
    3. Citation Accuracy: Whether sources are properly cited
    4. Hallucination Detection: Whether response contains unsupported claims
    5. Query Alignment: How directly the response addresses the query
    
    Args:
        query: Original user query
        response: Agent's response to evaluate
        search_results: The actual search results used
        api_key: OpenAI API key
    
    Returns:
        Dictionary with objective evaluation metrics
    """
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        return {
            "error": "No OpenAI API key found. Cannot evaluate response.",
            "metrics": None
        }
    
    try:
        print("\n📊 Running objective evaluation...")
        
        # Prepare source context for evaluation
        source_context = "\n\n".join([
            f"Source {i+1}: {r['title']}\n{r['body']}"
            for i, r in enumerate(search_results)
        ])
        
        evaluation_prompt = f"""
You are an objective evaluator that measures response accuracy and quality using specific, measurable criteria.

USER QUERY: {query}

AVAILABLE SOURCES:
{source_context}

AGENT RESPONSE TO EVALUATE:
{response}

Evaluate the response using these OBJECTIVE metrics (provide numeric scores):

1. FACTUAL GROUNDING (0-100%): What percentage of claims in the response are directly supported by the provided sources? Count each claim and determine support.

2. INFORMATION COVERAGE (0-100%): What percentage of relevant key facts from the sources are included in the response?

3. SOURCE CITATION (0-100%): Are sources mentioned or cited? (100% if sources cited, 50% if implicit reference, 0% if no attribution)

4. HALLUCINATION CHECK (0-100%): Percentage of response that is NOT hallucinated (100% = no hallucinations, 0% = all hallucinated). Identify any claims not supported by sources.

5. QUERY ALIGNMENT (0-100%): How directly does the response answer the specific query asked?

6. COMPLETENESS (0-100%): Does the response provide a complete answer to the query based on available information?

Provide your evaluation in this EXACT JSON format:
{{
    "factual_grounding": <0-100>,
    "information_coverage": <0-100>,
    "source_citation": <0-100>,
    "hallucination_check": <0-100>,
    "query_alignment": <0-100>,
    "completeness": <0-100>,
    "overall_accuracy": <average of all metrics>,
    "unsupported_claims": ["list", "of", "any", "unsupported", "claims"],
    "missing_information": ["list", "of", "key", "facts", "from", "sources", "not", "included"],
    "strengths": ["specific", "strength", "1", "strength", "2"],
    "improvements": ["specific", "improvement", "1", "improvement", "2"]
}}

Respond ONLY with the JSON, no other text.
"""
        
        client = OpenAI(api_key=api_key)
        
        eval_response = client.chat.completions.create(
            model="gpt-4",  # Use GPT-4 for more accurate evaluation
            messages=[
                {"role": "system", "content": "You are an objective evaluator that provides precise, measurable metrics. Always respond with valid JSON only."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.1,  # Very low temperature for consistency
            max_tokens=800
        )
        
        evaluation_text = eval_response.choices[0].message.content.strip()
        
        # Extract JSON from response (handle potential markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', evaluation_text)
        if json_match:
            evaluation_json = json.loads(json_match.group())
        else:
            raise ValueError("Could not parse JSON from evaluation response")
        
        return {
            "metrics": evaluation_json,
            "raw_evaluation": evaluation_text
        }
        
    except Exception as e:
        print(f"⚠️  Evaluation error: {e}")
        return {
            "error": f"Error during evaluation: {e}",
            "metrics": None
        }


def print_objective_evaluation(evaluation: dict):
    """Pretty print the objective evaluation results."""
    print("\n" + "="*70)
    print("📊 OBJECTIVE RESPONSE EVALUATION")
    print("="*70)
    
    if "error" in evaluation and evaluation["error"]:
        print(f"❌ {evaluation['error']}")
        return
    
    metrics = evaluation.get("metrics")
    if not metrics:
        print("❌ No metrics available")
        return
    
    # Print accuracy metrics
    print("\n📈 ACCURACY METRICS (0-100%):")
    print("-" * 70)
    print(f"  Factual Grounding:     {metrics.get('factual_grounding', 'N/A')}%  (Claims supported by sources)")
    print(f"  Information Coverage:  {metrics.get('information_coverage', 'N/A')}%  (Key facts included)")
    print(f"  Source Citation:       {metrics.get('source_citation', 'N/A')}%  (Proper attribution)")
    print(f"  Hallucination Check:   {metrics.get('hallucination_check', 'N/A')}%  (No false claims)")
    print(f"  Query Alignment:       {metrics.get('query_alignment', 'N/A')}%  (Answers the query)")
    print(f"  Completeness:          {metrics.get('completeness', 'N/A')}%  (Complete answer)")
    print("-" * 70)
    print(f"  📊 OVERALL ACCURACY:    {metrics.get('overall_accuracy', 'N/A')}%")
    print("="*70)
    
    # Print detailed findings
    unsupported = metrics.get('unsupported_claims', [])
    if unsupported and unsupported[0]:  # Check if not empty
        print("\n⚠️  UNSUPPORTED CLAIMS DETECTED:")
        for i, claim in enumerate(unsupported, 1):
            print(f"  {i}. {claim}")
    
    missing = metrics.get('missing_information', [])
    if missing and missing[0]:  # Check if not empty
        print("\n📋 MISSING KEY INFORMATION:")
        for i, info in enumerate(missing, 1):
            print(f"  {i}. {info}")
    
    strengths = metrics.get('strengths', [])
    if strengths:
        print("\n✅ STRENGTHS:")
        for i, strength in enumerate(strengths, 1):
            print(f"  {i}. {strength}")
    
    improvements = metrics.get('improvements', [])
    if improvements:
        print("\n🔧 SUGGESTED IMPROVEMENTS:")
        for i, improvement in enumerate(improvements, 1):
            print(f"  {i}. {improvement}")
    
    print("\n" + "="*70)


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
        if api_key and search_results:
            evaluate = input("\n📊 Run objective evaluation? (y/n): ").strip().lower()
            if evaluate == 'y':
                evaluation = evaluate_response_objective(query, response, search_results, api_key=api_key)
                print_objective_evaluation(evaluation)
        else:
            print("\n⚠️  Evaluation requires OpenAI API key and search results.")
        
        # Ask if user wants to continue
        continue_choice = input("\n🔄 Ask another question? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
