from duckduckgo_search import DDGS
import openai
import os

def search_web(query: str, max_results: int = 10) -> list[dict]:
    """
    Conducts web searches using DuckDuckGo and returns a list of webpages.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        List of dictionaries containing search results with keys:
        - title: The title of the webpage
        - href: The URL of the webpage
        - body: Snippet/description of the webpage
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
    
    Args:
        query: The user's question or search query
        api_key: OpenAI API key (or set OPENAI_API_KEY environment variable)
        max_results: Number of search results to retrieve (default: 5)
    
    Returns:
        Generated answer based on retrieved information
    """
    # Step 1: Retrieve information from web search
    print(f"Searching for: {query}")
    search_results = search_web(query, max_results=max_results)
    
    if not search_results:
        return "No search results found for your query."
    
    # Step 2: Prepare context from search results
    context = "Here are relevant web search results:\n\n"
    for i, result in enumerate(search_results, 1):
        context += f"{i}. {result['title']}\n"
        context += f"   URL: {result['href']}\n"
        context += f"   Content: {result['body']}\n\n"
    
    # Step 3: Generate answer using LLM (OpenAI)
    if api_key:
        openai.api_key = api_key
    else:
        openai.api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai.api_key:
        # If no API key, return search results only
        print("\nNo OpenAI API key found. Returning search results only.\n")
        return context
    
    try:
        # Augmented generation with retrieved context
        response = openai.ChatCompletion.create(
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
        print(f"\nError calling OpenAI API: {e}")
        print("Returning search results only.\n")
        return context


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage (without OpenAI API - returns search results)
    query = "What is Microsoft Foundry?"
    answer = rag_agent(query, max_results=3)
    print(f"\nQuestion: {query}")
    print(f"\nAnswer:\n{answer}")
    
    # Example 2: With OpenAI API key (uncomment and add your key)
    api_key = "Your API key"  # Replace with your actual OpenAI API key
    query = "What are the benefits of renewable energy?"
    answer = rag_agent(query, api_key=api_key, max_results=5)
    print(f"\nQuestion: {query}")
    print(f"\nAnswer:\n{answer}")
