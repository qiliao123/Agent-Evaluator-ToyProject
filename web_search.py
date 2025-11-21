from duckduckgo_search import DDGS

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


# Example usage
if __name__ == "__main__":
    query = "microsoft foundry"  # Change this to whatever you want to search
    webpages = search_web(query, max_results=5)
    
    print(f"Search results for: {query}\n")
    for i, page in enumerate(webpages, 1):
        print(f"{i}. {page['title']}")
        print(f"   URL: {page['href']}")
        print(f"   Description: {page['body'][:100]}...")
        print()
