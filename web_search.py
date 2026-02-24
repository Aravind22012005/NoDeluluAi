from serpapi import GoogleSearch
from config import SERPAPI_KEY

_cache = {}

def search_web(query):
    if query in _cache:
        return _cache[query]

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 3
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    snippets = []

    if "organic_results" in results:
        for result in results["organic_results"]:
            if "snippet" in result:
                snippets.append(result["snippet"])

    combined = " ".join(snippets)

    _cache[query] = combined
    return combined