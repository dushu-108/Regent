from dotenv import load_dotenv
load_dotenv()
from langchain_tavily import TavilySearch
from langchain.tools import tool
from bs4 import BeautifulSoup 
import requests
from rich import print

search = TavilySearch(max_results=5)

@tool
def web_search(query:str) -> str:
    """Search the web for recent and reliable information on a topic. Return's titles, urls and snippets"""
    responses = search.invoke({"query": query})
    out = []
    for r in responses['results']:
        out.append(
            f"""Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}\n"""
        )

    return "\n-------\n".join(out)

@tool
def scrape_url(url:str) -> str:
    """Scrapes a given URL and returns its content."""
    try:
        response = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script','style','nav', 'footer']):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape the URL : {str(e)}"

