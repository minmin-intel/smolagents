import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
from duckduckgo_search import DDGS
# from googlesearch import search as gsearch
from langchain_google_community import GoogleSearchAPIWrapper
import os
from openai import OpenAI, AsyncOpenAI
import time
from smolagents import Tool


WORKDIR = os.getenv("WORKDIR")
DATAPATH= os.path.join(WORKDIR, "datasets/test_web_search/")

class DuckDuckGoSearchEngine:
    # require duckduckgo_search==6.3.5 to avoid rate limit error
    async def perform_search(self, query, num_results=3, *args, **kwargs):
        """DuckDuckGo search engine."""
        print(f"Searching DuckDuckGo for: {query}")
        results = DDGS().text(query, max_results=num_results)
        urls = [result["href"] for result in results]
        time.sleep(10)
        return urls
    
class GoogleSearchEngine:
    async def perform_search(self, query, num_results=3, *args, **kwargs):
        """Google search engine."""
        search = GoogleSearchAPIWrapper()
        results = search.results(query, num_results = num_results)
        links = [result["link"] for result in results]
        for link in links:
            # remove huggingface links that may contain benchmark answers
            if "https://huggingface.co/" in link:
                links.remove(link)        
        print(f"Google search results: {links}")
        print(f"Found {len(links)} results.")
        time.sleep(5)
        return links
    
class WebCrawler:
    def __init__(self, browser_config=None, run_config=None):
        if browser_config is None:
            browser_config = BrowserConfig(verbose=True)
        if run_config is None:
            run_config = CrawlerRunConfig(
                # Content filtering
                word_count_threshold=10,
                excluded_tags=['form', 'header'],
                exclude_external_links=True,

                # Content processing
                process_iframes=True,
                remove_overlay_elements=True,

                # Cache control
                cache_mode=CacheMode.ENABLED  # Use cache if available
            )
        self.browser_config = browser_config
        self.run_config = run_config
        self.crawler = AsyncWebCrawler(config=self.browser_config)    
    async def crawl(self, url):
        async with self.crawler as crawler:
            result = await crawler.arun(
                url=url,
                config=self.run_config
            )

            if result.success:
                # # Print clean content
                # print("Content:", result.markdown[:500])  # First 500 chars
                # output_path = os.path.join(DATAPATH, "crawl4ai_content.md")
                # with open(output_path, "w") as f:
                #     f.write(result.markdown)
                return {"url":url,"content":result.markdown}

                # # Process images
                # for image in result.media["images"]:
                #     print(f"Found image: {image['src']}")

                # # Process links
                # for link in result.links["internal"]:
                #     print(f"Internal link: {link['href']}")

            else:
                print(f"Crawl failed: {result.error_message}")
                return f"Web crawl failed with url: {url}"


class LLMExtractor:
    def __init__(self, provider="together", model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        if provider == "together":
            self._api_key = os.getenv("TOGETHER_API_KEY")
            self._url = "https://api.together.xyz/v1"
        else:
            raise ValueError(f"Model provider {provider} is not supported.")
        
        self.client = OpenAI(
            timeout=180,
            max_retries=3,
            api_key=self._api_key,
            base_url=self._url,
        )
        # self._async_client = AsyncOpenAI(
        #     timeout=180,
        #     max_retries=3,
        #     api_key=self._api_key,
        #     base_url=self._url,
        # )

        self.model_name = model_name

    def run(self, webpages, query):
        # webpages: list of dicts with keys: url, content
        context = ""

        for i, webpage in enumerate(webpages):
            if isinstance(webpage, dict):
                url = webpage["url"]
                content = webpage["content"]
                breaker = "="*10
                context += f"[{i}] source: {url}\n{content}\n{breaker}\n"
            else:
                pass
        
        # print(f"Context:\n{context}")

        user_msg = """\
Below are some pages crawled from the web. Please read through them carefully and answer the question.
Web content:
{context}
===== End of web content =====
Question: {question}
"""

        # # Get response information
        if context:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": user_msg.format(context=context, question=query)}],
                max_tokens=2048,
                temperature=0.2,
                )
            resp_msg = resp.choices[0].message.content
            prompt_tokens = resp.usage.prompt_tokens
            completion_tokens = resp.usage.completion_tokens
        else:
            resp_msg = "No web content found."
            prompt_tokens = 0
            completion_tokens = 0
        return resp_msg, prompt_tokens, completion_tokens


def search_web(query: str) -> str:
    r"""
    Search the web for information on a given query.
    Args:
        query (str): Detailed search query. Only include the query, not any site info.
    Returns:
        str: web search result.
    """
    ddg = DuckDuckGoSearchEngine()
    crawler = WebCrawler()
    results = asyncio.run(ddg.perform_search(query))
    print(results)
    print(f"Found {len(results)} results.")

    webpages = []
    for url in results:
        print(f"Crawling {url}")
        web_dict = asyncio.run(crawler.crawl(url))
        if isinstance(web_dict, dict):
            webpages.append(web_dict)
    
    llm = LLMExtractor(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    response, prompt_tokens, completion_tokens = llm.run(webpages, query)
    print(f"Search web - Response: {response}")
    print(f"Search web - Prompt tokens: {prompt_tokens}")
    print(f"Search web - Completion tokens: {completion_tokens}")
    return response

class SearchWebTool(Tool):
    name = "search_web"
    description = "Search the web for content related to a given query. Output crawled web content."
    inputs = {"query": {"type": "string", "description": "The web search query to perform. Should be detailed."}}
    output_type = "string"
    def __init__(self, search_engine="google"):
        super().__init__()
        # self.ddg = DuckDuckGoSearchEngine()
        if search_engine == "google":
            self.search_engine = GoogleSearchEngine()
        else:
            self.search_engine = DuckDuckGoSearchEngine()
        self.crawler = WebCrawler()
        self.llm = LLMExtractor(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    
    def forward(self, query:str) -> str:
        results = asyncio.run(self.search_engine.perform_search(query))
        print(f"Found {len(results)} results.")

        webpages = []
        for url in results:
            print(f"Crawling {url}")
            web_dict = asyncio.run(self.crawler.crawl(url))
            if isinstance(web_dict, dict):
                webpages.append(web_dict)
        
        response, prompt_tokens, completion_tokens = self.llm.run(webpages, query)
        print(f"Search web - Response: {response}")
        print(f"Search web - Prompt tokens: {prompt_tokens}")
        print(f"Search web - Completion tokens: {completion_tokens}")
        return response

        # context = ""

        # for i, webpage in enumerate(webpages):
        #     if isinstance(webpage, dict):
        #         url = webpage["url"]
        #         content = webpage["content"]
        #         breaker = "="*10
        #         context += f"[{i+1}] source: {url}\n{content}\n{breaker}\n"
        #     else:
        #         pass
        # return context

class GetContentTool(Tool):
    name = "anwer_question_about_url"
    description = "Get the content of a webpage. And answer a question about the content."
    inputs = {"url": {"type": "string", "description": "The URL of the webpage to crawl."},
              "question": {"type": "string", "description": "The question to answer about the content."}}
    output_type = "string"
    def __init__(self):
        super().__init__()
        self.crawler = WebCrawler()
        self.llm = LLMExtractor(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    
    def forward(self, url:str, question: str) -> str:
        result = asyncio.run(self.crawler.crawl(url))
        answer, prompt_tokens, completion_tokens = self.llm.run([result], question)
        print(f"Get content - Response: {answer}")
        print(f"Get content - Prompt tokens: {prompt_tokens}")
        print(f"Get content - Completion tokens: {completion_tokens}")
        return answer


if __name__ == "__main__":

    search_web_tool = SearchWebTool()
    get_content_tool = GetContentTool()

    # query_list = [
    #     # "weather in New York",
    #     # "best restaurants in New York",
    #     # "top tourist attractions in New York",
    #     # "best hotels in New York",
    #     # "best schools in New York",
    #     # "British Museum's collection 2012,5015.17",
    #     "url of Scikit-Learn July 2017 changelog",
    # ]

    # sleep_time = 5
    # for query in query_list:
    #     result = search_web_tool.forward(query)
    #     print(result)
    #     print("="*50)
    #     # time.sleep(sleep_time)

    query = "link to Scikit-Learn July 2017 changelog"
    result = search_web_tool.forward(query)
    print(f"**** Search WEB RESULT: {result}")

    # url = "https://scikit-learn.org/0.21/whats_new/v0.19.html#version-0-19"
    # content = get_content_tool.forward(url)
    # print(content)
    
    question = "what other predictor base command received a bug fix?"
    content = get_content_tool.forward(result, question)
    print(f"***Answer: {content}")


