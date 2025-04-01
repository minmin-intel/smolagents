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
You should provide a short answer, a detailed answer, and additional context in the following format:
1. Short answer: <short answer>
2. Detailed answer: <detailed answer>
3. Additional context: <additional context>

===== Start of web content =====
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


def crawl_web(query: str) -> str:
    r"""
    Crawl the web for information on a given query. Returns summarized text from the crawled web pages.
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

class CrawlWebTool(Tool):
    name = "search_web"
    description = "Get answer to the query with a web search agent. Use it as if you are asking a question to a human. Do not include any site info in the query."
    inputs = {"query": {"type": "string", "description": "The query should be detailed and include all the info you want to search for."}}
    output_type = "string"
    def __init__(self):
        super().__init__()
        # self.ddg = DuckDuckGoSearchEngine()
        # if search_engine == "google":
        #     self.search_engine = GoogleSearchEngine()
        # else:
        #     self.search_engine = DuckDuckGoSearchEngine()
        self.crawler = WebCrawler()
        self.llm = LLMExtractor(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    
    def forward(self, query:str) -> str:
        reformat_query = query
        if "site:" in query:
            reformat_query = query.split("site:")[0].strip()
        if "google:" in query:
            reformat_query = query.split("google:")[0].strip()
        
        if reformat_query == "":
            reformat_query = query

        try:
            search_engine = DuckDuckGoSearchEngine()
            results = asyncio.run(search_engine.perform_search(reformat_query))
        except:
            search_engine = GoogleSearchEngine()
            results = asyncio.run(search_engine.perform_search(reformat_query))
        if results is None:
            print(f"Search failed for query: {query}")
            return "Search failed. No results found."
        
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

    search_web_tool = CrawlWebTool()
    # query = "NIH clinical trial on H. pylori in acne vulgaris patients from Jan-May 2018"
    query = "clinical trial H. pylori acne vulgaris patients January to May 2018" #site:clinicaltrials.gov"
    result = search_web_tool.forward(query)
    print(result)
    # get_content_tool = GetContentTool()

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

    # query = "link to Scikit-Learn July 2017 changelog"
    # # ['https://scikit-learn.org/', 'https://stackoverflow.com/questions/33679938/how-to-upgrade-scikit-learn-package-in-anaconda', 'https://scikit-learn.org/0.19/whats_new.html']

    # result = search_web_tool.forward(query)
    # print(f"**** Search WEB RESULT: {result}")

    # # url = "https://scikit-learn.org/0.21/whats_new/v0.19.html#version-0-19"
    # # content = get_content_tool.forward(url)
    # # print(content)
    
    # question = "what other predictor base command received a bug fix?"
    # content = get_content_tool.forward(result, question)
    # print(f"***Answer: {content}")


