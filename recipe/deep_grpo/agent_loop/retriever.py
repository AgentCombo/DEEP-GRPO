import logging
import os
import asyncio
import httpx


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Retriever:
    def __init__(self, search_url: str, concurrency_limit: int = 128, max_tool_response_length: int = 1024) -> None:
        self.search_url = search_url
        self.client = httpx.AsyncClient(timeout=20.0)
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.max_tool_response_length = max_tool_response_length

    async def close(self):
        await self.client.aclose()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        if retrieval_result is None:
            logger.error("WARNING!!! retrieval_result is None")
        elif len(retrieval_result) > 0 and isinstance(retrieval_result[0], dict):
            for idx, doc_item in enumerate(retrieval_result):
                
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        else:
            for tmp in retrieval_result:
                format_reference += tmp

        return format_reference

    def _trim_output(self, text: str) -> str:
        if len(text) > self.max_tool_response_length:
            text = text[:self.max_tool_response_length]
        return text
    
    async def _search(self, query):
        payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
        async with self.semaphore:
            response = await self.client.post(self.search_url, json=payload)
            response.raise_for_status()
            return response.json()

    async def search(self, query: str) -> str:
        try:
            response_data = await self._search(query)
            results = response_data['result']
            assert len(results) == 1
            result = results[0]
            return self._trim_output(self._passages2string(result))
        except httpx.HTTPStatusError as e:
            logger.error(f"Retrieve API request failed, status code: {e.response.status_code}, response: {e.response.text}")
            return ""
        except Exception as e:
            logger.error(f"Retrieve API request failed: {type(e).__name__} - {e}")
            return ""


async def main():
    retriever = Retriever(search_url=os.environ.get('RETRIEVER_URL', 'http://127.0.0.1:8000/retrieve'))
    
    try:
        search_result = await retriever.search("China")
        print(search_result)
    finally:
        await retriever.close()


if __name__ == '__main__':
    asyncio.run(main())
