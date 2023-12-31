import logging
import re
from typing import List, Optional
import time
import os

# from langchain.document_loaders import AsyncHtmlLoader
from WebRetrieverAccelerate.MyAsyncHtmlLoader import AsyncHtmlLoader
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.document_transformers import Html2TextTransformer
from langchain.llms import LlamaCpp
from langchain.llms.base import BaseLLM
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BaseRetriever, Document
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.utilities import GoogleSearchAPIWrapper

logger = logging.getLogger(__name__)


class SearchQueries(BaseModel):
    """Search queries to research for the user's goal."""

    queries: List[str] = Field(
        ..., description="List of search queries to look up on Google"
    )


DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n 你現在是一個助理，你的職責是改善Google搜尋結果。 \
\n <</SYS>> \n\n [INST] 請以numbered list of questions的型式，產出三個與下列問題相似的Google搜尋查詢，並在每一個問題最後加上問號： \
\n\n {question} [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你現在是一個助理，你的職責是改善Google搜尋結果。 \
請以numbered list of questions的型式，產出三個與下列問題相似的Google搜尋查詢，並在每一個問題最後加上問號： {question}""",
)


class LineList(BaseModel):
    """List of questions."""

    lines: List[str] = Field(description="Questions")


class QuestionListOutputParser(PydanticOutputParser):
    """Output parser for a list of numbered questions."""

    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = re.findall(r"\d+\..*?(?:\n|$)", text)
        return LineList(lines=lines)


class WebResearchRetriever(BaseRetriever):
    """`Google Search API` retriever."""

    # Inputs
    vectorstore: VectorStore = Field(
        ..., description="Vector store for storing web pages"
    )
    llm_chain: LLMChain
    search: GoogleSearchAPIWrapper = Field(..., description="Google Search API Wrapper")
    num_search_results: int = Field(1, description="Number of pages per Google search")
    text_splitter: TextSplitter = Field(
        RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for splitting web pages into chunks",
    )
    url_database: List[str] = Field(
        default_factory=list, description="List of processed URLs"
    )

    @classmethod
    def from_llm(
            cls,
            vectorstore: VectorStore,
            llm: BaseLLM,
            search: GoogleSearchAPIWrapper,
            prompt: Optional[BasePromptTemplate] = None,
            num_search_results: int = 1,
            text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=150
            ),
    ) -> "WebResearchRetriever":
        """Initialize from llm using default template.

        Args:
            vectorstore: Vector store for storing web pages
            llm: llm for search question generation
            search: GoogleSearchAPIWrapper
            prompt: prompt to generating search questions
            num_search_results: Number of pages per Google search
            text_splitter: Text splitter for splitting web pages into chunks

        Returns:
            WebResearchRetriever
        """

        if not prompt:
            QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
                default_prompt=DEFAULT_SEARCH_PROMPT,
                conditionals=[
                    (lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)
                ],
            )
            prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)

        # Use chat model prompt
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=QuestionListOutputParser(),
        )

        return cls(
            vectorstore=vectorstore,
            llm_chain=llm_chain,
            search=search,
            num_search_results=num_search_results,
            text_splitter=text_splitter,
        )

    def clean_search_query(self, query: str) -> str:
        # Some search tools (e.g., Google) will
        # fail to return results if query has a
        # leading digit: 1. "LangCh..."
        # Check if the first character is a digit
        if query[0].isdigit():
            # Find the position of the first quote
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                # Extract the part of the string after the quote
                query = query[first_quote_pos + 1:]
                # Remove the trailing quote if present
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()

    def search_tool(self, query: str, num_search_results: int = 1) -> List[dict]:
        """Returns num_search_results pages per Google search."""
        query_clean = self.clean_search_query(query)
        result = self.search.results(query_clean, num_search_results)
        return result

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Search Google for documents related to the query input.

        Args:
            query: user query

        Returns:
            Relevant documents from all various urls.
        """

        # Get search questions
        logger.info("Generating questions for Google Search ...")
        result = self.llm_chain({"question": query})
        print("\nOriginal Question: ", query)
        logger.info(f"Questions for Google Search (raw): {result}")
        questions = getattr(result["text"], "lines", [])
        logger.info(f"Questions for Google Search: {questions}")
        print("Generated Questions: ", questions, "\n")

        # Get urls
        logger.info("Searching for relevant urls...")
        urls_to_look = []
        for query in questions:
            # Google search
            search_results = self.search_tool(query, self.num_search_results)
            logger.info("Searching for relevant urls...")
            logger.info(f"Search results: {search_results}")
            for res in search_results:
                if res.get("link", None):
                    urls_to_look.append(res["link"])

        # Relevant urls
        urls = set(urls_to_look)

        # Check for any new urls that we have not processed
        new_urls = list(urls.difference(self.url_database))

        logger.info(f"New URLs to load: {new_urls}")
        # Load, split, and add new urls to vectorstore
        if new_urls:
            start_time = time.time()
            loader = AsyncHtmlLoader(new_urls)
            print("--- AsyncHtmlLoader in %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            html2text = Html2TextTransformer()
            print("--- Html2TextTransformer in %s seconds ---" % (time.time() - start_time))

            logger.info("Indexing new urls...")
            start_time = time.time()
            docs = loader.load()
            print("--- html loader in %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            docs = list(html2text.transform_documents(docs))
            print("--- html2text in %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            docs = self.text_splitter.split_documents(docs)
            print("--- text_splitter in %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            # self.vectorstore.add_documents(docs)
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            embeddings = OpenAIEmbeddings(deployment=os.environ["OPENAI_EMBEDDING_ENGINE"]).embed_documents([doc.page_content for doc in docs])
            self.vectorstore._FAISS__add(texts, embeddings, metadatas)
            print("--- vectorstore.add_documents in %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            self.url_database.extend(new_urls)
            print("--- url_database.extend in %s seconds ---" % (time.time() - start_time))

        # Search for relevant splits
        # TODO: make this async
        logger.info("Grabbing most relevant splits from urls...")
        docs = []
        start_time = time.time()
        for query in questions:
            docs.extend(self.vectorstore.similarity_search(query))
        print("--- similarity_search in %s seconds ---" % (time.time() - start_time))

        # Get unique docs
        start_time = time.time()
        unique_documents_dict = {
            (doc.page_content, tuple(sorted(doc.metadata.items()))): doc for doc in docs
        }
        unique_documents = list(unique_documents_dict.values())
        print("--- doc.metadata.items() in %s seconds ---" % (time.time() - start_time))
        return unique_documents

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise NotImplementedError
