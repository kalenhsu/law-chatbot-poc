import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.prompts import ChatMessagePromptTemplate, PromptTemplate

from dotenv import load_dotenv
from io import StringIO
import logging
import os

# %%
st.set_page_config(page_title="test search", page_icon="🌐")


def settings():
    # Vectorstore
    import faiss
    from langchain.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.docstore import InMemoryDocstore
    embeddings_model = OpenAIEmbeddings(deployment=os.environ["OPENAI_EMBEDDING_ENGINE"])
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # LLM
    from langchain.chat_models import AzureChatOpenAI
    openai_llm = AzureChatOpenAI(
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        deployment_name=os.environ["OPENAI_ENGINE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_type=os.environ["OPENAI_API_TYPE"],
    )

    # Search
    from langchain.utilities import GoogleSearchAPIWrapper
    search = GoogleSearchAPIWrapper()

    # Initialize 
    web_retriever_with_openai = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=openai_llm,
        search=search,
        num_search_results=5
    )

    return web_retriever_with_openai, openai_llm


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


def get_generated_question(log_text):
    import ast

    io_text = str(log_text)
    question_pattern = "INFO:langchain.retrievers.web_research:Questions for Google Search:"
    io_text = io_text[io_text.rfind(question_pattern):]
    generated_q_start = io_text.find("[")
    generated_q_end = io_text.find("]")
    generated_q = io_text[generated_q_start:generated_q_end + 1].replace("\\\n", "")
    generated_q = "\n".join(ast.literal_eval(generated_q))

    return generated_q


# %%
load_dotenv()

# Make retriever and llm
if "retriever" not in st.session_state:
    st.session_state["retriever"], st.session_state["llm"] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

# User input 
question = st.text_input("`Ask a question:`")

if question:
    prompt = """
    客戶詢問的問題是""" + question + """

    查詢到的資料是：

    {summaries}

    使用中文回覆客戶，並翻譯成繁體中文
    
    If there is insufficient context, write the reason how you cannot answer. 
    """
    chat_message_prompt = PromptTemplate(template=prompt, input_variables=["summaries"])

    # Set up logging
    main_handler = logging.StreamHandler()
    main_handler.setLevel(logging.INFO)

    log_stream = StringIO()
    stream_handler = logging.StreamHandler(stream=log_stream)

    logging.basicConfig(handlers=[main_handler, stream_handler])
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever,
                                                           chain_type_kwargs={"prompt": chat_message_prompt})

    # Write answer and sources
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
    result = qa_chain({"question": f"{question} {os.environ['question']}"},
                      callbacks=[retrieval_streamer_cb, stream_handler])

    answer.info("`Answer:`\n\n" + result["answer"])
    print("\n\n\n\n\n", log_stream.getvalue(), "\n\n\n\n\n")
    generated_question_by_llm = get_generated_question(log_text=log_stream.getvalue())
    st.info("`Related Questions:`\n\n" + generated_question_by_llm)
    # st.info("`Sources:`\n\n" + result["sources"])

    log_stream.truncate(0)
    log_stream.seek(0)
    log_stream = StringIO()
    # stream_handler.setStream(log_stream)
