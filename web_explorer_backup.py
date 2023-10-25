import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import ChatMessagePromptTemplate, PromptTemplate

from WebRetrieverAccelerate.my_web_research import WebResearchRetriever

from dotenv import load_dotenv
import json
import logging
import os
import requests

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


def query_with_button_value(input_value, llm, web_retriever, role):
    output_prompt = \
        f"現在你是一個{role['position']}的角色，會用{role['tone']}的口吻回答問題。\n" \
        + "當有用戶詢問的問題是「" + input_value + """」，
        
        且你查詢到的資料是：
    
        {summaries}
    
        請參考上述問題及資料，使用繁體中文、以列點的方式，用""" + role['tone'] + """的口吻回覆用戶。
        若查詢到的資料不足以回答，請說明無法回答的原因、並列舉建議之提問。
        """
    output_prompt_template = PromptTemplate(template=output_prompt, input_variables=["summaries"])

    # Set up logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever,
                                                           chain_type_kwargs={"prompt": output_prompt_template})

    # Write answer and sources
    output_answer = st.empty()
    placeholder = st.empty()
    placeholder.text("詢問中，請稍候...")
    stream_handler = StreamHandler(output_answer, initial_text="`Answer:`\n\n")

    try:
        with open("role-setting.json") as j:
            disclaimer_info = json.load(j)
        result = qa_chain({"question": f"{input_value}？ {os.environ['question']}"}, callbacks=[stream_handler])
        placeholder.empty()
        output_answer.info("`回答:`\n\n" + result["answer"] \
                           + f"\n\n以上資訊僅供參考，如有需要更精準資訊請諮詢專業律師或{disclaimer_info['服務單位']}服務團隊。" \
                           + f"\n\n{disclaimer_info['服務單位']}電話：{disclaimer_info['服務電話']}" \
                           + f"\n\n{disclaimer_info['服務單位']}官網：{disclaimer_info['服務官網']}")

    except requests.Timeout as timeErr:
        placeholder.empty()
        output_answer.info("`警告訊息:`\n\n" + "非常抱歉，當前伺服器過於擁擠。\n請稍待一段時間、並重新整理頁面再做嘗試。")
        print(timeErr)

    except Exception as e:
        placeholder.empty()
        output_answer.info("`警告訊息:`\n\n" + "非常抱歉，當前系統遭遇到問題。\n請稍待一段時間、並重新整理頁面再做嘗試。")
        print(e)

    return True


def query_with_button_value2(input_value, llm, web_retriever, role):
    output_prompt = \
        f"現在你是一個{role['position']}的角色，會用{role['tone']}的口吻回答問題。\n" \
        + "當有用戶詢問的問題是「" + input_value + """  李彥秀使否對該議題有貢獻？」，
        
        且你查詢到的資料是：
    
        {summaries}
    
        請參考上述問題及資料，使用繁體中文、以列點的方式，用""" + role['tone'] + """的口吻回覆用戶。
        若查詢到的資料不足以回答，請說明無法回答的原因、並列舉建議之提問。
        """
    output_prompt_template = PromptTemplate(template=output_prompt, input_variables=["summaries"])

    # Set up logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever,
                                                           chain_type_kwargs={"prompt": output_prompt_template})

    # Write answer and sources
    output_answer = st.empty()
    placeholder = st.empty()
    placeholder.text("詢問中，請稍候...")
    stream_handler = StreamHandler(output_answer, initial_text="`Answer:`\n\n")

    try:
        with open("role-setting.json") as j:
            disclaimer_info = json.load(j)
        result = qa_chain({"question": f"{input_value} 李彥秀的貢獻為何？"}, callbacks=[stream_handler])
        placeholder.empty()
        output_answer.info("`回答:`\n\n" + result["answer"] \
                           + f"\n\n以上資訊僅供參考，如有需要更精準資訊請諮詢專業律師或{disclaimer_info['服務單位']}服務團隊。" \
                           + f"\n\n{disclaimer_info['服務單位']}電話：{disclaimer_info['服務電話']}" \
                           + f"\n\n{disclaimer_info['服務單位']}官網：{disclaimer_info['服務官網']}")

    except requests.Timeout as timeErr:
        placeholder.empty()
        output_answer.info("`警告訊息:`\n\n" + "非常抱歉，當前伺服器過於擁擠。\n請稍待一段時間、並重新整理頁面再做嘗試。")
        print(timeErr)

    except Exception as e:
        placeholder.empty()
        output_answer.info("`警告訊息:`\n\n" + "非常抱歉，當前系統遭遇到問題。\n請稍待一段時間、並重新整理頁面再做嘗試。")
        print(e)

    return True


# %%
load_dotenv()

# Make retriever and llm
if "retriever" not in st.session_state:
    st.session_state["retriever"], st.session_state["llm"] = settings()
my_retriever = st.session_state.retriever
my_llm = st.session_state.llm

# User input
st.title("國眾法律聊天機器人")
st.header("Leosys Law Chatbot\n")
option_role = {
    "position": st.selectbox("請選擇您期望的服務角色", ("專業律師", "立法委員", "高級工程師")),
    "tone": st.selectbox("請選擇您期望的服務口吻", ("和藹親切", "專業冷酷", "熱情開放"))
}
init_status = 0
input_text_container = st.empty()
question = input_text_container.text_input("`請輸入您的問題👇`")
example_question_1 = st.button("建議問題：幾月繳牌照稅？")
example_question_2 = st.button("建議問題：酒駕罰多少？")
example_question_3 = st.button("建議問題：網路購物可以退貨嗎？")

if example_question_1:
    input_text_container.text_input("`請輸入您的問題👇`", "幾月繳牌照稅？")
    query_with_button_value("幾月繳牌照稅？", my_llm, my_retriever, option_role)
    question = False

if example_question_2:
    input_text_container.text_input("`請輸入您的問題👇`", "酒駕罰多少？")
    query_with_button_value("酒駕罰多少？", my_llm, my_retriever, option_role)
    question = False

if example_question_3:
    input_text_container.text_input("`請輸入您的問題👇`", "網路購物可以退貨嗎？")
    query_with_button_value("網路購物可以退貨嗎？", my_llm, my_retriever, option_role)
    question = False

if question and (question not in ["幾月繳牌照稅？", "酒駕罰多少？", "網路購物可以退貨嗎？"]):
    query_with_button_value(question, my_llm, my_retriever, option_role)
    question = False
