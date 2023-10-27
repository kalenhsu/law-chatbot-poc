import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

from WebRetrieverAccelerate.my_web_research import WebResearchRetriever

from dotenv import load_dotenv
import json
import logging
import os
import requests
import time

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
        streaming=True,
    )

    # Search
    from langchain.utilities import GoogleSearchAPIWrapper
    search = GoogleSearchAPIWrapper()

    # Initialize 
    web_retriever_with_openai = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=openai_llm,
        search=search,
        num_search_results=4
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


def check_similar_politics(politics_doc, input_q, llm):
    with open(politics_doc) as j:
        politics = json.load(j)
    n_politics = str(len(politics.keys()))
    politics = "\n\n".join([
        title+"：\n"+str(politics[title]).replace("', '", "\n")[2:-2] for title in politics.keys()
    ])
    politics_prompt = f"{politics}\n\n請問以上{n_politics}"+"""點政見，何者跟「{question}」最有關聯？
            
            請在第一行加上「有關聯」三個字、並在第二行將最有關的政見印出，並在第三行以一百五十字以內解釋兩者之間如何關聯。
            若全部都無關的話在第一行加上「無關聯」三個字。
            """.format(question=input_q)
    politics_msg = HumanMessage(content=politics_prompt)
    result = llm(messages=[politics_msg]).to_json()["kwargs"]["content"]

    return result


def query_with_button_value(input_value, llm, web_retriever):
    ttl_start_time = time.time()
    with open("role-setting.json") as j:
        role_setting_info = json.load(j)

    output_prompt = \
        f"請扮演一個台灣地區的{role_setting_info['服務角色']}的角色，會用{role_setting_info['服務口吻']}的口吻回答問題。\n" \
        + "當用戶詢問的問題是「" + input_value + """」，
        
        且你查詢到的資料是：
    
        {summaries}
    
        請參考上述問題及資料，於四百字以內、使用繁體中文、以列點的方式，用""" + role_setting_info['服務口吻'] + """的口吻回覆用戶的所有問題。
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
        result = qa_chain({"question": f"{input_value} {os.environ['question']}"},
                          callbacks=[stream_handler])
        output_answer.info("`回答:`\n\n" + result["answer"] \
                           + f"\n\n以上資訊僅供參考，如有需要更精準資訊請諮詢專業律師或{role_setting_info['服務單位']}服務團隊。" \
                           + f"\n\n{role_setting_info['服務單位']}電話：{role_setting_info['服務電話']}" \
                           + f"\n\n{role_setting_info['服務單位']}官網：{role_setting_info['服務官網']}")
        print("--- Basic Answering in %s seconds ---" % (time.time() - ttl_start_time))

        politics_result = check_similar_politics(politics_doc="politics.json", input_q=input_value, llm=llm)
        print("check_similar_politics: ", politics_result)
        if politics_result[:3] == "有關聯":
            politics_result = politics_result[3:].replace('\n', '\n\n')
            st.info('`Note:`\n\n' \
                    + f"以上提問「{input_value}」與{role_setting_info['服務單位']}之政見有關聯。\n" \
                    + f"\n\n{role_setting_info['服務單位']}曾經提出：{politics_result}")
        placeholder.empty()

    except requests.Timeout as timeErr:
        placeholder.empty()
        output_answer.info("`警告訊息:`\n\n" + "非常抱歉，當前伺服器過於擁擠。\n請稍待一段時間、並重新整理頁面再做嘗試。")
        print(timeErr)

    except Exception as e:
        placeholder.empty()
        output_answer.info("`警告訊息:`\n\n" + "非常抱歉，當前系統遭遇到問題。\n請稍待一段時間、並重新整理頁面再做嘗試。")
        print(e)

    print("--- Total Running in %s seconds ---" % (time.time() - ttl_start_time))
    print("Done. \n\n")

    return True


# %% Streamlit Run
load_dotenv()

# Make retriever and llm
if "retriever" not in st.session_state:
    st.session_state["retriever"], st.session_state["llm"] = settings()
my_retriever = st.session_state.retriever
my_llm = st.session_state.llm

# User input
st.title("國眾法律聊天機器人")
st.header("Leosys Law Chatbot\n")
input_text_container = st.empty()
question = input_text_container.text_input("`請輸入您的問題👇`")
example_question_1 = st.button("建議問題：幾月繳汽機車牌照稅？")
example_question_2 = st.button("建議問題：酒駕罰多少？")
example_question_3 = st.button("建議問題：網路購物可以退貨嗎？")

if example_question_1:
    input_text_container.text_input("`請輸入您的問題👇`", "幾月繳汽機車牌照稅？")
    query_with_button_value("幾月繳汽機車牌照稅？", my_llm, my_retriever)
    question = False

if example_question_2:
    input_text_container.text_input("`請輸入您的問題👇`", "酒駕罰多少？")
    query_with_button_value("酒駕罰多少？", my_llm, my_retriever)
    question = False

if example_question_3:
    input_text_container.text_input("`請輸入您的問題👇`", "網路購物可以退貨嗎？")
    query_with_button_value("網路購物可以退貨嗎？", my_llm, my_retriever)
    question = False

if question and (question not in ["幾月繳汽機車牌照稅？", "酒駕罰多少？", "網路購物可以退貨嗎？"]):
    query_with_button_value(question, my_llm, my_retriever)
    question = False
