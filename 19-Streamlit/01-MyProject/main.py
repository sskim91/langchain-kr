import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain import hub

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

st.title("나만의 첫 번째 Streamlit 앱")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화 기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

with st.sidebar:
    clear_btn = st.button("대화 기록 초기화")

    selected_prompt = st.selectbox(
        "프롬프트 유형을 선택하세요", ("기본 프롬프트", "SNS 게시글", "요약"), index=0
    )


# 이전 대화를 출력
def print_message():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, content):
    """대화 메시지를 추가하는 함수"""
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 체인 생성 함수
def create_chain(prompt_type):
    # prompt | llm | output_parser

    # 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 친절한 AI 어시스턴트입니다. 다음의 질문에 간결하게 답변해 주세요",
            ),
            ("user", "#Question:\n{question}"),
        ]
    )

    if prompt_type == "SNS 게시글":
        # SNS 게시글 프롬프트
        prompt = load_prompt("prompts/sns.yaml")  # 프롬프트 로드
    elif prompt_type == "요약":
        # 요약 프롬프트
        prompt = hub.pull("teddynote/chain-of-density-korean:946ed62d")  # 프롬프트 로드

    # GPT
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser
    return chain


if clear_btn:
    # 대화 기록 초기화
    st.session_state["messages"] = []
    st.success("대화 기록이 초기화되었습니다.")

# 이전 대화 기록 출력
print_message()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 사용자의 입력을 기반으로 한 출력
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)
    # 체인 생성
    chain = create_chain(selected_prompt)

    # 스트리밍 호출
    response = chain.stream({"question": user_input})

    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화 기록에 추가
    add_message("user", user_input)
    add_message("assistant", ai_answer)
