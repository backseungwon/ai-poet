from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatResult, ChatGeneration
from openai import OpenAI
import streamlit as st

# Streamlit Secrets에서 API Key와 Project ID 불러오기
api_key = st.secrets["OPENAI_API_KEY"]
project_id = st.secrets["OPENAI_PROJECT_ID"]

# OpenAI 클라이언트 생성 (project 지원)
client = OpenAI(api_key=api_key, project=project_id)

# 커스텀 ChatOpenAI 클래스 정의
class CustomChatOpenAI(ChatOpenAI):
    def __init__(self, client, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # OpenAI 공식 SDK 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.type, "content": m.content} for m in messages],
            temperature=self.temperature,
        )

        # LangChain이 요구하는 ChatResult로 변환
        content = response.choices[0].message.content
        generation = ChatGeneration(message=self._to_chat_message("assistant", content))
        return ChatResult(generations=[generation])

# ChatOpenAI 초기화
llm = CustomChatOpenAI(client=client, model="gpt-3.5-turbo", temperature=0.7)

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

# 문자열 출력 파서
output_parser = StrOutputParser()

# LLM 체인 구성
chain = prompt | llm | output_parser

# LLM 체인 테스트 실행
content = "코딩"
result = chain.invoke({"input": content + "에 대한 시를 써줘"})
print(result)

# Streamlit UI
st.title("인공지능 시인")

# 시 주제 입력 필드
content = st.text_input("시의 주제를 입력해주세요")
st.write("시의 주제는", content)

# 시 작성 요청하기
if st.button("시 작성 요청하기"):
    with st.spinner("AI가 시를 작성 중입니다..."):
        result = chain.invoke({"input": content + "에 대한 시를 써줘"})
        st.write(result)