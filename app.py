import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import os 
os.environ["OPENAI_API_KEY"] = "######"

def upload_htmls():
  loader = DirectoryLoader(path='hr_policies')
  documents = loader.load()
  print(f"Loaded Documents : {len(documents)}")

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  texts = text_splitter.split_documents(documents)

  embeddings = OpenAIEmbeddings()
  db = FAISS.from_documents(texts, embeddings)

  db.save_local('hr_policies_db')

def query(question,chat_history):
  embeddings = OpenAIEmbeddings()
  db = FAISS.load_local('hr_policies_db', embeddings,allow_dangerous_deserialization=True)
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

  retrieval_chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=db.as_retriever(),
      return_source_documents=True
  )

  result = retrieval_chain({"question": question, "chat_history": chat_history})
  return result

def show_ui():
  st.title("Your truly Human Resource assistant")
#   st.image("chat_image.png")
  st.subheader("Please enter you HR query ")

  if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_history = []

  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  if prompt := st.chat_input("Enter your HR Policy related query "):
    with st.spinner("Working on your query..."):
      response = query(question=prompt, chat_history=st.session_state.chat_history)
      with st.chat_message("user"):
        st.markdown(prompt)
      with st.chat_message("assistant"):
        st.markdown(response["answer"])

      st.session_state.messages.append({"role":"user","content":prompt})
      st.session_state.messages.append({"role":"assistant","content":response["answer"]})
      st.session_state.chat_history.extend([(prompt,response['answer'])])

if __name__ == "__main__":
    # upload_htmls()
    show_ui()