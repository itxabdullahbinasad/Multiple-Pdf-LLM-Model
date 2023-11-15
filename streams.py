import streamlit as st 
import os 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# import langchain 
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import langchain.llms as llms
from htmlTemplates import bot_template, user_template 
from dotenv import load_dotenv


google_api="AIzaSyCS1WxgSqNqEVPWjgMEYU1d44UHld8RT8w"


def get_pdf_text(pdf_docs):
   
    text=""
    for pdf in pdf_docs:
        pdf_text_extract=PdfReader(pdf)
        for page in pdf_text_extract.pages:
            text+=page.extract_text()
        return text
    return get_pdf_text

# Converting raw pdf text into text chunks 

def get_text_chunks(raw_text):

    text_splitter=CharacterTextSplitter(separator="\n" , chunk_size=1000 , chunk_overlap=200 , length_function=len)
    text_chunks=text_splitter.split_text(raw_text)

    return text_chunks

# Getting the embeddings and retrieval system 
def get_vectorstore(chunks):
    embeddings=HuggingFaceInstructEmbeddings()
    vector_store=FAISS.from_texts(texts=chunks , embedding=embeddings)
    return vector_store

# Preparing the llms 
def get_conversation(vectors):
    llms=GooglePalm(google_api_key=google_api)
    memory=ConversationBufferMemory(memory_key="chat_history" , return_messages=True)
    conversational_chain=ConversationalRetrievalChain.from_llm(llm=llms , retriever=vectors.as_retriever(), memory=memory)
    return conversational_chain

# User Question 

def get_userinput(user_question):
    response=st.session_state.conversation({"question":user_question})
    st.session_state.chat_history=response["chat_history"]
    for i , message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    st.set_page_config(page_title="Chat with Multiple Pdfs" , page_icon=":ice:" , initial_sidebar_state="expanded")
    st.header("Chat with multiple PDF's :🧊:")
    user_question= st.text_input("Ask a question about your documents")

    if user_question:
        get_userinput(user_question)

    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = None

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"]=None

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader("Upload Your Files Here" , accept_multiple_files=True)
        if st.button("Process"):
                
            with st.spinner("Processing"):
            #get pdf text raw

                raw_text=get_pdf_text(pdf_docs)
                st.write(raw_text)
                print(type(raw_text))
            # convert the text into chunks 
                chunks=get_text_chunks(raw_text)
                st.write(chunks)
                #Create a vector db and store them there and add retrieval system
                vectors = get_vectorstore(chunks)
                st.write(vectors)
                # Creating llm and chains
                print("Before setting conversation:", st.session_state.conversation)

                st.session_state.conversation = get_conversation(vectors)
                
                print("After setting conversation:", st.session_state.conversation)


if __name__=="__main__":
    main()