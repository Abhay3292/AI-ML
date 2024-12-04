import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser

import torch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv

import os
import faiss
import tempfile
import time
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

#streamlit settings
st.set_page_config(page_title="Chat with documents 📚", page_icon='📚')
st.title('Chat with documents 📚')

model_class='hf_hub' # @param ['hf_hub','openai','ollama']

def model_hf_hub(model='meta-llama/Meta-Llama-3-8B-Instruct', temperature=0.1):
    return HuggingFaceEndpoint(
        repo_id=model,
        temperature=temperature,
        return_full_text=False,
        max_new_tokens=512,
        model_kwargs={
        #"max_length":64
        #"stop": ["<|eot_id|>"]
    }
    )

def model_openai(model = 'pt-4o-mini', temperature=0.1):
    return ChatOpenAI(
        model=model,
        temperature=temperature
    )

def model_ollama(model='phi3', temperature=0.1):
    return ChatOllama(
        model=model,
        temperature=temperature
    )

#create side panel in interface

uploads = st.sidebar.file_uploader(
    label='Upload files', type=['pdf'],
    accept_multiple_files=True
)

if not uploads:
    st.info('Please send some files to continue')
    st.stop()

def config_retrieval(uploads):
    #load
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, 'wb') as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    #split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    #embedding
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

    #store
    vectorestore = FAISS.from_documents(splits,embeddings)
    vectorestore.save_local('vectorestore/db_faiss')

    #retrieve
    retriever = vectorestore.as_retriever(search_type='mmr', 
                                          search_kwargs={'k':3, 'fetch_k':4})
    
    return retriever

def config_rag_chain(model_class, retriever):
    if model_class == 'hf_hub':
        llm =  model_hf_hub()
    if model_class == 'openai':
        llm = model_openai()
    elif model_class == 'ollama':
        llm = model_ollama()
    
    # Prompt definition
    if model_class.startswith("hf"):
        token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        token_s, token_e = "", ""

    # Contextualization prompt
    context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = 'Question: {input}' + token_e
    context_q_prompt = ChatPromptTemplate.from_messages([
        ('system', context_q_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', context_q_user_prompt)
    ])

    #Chain for contextualization
    history_aware_retriever = create_history_aware_retriever(llm=llm,
                                                             retriever=retriever,
                                                             prompt=context_q_prompt)
    
    # Q&A Prompt
    qa_prompt_template = """You are a helpful virtual assistant answering general questions.
    Use the following bits of retrieved context to answer the question.
    If you don't know the answer, just say you don't know. Keep your answer concise.
    Answer in English. \n\n
    Question: {input} \n
    Context: {context}"""

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    # Configure LLM and Chain for Q&A

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain,
    )

    return rag_chain

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content='Hi, I am your virtual assistant! How can I help you?')
    ]

if 'doc_list' not in st.session_state:
    st.session_state.doc_list = None

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Enter your message here...")
if user_query is not None and user_query != '' and uploads is not None:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        if st.session_state.doc_list != uploads:
            st.session_state.doc_list = uploads
            st.session_state.retriever = config_retrieval(uploads)
        rag_chain = config_rag_chain(model_class, st.session_state.retriever)
        result = rag_chain.invoke({'input': user_query, 
                                   'chat_history': st.session_state.chat_history})
        resp = result['answer']
        st.write(resp)

         # show the source
        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', 'Page not specified')

            ref = f":link: Source {idx}: *{file} - p. {page}*"
            print(ref)
            with st.popover(ref):
                st.caption(doc.page_content)

    st.session_state.chat_history.append(AIMessage(content=resp))

    
