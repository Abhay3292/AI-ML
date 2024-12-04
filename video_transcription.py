import os
import io
import getpass
from langchain_community.document_loaders import YoutubeLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv



load_dotenv()

#Loading the transcript
#video : https://www.youtube.com/watch?v=NJcOoj5WBtU


def get_video_title(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features='html.parser')

    link = soup.find_all(name='title')[0]
    title = str(link)
    title = title.replace('<title>','').replace('</title>','')

    return title



# video_infos = f'''Video info:
# Title: {video_title}
# Transcript : {transcript}
# '''


def save_transcript_to_file(video_infos, video_title):
    with io.open(video_title+'_transcript.txt','w',encoding='utf-8') as f:
        f.write(video_infos)

#save_transcript(video_infos,video_title,infos)

def model_hf_hub(model='meta-llama/Meta-Llama-3-8B-Instruct', temperature=0.1):
    return HuggingFaceEndpoint(
        repo_id=model,
        temperature=temperature,
        return_full_text=False,
        max_new_tokens=512
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

model_class='hf_hub' # @param ['hf_hub','openai','ollama']


def llm_chain(model_class):
    system_prompt = 'You are a helpful virtual assistant answering a query based on a video transcript, which will be provided below.'
    inputs = "Query: {query} \n Transcription: {transcript}"

    if model_class.startswith("hf"):
        user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>".format(inputs)
    else:
        user_prompt = "{}".format(inputs)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
          ("user", user_prompt)
    ])

    llm = model_hf_hub()
    if model_class == 'openai':
        llm = model_openai()
    elif model_class == 'ollama':
        llm = model_ollama()
    chain = prompt_template | llm | StrOutputParser()
    return chain

def get_video_info(url,language='en', translation=None,save_transcript=False):
    video_loader = YoutubeLoader.from_youtube_url(
        url, 
        language = language,
        translation=translation
    )
    infos = video_loader.load()[0]
    transcript = infos.page_content
    video_title = get_video_title(url)
    if save_transcript:
        save_transcript_to_file(transcript,video_title)
    return transcript,video_title


def interpret_video(url, query='summarize', model_class='hf_hub', language='en', translation=None, save_transcript=False):
    try:
        transcript , video_title = get_video_info(url,language,translation,save_transcript)
        chain = llm_chain(model_class)
        res = chain.invoke({'transcript': transcript, 'query': query})
        print(res)
    except Exception as e:
        print('Error loading transcript')
        print(e)

interpret_video(url='https://www.youtube.com/watch?v=Fe4ZSFlYfNU', query='summarize', model_class='hf_hub', language='en')

