import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit App

st.set_page_config(page_title="LangChain Groq", page_icon=":robot:")
st.title ("Langchain: Summarize text from Youtube or website")
st.subheader('Summarize URL')




# Get the GROQ API Key and url fields to be summarize 

with st.sidebar:
    groq_api_key = st.text_input("GROQ API KEY", type= "password")
    
generic_url = st.text_input("URL", label_visibility="collapsed")

llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the content from YT or Website"):
    ## Validate 
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information")
    elif not validators.url(generic_url):
        st.error("Please provide the valid url")
    
    else:
        try:
            with st.spinner("Waiting ...."):
                # Loading the websit or YT data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify = False)
                data = loader.load()
                
                ## Chain for summarization 
                
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(data)
                
                
                st.success(output_summary)
        except Exception as e:
            st.error(f"Error: {e}")
