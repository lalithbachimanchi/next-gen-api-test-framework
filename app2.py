import os
import streamlit as st
import replicate

# from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI

# load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def generate_resp_from_llm(model, open_api_key, prompt):

    llm = None

    if model == 'LLAMA2':

        os.environ['REPLICATE_API_TOKEN'] = open_api_key

        output = replicate.run(
            "replicate/llama-7b:ac808388e2e9d8ed35a5bf2eaa7d83f0ad53f9e3df31a42e4eb0a0c3249b3165",
            input={
                "debug": False,
                "top_p": 0.95,
                "prompt": prompt,
                "max_length": 500,
                "temperature": 0.8,
                "repetition_penalty": 1
            }
        )
        print("".join(output))
        st.markdown("".join(output))

        return output

    elif model == 'Gemini':

        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=open_api_key)

    elif model == 'OpenAI':
        llm = OpenAI(openai_api_key=open_api_key)


    tweet_prompt = PromptTemplate.from_template(f"You are a helpful coding assistant.{prompt}")

    llm_chain = LLMChain(llm=llm, prompt=tweet_prompt, verbose=True)

    resp =  llm_chain.run(topic=topic)

    st.markdown(resp)

    return resp

if __name__=="__main__":
    topic = "how ai is really cool"
    llm_model = st.radio(label='Select LLM Model', options=['OpenAI', 'Gemini', 'LLAMA2'])
    open_api_key = st.text_input("Enter LLM Auth Key", type="password")
    prompt = st.text_input("What do you want to know?")
    submit = st.button("Submit")
    if submit:
        with st.spinner('Generating Response...'):
            generate_resp_from_llm(llm_model,open_api_key, prompt=prompt)
    # resp = tweet_chain.run(topic=topic)
    # print(resp)
