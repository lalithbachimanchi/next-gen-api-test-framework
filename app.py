import datetime
import replicate
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
import openai
from openai import OpenAI
import os
import logging
import json
import copy
import time
import pandas as pd
from io import StringIO
from glom import glom, PathAccessError
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader, select_autoescape
import html2text

_logger = logging.getLogger(__name__)


api_key = os.environ.get("OPENAI_API_KEY")
org_id = os.environ.get("OPENAI_ORG_ID")
max_tokens = int(os.environ.get('OPENAI_MAX_TOKENS', '4000'))
model = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')


@st.cache_data
def gpt_response(prompt):

    messages = [{"role": "system",
                 "content": " Generate programming code. Dont give explanation"}]

    messages.append({"role": "system",
                 "content": "Generate test automation code"})
    if prompt:
        messages.append({"role": "user", "content": f"{prompt}"})

    print(messages)
    try:
        client = OpenAI(api_key=st.session_state['AUTH_KEY'])

        response = client.chat.completions.create(
            model=model,
            messages= messages,
            max_tokens=int(max_tokens),
        )
        print(f"Tokens Used for Generating Code: {response.usage.completion_tokens}")
        st.write(f"Tokens Used: {response.usage.completion_tokens}")

        print(f"GPT Response: {response.json()}")

        return response.choices[0].message.content
    except Exception as e:
        print(f'Exception Occurred On GPT API: {e}')
        return None


@st.cache_data
def gemini_response(prompt):

    genai.configure(api_key=st.session_state['AUTH_KEY'])

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)

    print(response.text)

    return response.text


def generate_resp_from_llm(model, prompt):

    # llm = None

    if model == 'LLAMA2':

        os.environ['REPLICATE_API_TOKEN'] = st.session_state['REPLICATE_API_TOKEN']

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

        return "".join(output)

    elif model == 'Gemini':

        return gemini_response(prompt)

        # llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=st.session_state['AUTH_KEY'])

    elif model == 'OpenAI':
        return gpt_response(prompt)

    #     llm = OpenAI(openai_api_key=st.session_state['AUTH_KEY'])
    #
    # prompt = PromptTemplate.from_template(f"You are a helpful coding assistant.{prompt}")
    #
    # llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    #
    # resp = llm_chain.run(topic=prompt)
    #
    # st.markdown(resp)

    # return resp


def process_swagger_json(swagger_json):
    swagger_result = {}
    for path, path_values in swagger_json.get('paths').items():
        if path not in swagger_result:
            swagger_result[path] = []
        for each_method, method_value in path_values.items():

            try:
                schema = glom(method_value, 'requestBody.content.application/json.schema.$ref')
                model = schema.split('/')[-1] if schema else None
            except PathAccessError:
                schema = None
                model = None

            swagger_result[path].append({'method': each_method,
                                         'parameters':
                                             [each_param['name'] for each_param in method_value['parameters'] if
                                              each_param['in'] == 'query'],
                                         'schema': schema,
                                         'model': model,
                                         'response_codes': list(method_value.get('responses').keys())
                                         })
    model_structure_temp = {}
    swagger_components = swagger_json['components'] if 'components' in swagger_json else ''
    model_schemas = swagger_json.get('components')['schemas'] if swagger_components else None
    if model_schemas:
        for model_name, model_body in model_schemas.items():
            if model_name not in model_structure_temp:
                model_structure_temp[model_name] = {}
            if 'enum' in model_body:
                model_structure_temp[model_name] = model_body
                continue
            if 'properties' in model_body:
                for property_name, property_body in model_body.get('properties').items():
                    if 'type' in property_body:
                        if property_body['type'] == 'array' and '$ref' in property_body['items']:
                            sub_model = property_body['items']['$ref'].split('/')[-1]
                            model_structure_temp[model_name][property_name] = {"type": "sub_model",
                                                                               "model_name": sub_model}
                            continue
                    if '$ref' in property_body:
                        sub_model = property_body['$ref'].split('/')[-1]
                        model_structure_temp[model_name][property_name] = {"type": "sub_model",
                                                                           "model_name": sub_model}

                    else:
                        model_structure_temp[model_name][property_name] = property_body

    model_structure_final = copy.deepcopy(model_structure_temp)

    for model_name, model_body in model_structure_final.items():
        for property_name, property_body in model_body.items():
            if 'type' in property_body and property_body['type'] == "sub_model":
                model_structure_final[model_name][property_name]['details'] = model_structure_temp[
                    property_body['model_name']]

    st.session_state['model_structure_final'] = model_structure_final
    st.session_state['swagger_result'] = swagger_result


@st.cache_data
def render_template(api_endpoint_choices):
    print('\n\n\-----api_endpoint_choices', api_endpoint_choices)
    selected_prompts = {}
    response_codes = None
    request_body = None
    selected_prompt_messages = []
    for each_endpoint in api_endpoint_choices:
        data = []
        for each_method in st.session_state['swagger_result'][each_endpoint]:
            if 'schema' in each_method and each_method['schema']:
                schema_name = each_method['schema'].split('/')[-1]
                request_body = st.session_state['model_structure_final'][schema_name]
                response_codes = [c for c in each_method['response_codes'] if c != 'default']
            temp_dict = {'method': each_method['method'],
                         'response_codes': response_codes,
                         'request_body': request_body,
                         'endpoint': each_endpoint}
            data.append(temp_dict)
        selected_prompts.update({each_endpoint: data})

    print(selected_prompts)
    for each_url, each_prompt in selected_prompts.items():
        print(each_url)
        env = Environment(
            loader=FileSystemLoader('.'),
            autoescape=select_autoescape(['html'])
        )
        # Load the HTML template from the file
        template = env.get_template('prompt_template.html')

        context = {
            'framework': st.session_state['framework'],
            'end_point': each_url,
            'prompts': each_prompt,
            'base_url': st.session_state['swagger_json']['servers'][0]['url'],
        }

        # st.json(context)
        rendered_html = template.render(context)

        html_string = html2text.html2text(rendered_html)
        selected_prompt_messages.append(html_string)

    return selected_prompt_messages


def process_json(uploaded_file):
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        st.session_state['swagger_json'] = json.loads(string_data)
    except:
        st.error('There was an error processing the uploaded JSON. Please upload a valid JSON', icon="🚨")
        st.stop()


def main():
    if 'swagger_json' not in st.session_state:
        st.session_state['swagger_json'] = None
    if 'framework' not in st.session_state:
        st.session_state['framework'] = ''
    if 'selected_api_endpoint_choices' not in st.session_state:
        st.session_state['selected_api_endpoint_choices'] = []
    if 'selected_prompt_names' not in st.session_state:
        st.session_state['selected_prompt_names'] = []
    if 'swagger_result' not in st.session_state:
        st.session_state['swagger_result'] = {}
    st.set_page_config(page_title="Code Generator")
    st.title('NextGen API Test Framework')
    with st.sidebar:
        with st.form(key="user_choices", clear_on_submit=False):
            print('inside user_choices form')
            llm_model = st.radio(label='Select LLM Model', options=['OpenAI', 'Gemini', 'LLAMA2'])
            auth_key = st.text_input("Enter LLM Auth Key", type="password")
            test_type = st.selectbox("Test Type", ["API Tests"])
            st.session_state['framework'] = st.selectbox(
                "Framework", ["Python Pytest", "Python Robot", "Karate", "Cypress", "K6", "Test NG"])
            uploaded_file = st.file_uploader("Choose Swagger JSON file")
            parse_apis_submitted = st.form_submit_button("Parse APIs")

        if parse_apis_submitted:
            if not auth_key:
                st.error('Please enter your OpenAI API key!', icon='⚠')
                # st.stop()

            if uploaded_file is None:
                st.error('Swagger JSON Schema is mandatory to proceed!', icon="🚨")
                st.stop()

            if llm_model == 'LLAMA2':
                if 'REPLICATE_API_TOKEN' not in st.session_state:
                    st.session_state['REPLICATE_API_TOKEN'] = auth_key
            else:
                if 'AUTH_KEY' not in st.session_state:
                    st.session_state['AUTH_KEY'] = auth_key
            if 'llm_model' not in st.session_state:
                st.session_state['llm_model'] = llm_model

    if uploaded_file is not None:
        with st.spinner('Processing JSON...'):
            process_json(uploaded_file)

        if not st.session_state['swagger_json']:
            st.error("Uploaded JSON file is empty")
            st.stop()

    if st.session_state['swagger_json']:
        process_swagger_json(st.session_state['swagger_json'])
        with st.form(key="select_apis", clear_on_submit=True):
            print('inside select_apis form')
            print(st.session_state['selected_api_endpoint_choices'])

            api_options_to_show = [op for op in list(st.session_state['swagger_result'].keys()) if op not in st.session_state.get('selected_api_endpoint_choices')]

            print('\n\n', api_options_to_show, '\n\n\n')
            api_endpoint_choices = st.multiselect(label='Select the endpoints to generate API Tests', options=api_options_to_show, max_selections=3)
            api_submitted = st.form_submit_button("Vectorize Input")

        if api_submitted:
            st.session_state['selected_api_endpoint_choices'].extend(api_endpoint_choices)

            selected_prompt_messages = render_template(api_endpoint_choices)

            st.session_state.selected_prompt_names = []
            for index, value in enumerate(selected_prompt_messages, start=1):
                # user_prompt_value = None
                print(index, f"user_prompt_{index}")

                user_prompt_value = st.session_state[f"user_prompt_{index}"]\
                    if st.session_state.get(f"user_prompt_{index}") else value

                st.session_state[f"prompt_user_{index}"] = st.text_area(
                    label=f'LLM Prompt {index}', value=user_prompt_value, height=600, key=f'user_prompt_{index}')
                st.session_state.selected_prompt_names.append(f"user_prompt_{index}")
    if st.session_state.selected_prompt_names:
        with st.form(key="generate_llm_response", clear_on_submit=False):
            print('inside generate_llm_response')
            llm_resp_submitted = st.form_submit_button("Invoke")

        if llm_resp_submitted:

            print('inside generate code from llm')

            print(st.session_state.selected_prompt_names, '\n\n\n')

            st.session_state.downloadable_files = []
            resp_strings = []
            for each_selected_prompt in st.session_state.selected_prompt_names:
                with st.spinner("Langchain"):
                    # resp = langchain_output(prompt=str(st.session_state[each_selected_prompt]))
                    resp = generate_resp_from_llm(model=st.session_state['llm_model'], prompt=str(st.session_state[each_selected_prompt]))
                st.write(f"Response for {each_selected_prompt}")
                st.markdown(resp)
                resp_strings.append(resp)
                del st.session_state[each_selected_prompt]

            file_result = '\n'.join(resp_strings)
            st.download_button(
                label="Download Response",
                data=file_result,
                file_name=f"llm_response_{datetime.datetime.now()}.txt",
                mime="text/plain",
                key=f"download_button",
            )
            del st.session_state['selected_prompt_names']




if __name__ == "__main__":
    main()
