import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()
KEY=os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(openai_api_key=KEY,model_name="gpt-3.5-turbo", temperature=0.3)

TEMPLATE="""
You are an expert in coding and solving problem. kindly help and solving problem. Iam learning how to code and want to see multiple approaches to solve problem. Please Code for {text} with both brute force and optimized approach, with clear time and space complexity of each, in {language}. Include explanation: {explain}
"""

EXPLANATION_TEMPLATE="""
You are an expert programmer. Kindly explain this code: {text}
"""

DEBUG_TEMPLATE="""
You are an expert coder. Kindly debug this code: {text}
"""

def generate_code(text, language, explain):
    code_generation_prompt = PromptTemplate(
        input_variables=["text", "language", "explain"],
        template=TEMPLATE
    )
    code_chain = LLMChain(llm=llm, prompt=code_generation_prompt, output_key="code", verbose=True)
    response = code_chain(
        {
            "text": text,
            "language": language,
            "explain": explain,
        }
    )
    return response["code"]

def generate_explanation(text):
    explanation_prompt = PromptTemplate(
        input_variables=["text"],
        template=EXPLANATION_TEMPLATE
    )
    explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt, output_key="explanation", verbose=True)
    response = explanation_chain(
        {
            "text": text,
        }
    )
    return response["explanation"]

def debug_code(text):
    debug_prompt = PromptTemplate(
        input_variables=["text"],
        template=DEBUG_TEMPLATE
    )
    debug_chain = LLMChain(llm=llm, prompt=debug_prompt, output_key="debug_info", verbose=True)
    response = debug_chain(
        {
            "text": text,
        }
    )
    return response["debug_info"]

# Streamlit app
st.title("Coding Assistant")

# User selects the operation
operation = st.selectbox("What do you want to do?", ["Code Generation", "Code Explanation", "Code Debugging"])

# User inputs
text = st.text_input("Enter the problem statement:")


if operation == "Code Generation":
    language = st.selectbox("Select the programming language:", ["C", "C++", "Java", "Python", "JavaScript", "Go"])
    explain = st.checkbox("Include explanation")
    if text:
        code = generate_code(text, language, explain)
        
        # Save history
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append(code)
        
        # Display generated code
        st.subheader("Generated Code:")
        st.write(code, language=language)
        
        # Add download button for the code
        st.download_button(
            label="Download Code",
            data=code,
            file_name='code.txt',
            mime='text/plain',
        )
        
        # Display history
        st.subheader("History:")
        for i, item in enumerate(st.session_state['history']):
            st.code(item, language=language)

elif operation == "Code Explanation":
    if text:  # Check if the user has entered a problem statement
        explanation = generate_explanation(text)
        st.subheader("Generated Explanation:")
        st.write(explanation)
elif operation == "Code Debugging":
    if text:  # Check if the user has entered a problem statement
        debug_info = debug_code(text)
        st.subheader("Debug Information:")
        st.write(debug_info)
