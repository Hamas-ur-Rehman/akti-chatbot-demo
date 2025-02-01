import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage,HumanMessage,ToolMessage
from langchain_core.tools import tool
from datetime import datetime
from weather import get_weather
from data_retriever import pdf_database,csv_database,web_database

@tool
def get_latest_datetime():
    """
    Returns the latest date and time as a formatted string.
    Format: YYYY-MM-DD HH:MM:SS
    """
    current_datetime = datetime.now()
    return current_datetime.strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_latest_weather(location_name):
    """
    Returns the latest weather information for a given location.
    """
    return get_weather(location_name)

@tool
def get_pdf_data(question):
    """
    Returns the data from the pdf files : data is about EMR records
    """
    return pdf_database(question)

@tool
def get_csv_data(question):
    """
    Returns the data from the csv files : data is about insurance records of humans
    """
    return csv_database(question)

@tool
def get_web_data(question):
    """
    Returns the data from the web pages : data is about news
    """
    return web_database(question)


tools = [
    get_latest_datetime,
    get_latest_weather,
    get_pdf_data,
    get_csv_data,
    get_web_data
    ]

llm = ChatOpenAI(
    temperature=0.5,
).bind_tools(tools)

PROMPT = """You are a funny chatbot that can answer any question.
You can answer any question, no matter how difficult it is.
You can also answer questions that are not related to the topic.
You always use emojis and jokes in your answers
"""
messages = [SystemMessage(PROMPT)]

def chat(question):
    messages.append(HumanMessage(question))
    response = llm.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'get_latest_datetime':
                messages.append(ToolMessage(get_latest_datetime.invoke(input=""),tool_call_id=tool_call['id']))
            elif tool_call['name'] == 'get_latest_weather':
                location_name = tool_call['args']['location_name']
                messages.append(ToolMessage(get_latest_weather.invoke(input=location_name),tool_call_id=tool_call['id']))
            elif tool_call['name'] == 'get_pdf_data':
                question = tool_call['args']['question']
                messages.append(ToolMessage(get_pdf_data.invoke(input=question),tool_call_id=tool_call['id']))
            elif tool_call['name'] == 'get_csv_data':
                question = tool_call['args']['question']
                messages.append(ToolMessage(get_csv_data.invoke(input=question),tool_call_id=tool_call['id']))
            elif tool_call['name'] == 'get_web_data':
                question = tool_call['args']['question']
                messages.append(ToolMessage(get_web_data.invoke(input=question),tool_call_id=tool_call['id']))
                
        response = llm.invoke(messages)
        messages.append(response)

    # print("Chatbot:", response.content,"\n")
    return response.content

