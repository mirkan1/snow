import os
import openai
import json
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"],)
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from flask import Flask, redirect, render_template, request, url_for, session, jsonify
from langchain.tools import BaseTool

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
app.secret_key = os.urandom(24)

# @tool
def get_contract_function_names():
    """Returns the names of the contract functions."""
    abi = open("abi.json", "r")

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

get_contract_function_names()
tools = [get_word_length]

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a powerful assistant that onboards cryptocurrency (crypto) newcomers, specifically those who are around 12-18 years old, in easy and simple terms. You provide simple answers that are understandable. You also can use your tools to answer the questions. If you do not know the answer to a question, you truthfully say you do not know."),
    MessagesPlaceholder(variable_name=MEMORY_KEY),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

chat_history = []

llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)

agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps']),
    "chat_history": lambda x: x["chat_history"]
} | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.route("/", methods=("GET", "POST"))
def answer_chat():
    if request.method == "POST":
        user_input = request.form["question"]
        response_text = interact_with_chat(user_input)
        print("response text is ", response_text)

        if 'chat_history' not in session:
            session['chat_history'] = []

        ai_response = response_text['output']


        # Add the user's question and AI's response to chat history
        session['chat_history'].append({
            'question': user_input,
            'response': ai_response
        })
        session.modified = True
        return redirect(url_for("answer_chat"))

    return render_template("index.html", chat_history=session.get('chat_history', []))

def interact_with_chat(user_input):
    result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result['output']))
    return result

@app.route('/metamask', methods=['POST'])
def handle_data():
    data = request.json
    ethereum_address = data.get('ethereumAddress')
    return jsonify({"message": "Data received successfully!"})

@app.route("/clear_chat", methods=["GET"])
def clear_chat():
    session['chat_history'] = []
    return redirect(url_for("answer_chat"))

if __name__ == "__main__":
    app.run(port=5000)
