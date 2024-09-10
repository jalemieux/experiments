from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_community.tools import GoogleSerperResults

# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=PythonREPL().run,
)

search = GoogleSerperResults()


prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")


llm = ChatOpenAI(model="gpt-3.5-turbo")

agent = create_react_agent(llm, tools=[repl_tool, search], prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=[repl_tool, search], verbose=True, return_intermediate_steps=True)

response = agent_executor.invoke({"input": "what makes a biying group a good one on Adobe Journey Optimizer B2B Edition"})

print(response)
