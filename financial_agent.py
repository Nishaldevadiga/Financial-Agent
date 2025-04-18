from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

load_dotenv()

##web search Agent
web_search_agent=Agent(
    name="web search Agent",
    role="Search the web for the information",
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

##Financial Agent
finance_agent=Agent(
    name="Finance AI Agent",
     model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instructions=["use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
     model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    instructions=["Always include sources","use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)