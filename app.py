import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================
# CHECK API KEY
# ============================================================

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. "
        "Add it to your .env file."
    )


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Financial Research AI Agent",
    description="LLM-powered financial research chatbot",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# FINANCIAL AGENT SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a Financial Research AI Agent.

Your job is to help users understand companies,
businesses, financial concepts, and financial analysis.

You can help with:

1. Company analysis
2. Revenue analysis
3. Profit analysis
4. Profit margin calculations
5. Revenue growth calculations
6. Debt analysis
7. Financial ratios
8. Cash flow concepts
9. Business growth analysis
10. Industry analysis
11. Startup analysis
12. Financial risk analysis
13. Management strategy
14. Financial forecasting concepts
15. Financial mathematics

IMPORTANT RULES:

- Explain calculations clearly.
- Show formulas when performing calculations.
- Separate facts from assumptions.
- Do not pretend to have real-time financial data.
- Do not invent current stock prices, earnings,
  financial statements, or market data.
- If the user asks for current data, clearly say
  that real-time financial data is not connected.
- Do not provide personalized investment advice.
- Explain financial concepts in a beginner-friendly way.
- When useful, provide tables or bullet points.
- If the user's question is unclear, ask for clarification.

You are a research and analysis assistant,
not a financial advisor.
"""


# ============================================================
# LLM AGENT
# ============================================================

def financial_agent(user_message: str) -> str:

    response = client.responses.create(

        model="gpt-5",

        instructions=SYSTEM_PROMPT,

        input=user_message
    )

    return response.output_text


# ============================================================
# CHAT API
# ============================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    try:

        user_message = request.message.strip()

        if not user_message:

            return {
                "success": False,
                "response": "Please enter a financial question."
            }


        response = financial_agent(
            user_message
        )


        return {
            "success": True,
            "response": response
        }


    except Exception as error:

        return {
            "success": False,
            "response": (
                "The financial agent encountered an error.\n\n"
                f"Error: {str(error)}"
            )
        }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "agent": "Financial Research AI Agent"
    }
    
