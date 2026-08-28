"""
AI Real Estate Agent — LangGraph + Flask chatbot
--------------------------------------------------
A single-file agentic chatbot that ONLY answers questions related to
real estate (buying, selling, renting, mortgages, property valuation,
market trends, home inspection, real-estate law basics, etc.).

Any question outside the real-estate domain is politely declined by
the agent's own routing logic — this is enforced INSIDE the LangGraph
graph (a "topic guard" node), not just via a system prompt.

Deploy on Render:
  Build Command : pip install -r requirements.txt
  Start Command : python app.py
  Env Vars      : OPENAI_API_KEY = <your key>

Render provides the app a PORT env var — this app binds to it and to
0.0.0.0 so the Render URL opens the chat UI directly.
"""

import os
from typing import TypedDict, List

from flask import Flask, request, jsonify, Response
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ---------------------------------------------------------------------
# 1. LLM
# ---------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

REAL_ESTATE_SYSTEM_PROMPT = """You are "EstateBot", an AI Real Estate Agent assistant.
You ONLY help with real-estate related topics such as:
- Buying, selling, or renting residential/commercial property
- Property valuation, pricing trends, and market analysis
- Mortgages, loans, EMIs, down payments, and financing
- Home inspection, legal documents (sale deed, RERA, title check)
- Neighborhood/locality comparisons, amenities, and investment advice
- Negotiation tips and listing descriptions

Stay concise, professional, and helpful. Ask clarifying questions
(budget, location, property type) when useful. Never answer questions
unrelated to real estate — the system will already filter those out,
but if one slips through, politely redirect the user back to real
estate topics.
"""

# ---------------------------------------------------------------------
# 2. Graph State
# ---------------------------------------------------------------------
class AgentState(TypedDict):
    messages: List[dict]      # chat history: [{"role": "user"/"assistant", "content": str}]
    user_input: str
    is_real_estate: bool
    response: str


# ---------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------
def topic_guard_node(state: AgentState) -> AgentState:
    """Rule-based + LLM-assisted check: is this query about real estate?"""
    query = state["user_input"].lower()

    real_estate_keywords = [
        "property", "house", "home", "apartment", "flat", "rent", "lease",
        "mortgage", "loan", "emi", "real estate", "realtor", "broker",
        "buy", "sell", "listing", "valuation", "price", "market", "land",
        "plot", "villa", "condo", "tenant", "landlord", "square feet",
        "sqft", "down payment", "interest rate", "deed", "rera", "agent",
        "neighborhood", "locality", "investment property", "resale",
        "commercial space", "office space", "closing cost", "appraisal",
    ]

    if any(kw in query for kw in real_estate_keywords):
        state["is_real_estate"] = True
        return state

    # Fallback: ask the LLM to classify ambiguous queries
    classification = llm.invoke([
        SystemMessage(content=(
            "Reply with only one word: 'YES' if the following user message "
            "is related to real estate (property, housing, rent, buying, "
            "selling, mortgages, etc.), or 'NO' if it is not."
        )),
        HumanMessage(content=state["user_input"]),
    ])
    state["is_real_estate"] = "YES" in classification.content.upper()
    return state


def real_estate_agent_node(state: AgentState) -> AgentState:
    """Main agentic node — answers using conversation history."""
    history = [SystemMessage(content=REAL_ESTATE_SYSTEM_PROMPT)]
    for m in state["messages"]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    history.append(HumanMessage(content=state["user_input"]))

    result = llm.invoke(history)
    state["response"] = result.content
    return state


def refusal_node(state: AgentState) -> AgentState:
    """Runs when the query is NOT about real estate."""
    state["response"] = (
        "I'm EstateBot, an AI Real Estate Agent 🏠 — I can only help with "
        "real-estate topics like buying, selling, renting, mortgages, "
        "property valuation, or market trends. Could you ask me something "
        "related to real estate?"
    )
    return state


def route_after_guard(state: AgentState) -> str:
    return "agent" if state["is_real_estate"] else "refuse"


# ---------------------------------------------------------------------
# 4. Build LangGraph
# ---------------------------------------------------------------------
graph = StateGraph(AgentState)
graph.add_node("guard", topic_guard_node)
graph.add_node("agent", real_estate_agent_node)
graph.add_node("refuse", refusal_node)

graph.set_entry_point("guard")
graph.add_conditional_edges("guard", route_after_guard, {"agent": "agent", "refuse": "refuse"})
graph.add_edge("agent", END)
graph.add_edge("refuse", END)

compiled_graph = graph.compile()

# ---------------------------------------------------------------------
# 5. Flask app (serves chat UI + API on the same Render URL)
# ---------------------------------------------------------------------
app = Flask(__name__)

# In-memory session history (single-user demo; resets on restart)
chat_history: List[dict] = []

CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Real Estate Agent</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: -apple-system, Arial, sans-serif; background:#0f172a; margin:0; color:#e2e8f0; }
  .wrap { max-width:640px; margin:0 auto; height:100vh; display:flex; flex-direction:column; }
  header { padding:16px; background:#1e293b; text-align:center; font-size:20px; font-weight:600; }
  #chat { flex:1; overflow-y:auto; padding:16px; }
  .msg { margin:8px 0; padding:10px 14px; border-radius:14px; max-width:80%; line-height:1.4; }
  .user { background:#2563eb; margin-left:auto; }
  .bot { background:#334155; margin-right:auto; }
  form { display:flex; padding:12px; background:#1e293b; }
  input { flex:1; padding:10px; border-radius:8px; border:none; margin-right:8px; }
  button { padding:10px 18px; border:none; border-radius:8px; background:#2563eb; color:#fff; cursor:pointer; }
</style>
</head>
<body>
<div class="wrap">
  <header>🏠 AI Real Estate Agent</header>
  <div id="chat"></div>
  <form id="f">
    <input id="i" placeholder="Ask about buying, renting, mortgages..." autocomplete="off" />
    <button type="submit">Send</button>
  </form>
</div>
<script>
const chat = document.getElementById('chat');
const form = document.getElementById('f');
const input = document.getElementById('i');

function addMsg(text, cls) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addMsg(text, 'user');
  input.value = '';
  const res = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text})
  });
  const data = await res.json();
  addMsg(data.response, 'bot');
});
</script>
</body>
</html>
"""


@app.route("/")
def index() -> Response:
    return Response(CHAT_HTML, mimetype="text/html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    state: AgentState = {
        "messages": chat_history.copy(),
        "user_input": user_message,
        "is_real_estate": False,
        "response": "",
    }

    result = compiled_graph.invoke(state)

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": result["response"]})

    return jsonify({"response": result["response"]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
