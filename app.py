import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import requests
import json
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

# --- 1. Define Tools ---
@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""
    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali"
    }
    return movies.get(genre.lower(), "No movies found for that genre")


@tool
def change__to_f(temp_c: float) -> float:
  """converts the cel temp to F temperature"""
  return temp_c * (1.8) + 32


@tool
def get_weather(city: str) -> str:
    """Get current temperature for a given city name."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": city, "count": 1}
    geo_response = requests.get(geo_url, params=geo_params).json()
    if "results" not in geo_response:
        return f"Could not find weather data for city: {city}"
    location = geo_response["results"][0]
    latitude = location["latitude"]
    longitude = location["longitude"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "celsius"
    }
    weather_response = requests.get(weather_url, params=weather_params).json()["current"]

    result = {
        "resolved_city": location["name"],
        "temperature_celsius": weather_response["temperature_2m"],
        "weather_code": weather_response["weather_code"]
    }
    return json.dumps(result)

tools = [get_weather, search_movies, change__to_f]

# --- 2. Initialize Model & Agent ---
# Retrieve the key from the OS environment instead of Colab's userdata
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to Indian weather and cinema. "
        "For any other roles, topics, questions, or general knowledge outside of Indian weather and movies, "
        "you must say exactly: 'I am not authorized to answer questions outside of Indian weather and cinema.'"
    )
)

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}

def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    # Case 1: top-level messages (normal final state)
    messages = agent_output.get("messages")

    # Case 2: nested under a node name, e.g. {"model": {"messages": [...]}}
    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))

    return str(agent_output)

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)

# --- 3. FastAPI App ---
##Need To Code
app=FastAPI(title="Indian Weather and Cinem Agents")
add_rountes(app, formatted_agent_chain, path="/agent",playground_type="default")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
