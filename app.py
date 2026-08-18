import json
import os
import requests
import uvicorn
from fastapi import FastAPI
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
from pydantic import BaseModel, Field


# --- 1. Define Tools (All returning pure strings) ---

def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""
    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
    }
    return movies.get(genre.lower(), "No movies found for that genre")



def change__to_f(temp_c: float) -> str:
    """Converts the Celsius temperature to Fahrenheit temperature and returns a string."""
    temp_f = temp_c * 1.8 + 32
    return f"{temp_f:.2f} °F"


def get_weather(city: str) -> str:
    """Get current temperature for a given city name."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": city, "count": 1}

    try:
        geo_response = requests.get(geo_url, params=geo_params).json()
        if "results" not in geo_response or not geo_response["results"]:
            return f"Could not find weather data for city: {city}"

        location = geo_response["results"][0]
        latitude = location["latitude"]
        longitude = location["longitude"]

        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius",
        }
        weather_response = requests.get(
            weather_url, params=weather_params
        ).json()["current"]

        result = {
            "resolved_city": str(location.get("name")),
            "temperature_celsius": str(
                weather_response.get("temperature_2m")
            ),
            "weather_code": str(weather_response.get("weather_code")),
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error retrieving weather data: {str(e)}"


tools = [get_weather, search_movies, change__to_f]

# --- 2. Initialize Model & Agent ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it", api_key=GEMINI_API_KEY, temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to Indian weather and cinema. "
        "Always format your answers in plain text strings. "
        "For any other roles, topics, questions, or general knowledge outside of Indian weather and movies, "
        "you must say exactly: 'I am not authorized to answer questions outside of Indian weather and cinema.'"
    ),
)


class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", str(user_input))]}


def extract_text_response(agent_output) -> str:
    """Guarantees a clean, single string return value."""
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        content = getattr(last, "content", last)

        # Handle multimodal/structured chunk lists if returned by LangChain
        if isinstance(content, list):
            text_parts = [
                part.get("text", str(part))
                if isinstance(part, dict)
                else str(part)
                for part in content
            ]
            return "".join(text_parts)

        return str(content)

    return str(agent_output)


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)

# --- 3. FastAPI App ---
app = FastAPI(title="Indian Weather and Cinema Agents")

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
