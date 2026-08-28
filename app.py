import os
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
from pydantic import BaseModel, Field

# --- 1. Define Tools ---
@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""
    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
    }
    return movies.get(genre.lower(), "No movies found for that genre")


@tool
def change_to_f(temp_c: float) -> str:
    """Converts the Celsius temperature to Fahrenheit temperature string."""
    temp_f = temp_c * 1.8 + 32
    return f"{temp_f:.2f} °F"


@tool
def get_weather(city: str) -> str:
    """Get current temperature for a given city name."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": city, "count": 1}

    try:
        geo_response = requests.get(geo_url, params=geo_params).json()
        if "results" not in geo_response or not geo_response.get("results"):
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
        ).json().get("current", {})

        city_name = location.get("name")
        temp = weather_response.get("temperature_2m")
        code = weather_response.get("weather_code")
        return f"Weather in {city_name}: {temp}°C, weather code: {code}."
    except Exception as e:
        return f"Error retrieving weather data: {str(e)}"


tools = [get_weather, search_movies, change_to_f]

# --- 2. Initialize Gemini Model & Agent ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant restricted ONLY to Indian weather and cinema. "
            "Always respond in plain conversational text sentences. Never output JSON, dictionaries, or raw structures. "
            "For any other roles, topics, questions, or general knowledge outside of Indian weather and movies, "
            "you must say exactly: 'I am not authorized to answer questions outside of Indian weather and cinema.'",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


# --- 3. FastAPI Application ---
app = FastAPI(title="Indian Weather and Cinema Agents")

add_routes(
    app,
    agent_executor.with_types(input_type=AgentInput),
    path="/agent",
)


@app.post("/chat", response_class=PlainTextResponse)
async def chat_endpoint(data: AgentInput) -> str:
    result = await agent_executor.ainvoke({"input": data.input})
    return str(result.get("output", ""))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
