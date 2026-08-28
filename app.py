import os
import requests
import uvicorn
import threading
import nest_asyncio
import socket

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field
from google.colab import userdata

nest_asyncio.apply()


# ============================================================
# 1. DEFINE TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
        "horror": "Tumbbad, Stree, Bhool Bhulaiyaa",
        "thriller": "Drishyam, Andhadhun, Ratsasan",
        "romance": "Jab We Met, Sita Ramam, 96",
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre."
    )


@tool
def change_to_f(temp_c: float) -> str:
    """Convert Celsius temperature to Fahrenheit."""

    temp_f = temp_c * 1.8 + 32

    return f"{temp_f:.2f} °F"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for an Indian city."""

    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    try:

        # ----------------------------------------------------
        # Find city coordinates
        # ----------------------------------------------------

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find the city: {city}"

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        city_name = location.get("name", city)

        # ----------------------------------------------------
        # Get weather
        # ----------------------------------------------------

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        current_weather = weather_data.get("current", {})

        temperature = current_weather.get("temperature_2m")
        weather_code = current_weather.get("weather_code")

        return (
            f"Weather in {city_name}: "
            f"{temperature}°C, "
            f"weather code: {weather_code}."
        )

    except requests.RequestException as e:

        return f"Weather service error: {str(e)}"

    except Exception as e:

        return f"Error retrieving weather data: {str(e)}"


# ============================================================
# 2. TOOL LIST
# ============================================================

tools = [
    get_weather,
    search_movies,
    change_to_f,
]


# ============================================================
# 3. GEMINI API KEY
# ============================================================

GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# 4. INITIALIZE GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)


# ============================================================
# 5. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a specialized assistant.

You are ONLY authorized to answer questions related to:

1. Indian weather
2. Indian movies and cinema
3. Temperature conversion when required for weather

You have access to tools for:
- Getting weather information
- Searching Indian movies by genre
- Converting Celsius to Fahrenheit

For weather questions, use the weather tool.

For Indian movie questions, use the movie search tool when appropriate.

For temperature conversion, use the temperature conversion tool when appropriate.

Always respond in plain conversational text.

Never output JSON.
Never output dictionaries.
Never output raw Python structures.

For any question outside Indian weather and Indian cinema, respond exactly:

I am not authorized to answer questions outside of Indian weather and cinema.
"""


# ============================================================
# 6. CREATE LANGCHAIN AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# 7. REQUEST MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Your message to the agent"
    )


# ============================================================
# 8. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Indian Weather and Cinema Agent",
    description="LangChain + Gemini + FastAPI Agent",
    version="1.0.0",
)


# ============================================================
# 9. CHAT ENDPOINT
# ============================================================

@app.post(
    "/chat",
    response_class=PlainTextResponse
)
async def chat_endpoint(data: AgentInput):

    try:

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": data.input,
                    }
                ]
            }
        )

        messages = result.get("messages", [])

        if not messages:
            return "No response generated."

        # Last message generated by the agent
        final_message = messages[-1]

        content = final_message.content

        # ----------------------------------------------------
        # Handle normal string content
        # ----------------------------------------------------

        if isinstance(content, str):
            return content

        # ----------------------------------------------------
        # Handle Gemini structured content
        # ----------------------------------------------------

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":
                        text_parts.append(
                            item.get("text", "")
                        )

                elif isinstance(item, str):

                    text_parts.append(item)

            if text_parts:
                return " ".join(text_parts)

        return str(content)

    except Exception as e:

        return f"Error: {str(e)}"


# ============================================================
# 10. HEALTH CHECK
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "running",
        "message": "Indian Weather and Cinema Agent is running."
    }


# ============================================================
# 11. RUN SERVER
# ============================================================

def find_free_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))
        return s.getsockname()[1]

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            find_free_port() # Use a dynamic port if default is not set or in use
        )
    )

    # Run in a separate thread to avoid Colab event loop conflict
    thread = threading.Thread(target=uvicorn.run, kwargs={"app": app, "host": "0.0.0.0", "port": port})
    thread.start()
    print(f"Server running on http://0.0.0.0:{port}")
