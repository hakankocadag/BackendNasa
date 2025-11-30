# ☔ Will It Rain on My Parade? - NASA Hackathon Backend API

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-green)

A robust, AI-enhanced backend API developed during the NASA Space Apps Hackathon. This project provides climatological analysis to help event planners assess the risk of disruptive weather, using historical weather data and AI-generated advice.

> ⚠️ **Note:** This project was backend-focused for the NASA Space Apps Hackathon. There is **no frontend** in this repository. All functionality is accessible via API endpoints.  
> While we wished to further expand this project, this repository documents the completed hackathon submission.  
> Generative AI usage was **permissible** in the hackathon. For this project, AI (Google Gemini 2 and GPT-5) was used **to assist in writing code** as well as for AI-driven advice generation.

---

## 🌟 Key Features

- **Climatology Query Engine:** Calculates historical averages and probabilities for a target date using a user-defined window (e.g., 7 days before and after the date across all years).  
- **Geospatial Matching:** Uses the Haversine formula to find the nearest historical weather station data based on user-provided latitude and longitude.  
- **Circular Date Handling:** Correctly handles the circular nature of the Day of Year (DOY) for climatology windows that wrap around the start/end of the year.  
- **AI-Powered Advice:** Integrates with the Google Gemini API to transform calculated probabilities (e.g., 40% chance of rain) into a single, conversational paragraph suitable for event organizers.  
- **Real-time Data Access:** `/climatology/series` endpoint allows retrieval of all historical data points used in calculations for visualization and debugging.  
- **Robust Backend:** Built on FastAPI with asynchronous I/O (`httpx`) and retry logic for external API calls.

---

## 💻 Architecture & Technology Stack

- **Framework:** FastAPI for API development and Pydantic validation  
- **Data Science:** Pandas and NumPy for data loading, filtering, and statistical analysis  
- **AI Integration:** Google Gemini API (`gemini-2.5-flash-preview-05-20`) for human-friendly advice  
- **Geospatial Logic:** Haversine distance calculation for location mapping  
- **Asynchronous I/O:** `httpx` for non-blocking API calls

---

## 🛠️ Setup and Installation

### Prerequisites

- Python 3.9+  
- Local copy of historical weather CSV (default: `1abilene_weather_january_2024.csv`)  
- Gemini API Key from Google AI Studio  

### 1. Environment and Dependencies

```bash
# Clone the repository
git clone <repository-link>
cd backendNasa

# Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate # Windows

# Install required packages
pip install -r requirements.txt
