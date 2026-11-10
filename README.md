Stil Ajani 📸

Stil Ajani is an AI-powered photo editing assistant that analyzes photographs and generates personalized editing recommendations using Google’s Gemini API.

The project aims to simplify photo editing by combining artificial intelligence with image processing techniques.

With a single command it produces:

An edited photo (_edit.jpg)
Adjustment values for Lightroom / Photoshop (_degerler.json)
A preset file you can import into Lightroom (_preset.xmp

✨ Features

* AI-powered image analysis using Google Gemini.
* Intelligent photo editing recommendations.
* Automatic adjustment of image parameters.
* Iterative AI feedback for improving editing results.
* Image processing with Python.

🛠️ Technologies

* Python
* Google Gemini API
* Google Gen AI SDK
* Pillow (PIL)
* NumPy

⚙️ Installation

Clone the repository:

git clone https://github.com/FeyzaOzturk1/stil-agent
cd stil-ajani

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install google-genai pillow numpy

🔑 API Configuration

Create a .env file in the project directory and add your Gemini API key:

GEMINI_API_KEY=your_api_key_here

Make sure .env is included in .gitignore.

Choosing a model🔑

The default model is defined by the MODEL line in the code. You can also set the model name through the GEMINI_MODEL environment variable without editing the code. This project has been run with gemini-3.5-flash:

Run the application✨

python3 stil_ajani_gemini.py "path/to/your/image.png"


Using it in Lightroom✨

You can import _preset.xmp into Lightroom as a preset and apply it to your photo with one click:

Lightroom Classic: Presets panel, "+" icon, Import Presets
Lightroom (cloud / mobile): Presets, three-dot menu, Import Presets

Note: the analysis text is written in Turkish by default. To get it in English, change the word "Turkish" to "English" near the end of the SYSTEM prompt in the code

Usage


