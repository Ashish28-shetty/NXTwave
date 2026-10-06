import base64
import os
import tempfile

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Keep API keys outside the source code.
MURF_API_KEY = os.getenv("MURF_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")

if not MURF_API_KEY:
    raise RuntimeError("MURF_API_KEY is not set. Add it to your environment or .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

PROMPTS = {
    "Summary": """
You are a professional tourist guide.
Provide a high-level overview of "{place}" in {language}.

Focus on:
- The historical significance
- Why the place is famous
- Key architectural or cultural highlights

Keep the explanation concise, engaging, and easy to follow.
Avoid excessive details and dates.
Limit the response to around 200 words.

Respond ONLY in {language}.
""",
    "Detailed": """
You are a professional tourist guide.
Provide a detailed and immersive explanation of "{place}" in {language}.

Cover:
- Historical background and timeline
- Architectural design and unique features
- Cultural importance and notable events
- Interesting facts and visitor insights

Explain concepts clearly and in a storytelling manner.
Include relevant details and examples to create a rich experience.
Limit the response to around 400 words.

Respond ONLY in {language}.
""",
}


def generate_description(place, answer_type, language):
    """Generate the place description with Gemini using the selected response type."""
    if answer_type not in PROMPTS:
        raise ValueError("answerType must be Summary or Detailed")

    prompt = PROMPTS[answer_type].format(place=place, language=language)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty description.")

    return response.text.strip()


def generate_speech(text, voice_id, locale):
    """Convert the generated description into MP3 audio using Murf Falcon."""
    url = "https://global.api.murf.ai/v1/speech/stream"
    headers = {
        "api-key": MURF_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "voice_id": voice_id,
        "text": text,
        "locale": locale,
        "model": "FALCON",
        "format": "MP3",
        "sampleRate": 24000,
        "channelType": "MONO",
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120,
        stream=True,
    )

    if response.status_code != 200:
        error_text = response.text[:500]
        raise RuntimeError(
            f"Murf API request failed ({response.status_code}): {error_text}"
        )

    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_audio.close()

    try:
        with open(temp_audio.name, "wb") as audio_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    audio_file.write(chunk)
        return temp_audio.name
    except Exception:
        try:
            os.remove(temp_audio.name)
        except OSError:
            pass
        raise


@app.route("/generate-audio-guide", methods=["POST"])
def generate_audio_guide():
    """Generate the selected place description and its Murf audio guide."""
    data = request.get_json(silent=True) or {}

    required_fields = ["place", "answerType", "language", "voiceId", "locale"]
    missing = [field for field in required_fields if not data.get(field)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    place = data["place"].strip()
    answer_type = data["answerType"]
    language = data["language"]
    voice_id = data["voiceId"]
    locale = data["locale"]

    audio_path = None

    try:
        # Step 1: Gemini creates the historical description.
        text_description = generate_description(place, answer_type, language)

        # Step 2: Murf converts the description into audio.
        audio_path = generate_speech(text_description, voice_id, locale)

        # Step 3: Encode the MP3 so the frontend can receive it in JSON.
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        # Step 4: Return both transcript and audio to the frontend.
        return jsonify({
            "description": text_description,
            "audioBase64": encoded_audio,
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Audio guide generation failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
