import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()

def generate_report(detections):
    prompt = f"""Act as an expert Telecom Tower Structural Inspector and write a short "Fleet-Wide Executive Summary" based on the consolidated detection data from multiple images below.
STRICTLY FORBIDDEN: Do not output raw bounding box coordinates or complex numbers.
Mention the total joints detected, overall confidence, and give a simulated 'Structural Health Status' (e.g., 'Structurally Sound' or 'Needs Manual Inspection' if confidence is low).
Focus on conveying a high-level overview of all the combined image data provided.
IMPORTANT: Do NOT use Markdown formatting, asterisks (**), or special symbols. Provide the response in plain, standard text only.

Detection data:
{detections}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
