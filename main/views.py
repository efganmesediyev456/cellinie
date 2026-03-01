import tempfile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import whisper
from openai import OpenAI
import os
from dotenv import load_dotenv
from django.shortcuts import render


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
whisper_model = whisper.load_model("base")

@csrf_exempt
def voice_view(request):
    if request.method == "POST":
        audio_file = request.FILES.get("audio")

        if not audio_file:
            return JsonResponse({"error": "Audio faylı göndərilməyib"}, status=400)

        # Müvəqqəti fayl kimi saxla
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            for chunk in audio_file.chunks():
                temp_audio.write(chunk)
            temp_audio_path = temp_audio.name

        # 1️⃣ Whisper ilə səsi mətnə çevir
        result = whisper_model.transcribe(temp_audio_path, language="az")  # Azərbaycan dili
        user_text = result["text"]

        print(user_text)

        # 2️⃣ ChatGPT cavabı al
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": user_text}
            ]
        )

        ai_answer = response.choices[0].message.content

        print(ai_answer)

        return JsonResponse({
            "text": user_text,
            "answer": ai_answer
        })

def home(request):
    return render(request, 'main/home.html')