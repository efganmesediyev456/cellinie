import tempfile
import os
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


@csrf_exempt
def voice_view(request):
    if request.method == "POST":
        audio_file = request.FILES.get("audio")

        if not audio_file:
            return JsonResponse({"error": "Audio faylı göndərilməyib"}, status=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            for chunk in audio_file.chunks():
                temp_audio.write(chunk)
            temp_audio_path = temp_audio.name

        try:
            # Whisper API ilə səsi mətnə çevir
            with open(temp_audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="az",
                )
            user_text = transcript.text

            # ChatGPT cavabı al
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": user_text}
                ]
            )
            ai_answer = response.choices[0].message.content

            return JsonResponse({
                "text": user_text,
                "answer": ai_answer
            })
        finally:
            os.unlink(temp_audio_path)


@csrf_exempt
def tts_view(request):
    if request.method == "POST":
        import json
        body = json.loads(request.body)
        text = body.get("text", "")

        if not text:
            return JsonResponse({"error": "Mətn göndərilməyib"}, status=400)

        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
        )

        audio_content = response.content
        return HttpResponse(audio_content, content_type="audio/mpeg")


def home(request):
    return render(request, 'main/home.html')
