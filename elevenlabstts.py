from elevenlabs import generate, play
from elevenlabs import set_api_key
import os


voice = "Bella"
set_api_key(os.environ["ELEVENLABS_API_KEY"])

def narrate(text):
    audio = generate(
    voice=voice, text=text
    )
    play(audio)

narrate("Hello I am Erica. I am a virtual assistant.")
