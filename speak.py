import os
from gtts import gTTS
from elevenlabs import set_api_key
from elevenlabs import generate, play

# LANGUAGE = 'en'
# TMP_ERICA_AUDIO_FILE = '/tmp/erica_speak.mp3'

# def erica_speak(message: str):
    # myobj = gTTS(text=message, lang=LANGUAGE, slow=False)
    # myobj.save(TMP_ERICA_AUDIO_FILE)
    # os.system("mpg321 " + TMP_ERICA_AUDIO_FILE + " > /dev/null 2>&1")

voice = "Bella"
set_api_key(os.environ["ELEVENLABS_API_KEY"])

def erica_speak(text):
    audio = generate(
    voice=voice, text=text
    )
    play(audio)

if __name__ == '__main__':
    erica_speak('Hello, my name is Erica')
