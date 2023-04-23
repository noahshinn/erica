import os
from gtts import gTTS

LANGUAGE = 'en'
TMP_ERICA_AUDIO_FILE = '/tmp/erica_speak.mp3'

def erica_speak(message: str):
    myobj = gTTS(text=message, lang=LANGUAGE, slow=False)
    myobj.save(TMP_ERICA_AUDIO_FILE)
    os.system("mpg321 " + TMP_ERICA_AUDIO_FILE + " > /dev/null 2>&1")

if __name__ == '__main__':
    erica_speak('Hello, my name is Erica')
