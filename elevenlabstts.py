from elevenlabs import generate, play
voice = "Bella"

def narrate(text):
    audio = generate(
    voice=voice, text=text
    )
    play(audio)

#example usage: narrate("Hello I am erica. I am a virtual assistant.") If longer than like a sentence, you have to use 