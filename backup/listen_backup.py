import io
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from sys import platform

from typing import Tuple

# Adapted from https://github.com/davabase/whisper_real_time 

MODEL = "small"
NON_ENGLISH = False
ENERGY_THRESHOLD = 250
RECORD_TIMEOUT = 5
PHRASE_TIMEOUT = 3 
DEFAULT_MICROPHONE = "pulse"
ERICA_IDENTIFICATION_KEYWORDS = ["Erica", "Erika","erika","erica"]

if MODEL != "large" and not NON_ENGLISH:
    MODEL = MODEL + ".en"
audio_model = whisper.load_model(MODEL)

if 'linux' in platform:
    mic_name = DEFAULT_MICROPHONE
    if not mic_name or mic_name == 'list':
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")   
    else:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                audio_source = sr.Microphone(sample_rate=16000, device_index=index)
                break
else:
    audio_source = sr.Microphone(sample_rate=16000)

def erica_listen() -> Tuple[bool, str]:
    print("listening")
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = ENERGY_THRESHOLD
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    record_timeout = RECORD_TIMEOUT
    phrase_timeout = PHRASE_TIMEOUT

    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    with audio_source:
        recorder.adjust_for_ambient_noise(audio_source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(audio_source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    use_voice = False
    transcription = ""

    print("listening...")
    try:
        now = datetime.utcnow()
        # Pull raw recorded audio from the queue.
        if not data_queue.empty():
            print('here')
            phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                phrase_complete = True
            # This is the last time we received new audio data from the queue.
            phrase_time = now

            # Concatenate our current audio data with the latest audio data.
            while not data_queue.empty():
                data = data_queue.get()
                last_sample += data

            # Use AudioData to convert the raw data to wav data.
            audio_data = sr.AudioData(last_sample, audio_source.SAMPLE_RATE, audio_source.SAMPLE_WIDTH)
            wav_data = io.BytesIO(audio_data.get_wav_data())

            # Write wav data to the temporary file as bytes.
            with open(temp_file, 'w+b') as f:
                f.write(wav_data.read())

            # Read the transcription.
            result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
            text = result['text'].strip()
            use_voice = False
            for keyword in ERICA_IDENTIFICATION_KEYWORDS:
                if keyword in text:
                    use_voice = True
                    break
            transcription = text

            if phrase_complete:
                # Stop listening in the background
                # audio_source.stop()
                return use_voice, transcription
    except Exception as e:
        print(e)
    
    # If phrase_timeout is hit and the function hasn't returned yet, stop listening in the background
    # audio_source.stop()
    return use_voice, transcription

# def erica_listen() -> Tuple[bool, str]:
    # print("listening")
    # phrase_time = None
    # # Current raw audio bytes.
    # last_sample = bytes()
    # # Thread safe Queue for passing data from the threaded recording callback.
    # data_queue = Queue()
    # # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    # recorder = sr.Recognizer()
    # recorder.energy_threshold = ENERGY_THRESHOLD
    # # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    # recorder.dynamic_energy_threshold = False

    # record_timeout = RECORD_TIMEOUT
    # phrase_timeout = PHRASE_TIMEOUT

    # temp_file = NamedTemporaryFile().name
    # transcription = ['']
    
    # with audio_source:
        # recorder.adjust_for_ambient_noise(audio_source)

    # def record_callback(_, audio:sr.AudioData) -> None:
        # """
        # Threaded callback function to recieve audio data when recordings finish.
        # audio: An AudioData containing the recorded bytes.
        # """
        # # Grab the raw bytes and push it into the thread safe queue.
        # data = audio.get_raw_data()
        # data_queue.put(data)

    # # Create a background thread that will pass us raw audio bytes.
    # # We could do this manually but SpeechRecognizer provides a nice helper.
    # recorder.listen_in_background(audio_source, record_callback, phrase_time_limit=record_timeout)

    # # Cue the user that we're ready to go.
    # print("Model loaded.\n")

    # use_voice = False
    # transcription = ""

    # print("listening...")
    # try:
        # now = datetime.utcnow()
        # # Pull raw recorded audio from the queue.
        # if not data_queue.empty():
            # phrase_complete = False
            # # If enough time has passed between recordings, consider the phrase complete.
            # # Clear the current working audio buffer to start over with the new data.
            # if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                # phrase_complete = True
            # # This is the last time we received new audio data from the queue.
            # phrase_time = now

            # # Concatenate our current audio data with the latest audio data.
            # while not data_queue.empty():
                # data = data_queue.get()
                # last_sample += data

            # # Use AudioData to convert the raw data to wav data.
            # audio_data = sr.AudioData(last_sample, audio_source.SAMPLE_RATE, audio_source.SAMPLE_WIDTH)
            # wav_data = io.BytesIO(audio_data.get_wav_data())

            # # Write wav data to the temporary file as bytes.
            # with open(temp_file, 'w+b') as f:
                # f.write(wav_data.read())

            # # Read the transcription.
            # result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
            # text = result['text'].strip()
            # use_voice = False
            # for keyword in ERICA_IDENTIFICATION_KEYWORDS:
                # if keyword in text:
                    # use_voice = True
                    # break
            # transcription = text

            # if phrase_complete:
                # # Stop listening in the background
                # recorder.stop()
                # return use_voice, transcription
    # except Exception as e:
        # print(e)
    
    # # If phrase_timeout is hit and the function hasn't returned yet, stop listening in the background
    # recorder.stop()
    # return use_voice, transcription

if __name__ == "__main__":
    use_voice, transcription = erica_listen()
    print(f"Use voice: {use_voice}\nTranscription: {transcription}")
