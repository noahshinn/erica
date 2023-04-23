import io
import torch
import whisper
import argparse
from time import sleep
from queue import Queue
from sys import platform
import speech_recognition as sr

from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile

from speak import erica_speak
from scrape import get_text_from_url
from sight import get_context, does_contain_url
from utils import erica_chat, compress_for_memory, reflect_on_memory

from typing import List

#Most of this code is from https://github.com/davabase/whisper_real_time 

TEXT_MODEL = "gpt-3.5-turbo"
AUDIO_MODEL = "small"

def handle_message(message: str, memory: List[dict]):
    context = get_context()
    is_url_present, url = does_contain_url(context)
    web_content = None
    should_speak = "use your voice" in message or "speak" in message
    should_reflect = "reflect" in message
    if should_reflect:
        response = reflect_on_memory(TEXT_MODEL, memory)
    else:
        if is_url_present:
            print(url)
            url_present_message = "I see a url present. I'm going to grab the content from there"
            if should_speak:
                erica_speak(url_present_message)
            else:
                print(url_present_message)
            if url.endswith(".pdf"):
                pdf_present_message = f"I can see that `{url}` is a pdf. I'm going to try to scrape the text from it. Give me a sec"
                if should_speak:
                    erica_speak(pdf_present_message)
                else:
                    print(pdf_present_message)
            web_content = get_text_from_url(url)

        if is_url_present and web_content:
            context = f"Web content:\n\n{web_content}\n-----"
        response = str(erica_chat(TEXT_MODEL, message, context))
    if should_speak:
        erica_speak(response)
    else:
        print(f"Erica: {response}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=AUDIO_MODEL, help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=250,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=5,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=1,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    keywords=["erica", "erika", "eric", "rica"]
    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    
    # Important for linux users. 
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")   
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)
        
    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    with source:
        recorder.adjust_for_ambient_noise(source)

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
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    memory = []

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()
                for keyword in keywords:
                    if keyword in text.lower():
                        handle_message(text, memory)
                        task_summary = compress_for_memory(TEXT_MODEL, text)
                        memory += [{
                            "time": datetime.now(),
                            "task_summary": task_summary
                        }]
                        #### ERICA NON-VOICE PROCESS
                        if("use your voice" in text):
                            print("Voice Request Detected: ", text) #### ERICA VOICE PROCESS HERE


                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                    
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                # os.system('cls' if os.name=='nt' else 'clear')
                # for line in transcription:
                    # print(line)
                    
            
                # Flush stdout.
                # print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.1)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
