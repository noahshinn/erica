import re
import os
import openai
import jsonlines
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import subprocess

from typing import Tuple, List, Union

def does_contain_url(string: str) -> Tuple[bool, str]:
    """
    Check if a string contains a URL.

    Returns a tuple of a boolean and the URL if found.
    """
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    if not url:
        return False, ""
    else:
        return True, url[0][0]

# uses Popen to run a command with a timeout
def run_with_timeout(cmd: List[str], timeout_sec: int) -> subprocess.Popen:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, close_fds=True)
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    return proc

# def erica_chat(message: str, context: str) -> str:
    # try:
        # cmd = ["erica-chat", message, context]
        # proc = run_with_timeout(cmd, 10)
        # status = proc.returncode
        # if status != 0:
            # print(status)
            # print("status not 0")
            # return "Sorry, there was an error."
        # assert not proc.stdout is None
        # out = proc.stdout.read().decode("utf-8")
        # out_status = out.startswith("OK")
        # if not out_status:
            # print(out_status)
            # print("out status not OK")
            # return "Sorry, there was an error."
        # return "".join(out.split("OK\n")[1:]).strip()
    # except Exception as e:
        # print(e)
        # return "Sorry, there was an error."

MODEL = "gpt-3.5-turbo"
SYSTEM_MESSAGE = "You are an Agent named Erica. You are smart, intelligent, helpful, and very good at programming. You will be frequently given requests to write code. You should do your best to write the code in the language that the user specifies. You live in the operating system of the user. If the user asks who you are, answer with the answer given below. If the user asks a question that is relevant to the context that you are given, reference the text as much as possible in your answer. When necessary, start your responses with \"Based on the content on your screen, ...\" If you are writing code or a summary, don't start your responses like that. Just give your answer."
MAX_TOKENS = 1024
TEMPERATURE = 0.7

# @retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def erica_chat(
    model: str,
    message: str,
    context: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    num_comps=1,
) -> Union[List[str], str]:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            { "role": "system", "content": SYSTEM_MESSAGE },
            { "role": "user", "content": "You should never admit that you don't know how to do something." },
            { "role": "assistant", "content": "Ok!" },
            { "role": "assistant", "content": "I am an Agent named Erica. I can see all of the text on your screen. Don't mind me! I'm just living here in the background." },
            { "role": "user", "content": f"Context: {context}" },
            { "role": "user", "content": message },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore

if __name__ == "__main__":
    print(erica_chat(model=MODEL, message="Can you write a script to train a GAN with PyTorch", context="No context"))
