import os

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import retry

load_dotenv()

proxy = os.getenv("PROXY")
os.environ["http_proxy"] = proxy
os.environ["HTTP_PROXY"] = proxy
os.environ["https_proxy"] = proxy
os.environ["HTTPS_PROXY"] = proxy

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

try:
    name = ""
    print("Available models:")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)
        if "1.5" in m.name:
            name = m.name

    if name:
        model = genai.GenerativeModel(name)

        response = model.generate_content(
            "Test", request_options={"retry": retry.Retry()}
        ).text.strip()

        print(f"{'-'*80}")

        print(
            "The expected answer is something like:\n"
            "'Test received. I am online and responding. How can I assist you today?'"
        )

        print(f"{'-'*80}")

        print(f"The actual answer is:\n" f"'{response}'")

        print(f"{'-'*80}")
    else:
        print("No suitable models?")

except Exception as e:
    print(e)
