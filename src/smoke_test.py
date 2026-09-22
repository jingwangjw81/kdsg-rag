"""Step 1 check: are the API keys set and do the LLM providers answer?"""
import os

def check(name):
    v = os.getenv(name)
    print(f"{name}: {'set' if v else 'MISSING'}")
    return bool(v)

ok_g = check("GOOGLE_API_KEY")
ok_m = check("MISTRAL_API_KEY")

if ok_g:
    from llama_index.llms.google_genai import GoogleGenAI
    r = GoogleGenAI(model="models/gemini-3.6-flash").complete("Antworte mit einem Wort: Bern?") # Not working: gemini-2.5-flash
    print("Gemini:", str(r).strip())

if ok_m:
    from llama_index.llms.mistralai import MistralAI
    r = MistralAI(model="codestral-2508").complete("Antworte mit einem Wort: Bern?") # Not working: mistral-small-2506, mistral-small-latest
    print("Mistral:", str(r).strip())

    # from mistralai.client import Mistral
    # client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    # response = client.chat.complete(
    #     model="mistral-small-latest",
    #     messages=[
    #         {"role": "user", "content": "Hello"}
    #     ],
    # )
    # print(response.choices[0].message.content)

print("Smoke test finished.")
