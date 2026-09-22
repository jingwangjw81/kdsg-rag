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
    # r = GoogleGenAI(model="models/gemini-2.5-flash").complete("Antworte mit einem Wort: Bern?")
    # print("Gemini:", str(r).strip())

if ok_m:
    from llama_index.llms.mistralai import MistralAI
    r = MistralAI(model="mistral-small-latest").complete("Antworte mit einem Wort: Bern?")
    print("Mistral:", str(r).strip())

print("Smoke test finished.")
