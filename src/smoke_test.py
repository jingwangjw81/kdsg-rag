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
    r = MistralAI(model="ministral-3b-2512").complete("Antworte mit einem Wort: Bern?")
    print("Mistral:", str(r).strip())

    # llm = MistralAI(model="mistral-small-latest")
    # response = llm.complete("Hello")
    # print(response)

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
