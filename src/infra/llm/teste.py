from google import genai
from google.genai import types
from dotenv import load_dotenv

import requests

load_dotenv()

image_path = "https://http2.mlstatic.com/D_NQ_NP_859836-MLA83020758411_032025-O.webp"
image_bytes = requests.get(image_path).content
image = types.Part.from_bytes(
  data=image_bytes, mime_type="image/jpeg"
)

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=["What is this image?", image],
)

print(response.text)

usage = response.usage_metadata
if usage:
  input_tokens = getattr(usage, "prompt_token_count", None) or getattr(usage, "input_token_count", None)
  output_tokens = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_token_count", None)
  total_tokens = getattr(usage, "total_token_count", None)
  print(
    f"[TOKENS] input={input_tokens} output={output_tokens} total={total_tokens}"
  )
else:
  print("[TOKENS] usage_metadata não retornado pela API.")