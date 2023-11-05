from PIL import Image
from pyprojroot import here
from transformers import Blip2ForConditionalGeneration, Blip2Processor

model_name = "Salesforce/blip2-flan-t5-xl"

processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder=str(here("data/accelerate/offload")),
)

image = Image.open(here("data/test-images/img.png"))

question = "What is on the picture?"
inputs = processor(image, question, return_tensors="pt").to("cpu")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
