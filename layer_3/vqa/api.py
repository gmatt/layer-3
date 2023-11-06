from io import BytesIO
from typing import Optional

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from lavis.models import load_model_and_preprocess
from pydantic import BaseModel

app = FastAPI()


class ResponseModel(BaseModel):
    answer: str


model: Optional[torch.nn.Module]
vis_processors: Optional[dict]
# TODO Enable GPU.
device = "cpu"


@app.on_event("startup")
async def startup_event():
    global model, vis_processors

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xl",
        is_eval=True,
        device=device,
    )


@app.post("/generate")
async def localize(
    question: str = Form(...),
    image: UploadFile = File(...),
) -> ResponseModel:
    raw_image = Image.open(BytesIO(await image.read())).convert("RGB")
    # plt.imshow(raw_image)
    # plt.show()

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    answers = model.generate(
        {
            "image": image,
            "prompt": question,
        }
    )

    return ResponseModel(
        answer=answers[0],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
