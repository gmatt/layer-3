import PIL
import numpy as np
import pyscreeze
import torch
from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from matplotlib import pyplot as plt

# Monkey patch for bug
# https://stackoverflow.com/questions/76361049/how-to-fix-typeerror-not-supported-between-instances-of-str-and-int-wh


__PIL_TUPLE_VERSION = tuple(int(x) for x in PIL.__version__.split("."))
pyscreeze.PIL__version__ = __PIL_TUPLE_VERSION

import pyautogui


def perform_action(action: str) -> None:
    screenWidth, screenHeight = pyautogui.size()  # Get the size of the primary monitor.

    # (
    #     currentMouseX,
    #     currentMouseY,
    # ) = pyautogui.position()  # Get the XY position of the mouse.

    # pyautogui.moveTo(100, 150)  # Move the mouse to XY coordinates.
    #
    # pyautogui.click()  # Click the mouse.
    # pyautogui.click(100, 200)

    image = pyautogui.screenshot()

    plt.imshow(image)
    plt.show()

    # Next action model

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    action_model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5",
        model_type="pretrain_flant5xl",
        is_eval=True,
        device=device,
    )

    raw_image = image.convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    #     prompt = f"""Question: I'm performing the following task: {action}
    # Currently I see this on my screen.
    # What should I do next?
    # Answer:"""
    prompt = f"""I would like to do the following: {action}. Where should I click next? Answer:"""

    answer = action_model.generate(
        {
            "image": image,
            "prompt": prompt,
        }
    )

    print(answer)

    # Localization model

    caption = answer[0]

    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching",
        "large",
        device=device,
        is_eval=True,
    )

    dst_w = 720
    w, h = raw_image.size
    scaling_factor = dst_w / w

    resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption)
    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=7)
    avg_gradcam = getAttMap(norm_img, gradcam[0][1].cpu().numpy(), blur=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(avg_gradcam)
    fig.show()

    avg_gradcam = getAttMap(
        norm_img, gradcam[0][1].cpu().numpy(), blur=True, overlap=False
    )

    coordinates = (
        np.array(np.unravel_index(avg_gradcam.argmax(), avg_gradcam.shape))
        * scaling_factor
    )

    pyautogui.moveTo(coordinates[0], coordinates[1])


if __name__ == "__main__":
    perform_action("Open Spotify.")
