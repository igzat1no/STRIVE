import base64
import os

import cv2
from openai import OpenAI
from pydantic import BaseModel

from constants import DETECT_OBJECTS, GEMINI_API_KEY, MODEL_NAME
from cv_utils.visualizer import visualize_mask


def _get_client_and_model(vlm: str) -> tuple[OpenAI, str]:
    if vlm == "gemini":
        return (
            OpenAI(
                api_key=GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            MODEL_NAME,
        )
    if vlm == "openai":
        return OpenAI(), "gpt-4o"
    raise ValueError(f"Invalid VLM: {vlm}")


def _encode_image_base64(image) -> str:
    image_jpg = cv2.imencode(".jpg", image)[1]
    return base64.b64encode(image_jpg).decode("utf-8")


def _step_dir(save_dir, episode_idx, episode_step) -> str:
    path = f"{save_dir}/episode-{episode_idx}/detection/step_{episode_step}"
    os.makedirs(path, exist_ok=True)
    return path


def ask_gpt_object_in_box(img, boxes, save_dir, episode_idx, episode_step, ind, vlm):
    step_dir = _step_dir(save_dir, episode_idx, episode_step)
    img_vis = visualize_mask(img, boxes, None, None, None)

    cv2.imwrite(
        f"{step_dir}/real_C_image_bbox_for_gpt_{ind}.jpg",
        img_vis,
    )

    img_vis_base64 = _encode_image_base64(img_vis)

    # crop the image to the bounding box
    box = boxes[0]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    # bigger the bbox by 5 pixels
    x1 = max(0, x1 - 5)
    y1 = max(0, y1 - 5)
    x2 = min(img.shape[1], x2 + 5)
    y2 = min(img.shape[0], y2 + 5)

    img_cropped = img[y1:y2, x1:x2]

    cv2.imwrite(
        f"{step_dir}/real_C_image_cropped_for_gpt_{ind}.jpg",
        img_cropped,
    )

    img_cropped_base64 = _encode_image_base64(img_cropped)

    class Step(BaseModel):
        explanation: str
        output: str

    class DetResult(BaseModel):
        steps: list[Step]
        res: str

    PROMPT = f"""
    I will provide you an image with one bounding box drawn on it and the cropped image inside the bounding box. For this bounding box, I want you to reason step-by-step and consider surrounding context to determine what object is inside this bounding box.

    Details:
    - The image is input as a base64 string. The bounding box is visually drawn on the image.

    Your goal:
    - Choose the most appropriate label from the following predefined object list for the object inside the bounding box.
    - If you are unsure, respond with `"unknown"`.
    - Output a JSON object **without markdown**.

    Pre-defined object list: {DETECT_OBJECTS}
    """

    prompt_info = f'box: {boxes[0].tolist()}'

    with open(
            f"{step_dir}/real_C_image_gpt_input_{ind}.txt",
            "a",
    ) as f:
        f.write(f'Input: {PROMPT}\n')
        f.write(prompt_info)
        f.write(f'\n')
        f.write(f'\n')

    client, model_name = _get_client_and_model(vlm)
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[{
            "role": "system",
            "content": PROMPT
        }, {
            "role":
                "user",
            "content": [
                {
                    "type": "text",
                    "text": 'This is the whole image with the bounding box.'
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_vis_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": 'This is the cropped image inside the bounding box.'
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_cropped_base64}"
                    }
                },
            ]
        }],
        response_format=DetResult,
    )

    answer = completion.choices[0].message.parsed
    if answer is None:
        raise ValueError("VLM returned empty parsed response for object detection.")
    nwlabels = [answer.res]
    img_vis = visualize_mask(img, boxes, None, nwlabels, None)
    cv2.imwrite(
        f"{step_dir}/real_C_image_gpt_output_{ind}.jpg",
        img_vis,
    )

    with open(
            f"{step_dir}/real_C_image_gpt_output_{ind}.txt",
            "w",
    ) as f:
        f.write(f'Answer: {answer}\n')
        f.write(f'\n')
        f.write(f'\n')

    return answer.res


def refine_tag_with_target(res, target, save_dir, episode_idx, episode_step, ind, vlm):
    tags = [item.tag for item in res]

    class Result(BaseModel):
        res: list[str]

    PROMPT = f"Here is a list of words and a target word. For each word in the list, if it has the same meaning as the target, please replace it with the target. Otherwise, keep it unchanged."
    INPUT = f"List: {tags}\nTarget: {target}\n"

    step_dir = _step_dir(save_dir, episode_idx, episode_step)
    with open(
            f"{step_dir}/refine_prompt_{ind}.txt",
            "w",
    ) as f:
        f.write(f'Prompt: {PROMPT}\n')
        f.write(f'Input: {INPUT}\n')

    client, model_name = _get_client_and_model(vlm)
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "user",
                "content": INPUT
            },
        ],
        response_format=Result,
    )

    ans = completion.choices[0].message.parsed
    if ans is None:
        raise ValueError("VLM returned empty parsed response for tag refinement.")

    with open(
            f"{step_dir}/refine_output_{ind}.txt",
            "w",
    ) as f:
        f.write(f'Answer: {ans}\n')
        f.write(f'\n')
        f.write(f'\n')

    assert len(ans.res) == len(tags)
    for i, item in enumerate(res):
        item.tag = ans.res[i]
    return res


def refine_tag_with_target_obj_list(res, target, save_dir, episode_idx, episode_step, ind,
                                    vlm):

    class Step(BaseModel):
        explanation: str
        output: str

    class Result(BaseModel):
        steps: list[Step]
        output: str

    object_list = list(DETECT_OBJECTS)
    if target not in object_list:
        object_list.append(target)

    PROMPT = f"""
              Here is a predefined object list: {object_list}.
              You will be given one tag. You need to find the object in the list that has the closest meaning to this tag. If you find the object, please output the object. Otherwise, output "unknown". If you are not sure, please output "unknown".
             """

    INPUT = f"""
            tag1: {res}
            """

    step_dir = _step_dir(save_dir, episode_idx, episode_step)
    with open(
            f"{step_dir}/refine_prompt_{ind}.txt",
            "w",
    ) as f:
        f.write(PROMPT)
        f.write(f'\n')
        f.write(INPUT)
        f.write(f'\n')

    client, model_name = _get_client_and_model(vlm)
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "user",
                "content": INPUT
            },
        ],
        response_format=Result,
    )
    ans = completion.choices[0].message.parsed
    if ans is None:
        raise ValueError("VLM returned empty parsed response for object-list refinement.")
    res = ans.output

    with open(
            f"{step_dir}/refine_output_{ind}.txt",
            "w",
    ) as f:
        f.write(f'Answer: {ans}\n')
        f.write(f'\n')
        f.write(f'\n')

    return res


def ask_gpt_similar_objects(obj_list, target, vlm="gemini"):
    obj_str = ", ".join(obj_list)

    prompt = f"""
    Here is a predefined object list: {obj_str}.

    You are given a target object. Your task is to identify all objects in the object list that have the same meaning as the target object.

    Your response should include:
    - `object_list`: a list of objects that have the same meaning with target object. Follow the python list format, e.g., ['object1', 'object2', 'object3'].
    """

    user_input = f"""
    Target object: {target}
    """

    class Step(BaseModel):
        explanation: str
        output: str

    class Result(BaseModel):
        steps: list[Step]
        object_list: list[str]

    client, model_name = _get_client_and_model(vlm)
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
        response_format=Result,
    )

    answer = completion.choices[0].message.parsed
    if answer is None:
        raise ValueError("VLM returned empty parsed response for similar-objects query.")

    result = answer.object_list
    if target not in result:
        result.append(target)
    return result


def check_again_object_in_bbox(img_vis, target, save_dir, episode_idx, episode_step, vlm):
    check_dir = f"{save_dir}/episode-{episode_idx}/check_again"
    os.makedirs(check_dir, exist_ok=True)

    class Step(BaseModel):
        explanation: str
        output: str

    class Result(BaseModel):
        steps: list[Step]
        flag: bool

    prompt = """
    I will give you an image with a bbox drawn on it and an object class label. Your task is to determine whether the object within the bbox is the given object class.

    The image is input as a base64 string. Please notice that the bbox may only cover part of the object. You should use common-sense reasoning to determine whether the main object in the bbox is the given class.

    Instructions:
    1. Carefully examine the RGB image and the region specified by the bbox.
    2. Use visual cues and common-sense reasoning to assess whether the object matches the given class.
    3. Consider the surrounding context of the image and the object class label.
    4. Make your decision through step-by-step observation and reasoning.

    Your response should include:
    - 'steps': the process of chain of thought reasoning.
    - `flag`: a boolean value. If the object in the bbox is the given class, output True. If the object is not the given class, output False.
    """

    prompt_info = (
        f"Whether the object within the bbox in the above image is {target}? "
        "If yes, please output True in the flag field. If no, please output False in the flag field."
    )
    img_base64 = _encode_image_base64(img_vis)

    with open(f"{check_dir}/prompt_{episode_step}.txt", "w") as f:
        f.write(f"Input: {prompt}\n")
        f.write(f"Target: {prompt_info}\n\n")

    client, model_name = _get_client_and_model(vlm)
    messages = [{
        "role": "system",
        "content": prompt
    }, {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            },
            {
                "type": "text",
                "text": prompt_info
            },
        ],
    }]

    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=Result,
        )
    except Exception:
        if vlm != "gemini":
            raise
        completion = client.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            messages=messages,
            response_format=Result,
        )

    answer = completion.choices[0].message.parsed
    if answer is None:
        raise ValueError("VLM returned empty parsed response for check-again.")

    with open(f"{check_dir}/answer_{episode_step}.txt", "w") as f:
        f.write(f"Answer: {answer}\n\n")

    return answer.flag
