import os
# from cv_utils.object_list.matterport_categories_1_10 import categories
from cv_utils.object_list.nyu_categories import categories

DETECT_OBJECTS = [cat['name'].lower() for cat in categories]
INTEREST_OBJECTS = ['bed', 'chair', 'toilet', 'potted_plant', 'tv_monitor', 'sofa']

MODEL_NAME = 'gemini-2.5-flash'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")
