VQA_CLEAN_NI = [
    'aokvqa(cauldron,llava_format)',
    'VizWiz(MathV360K)',
    'ai2d(cauldron,llava_format)',
    'hme100k',
    'iam(cauldron)',
    'iiit5k',
    'scienceqa(cauldron,llava_format)',
    'scienceqa(nona_context)',
    'st_vqa(cauldron,llava_format)',
    'tallyqa(cauldron,llava_format)',
    'textcaps',
    'textocr(gpt4v)',
    'vision_flan(filtered)',
    'visualmrc(cauldron)',
    'vqarad(cauldron,llava_format)',
    'vsr(cauldron,llava_format)',
    'visual7w(cauldron,llava_format)',
]

VQA_CLEAN_SI = [
    'clevr(cauldron,llava_format)',
    # 'geo3k',
    'geomverse(cauldron)',
    'hitab(cauldron,llava_format)',
    'iconqa(cauldron,llava_format)',
    'chart2text(cauldron)',
    'chartqa(cauldron,llava_format)',
    'dvqa(cauldron,llava_format)',
    'figureqa(cauldron,llava_format)',
    'hateful_memes(cauldron,llava_format)',
    'infographic_vqa',
    'intergps(cauldron,llava_format)',
    'mapqa(cauldron,llava_format)',
    'multihiertt(cauldron)',
    'orand_car_a',
    'raven(cauldron)',
    'rendered_text(cauldron)',
    'robut_sqa(cauldron)',
    'robut_wikisql(cauldron)',
    'robut_wtq(cauldron,llava_format)',
    'screen2words(cauldron)',
    'sroie',
    'tabmwp(cauldron)',
    'tqa(cauldron,llava_format)',
    'websight(cauldron)',
    'vistext(cauldron)'
]

VQA_GPT_NI = [
    'PMC-VQA(MathV360K)',
    'image_textualization(filtered)',
    'sharegpt4o',
    'sharegpt4v(coco)',
    'sharegpt4v(knowledge)',
    'sharegpt4v(llava)',
    'sharegpt4v(sam)',
    'ai2d(gpt4v)',
    'ai2d(internvl)',
    'allava_instruct_laion4v',
    'allava_instruct_vflan4v',
    'llavar_gpt4_20k',
    'lrv_chart',
    'lrv_normal(filtered)',
    'mavis_math_metagen',
    'mavis_math_rule_geo',
]

VQA_GPT_SI = [
    'CLEVR-Math(MathV360K)',
    'IconQA(MathV360K)',
    'FigureQA(MathV360K)',
    'GEOS(MathV360K)',
    'GeoQA+(MathV360K)',
    'Geometry3K(MathV360K)',
    'geo170k(align)',
    'geo170k(qa)',
    'MapQA(MathV360K)',
    'Super-CLEVR(MathV360K)',
]

language = [
    'magpie_pro(l3_80b_mt)',
    'magpie_pro(l3_80b_st)',
    'magpie_pro(qwen2_72b_st)',
    'mathqa',
]

UNKNOWN = [
    'TabMWP(MathV360K)',
    'UniGeo(MathV360K)',
    'diagram_image_to_text(cauldron)',
    'infographic(gpt4v)',
    'infographic_vqa_llava_format',
    'k12_printing',
]


# Image Sources:
# Visual Genome: MS-COCO (108,077), YFCC100M (328,000)
# Coco: COCO (330k)
# ADE20K: ADE20K (20k)

# Localized Narratives: COCO (123k), ADE20K (20k), Flickr30k (32k), Open Images Train (504k)
# refcocog: COCO (42k)
# LVIS: COCO (100k)
# VQA v2: COCO (123k)
# A-OKVQA: COCO (25k)
# OK-VQA: COCO (14k)
# Visual7W: COCO (47k)
# vsr: COCO (123k)

# GQA: Visual Genome
# ST-VQA: COCO (7.5k), Visual Genome (8.4k)
# TallyQA: COCO and Visual Genome

# TextVQA: Open Images (28k)
# TextCaps: Open Images (28k)
# TextOCR: Open Images (28k)




# VQAv2
# COCO-QA
# Visual7W
# A-OKVQA
# TallyQA
# OK-VQA
# HatefulMemes
# LNarratives
# Screen2Words
# VSR	
# RenderedText
# DocVQA
# TextCaps
# TextVQA
# ST-VQA
# OCR-VQA
# VisualMRC
# IAM
# InfoVQA
# Chart2Text
# DVQA
# VisText
# ChartQA
# PlotQA
# FigureQA
# Tab
# TabMWP
# TAT-Q
# HiTab	
# MultiHiertt	
# FinQA	
# WikiSQL
# SQA	
# WTQ
# GeomVerse
# CLEVR-Math
# CLEVR
# IconQA
# RAVEN
# Inter-GP
# Textbook/ac
# AI2D
# TQA
# ScienceQA
# Differences 
# NLVR2
# GSD
# Spot the diff	
# Sc
# WebSight
# DaTikz


# {
#     "image_path": [
#         {
#             "image_id": str,
#             "image_source": str,
#             "bboxes": List[[x1, y1, x2, y2]],
#             "masks": List[numpy.ndarray],
#             "captions": List[str],
#             "QA": List[str]
#         }
#     ],
#     ...
# }



# GPT-4 Style Conversions:
# LLaVA:
# - Complex Reasoning: 
#       """create a plausible question about the image, and provide the answer in detail
#       To answer such questions, one should require first understanding the visual content, then based on the background knowledge or reasoning, either explain why the things are happening that way, or provide guides and help to user's request.  Make the question challenging by not including the visual content details in the question so that the user needs to reason about that first."""
# - Conversation:
#       """Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question.
#       Ask diverse questions and give corresponding answers.
#       Include questions asking about the visual content of the image, including the object types, counting the objects, object actions, object locations, relative positions between objects, etc. Only include questions that have definite answers:
#       (1) one can see the content in the image that the question asks about and can answer confidently;
#       (2) one can determine confidently from the image that it is not in the image.
#       Do not ask any question that cannot be answered confidently.
#       Also include complex questions that are relevant to the content in the image, for example, asking about background knowledge of the objects in the image, asking to discuss about events happening in the image, etc. Again, do not ask about uncertain details.
#       Provide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized.  You can include multiple paragraphs if necessary."""
# - Detail Description:
#       """Using the provided caption and bounding box information, describe the scene in a detailed manner."""
#
# DocOwl (DocVQA, InfoVQA, WTQ, and VisualMRC) (ChartQA and TextVQA) (CCpdf, RVL-CDIP, VisualMRC):
# - Convert ... 
# - Standard QA
# 
# MathV360K (FQA, GPS, MWP, TQA, VQA),:
# - Image Complexity Classification
#
# LVIS-INSTRUCT4V (adding LVIS)
# - question-answer lists
# - high-quality image descriptions
#
# ALLaVA
# - reasoning steps:
#         < start of description >
#         { description }
#         < end of description >
#         < start of candidate questions >
#         { candidate questions }
#         < end of candidate questions >
#         < start of question >
#         { question }
#         < end of question >
#         < start of answer >
#         { answer }
#         < end of answer >
#
# ShareGPT4V
# - Improves quality of descriptions
#
#



# TO ADD
# Bounding Boxes (grounding)
# Chain of reasoning
# Reference points on the image (Referring QA)


# Reasoning (why)
#   Social Reasoning
#   Commensense Reasoning
#   Abductive Reasoning


GPT_generated_datasets = {
    "ShareGPT4Video": "https://github.com/ShareGPT4Omni/ShareGPT4Video",
    "ShareGPT4V": "https://github.com/ShareGPT4Omni/ShareGPT4V",
    "UNK-VQA": "https://github.com/guoyang9/UNK-VQA",
    "ALLaVA-4V": "https://github.com/FreedomIntelligence/ALLaVA",
    "IDK": "https://github.com/ncsoft/idk",
    "CAP2QA": "https://github.com/ncsoft/cap2qa",
    "ViP-LLaVA-Instruct": "https://github.com/WisconsinAIVision/ViP-LLaVA",
    "LVIS-Instruct4V": "https://github.com/X2FD/LVIS-INSTRUCT4V",
    "ComVint": "https://github.com/RUCAIBox/ComVint",
    "SparklesDialogue": "https://github.com/HYPJUDY/Sparkles",
    "BuboGPT": "https://github.com/magic-research/bubogpt",
    "mPLUG-DocOwl": "https://github.com/X-PLUG/mPLUG-DocOwl",
    "SVIT": "https://github.com/BAAI-DCAI/Visual-Instruction-Tuning",
    "Polite Flamingo": "https://github.com/ChenDelong1999/polite-flamingo",
    "LLaVAR": "https://github.com/SALT-NLP/LLaVAR",
    "ChartLlama": "https://github.com/tingxueronghua/ChartLlama-code",
    "Macaw-LLM": "https://github.com/lyuchenyang/Macaw-LLM",
    "LLaVA-Med": "https://github.com/microsoft/LLaVA-Med",
    "LLaVA-Instruct-150K": "https://github.com/haotian-liu/LLaVA",
    "MathV360K": "https://github.com/HZQ950419/Math-LLaVA",
    "G-LLaVA": "https://github.com/pipilurj/G-LLaVA",
    "MAVIS": "https://github.com/ZrrSkywalker/MAVIS",
}