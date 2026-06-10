from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, inference
from llava.utils import disable_torch_init
import json
import os
import ipdb
from tqdm import tqdm

model_path = "../cache/llava-v1.5-13b"


llava_model_name = get_model_name_from_path(model_path)
model_base = None

disable_torch_init()

llava_model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base, llava_model_name
)

with open("../AMBER/data/query/query_all.json", "r") as f:
    AMBER_dataset = json.load(f)

with open("./visual_evidence/detr_renset101_object_and_box_context_AMBER.json", "r") as f:
    object_context_list = json.load(f)

with open("./visual_evidence/AMBER_sgg_relation_context_reltr_top50.json", "r") as f:
    relation_context_list = json.load(f)


def map_answer_to_choice(response):
    # Only keep the first sentence
    if response.find('.') != -1:
        response = response.split('.')[0]
    
    response = response.replace(',', '')
    if 'No' in response or 'not' in response or 'no' in response:
        return "No"
    elif 'Yes' in response or 'yes' in response:
        return "Yes"
    else:
        return "None"

log_list = []

for idx, item in enumerate(AMBER_dataset):
    print(f"processing question: {idx}/{len(AMBER_dataset)}")
    
    img_path = os.path.join("../AMBER/image", item["image"])

    # compose object context and relation context
    object_list = object_context_list[item["image"]]["object_context"]
    relation_list = relation_context_list[item["image"]]["relation_context"]

    object_context_prompt = ""
    if len(object_list) > 0:
        # object_context = []
        # for k in range(len(object_list)):
        #     object_context.append(f"{object_list[k]}")
        # object_context = ", ".join(object_context)
        # count object
        object_count_dict = {}
        for object in object_list:
            if object not in object_count_dict.keys():
                object_count_dict[object] = 1
            else:
                object_count_dict[object] += 1
        
        object_context = []
        for object in object_count_dict.keys():
            object_context.append(f"{object_count_dict[object]} {object}")
        object_context = ", ".join(object_context)
        object_context_prompt = f"You can see {object_context} in the image."
    
    relation_context_prompt = ""
    if len(relation_list) > 0:
        relation_context = []
        for k in range(len(relation_list)):
            if relation_list[k][1] not in ["has", "of"]:
                relation_context.append(f"{relation_list[k][0]} {relation_list[k][1]} {relation_list[k][2]}")
        if len(relation_context) > 0:
            relation_context = ", ".join(relation_context)
            relation_context_prompt = f"You can see {relation_context} in the image."
    
    USE_OBJECT_CONTEXT = True
    USE_RELATION_CONTEXT = False
    if USE_OBJECT_CONTEXT and USE_RELATION_CONTEXT:
        if object_context_prompt != "" and relation_context_prompt != "":
            context = f"{object_context_prompt} {relation_context_prompt}"
        elif object_context_prompt == "" and relation_context_prompt != "":
            context = relation_context_prompt
        elif object_context_prompt != "" and relation_context_prompt == "":
            context = object_context_prompt
        else:
            context = ""
    elif USE_OBJECT_CONTEXT and not USE_RELATION_CONTEXT:
        if object_context_prompt != "":
            context = object_context_prompt
        else:
            context = ""
    elif not USE_OBJECT_CONTEXT and USE_RELATION_CONTEXT:
        if relation_context_prompt != "":
            context = relation_context_prompt
        else:
            context = ""
    else:
        context = ""

    if context != "":
        prompt = f"""{context}\n{item["query"]}"""
    else:
        prompt = item["query"]

    args = type('Args', (), {
                "model_path": model_path,
                "model_base": model_base,
                "model_name": llava_model_name,
                "query": prompt,
                "conv_mode": None,
                "image_file": img_path,
                "sep": ",",
                "temperature": 0.2,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()

    response = inference(args, tokenizer, model, image_processor, llava_model_name)
    
    if item["id"] >= 1005:
        pred = map_answer_to_choice(response)
    else:
        pred = response

    print(prompt)
    print(f"##response_original: {response}")
    print(f"##response: {pred}")
    log_list.append({
        "id": item["id"],
        "prompt": prompt,
        "response_original": response,
        "response": pred
    })

with open(f'./results/AMBER/llava1.5_13b_with_detr_renset101_context.json', 'w', encoding='utf-8') as f2:
    json.dump(log_list, f2, ensure_ascii=False, indent=2)