import argparse
import os
import copy
from PIL import Image
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as mcolors

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    SamPredictor,
    sam_model_registry
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
    return boxes_filt, pred_phrases

def show_mask(mask, ax, c):
    # if random_color:
        #color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
        #color = np.array([30/255, 144/255, 255/255, 0.6])
    rgb = mcolors.to_rgba(c)[:3]
    color= np.array([rgb[0],rgb[1],rgb[2],0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label,color):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
        # Calculate the center of the bounding box
        
    # if(label=="flowers(0.25)"):
    #     center_x = x0 + w / 2 -65
    #     print("works")
    # else:
    #     center_x = x0 + w / 2 -40
    center_x = x0 + w / 2 -30
    
    center_y = y0 + h / 2 +30
    
    # Offset for label text to center it within the box
    label_x_offset = -len(label) * 2  # Adjust this value as needed
    label_y_offset = -10  # Adjust this value as needed
    
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))
    # ax.text(center_x + label_x_offset, center_y + label_y_offset, label, color=(0,0,0,1))
    ax.text(center_x + label_x_offset, center_y + label_y_offset, label, color=(0, 0, 0, 1), fontsize=8)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    # plt.figure(figsize=(12.8, 7.2))
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    # plt.axis('equal') 
    plt.axis('off')

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
    

if __name__ == "__main__":
    
    input_dir= 'Test Images/'
    colour_dir= 'images_1/'  
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    grounded_checkpoint = "./GroundingDINO/weights/groundingdino_swint_ogc.pth"  # change the path of the model
    sam_hq_checkpoint = "sam_hq_vit_l.pth"
    # text_prompt = "Hedge. Path. Flowers. Grass. Pole" #Up to 15 prompts for now Greenhouse. Net. Tractor. Bush. Flowers. Grass. Sky. Hill. Car. 
    text_prompt = "Fireplace. Tripod. Poster. Camera. Floor" #Up to 15 prompts for now Greenhouse. Net. Tractor. Bush. Flowers. Grass. Sky. Hill. Car. 
    colours= ['pink','orange','yellow','green','cyan','purple','white','red','olive','blue','brown','gray','coral','lavender','beige']
    output_dir = "segmented_1/"
    box_threshold = 0.2
    text_threshold = 0.2
    device = "cpu"
    prompts = text_prompt.split(". ")
    lowercase_prompts = [word.lower() for word in prompts]
    for j in range(41,50):
        num= str(j)
        if(j<10):
            col= input_dir+'00000'+num+'_color.png' 
            dep= input_dir+'00000'+num+'_aligned_depth.png' 
        elif(j<100):
            col= input_dir+'0000'+num+'_color.png' 
            dep= input_dir+'0000'+num+'_aligned_depth.png'
        elif(j<1000):
            col= input_dir+'000'+num+'_color.png' 
            dep= input_dir+'000'+num+'_aligned_depth.png'  
        else:
            col= input_dir+'00'+num+'_color.png' 
            dep= input_dir+'00'+num+'_aligned_depth.png' 

        
        
        # image_path= input_dir+colour_dir+'image_'+num+'.png' 
        image_pil, image = load_image(col)
        model = load_model(config_file, grounded_checkpoint, device=device)
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )
        sam = sam_model_registry["vit_l"](checkpoint=sam_hq_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        image  = cv2.imread(col)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        # plt.axis('equal') 
        boxcolors=[]
        for box, label in zip(boxes_filt, pred_phrases):
            colorUsed=''
            i=0
            for p in lowercase_prompts:
                l=label.split("(")[0]
                if(p==l):
                    colourUsed=colours[i]
                    boxcolors.append(colours[i])
                    break;
                i=i+1
            show_box(box.numpy(), plt.gca(), label,colourUsed)
        c=0;
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), boxcolors[c])
            # show_mask(mask.cpu().numpy(), plt.gca(), colours[c])
            c=c+1
        plt.axis('off')
        
        result_path = input_dir  + 'temp_segmented_' + num + '.png'

    
        plt.savefig(
            result_path, 
            bbox_inches="tight", 
            pad_inches=0.0,
            dpi=165.2
        )
        
        img = Image.open(result_path)  # Replace with the path to your image file

        # Convert the image to 24-bit depth
        img = img.convert("RGB")

        # Save the converted image
        result_path = input_dir + 'segmented_' + num + '.png'
        img.save(result_path)  # Replace with the desired path for the converted image
        # end for
            # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
        print(num+ ' done')


