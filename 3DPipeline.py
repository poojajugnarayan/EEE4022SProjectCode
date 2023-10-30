import argparse
import os
import copy
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as mcolors
import open3d as o3d
import sys
import matplotlib.pyplot as plt
import cv2
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import ( SamPredictor, sam_model_registry)


def load_image(image_path):
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
    center_x = x0 + w / 2
    center_y = y0 + h / 2
    
    # Offset for label text to center it within the box
    label_x_offset = -len(label) * 2  # Adjust this value as needed
    label_y_offset = -10  # Adjust this value as neede
    
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))
    ax.text(center_x + label_x_offset, center_y + label_y_offset, label, color=(0,0,0,1))

def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
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
    pcds = []
    merged_pcd = o3d.geometry.PointCloud()
    input_dir= 'Test Images/'
    colour_dir= 'images/'
    depth_dir= 'depth/'
    segmented_dir= 'segmented/'
    output_dir='Mesh Tests/'
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    grounded_checkpoint = "./GroundingDINO/weights/groundingdino_swint_ogc.pth"  # change the path of the model
    sam_hq_checkpoint = "sam_hq_vit_l.pth"
    text_prompt = "Hedge. Path. Flowers. Grass. Pole" #Up to 15 prompts for now Greenhouse. Net. Tractor. Bush. Flowers. Grass. Sky. Hill. Car. 
    colours= ['red','orange','yellow','green','blue','purple','white','pink','olive','cyan','brown','gray','coral','lavender','beige']
    box_threshold = 0.25
    text_threshold = 0.25
    device = "cpu"
    prompts = text_prompt.split(". ")
    lowercase_prompts = [word.lower() for word in prompts]
    
    
    for j in range(0,15):
        num= str(j)
        image_path= input_dir+colour_dir+'image_'+num+'.png' 
        image_pil, image = load_image(image_path)
        model = load_model(config_file, grounded_checkpoint, device=device)
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )
        sam = sam_model_registry["vit_l"](checkpoint=sam_hq_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        input_image  = cv2.imread(image_path)
        input_height, input_width, _ = input_image .shape
        output_image = copy.deepcopy(input_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(output_image)

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
            c=c+1
        plt.axis('off')
        result_path= input_dir+output_dir+'segmented_'+num+'.png'
        plt.savefig(
            result_path, 
            bbox_inches='tight', 
            pad_inches=0.0,
            dpi=(2560/31)
        )
        # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
        print(num+ ' done')

        col= input_dir+segmented_dir+'segmented_'+num+'.png'  
        dep= input_dir+depth_dir+'depth_'+num+'.png'
        color_raw = o3d.io.read_image(col)
        depth_raw = o3d.io.read_image(dep)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.005) 
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        merged_pcd += pcd
        pcds.append(pcd)
    merged_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(merged_pcd, depth=9)
    o3d.visualization.draw_geometries(pcds)
    o3d.visualization.draw_geometries([mesh])
    output_filename = output_dir+"mesh_"+num+".ply"
    o3d.io.write_triangle_mesh(output_filename,mesh)
    print(output_filename, " created.")

