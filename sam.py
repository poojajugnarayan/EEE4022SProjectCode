from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry_baseline, SamPredictor
import os
import sys
sys.path.append("..")


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'.png',
                    bbox_inches='tight', pad_inches=-0.1)
        plt.close()


def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=-0.1)
    plt.close()


if __name__ == "__main__":
   
    sam_checkpoint = "sam_vit_l_0b3195.pth"#sam_vit_h_4b8939.pth"
    model_type = "vit_l"
    device = "cpu"
    sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    image = cv2.imread('seqtestimages/image_5.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    

    # input_box = np.array([[4,13,1007,1023]])
    # input_point, input_label = None, None
    output_dir = 'seq_baseline_sam_result_not_auto/'
    os.makedirs(output_dir, exist_ok=True)
    
    input_point = np.array([[1100,300]])
    input_label = np.array([1])
    # input_point = np.array([[1100,345]])
    # input_label = np.array([1])
    # input_label = np.ones(input_point.shape[0])
    input_box = None
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    # plt.show()
    plt.savefig(output_dir+'image_5' + '.png', bbox_inches='tight', pad_inches=-0.1)
    plt.close()
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box = input_box,
        multimask_output=False,
    )
    
    result = 'output_5'        
    show_res(masks,scores,input_point, input_label, input_box, output_dir + result, image)
        
        

