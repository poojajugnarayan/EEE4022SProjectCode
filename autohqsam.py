# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
# import os
# import time
# import matplotlib.colors as mcolors


# def show_anns(anns, colours):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#     i=0;
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         rgb = mcolors.to_rgba(colours[i])[:3]
#         color_mask= np.array([rgb[0],rgb[1],rgb[2],0.6])
#         # color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#         i=i+1;
#         if(i==14):
#             i=0
#     ax.imshow(img)
    
# if __name__ == "__main__":
#     t=0
#     for i in range (8, 16):
#         start_time = time.time()
#         num=str(i)
#         sam_checkpoint = "sam_hq_vit_l.pth"
#         model_type = "vit_l"
#         device = "cpu"
#         colours= ['red','orange','yellow','green','blue','purple','white','pink','olive','cyan','brown','gray','coral','lavender','beige']
#         sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#         sam.to(device=device)
#         predictor = SamPredictor(sam)
#         image = cv2.imread('seqtestimages/image_'+num+'.png')
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask_generator = SamAutomaticMaskGenerator(
#             model=sam,
#             points_per_side=32,
#             pred_iou_thresh=0.8,
#             stability_score_thresh=0.9,
#             crop_n_layers=1,
#             crop_n_points_downscale_factor=2,
#             min_mask_region_area=100,  # Requires open-cv to run post-processing
#         )
#         masks = mask_generator.generate(image)
#         len(masks)
#         plt.figure(figsize=(20,20))
#         plt.imshow(image)
#         show_anns(masks, colours)
#         plt.axis('off')
#         output_dir = 'seq_hq'
#         os.makedirs(output_dir, exist_ok=True)
#         result = 'output'+num+'.jpg'
#         plt.savefig(
#             os.path.join(output_dir, result),
#             bbox_inches="tight", dpi=300, pad_inches=0.0
#         )
#         end_time = time.time()
#         execution_time = end_time - start_time
#         t=t+execution_time
#         print("The time taken for",num, "outdoor images using sam_hq_vit_l is",execution_time,"s") 
#         if (i==5 or i==10 or i==15):
#             print(t)
    
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import os
import time
import matplotlib.colors as mcolors


def show_anns(anns, colours):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    i=0;
    for ann in sorted_anns:
        m = ann['segmentation']
        rgb = mcolors.to_rgba(colours[i])[:3]
        color_mask= np.array([rgb[0],rgb[1],rgb[2],0.6])
        # color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        i=i+1;
        if(i==14):
            i=0
    ax.imshow(img)
    
if __name__ == "__main__":
    t=0
    for i in range (341,342):
        start_time = time.time()
        num=str(i)
        sam_checkpoint = "sam_hq_vit_l.pth"
        model_type = "vit_l"
        device = "cpu"
        colours= ['red','orange','yellow','green','blue','purple','white','pink','olive','cyan','brown','gray','coral','lavender','beige']
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        image = cv2.imread('indoor10test/'+num+'.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
        masks = mask_generator.generate(image)
        len(masks)
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks, colours)
        plt.axis('off')
        output_dir = 'indoor'
        os.makedirs(output_dir, exist_ok=True)
        result = 'indoor_hq'+num+'.jpg'
        plt.savefig(
            os.path.join(output_dir, result),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        end_time = time.time()
        execution_time = end_time - start_time
        t=t+execution_time
        print("The time taken for",num, " indoor images using hq_sam_vit_l is",execution_time,"s")   
        if (i==5 or i==10 or i==15):
            print(t)
    