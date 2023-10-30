# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from segment_anything import sam_model_registry_baseline, SamAutomaticMaskGenerator,SamPredictor
# import os
# import sys
# import matplotlib.colors as mcolors
# sys.path.append("..")
# import time


# def show_anns(anns, colours):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0],
#                    sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:, :, 3] = 0
#     i=0
#     for ann in sorted_anns: 
#         m = ann['segmentation']
#         rgb = mcolors.to_rgba(colours[i])[:3]
#         color_mask= np.array([rgb[0],rgb[1],rgb[2],0.6])
#         #color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#         i=i+1;
#         if(i==14):
#             i=0
#     ax.imshow(img)


# if __name__ == "__main__":
    
#     for i in range (1,16):
#         start_time = time.time()
#         num=str(i)
#         sam_checkpoint = "sam_vit_l_0b3195.pth"
#         model_type = "vit_l"
#         device = "cpu"
#         sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint)
#         sam.to(device=device)
#         predictor = SamPredictor(sam)
#         image = cv2.imread('seqtestimages/image_'+num+'.png')
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask_generator = SamAutomaticMaskGenerator(sam)
#         colours= ['red','orange','yellow','green','blue','purple','white','pink','olive','cyan','brown','gray','coral','lavender','beige']
#         masks = mask_generator.generate(image)
#         len(masks)
#         plt.figure(figsize=(20, 20))
#         plt.imshow(image)
#         show_anns(masks,colours)
#         plt.axis('off')
#         output_dir = 'seq_baseline_sam_result'
#         os.makedirs(output_dir, exist_ok=True)
#         result = 'output'+num+'.jpg'
#         plt.savefig(
#             os.path.join(output_dir, result),
#             bbox_inches="tight", dpi=300, pad_inches=0.0
#         )
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print("The time taken for ",num, " outdoor images using sam_vit_l_0b3195 is",execution_time,"s")
        
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from segment_anything import sam_model_registry_baseline, SamAutomaticMaskGenerator,SamPredictor
# import os
# import sys
# import matplotlib.colors as mcolors
# sys.path.append("..")
# import time


# def show_anns(anns, colours):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0],
#                    sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:, :, 3] = 0
#     i=0
#     for ann in sorted_anns: 
#         m = ann['segmentation']
#         rgb = mcolors.to_rgba(colours[i])[:3]
#         color_mask= np.array([rgb[0],rgb[1],rgb[2],0.6])
#         #color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#         i=i+1;
#         if(i==14):
#             i=0
#     ax.imshow(img)


# if __name__ == "__main__":
    
#     for i in range (9,16):
#         start_time = time.time()
#         num=str(i)
#         sam_checkpoint = "sam_vit_h_4b8939.pth"
#         model_type = "vit_h"
#         device = "cpu"
#         sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint)
#         sam.to(device=device)
#         predictor = SamPredictor(sam)
#         image = cv2.imread('seqtestimages/image_'+num+'.png')
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask_generator = SamAutomaticMaskGenerator(sam)
#         colours= ['red','orange','yellow','green','blue','purple','white','pink','olive','cyan','brown','gray','coral','lavender','beige']
#         masks = mask_generator.generate(image)
#         len(masks)
#         plt.figure(figsize=(20, 20))
#         plt.imshow(image)
#         show_anns(masks,colours)
#         plt.axis('off')
#         output_dir = 'seq_baseline_sam_result_h'
#         os.makedirs(output_dir, exist_ok=True)
#         result = 'output'+num+'.jpg'
#         plt.savefig(
#             os.path.join(output_dir, result),
#             bbox_inches="tight", dpi=300, pad_inches=0.0
#         )
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print("The time taken for",num, " outdoor images using sam_vit_h is",execution_time,"s")        



import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry_baseline, SamAutomaticMaskGenerator,SamPredictor
import os
import sys
import matplotlib.colors as mcolors
sys.path.append("..")
import time


def show_anns(anns, colours):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    i=0
    for ann in sorted_anns: 
        m = ann['segmentation']
        rgb = mcolors.to_rgba(colours[i])[:3]
        color_mask= np.array([rgb[0],rgb[1],rgb[2],0.6])
        #color_mask = np.concatenate([np.random.random(3), [0.35]])
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
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cpu"
        sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        image = cv2.imread('indoor10test/'+num+'.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_generator = SamAutomaticMaskGenerator(sam)
        colours= ['red','orange','yellow','green','blue','purple','white','pink','olive','cyan','brown','gray','coral','lavender','beige']
        masks = mask_generator.generate(image)
        len(masks)
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks,colours)
        plt.axis('off')
        output_dir = 'indoor'
        os.makedirs(output_dir, exist_ok=True)
        result = 'indoor_vit_h'+num+'.jpg'
        plt.savefig(
            os.path.join(output_dir, result),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        end_time = time.time()
        execution_time = end_time - start_time
        print("The time taken for",num, " indoor images using sam_vit_h is",execution_time,"s")   
        t=t+execution_time 
        if (i==5 or i==10 or i==15):
            print(t)     

# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from segment_anything import sam_model_registry_baseline, SamAutomaticMaskGenerator,SamPredictor
# import os
# import sys
# import matplotlib.colors as mcolors
# sys.path.append("..")
# import time


# def show_anns(anns, colours):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0],
#                    sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:, :, 3] = 0
#     i=0
#     for ann in sorted_anns: 
#         m = ann['segmentation']
#         rgb = mcolors.to_rgba(colours[i])[:3]
#         color_mask= np.array([rgb[0],rgb[1],rgb[2],0.6])
#         #color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#         i=i+1;
#         if(i==14):
#             i=0
#     ax.imshow(img)


# if __name__ == "__main__":
#     t=0
#     for i in range (341,342):
#         start_time = time.time()
#         num=str(i)
#         sam_checkpoint = "sam_vit_l_0b3195.pth"
#         model_type = "vit_l"
#         device = "cpu"
#         sam = sam_model_registry_baseline[model_type](checkpoint=sam_checkpoint)
#         sam.to(device=device)
#         predictor = SamPredictor(sam)
#         image = cv2.imread('indoor10test/'+num+'.png')
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask_generator = SamAutomaticMaskGenerator(sam)
#         colours= ['red','orange','yellow','green','blue','purple','white','pink','olive','cyan','brown','gray','coral','lavender','beige']
#         masks = mask_generator.generate(image)
#         len(masks)
#         plt.figure(figsize=(20, 20))
#         plt.imshow(image)
#         show_anns(masks,colours)
#         plt.axis('off')
#         output_dir = 'indoor'
#         os.makedirs(output_dir, exist_ok=True)
#         result = 'indoor_vit_l'+num+'.jpg'
#         plt.savefig(
#             os.path.join(output_dir, result),
#             bbox_inches="tight", dpi=300, pad_inches=0.0
#         )
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print("The time taken for",num, " indoor images using sam_vit_l is",execution_time,"s")   
#         t=t+execution_time 
#         if (i==5 or i==10 or i==15):
#             print(t)     
