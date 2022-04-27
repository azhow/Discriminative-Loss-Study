import os, sys
import json
import glob
import argparse
import numpy as np
import PIL.Image as Image

from panopticapi.utils import IdGenerator, save_json

try:
    # set up path for cityscapes scripts
    # sys.path.append('./cityscapesScripts/')
    from cityscapesscripts.helpers.labels import labels, id2label
except Exception:
    raise Exception("Please load Cityscapes scripts from https://github.com/mcordts/cityscapesScripts")

def convert_dataset_to_coco_format(input_folder, output_folder):
    output_file = os.path.join(output_folder, "converted.json")
    output_folder = os.path.join(output_folder, "images\\")

    if not os.path.exists(output_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(output_folder))
        os.makedirs(output_folder)

    instance_files = sorted(glob.glob(input_folder + "*/plant???*.png"))
    
    categories = []
    for idx, cat in enumerate([("background", [255, 0, 0], 0), ("leaf", [0, 255, 0], 1)]):
        # Create entry
        categories.append({
            'id': idx,
            'name': cat[0],
            'color': cat[1],
            'supercategory': cat[0],
            'isthing': cat[2]})

    categories_dict = {cat['id']: cat for cat in categories}

    images = []
    annotations = []

    for working_idx, f in enumerate(instance_files):
        if working_idx % 10 == 0:
            print(working_idx, len(instance_files))

        im = Image.open(f)
        original_format = np.array(im)
        palette = im.getpalette()
        index_bg = 0
        if palette is not None:
            palette = np.reshape( palette, (-1, 3) )
            # Find index of black colors (bg)
            index_bg = np.argmax(np.all(palette==0,axis=-1))

        # Image ID is FOLDER_plantXXXX
        folder = f.split('\\')[-2]
        image_id = f.split('\\')[-1].split('.')[-2]
        image_id = f'{folder}_{image_id}'

        # Image file name is FOLDER_plantXXXX.png
        image_filename = image_id + ".png"
        file_name = image_filename
    
        # image entry, id for image is its filename without extension
        images.append({"id": image_id,
                       "width": original_format.shape[1],
                       "height": original_format.shape[0],
                       "file_name": image_filename})

        pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)
        id_generator = IdGenerator(categories_dict)

        idx = 0
        
        l = np.unique(original_format)

        segm_info = []
        for el in l:
            semantic_id = 1 if el != index_bg else 0
            is_crowd = 0
            if semantic_id not in categories_dict:
                continue
            if categories_dict[semantic_id]['isthing'] == 0:
                is_crowd = 0
            mask = original_format == el
            segment_id, color = id_generator.get_id_and_color(semantic_id)
            pan_format[mask] = color

            area = np.sum(mask) # segment area computation

            # bbox computation for a segment
            hor = np.sum(mask, axis=0)
            hor_idx = np.nonzero(hor)[0]
            x = hor_idx[0]
            width = hor_idx[-1] - x + 1
            vert = np.sum(mask, axis=1)
            vert_idx = np.nonzero(vert)[0]
            y = vert_idx[0]
            height = vert_idx[-1] - y + 1
            bbox = [int(x), int(y), int(width), int(height)]

            segm_info.append({"id": int(segment_id),
                              "category_id": int(semantic_id),
                              "area": int(area),
                              "bbox": bbox,
                              "iscrowd": int(is_crowd)})

        annotations.append({'image_id': image_id,
                            'file_name': file_name,
                            "segments_info": segm_info})

        Image.fromarray(pan_format).save(os.path.join(output_folder, file_name))
    
    d = {'images': images, 'annotations': annotations, 'categories': categories,}
    save_json(d, output_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help="Input folder with the images to be converted")
    parser.add_argument('--output_folder', type=str, help="Output folder for the converted data")
    args = parser.parse_args()
    convert_dataset_to_coco_format(args.input_folder, args.output_folder)