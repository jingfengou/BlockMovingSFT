#!/usr/bin/env python3
# generate_dataset_handler.py
# Run this script to generate the blockmoving_dataset.py file

import os

# The content of blockmoving_dataset.py
# We use this approach to avoid issues with special characters in tool calls
CONTENT = """# data/blockmoving_dataset.py
# Dataset handler for BlockMoving COT data in parquet format

import json
import traceback
from io import BytesIO
from PIL import Image, ImageFile, PngImagePlugin
import PIL
import sys
import os

# Add MathCanvas path for imports
MATHCANVAS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'MathCanvas', 'BAGEL-Canvas')
if MATHCANVAS_PATH not in sys.path:
    sys.path.insert(0, MATHCANVAS_PATH)

try:
    from data.interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
    from data.data_utils import pil_img2rgb
except ImportError:
    print("Warning: Could not import from MathCanvas. Using standalone implementation.")
    InterleavedBaseIterableDataset = object
    ParquetStandardIterableDataset = object
    def pil_img2rgb(img):
        return img.convert('RGB') if img.mode != 'RGB' else img

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class BlockMovingCOTIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    '''Dataset handler for BlockMoving COT data in parquet format.'''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_of_image = self.tokenizer.convert_tokens_to_ids(chr(60) + '|vision_start|' + chr(62))
        self.im_start = self.tokenizer.convert_tokens_to_ids(chr(60) + '|im_start|' + chr(62))
        self.end_of_text = self.tokenizer.convert_tokens_to_ids(chr(60) + '|endoftext|' + chr(62))

    def _get_pil_image(self, image_data):
        '''Handle loading image from bytes or PIL Image.'''
        if isinstance(image_data, bytes):
            return Image.open(BytesIO(image_data))
        elif isinstance(image_data, Image.Image) or isinstance(image_data, PIL.PngImagePlugin.PngImageFile):
            return image_data
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            return Image.open(BytesIO(image_data['bytes']))
        else:
            raise TypeError(f"Unsupported image data type: {type(image_data)}")

    def parse_row(self, row):
        '''Parse a single parquet row into the required format.'''
        try:
            answer = row.get('answer', '')
            question_interleave_str = row.get('question_interleave', '[]')
            solution_interleave_str = row.get('solution_interleave', '[]')
            
            # Parse JSON strings
            if isinstance(question_interleave_str, str):
                question_interleave = json.loads(question_interleave_str)
            else:
                question_interleave = question_interleave_str
                
            if isinstance(solution_interleave_str, str):
                solution_interleave = json.loads(solution_interleave_str)
            else:
                solution_interleave = solution_interleave_str

            question_images = row.get('question_images', [])
            solution_images = row.get('solution_images', [])

            if not answer or len(question_interleave) == 0 or len(solution_interleave) == 0:
                return {}
            
            data = self._init_data()
            
            # 1. Process question part
            success = self._add_interleave_sequence(
                data, 
                question_interleave,
                question_images,
                section_type="question"
            )
            if not success:
                return {}
            
            # 2. Process solution part  
            success = self._add_interleave_sequence(
                data,
                solution_interleave,
                solution_images,
                section_type="solution"
            )
            if not success:
                return {}
            
            return data
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error parsing row: {e}")
            return {}

    def _add_interleave_sequence(self, data, interleave_list, images_list, section_type):
        '''Add interleaved text and image sequence in original order.'''
        
        try:
            image_count = 0
            
            for i, item in enumerate(interleave_list):
                if item['type'] == 'text':
                    text_content = item.get('content', '')
                    if text_content and text_content.strip():
                        special_token_label = None
                        
                        if section_type == "question":
                            need_loss = False
                        else:
                            need_loss = True
                            has_following_image = (i + 1 < len(interleave_list)) and (interleave_list[i + 1]['type'] == 'image')
                            is_last_item = (i == len(interleave_list) - 1)
                            
                            if has_following_image:
                                special_token_label = self.start_of_image
                            elif is_last_item:
                                special_token_label = self.end_of_text
                            else:
                                special_token_label = self.im_start
                        
                        data = self._add_text(
                            data,
                            text_content.strip(),
                            need_loss=need_loss,
                            enable_cfg=True,
                            special_token_label=special_token_label
                        )
                        
                elif item['type'] == 'image':
                    image_count += 1
                    
                    try:
                        image_index = item.get('index', 0)
                        if image_index < len(images_list):
                            image_data = images_list[image_index]
                            image = self._get_pil_image(image_data)
                            image = pil_img2rgb(image)
                        else:
                            print(f"Image index {image_index} out of bounds")
                            return False
                            
                        if section_type == "question":
                            data = self._add_image(
                                data,
                                image,
                                need_loss=False,
                                need_vae=True,
                                need_vit=True,
                                enable_cfg=True,
                            )
                        else:
                            data = self._add_image(
                                data,
                                image,
                                need_loss=True,
                                need_vae=True,  
                                need_vit=True,               
                                enable_cfg=True,
                            )
                                
                    except Exception as e:
                        print(f"Failed to load image: {e}")
                        traceback.print_exc()
                        return False
            
            return True
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error processing interleave sequence: {e}")
            return False
"""

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'data', 'blockmoving_dataset.py')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(CONTENT)
    
    print(f"Generated: {output_path}")

if __name__ == '__main__':
    main()
