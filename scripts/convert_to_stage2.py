"""
Convert enhanced_cot_cleaned.json to Stage 2 Parquet format for MathCanvas training.

Usage:
    conda activate bagel-canvas
    python convert_to_stage2.py
"""

import json
import os
import re
import pandas as pd
from PIL import Image
import io


# Configuration
QUESTION_PROMPT = """The top row of images shows different views of the initial state of a cube stack, while the bottom row shows different views of the final state after transformation. During the transformation process, blocks can move one unit in any direction (forward, backward, left, right, up, down). If the target position is empty, the block can move there directly; if the target position already has a block, they swap places. Blocks cannot float in the air. If a block is moved away from a position, any block above it will fall down until reaching a supporting surface. The xyz axes are shown in the diagram, and each block's position can be precisely identified using coordinates (x1,y1,z1). Which of the following transformation sequences can change the cube stack from the initial state to the final state shown in the diagram? Please answer from options A, B, C, or D."""
INPUT_JSON = '/home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/data/enhanced_cot_cleaned.json'
IMAGE_ROOT = '/home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/data/blockmoving'
OUTPUT_DIR = '/home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/data/output'


def parse_cot_text(cot_text):
    """
    Parses cot_text into a list of segments.
    Each segment is a dict: {'type': 'text'|'image', 'content': string}
    """
    segments = []
    # Regex to find <image>path</image> tags
    pattern = re.compile(r'(<image>.*?</image>)', re.DOTALL)
    parts = pattern.split(cot_text)
    
    for part in parts:
        if not part:
            continue
        
        if part.startswith('<image>') and part.endswith('</image>'):
            image_path = part[len('<image>'):-len('</image>')].strip()
            segments.append({'type': 'image', 'content': image_path})
        else:
            # Clean up text: strip leading/trailing whitespace
            cleaned_text = part.strip()
            if cleaned_text:
                segments.append({'type': 'text', 'content': cleaned_text})
            
    return segments


def load_image_bytes(image_root, rel_path):
    """Load image and return bytes."""
    full_path = os.path.join(image_root, rel_path)
    with open(full_path, 'rb') as f:
        img_bytes = f.read()
    # Validate image
    Image.open(io.BytesIO(img_bytes)).verify()
    return img_bytes


def load_task_options(image_root, task_id):
    """Load options from task's data.json file."""
    data_json_path = os.path.join(image_root, f'task_{task_id}', 'data.json')
    if os.path.exists(data_json_path):
        with open(data_json_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
        options = task_data.get('options', {})
        # Format options as text
        options_text = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
        return options_text
    return ""


def convert_item(item, image_root):
    """
    Convert a single item to Stage 2 format.
    
    Returns:
        dict with keys: answer, question_interleave, solution_interleave, 
                        question_images, solution_images
    """
    task_id = item['task_id']
    gt_answer = item['gt']
    cot_text = item['cot_text']
    
    segments = parse_cot_text(cot_text)
    
    # Initialize output structures
    question_interleave = []
    solution_interleave = []
    question_images = []
    solution_images = []
    
    # Find the first image (composite) for question
    first_image_idx = None
    for i, seg in enumerate(segments):
        if seg['type'] == 'image':
            first_image_idx = i
            break
    
    if first_image_idx is None:
        print(f"Warning: task {task_id} has no images, skipping")
        return None
    
    # Build Question:
    # 1. Text prompt with options
    options_text = load_task_options(image_root, task_id)
    if options_text:
        question_text = f"{QUESTION_PROMPT}\n\nOptions:\n{options_text}"
    else:
        question_text = QUESTION_PROMPT
    question_interleave.append({'type': 'text', 'content': question_text})
    
    # 2. First image (composite)
    first_image_path = segments[first_image_idx]['content']
    try:
        img_bytes = load_image_bytes(image_root, first_image_path)
        question_images.append({'bytes': img_bytes})
        question_interleave.append({'type': 'image', 'index': 0})
    except Exception as e:
        print(f"Error loading question image for task {task_id}: {e}")
        return None
    
    # Build Solution: everything after the first image
    solution_image_count = 0
    for seg in segments[first_image_idx + 1:]:
        if seg['type'] == 'text':
            solution_interleave.append({'type': 'text', 'content': seg['content']})
        elif seg['type'] == 'image':
            try:
                img_bytes = load_image_bytes(image_root, seg['content'])
                solution_images.append({'bytes': img_bytes})
                solution_interleave.append({'type': 'image', 'index': solution_image_count})
                solution_image_count += 1
            except Exception as e:
                print(f"Error loading solution image {seg['content']} for task {task_id}: {e}")
                return None
    
    return {
        'task_id': task_id,
        'answer': gt_answer,
        'question_interleave': question_interleave,
        'solution_interleave': solution_interleave,
        'question_images': question_images,
        'solution_images': solution_images,
    }


def main():
    print(f"Loading data from {INPUT_JSON}")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} items")
    
    output_rows = []
    failed_count = 0
    
    for i, item in enumerate(data):
        result = convert_item(item, IMAGE_ROOT)
        if result:
            output_rows.append(result)
        else:
            failed_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} items...")
    
    print(f"\nConversion complete: {len(output_rows)} success, {failed_count} failed")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save to parquet
    df = pd.DataFrame(output_rows)
    parquet_path = os.path.join(OUTPUT_DIR, 'train.parquet')
    df.to_parquet(parquet_path)
    print(f"Saved parquet to {parquet_path}")
    
    # Create info file
    info = {
        'num_samples': len(df),
        'file_path': parquet_path,
        'columns': list(df.columns)
    }
    info_path = os.path.join(OUTPUT_DIR, 'train_parquet_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Saved info to {info_path}")
    
    # Print sample for verification
    print("\n=== Sample Output (first item) ===")
    if len(output_rows) > 0:
        sample = output_rows[0]
        print(f"task_id: {sample['task_id']}")
        print(f"answer: {sample['answer']}")
        print(f"question_interleave: {len(sample['question_interleave'])} items")
        for item in sample['question_interleave']:
            if item['type'] == 'text':
                print(f"  - text: {item['content'][:50]}...")
            else:
                print(f"  - image: index={item['index']}")
        print(f"solution_interleave: {len(sample['solution_interleave'])} items")
        print(f"question_images: {len(sample['question_images'])} images")
        print(f"solution_images: {len(sample['solution_images'])} images")


if __name__ == '__main__':
    main()
