# convert_cot_to_parquet.py
# Convert BlockMoving COT data from enhanced_cot_cleaned.json to MathCanvas-compatible parquet format

import os
import re
import json
import argparse
from io import BytesIO
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def parse_cot_text(cot_text: str) -> tuple:
    """
    Parse COT text into question_interleave and solution_interleave format.
    
    The cot_text contains interleaved text and <image>path</image> tags.
    First image (usually composite.png) goes to question, rest goes to solution.
    
    Returns:
        question_interleave: list of {"type": "text/image", "content": str, "index": int}
        solution_interleave: list of {"type": "text/image", "content": str, "index": int}
        question_image_paths: list of relative image paths for question
        solution_image_paths: list of relative image paths for solution
    """
    # Pattern to split by image tags
    pattern = r'<image>(.*?)</image>'
    
    # Find all image paths
    image_paths = re.findall(pattern, cot_text)
    
    # Split text by image tags
    text_parts = re.split(pattern, cot_text)
    
    question_interleave = []
    solution_interleave = []
    question_image_paths = []
    solution_image_paths = []
    
    question_image_idx = 0
    solution_image_idx = 0
    
    # Track whether we've moved past the question section
    in_solution = False
    
    part_idx = 0
    image_idx = 0
    
    for i, part in enumerate(text_parts):
        if i % 2 == 0:
            # Text part
            text = part.strip()
            if text:
                if not in_solution:
                    # First text part before any image, or text after first image but still in question
                    # Usually this is empty or minimal for BlockMoving data
                    if question_interleave or question_image_paths:
                        # We've already started, this is now solution
                        in_solution = True
                        solution_interleave.append({
                            "type": "text",
                            "content": text
                        })
                    else:
                        # Pre-question text (usually empty)
                        question_interleave.append({
                            "type": "text",
                            "content": text
                        })
                else:
                    solution_interleave.append({
                        "type": "text",
                        "content": text
                    })
        else:
            # Image path part
            img_path = part.strip()
            if img_path:
                if not in_solution:
                    # First image (composite) goes to question
                    question_interleave.append({
                        "type": "image",
                        "content": img_path,
                        "index": question_image_idx
                    })
                    question_image_paths.append(img_path)
                    question_image_idx += 1
                    in_solution = True  # After first image, we're in solution mode
                else:
                    solution_interleave.append({
                        "type": "image",
                        "content": img_path,
                        "index": solution_image_idx
                    })
                    solution_image_paths.append(img_path)
                    solution_image_idx += 1
    
    return question_interleave, solution_interleave, question_image_paths, solution_image_paths


def load_image_as_bytes(image_path: str, base_dir: str) -> bytes:
    """Load an image file and return as bytes."""
    full_path = os.path.join(base_dir, image_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    
    with open(full_path, 'rb') as f:
        return f.read()


def convert_json_to_records(json_path: str, image_base_dir: str) -> list:
    """
    Convert enhanced_cot_cleaned.json to list of records for parquet.
    
    Args:
        json_path: Path to enhanced_cot_cleaned.json
        image_base_dir: Base directory containing task_X folders with images
    
    Returns:
        List of record dicts ready for parquet conversion
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    skipped_pred_mismatch = 0
    skipped_render_result = 0
    skipped_other = 0
    
    for item in tqdm(data, desc="Converting records"):
        task_id = item.get('task_id')
        gt = item.get('gt', '')
        pred = item.get('pred', '')
        cot_text = item.get('cot_text', '')
        
        # Only use correctly predicted samples (pred == gt)
        if pred != gt:
            skipped_pred_mismatch += 1
            continue
        
        # Skip records containing [render_result] placeholder
        if '[render_result]' in cot_text:
            skipped_render_result += 1
            continue
        
        if not cot_text:
            skipped_other += 1
            continue
        
        try:
            # Parse COT text
            question_interleave, solution_interleave, q_img_paths, s_img_paths = parse_cot_text(cot_text)
            
            # Load question images
            question_images = []
            for img_path in q_img_paths:
                try:
                    img_bytes = load_image_as_bytes(img_path, image_base_dir)
                    question_images.append(img_bytes)
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
                    continue
            
            # Load solution images
            solution_images = []
            for img_path in s_img_paths:
                try:
                    img_bytes = load_image_as_bytes(img_path, image_base_dir)
                    solution_images.append(img_bytes)
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
                    continue
            
            # Skip if no images loaded
            if not question_images and not solution_images:
                skipped_other += 1
                continue
            
            record = {
                'id': f"task_{task_id}",
                'answer': gt,
                'question_interleave': question_interleave,
                'solution_interleave': solution_interleave,
                'question_images': question_images,
                'solution_images': solution_images,
            }
            records.append(record)
            
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            skipped_other += 1
            continue
    
    print(f"Converted {len(records)} records")
    print(f"  Skipped (pred!=gt): {skipped_pred_mismatch}")
    print(f"  Skipped ([render_result]): {skipped_render_result}")
    print(f"  Skipped (other): {skipped_other}")
    return records


def save_to_parquet(records: list, output_dir: str, samples_per_file: int = 100):
    """
    Save records to parquet files.
    
    Args:
        records: List of record dicts
        output_dir: Output directory for parquet files
        samples_per_file: Number of samples per parquet file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define schema
    schema = pa.schema([
        ('id', pa.string()),
        ('answer', pa.string()),
        ('question_interleave', pa.string()),  # JSON string
        ('solution_interleave', pa.string()),  # JSON string
        ('question_images', pa.list_(pa.binary())),
        ('solution_images', pa.list_(pa.binary())),
    ])
    
    file_idx = 0
    parquet_info = []
    
    for i in range(0, len(records), samples_per_file):
        batch = records[i:i + samples_per_file]
        
        # Convert to arrow format
        ids = [r['id'] for r in batch]
        answers = [r['answer'] for r in batch]
        q_interleaves = [json.dumps(r['question_interleave'], ensure_ascii=False) for r in batch]
        s_interleaves = [json.dumps(r['solution_interleave'], ensure_ascii=False) for r in batch]
        q_images = [r['question_images'] for r in batch]
        s_images = [r['solution_images'] for r in batch]
        
        table = pa.table({
            'id': ids,
            'answer': answers,
            'question_interleave': q_interleaves,
            'solution_interleave': s_interleaves,
            'question_images': q_images,
            'solution_images': s_images,
        }, schema=schema)
        
        # Write parquet file
        filename = f"blockmoving_{file_idx:03d}.parquet"
        filepath = os.path.join(output_dir, filename)
        pq.write_table(table, filepath)
        
        parquet_info.append({
            'path': filename,
            'num_samples': len(batch),
            'start_idx': i,
            'end_idx': i + len(batch),
        })
        
        file_idx += 1
    
    # Save parquet info
    info_path = os.path.join(output_dir, 'train_parquet_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': len(records),
            'num_files': file_idx,
            'files': parquet_info,
        }, f, indent=2)
    
    print(f"Saved {file_idx} parquet files to {output_dir}")
    print(f"Parquet info saved to {info_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert BlockMoving COT data to parquet format')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to enhanced_cot_cleaned.json')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Base directory for images (default: same as input file directory)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for parquet files')
    parser.add_argument('--samples_per_file', type=int, default=100,
                        help='Number of samples per parquet file (default: 100)')
    
    args = parser.parse_args()
    
    image_dir = args.image_dir if args.image_dir else os.path.dirname(args.input)
    
    print(f"Input: {args.input}")
    print(f"Image directory: {image_dir}")
    print(f"Output: {args.output}")
    
    # Convert
    records = convert_json_to_records(args.input, image_dir)
    
    if records:
        save_to_parquet(records, args.output, args.samples_per_file)
    else:
        print("No records to convert!")


if __name__ == '__main__':
    main()
