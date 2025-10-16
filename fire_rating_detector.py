"""
Fire Rating Detection Script using OpenAI Vision API
Analyzes cropped door images to extract fire rating information.
"""

import os
import base64
from pathlib import Path
from typing import Dict, List, Optional
import json

try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai package not installed. Run: pip install openai")
    OpenAI = None


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string for API transmission.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_door_image(image_path: str, api_key: str, model: str = "gpt-5-nano") -> Dict:
    """
    Analyze a single door image to extract fire rating information using OpenAI Vision API.
    
    Args:
        image_path (str): Path to the cropped door image
        api_key (str): OpenAI API key
        model (str): OpenAI model to use (default: gpt-5-mini for vision)
    
    Returns:
        dict: Analysis result with structure:
            {
                "image_path": str,
                "image_name": str,
                "rating": str or None (e.g., "FD30", "FD60", "FD90", "FD120"),
                "needs_review": bool,
                "reason": str (explanation or reason for review),
                "confidence": str ("high", "medium", "low")
            }
    """
    if OpenAI is None:
        return {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "rating": None,
            "needs_review": True,
            "reason": "OpenAI package not installed",
            "confidence": "none"
        }
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Create prompt for fire door rating detection
        prompt = """You are analyzing a cropped image of a door from a floor plan to determine its fire rating.

Fire door ratings are typically indicated by a number in a red circle like:
- 30 (30-minute fire door)
- 60 (60-minute fire door) 
- 90 (90-minute fire door)
- 180 (120-minute fire door)

Other formats might include:
- "FD30", "FD60", "FD90", "FD120"

Please analyze this door image and respond with a JSON object containing:
1. "rating": The fire door rating if clearly visible (e.g., "30", "60", "90", "180"), or null if not found
2. "needs_review": true if you cannot confidently determine the rating or if the image is unclear, false otherwise
3. "reason": Brief explanation of what you found or why review is needed
4. "confidence": "high" if very confident, "medium" if somewhat confident, "low" if uncertain

Important: Only return ratings you can confidently identify. If the rating is not clearly visible, illegible, or ambiguous, set needs_review to true and rating to null.

Return ONLY the JSON object, no other text."""

        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.1  # Low temperature for more consistent results
        )
        
        # Extract and parse response
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            # Add image metadata
            result["image_path"] = image_path
            result["image_name"] = os.path.basename(image_path)
            
            # Validate required fields
            if "rating" not in result:
                result["rating"] = None
            if "needs_review" not in result:
                result["needs_review"] = True
            if "reason" not in result:
                result["reason"] = "Response missing required fields"
            if "confidence" not in result:
                result["confidence"] = "low"
            
            return result
            
        except json.JSONDecodeError:
            # If JSON parsing fails, flag for review
            return {
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "rating": None,
                "needs_review": True,
                "reason": f"Could not parse API response: {response_text[:100]}",
                "confidence": "none"
            }
    
    except Exception as e:
        # Handle any API or processing errors
        return {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "rating": None,
            "needs_review": True,
            "reason": f"Error analyzing image: {str(e)}",
            "confidence": "none"
        }


def process_door_folder(folder_path: str, api_key: str, model: str = "gpt-4o") -> List[Dict]:
    """
    Process all door images in a folder to extract fire ratings.
    
    Args:
        folder_path (str): Path to folder containing cropped door images
        api_key (str): OpenAI API key
        model (str): OpenAI model to use (default: gpt-4o)
    
    Returns:
        list: List of analysis results, one per image
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []
    
    # Find all image files (PNG, JPG, JPEG)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file_path in Path(folder_path).iterdir():
        if file_path.suffix.lower() in image_extensions and file_path.is_file():
            # Skip converted/enhanced images if they exist
            if not any(x in file_path.name.lower() for x in ['converted', 'enhanced']):
                image_files.append(str(file_path))
    
    if not image_files:
        print(f"No door images found in '{folder_path}'")
        return []
    
    # Sort files for consistent ordering
    image_files.sort()
    
    print(f"\n{'='*60}")
    print(f"Processing {len(image_files)} door images with OpenAI Vision API...")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Analyzing {os.path.basename(image_path)}...")
        
        result = analyze_door_image(image_path, api_key, model)
        results.append(result)
        
        # Print result
        if result["needs_review"]:
            print(f"  ⚠️  NEEDS REVIEW: {result['reason']}")
        else:
            print(f"  ✓ Rating: {result['rating']} (confidence: {result['confidence']})")
    
    print(f"\n{'='*60}")
    print(f"Completed analysis of {len(image_files)} images")
    print(f"{'='*60}\n")
    
    return results


def aggregate_ratings(results: List[Dict]) -> Dict:
    """
    Aggregate fire rating results into summary counts and items needing review.
    
    Args:
        results (list): List of analysis results from process_door_folder()
    
    Returns:
        dict: Aggregated results with structure:
            {
                "total_doors": int,
                "ratings": {"FD30": 5, "FD60": 3, ...},
                "needs_review": [
                    {
                        "image_name": str,
                        "image_path": str,
                        "reason": str
                    },
                    ...
                ],
                "review_count": int,
                "confidence_breakdown": {"high": 5, "medium": 2, "low": 1}
            }
    """
    total_doors = len(results)
    rating_counts = {}
    needs_review_list = []
    confidence_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
    
    for result in results:
        # Count ratings
        if result["rating"] and not result["needs_review"]:
            rating = result["rating"]
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        # Collect items needing review
        if result["needs_review"]:
            needs_review_list.append({
                "image_name": result["image_name"],
                "image_path": result["image_path"],
                "reason": result["reason"]
            })
        
        # Count confidence levels
        confidence = result.get("confidence", "none")
        if confidence in confidence_counts:
            confidence_counts[confidence] += 1
    
    return {
        "total_doors": total_doors,
        "ratings": rating_counts,
        "needs_review": needs_review_list,
        "review_count": len(needs_review_list),
        "confidence_breakdown": confidence_counts
    }


def print_summary(aggregated_results: Dict):
    """
    Print a formatted summary of the fire rating detection results.
    
    Args:
        aggregated_results (dict): Results from aggregate_ratings()
    """
    print("\n" + "="*60)
    print("FIRE RATING DETECTION SUMMARY")
    print("="*60)
    
    print(f"\nTotal doors analyzed: {aggregated_results['total_doors']}")
    
    if aggregated_results['ratings']:
        print("\nFire Rating Distribution:")
        for rating, count in sorted(aggregated_results['ratings'].items()):
            print(f"  {rating}: {count} door(s)")
    else:
        print("\nNo fire ratings detected with confidence.")
    
    if aggregated_results['review_count'] > 0:
        print(f"\n⚠️  Items flagged for human review: {aggregated_results['review_count']}")
        print("\nImages needing review:")
        for item in aggregated_results['needs_review']:
            print(f"  - {item['image_name']}: {item['reason']}")
    else:
        print("\n✓ All doors analyzed successfully with no review needed!")
    
    print("\nConfidence Breakdown:")
    for confidence, count in aggregated_results['confidence_breakdown'].items():
        if count > 0:
            print(f"  {confidence.capitalize()}: {count}")
    
    print("="*60 + "\n")


def main():
    """
    Example usage and testing function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze door images for fire ratings using OpenAI Vision API'
    )
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to folder containing cropped door images'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='OpenAI API key'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use (default: gpt-4o)'
    )
    
    args = parser.parse_args()
    
    # Process all door images
    results = process_door_folder(args.folder_path, args.api_key, args.model)
    
    # Aggregate results
    aggregated = aggregate_ratings(results)
    
    # Print summary
    print_summary(aggregated)
    
    # Also print raw JSON for debugging
    print("\nRaw JSON Output:")
    print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()

