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


def analyze_door_image(image_path: str, api_key: str, model: str = "gpt-4o", debug: bool = False) -> Dict:
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
NOTE: there can be other numbers - the above are just a few examples of some of the comon ratings. 

Other formats might include:
- "FD30", "FD60", "FD90", "FD120", etc. 

IMPORTANT: Some doors may have NO fire rating at all. This is normal for:
- Regular doors (not fire doors)
- Interior doors
- Non-fire-rated doors
These doors will look like a regular door with no red circle containing a number. 

Please analyze this door image and respond with a JSON object containing:
1. "rating": The fire door rating if clearly visible (e.g., "30", "60", "90", "180"), or "NO_RATING"" if you are confident this door has no fire rating
2. "needs_review": true if you cannot confidently determine whether there is a rating or not, false otherwise
3. "reason": Brief explanation of what you found or why review is needed
4. "confidence": "high" if very confident, "medium" if somewhat confident, "low" if uncertain

Important: Only return ratings you can confidently identify. If the rating is not clearly visible, illegible, or ambiguous, set needs_review to true and rating to null.
Guidelines:
- If you can clearly see a fire rating label, return that rating with high confidence
- If you can clearly see this is a regular door with no fire rating, return "NO_RATING" with high confidence (NOTE this will look like a regular door with no red circle containing a fire rating)
- If the image is unclear, illegible, or you cannot determine if there's a rating, set needs_review to true and rating to null
- If the door appears to be a double door with 2 ratings, return the rating for both doors separately (e.g. if there's one image of a double door with 2 ratings, you should return 2 doors with the corresponding rating for each door)

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
        try:
            if debug:
                # Debug: Print response structure
                print(f"    Debug: Response type: {type(response)}")
                print(f"    Debug: Choices type: {type(response.choices)}")
                print(f"    Debug: Choices length: {len(response.choices) if hasattr(response, 'choices') else 'No choices attr'}")
            
            if not response.choices:
                return {
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "rating": None,
                    "needs_review": True,
                    "reason": "No choices in API response",
                    "confidence": "none"
                }
            
            choice = response.choices[0]
            if debug:
                print(f"    Debug: Choice type: {type(choice)}")
                print(f"    Debug: Choice attributes: {dir(choice)}")
            
            if not hasattr(choice, 'message'):
                return {
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "rating": None,
                    "needs_review": True,
                    "reason": "No message in API response choice",
                    "confidence": "none"
                }
            
            message = choice.message
            if debug:
                print(f"    Debug: Message type: {type(message)}")
                print(f"    Debug: Message attributes: {dir(message)}")
            
            if not hasattr(message, 'content'):
                return {
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "rating": None,
                    "needs_review": True,
                    "reason": "No content in API response message",
                    "confidence": "none"
                }
            
            response_text = message.content.strip()
            if debug:
                print(f"    Debug: Response text length: {len(response_text)}")
                print(f"    Debug: Response text preview: {response_text[:100]}...")
            
        except (AttributeError, IndexError, KeyError) as e:
            return {
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "rating": None,
                "needs_review": True,
                "reason": f"Error extracting response from API: {str(e)}",
                "confidence": "none"
            }
        
        # Try to parse as JSON
        try:
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            parsed_response = json.loads(response_text)
            
            # Handle both single objects and arrays of objects
            if isinstance(parsed_response, list):
                # Multiple doors detected - return array of results
                results = []
                for i, door_result in enumerate(parsed_response):
                    # Add image metadata with door index
                    door_result["image_path"] = image_path
                    door_result["image_name"] = os.path.basename(image_path)
                    door_result["door_index"] = i + 1  # 1-based indexing
                    
                    # Validate required fields
                    if "rating" not in door_result:
                        door_result["rating"] = None
                    if "needs_review" not in door_result:
                        door_result["needs_review"] = True
                    if "reason" not in door_result:
                        door_result["reason"] = "Response missing required fields"
                    if "confidence" not in door_result:
                        door_result["confidence"] = "low"
                    
                    results.append(door_result)
                
                if debug:
                    print(f"    Debug: Found {len(results)} doors in single image")
                
                return results  # Return array of results
            else:
                # Single door - return single result
                result = parsed_response
                
                # Add image metadata
                result["image_path"] = image_path
                result["image_name"] = os.path.basename(image_path)
                result["door_index"] = 1  # Single door
                
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


def process_door_folder(folder_path: str, api_key: str, model: str = "gpt-4o", debug: bool = False) -> List[Dict]:
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
        
        result = analyze_door_image(image_path, api_key, model, debug)
        
        # Handle both single results and arrays of results
        if isinstance(result, list):
            # Multiple doors found in single image
            print(f"  Found {len(result)} doors in this image")
            for j, door_result in enumerate(result, 1):
                print(f"    Door {j}: ", end="")
                if door_result["needs_review"]:
                    print(f"⚠️  NEEDS REVIEW: {door_result['reason']}")
                else:
                    print(f"✓ Rating: {door_result['rating']} (confidence: {door_result['confidence']})")
            results.extend(result)  # Add all doors to results
        else:
            # Single door result
            if result["needs_review"]:
                print(f"  ⚠️  NEEDS REVIEW: {result['reason']}")
            else:
                print(f"  ✓ Rating: {result['rating']} (confidence: {result['confidence']})")
            results.append(result)
    
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
            # Handle "NO_RATING" as a special case
            if rating == "NO_RATING":
                rating_counts["NO_RATING"] = rating_counts.get("NO_RATING", 0) + 1
            else:
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

