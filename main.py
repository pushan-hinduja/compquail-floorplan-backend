"""
Fire Door Detection System - Testing Interface
Orchestrates the complete workflow: door detection + fire rating analysis
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from door_detector import detect_and_save_doors
from fire_rating_detector import process_door_folder, aggregate_ratings, print_summary

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


def process_floorplan_complete(pdf_path: str, output_folder: str, api_key: str, 
                              confidence: float = 0.25, padding: int = 20, 
                              min_size: Optional[int] = None, dpi: int = 600, 
                              enhance: bool = True, model: str = "gpt-4o", 
                              save_debug_images: bool = False, debug_api: bool = False, 
                              debug_doors: bool = False, use_adaptive_grid: bool = True) -> Dict:
    """
    Complete workflow: detect doors in PDF, analyze fire ratings, handle human review.
    
    Args:
        pdf_path (str): Path to the floor plan PDF
        output_folder (str): Folder to save door images and results
        api_key (str): OpenAI API key for fire rating analysis
        confidence (float): Door detection confidence threshold (default: 0.25)
        padding (int): Pixels to add around door detections (default: 20)
        min_size (int): Minimum size to upscale door images (default: None)
        dpi (int): DPI for PDF conversion (default: 600)
        enhance (bool): Apply image enhancement for better detection (default: True)
        model (str): OpenAI model to use (default: gpt-4o)
    
    Returns:
        dict: Complete results including door detection and fire rating analysis
    """
    print("🔥 FIRE DOOR DETECTION SYSTEM")
    print("=" * 50)
    print(f"Input PDF: {pdf_path}")
    print(f"Output folder: {output_folder}")
    print(f"OpenAI Model: {model}")
    print("=" * 50)
    
    # Validate input file
    if not os.path.exists(pdf_path):
        return {
            "error": f"Input file '{pdf_path}' does not exist",
            "success": False
        }
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Detect doors using YOLO model
    print("\n📋 STEP 1: Detecting doors in floor plan...")
    print("-" * 40)
    
    door_result = detect_and_save_doors(
        image_path=pdf_path,
        output_folder=output_folder,
        confidence=confidence,
        padding=padding,
        min_size=min_size,
        dpi=dpi,
        enhance=enhance,
        save_debug_images=save_debug_images,
        debug=debug_doors,
        use_adaptive_grid=use_adaptive_grid
    )
    
    if door_result["door_count"] == 0:
        return {
            "success": False,
            "error": "No doors detected in the floor plan",
            "door_detection": door_result
        }
    
    print(f"\n✅ Door detection completed!")
    print(f"   Doors found: {door_result['door_count']}")
    print(f"   Images saved: {len(door_result['door_images'])}")
    
    # Step 2: Analyze fire ratings using OpenAI
    print(f"\n🔥 STEP 2: Analyzing fire ratings...")
    print("-" * 40)
    
    fire_rating_results = process_door_folder(
        folder_path=door_result["output_folder"],
        api_key=api_key,
        model=model,
        debug=debug_api
    )
    
    if not fire_rating_results:
        return {
            "success": False,
            "error": "Failed to analyze fire ratings",
            "door_detection": door_result,
            "fire_rating_analysis": []
        }
    
    # Step 3: Aggregate results
    print(f"\n📊 STEP 3: Aggregating results...")
    print("-" * 40)
    
    aggregated_results = aggregate_ratings(fire_rating_results)
    
    # Print summary
    print_summary(aggregated_results)
    
    # Step 4: Handle human review if needed
    final_results = None
    if aggregated_results["review_count"] > 0:
        print(f"\n👤 STEP 4: Human review required for {aggregated_results['review_count']} images")
        print("-" * 50)
        
        final_results = handle_human_review(
            aggregated_results, 
            fire_rating_results, 
            door_result["output_folder"]
        )
    else:
        print(f"\n✅ STEP 4: No human review needed - all doors analyzed successfully!")
        final_results = aggregated_results
    
    # Prepare final response
    complete_result = {
        "success": True,
        "input_file": pdf_path,
        "output_folder": output_folder,
        "door_detection": door_result,
        "fire_rating_analysis": fire_rating_results,
        "aggregated_results": final_results,
        "needs_human_review": aggregated_results["review_count"] > 0
    }
    
    # Save results to JSON file
    results_file = os.path.join(output_folder, "complete_results.json")
    with open(results_file, 'w') as f:
        json.dump(complete_result, f, indent=2)
    
    print(f"\n💾 Complete results saved to: {results_file}")
    print("=" * 50)
    
    return complete_result


def handle_human_review(aggregated_results: Dict, fire_rating_results: List[Dict], 
                       output_folder: str) -> Dict:
    """
    Handle human review for images that need manual fire rating assessment.
    
    Args:
        aggregated_results (dict): Results from aggregate_ratings()
        fire_rating_results (list): Detailed results from process_door_folder()
        output_folder (str): Path to folder containing door images
    
    Returns:
        dict: Updated aggregated results with human-provided ratings
    """
    print("\nImages flagged for human review:")
    print("=" * 40)
    
    review_items = aggregated_results["needs_review"]
    manual_ratings = {}
    
    for i, item in enumerate(review_items, 1):
        print(f"\n[{i}/{len(review_items)}] {item['image_name']}")
        print(f"   Reason: {item['reason']}")
        print(f"   Full path: {item['image_path']}")
        
        # In a real implementation, you might display the image here
        # For CLI testing, we'll prompt for manual input
        while True:
            rating = input(f"   Enter fire rating (FD30/FD60/FD90/FD120/NO_RATING/UNKNOWN) or 'skip': ").strip().upper()
            
            if rating in ['FD30', 'FD60', 'FD90', 'FD120', 'NO_RATING', 'UNKNOWN', 'SKIP']:
                if rating != 'SKIP':
                    manual_ratings[item['image_name']] = rating
                else:
                    manual_ratings[item['image_name']] = None  # Mark as skipped
                break
            else:
                print("   Invalid input. Please enter FD30, FD60, FD90, FD120, NO_RATING, UNKNOWN, or skip.")
    
    # Update aggregated results with manual ratings
    updated_ratings = aggregated_results["ratings"].copy()
    
    for image_name, rating in manual_ratings.items():
        if rating is not None and rating != 'UNKNOWN':
            # Handle "NO_RATING" as a special case
            if rating == 'NO_RATING':
                updated_ratings["NO_RATING"] = updated_ratings.get("NO_RATING", 0) + 1
            else:
                updated_ratings[rating] = updated_ratings.get(rating, 0) + 1
    
    # Remove items that were manually reviewed
    remaining_review = [
        item for item in review_items 
        if item['image_name'] not in manual_ratings
    ]
    
    return {
        "total_doors": aggregated_results["total_doors"],
        "ratings": updated_ratings,
        "needs_review": remaining_review,
        "review_count": len(remaining_review),
        "confidence_breakdown": aggregated_results["confidence_breakdown"],
        "manual_reviews": manual_ratings,
        "human_feedback": {
            "total_reviewed": len(manual_ratings),
            "skipped": len([r for r in manual_ratings.values() if r is None]),
            "rated": len([r for r in manual_ratings.values() if r is not None]),
            "no_rating_count": len([r for r in manual_ratings.values() if r == "NO_RATING"])
        }
    }


def test_individual_components(pdf_path: str, output_folder: str, api_key: str):
    """
    Test individual components separately for debugging.
    
    Args:
        pdf_path (str): Path to test PDF
        output_folder (str): Output folder for results
        api_key (str): OpenAI API key
    """
    print("🧪 TESTING INDIVIDUAL COMPONENTS")
    print("=" * 50)
    
    # Test door detection only
    print("\n1. Testing door detection...")
    door_result = detect_and_save_doors(pdf_path, output_folder)
    print(f"   Result: {door_result}")
    
    if door_result["door_count"] > 0:
        # Test fire rating analysis only
        print("\n2. Testing fire rating analysis...")
        fire_results = process_door_folder(output_folder, api_key)
        print(f"   Results: {len(fire_results)} images analyzed")
        
        # Test aggregation
        print("\n3. Testing aggregation...")
        aggregated = aggregate_ratings(fire_results)
        print(f"   Aggregated: {aggregated}")
    else:
        print("   Skipping fire rating test - no doors detected")


def main():
    """
    Command-line interface for testing the fire door detection system.
    """
    parser = argparse.ArgumentParser(
        description='Fire Door Detection System - Complete Workflow Testing'
    )
    parser.add_argument(
        'pdf_path',
        type=str,
        help='Path to the floor plan PDF file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_output',
        help='Output folder for results (default: test_output)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('OPENAI_API_KEY'),
        help='OpenAI API key for fire rating analysis (can also be set via OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Door detection confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--padding',
        type=int,
        default=20,
        help='Pixels to add around door detections (default: 20)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=600,
        help='DPI for PDF conversion (default: 600)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use (default: gpt-4o)'
    )
    parser.add_argument(
        '--test-components',
        action='store_true',
        help='Test individual components separately'
    )
    parser.add_argument(
        '--no-enhance',
        action='store_true',
        help='Disable image enhancement'
    )
    parser.add_argument(
        '--debug-pdf',
        action='store_true',
        help='Save debug images from PDF conversion for troubleshooting'
    )
    parser.add_argument(
        '--pdf-dpi',
        type=int,
        default=600,
        help='DPI for PDF conversion (default: 600, higher = better quality but slower)'
    )
    parser.add_argument(
        '--debug-api',
        action='store_true',
        help='Enable debug output for OpenAI API responses'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output for YOLO door detection (shows all detections with coordinates)'
    )
    parser.add_argument(
        '--no-adaptive-grid',
        action='store_true',
        help='Disable adaptive grid processing for PDFs (use standard processing)'
    )
    parser.add_argument(
        '--grid-size',
        type=str,
        default='auto',
        help='Grid size for adaptive processing (e.g., 4x4, 6x6, auto)'
    )
    
    args = parser.parse_args()
    
    # Check if API key is provided
    if not args.api_key:
        print("❌ ERROR: OpenAI API key is required!")
        print("Set it via:")
        print("  1. Environment variable: export OPENAI_API_KEY='sk-your-key'")
        print("  2. Command line: --api-key sk-your-key")
        print("  3. Create a .env file with OPENAI_API_KEY=sk-your-key")
        return
    
    if args.test_components:
        test_individual_components(args.pdf_path, args.output, args.api_key)
    else:
        # Run complete workflow
        result = process_floorplan_complete(
            pdf_path=args.pdf_path,
            output_folder=args.output,
            api_key=args.api_key,
            confidence=args.confidence,
            padding=args.padding,
            dpi=args.pdf_dpi,  # Use PDF-specific DPI
            enhance=not args.no_enhance,
            model=args.model,
            save_debug_images=args.debug_pdf,
            debug_api=args.debug_api,
            debug_doors=args.debug,
            use_adaptive_grid=not args.no_adaptive_grid
        )
        
        if result["success"]:
            print("\n🎉 SUCCESS! Fire door detection completed.")
            print(f"📁 Results saved in: {args.output}")
            print(f"📄 JSON report: {os.path.join(args.output, 'complete_results.json')}")
        else:
            print(f"\n❌ ERROR: {result.get('error', 'Unknown error occurred')}")


if __name__ == "__main__":
    main()

