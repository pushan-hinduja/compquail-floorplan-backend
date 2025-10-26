"""
Test script for the Fire Door Detection API
"""

import requests
import json
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print("Health Check:", response.json())
    return response.status_code == 200

def test_analyze_with_pdf(pdf_path: str):
    """Test the analyze endpoint with a PDF file"""
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found")
        return None
    
    with open(pdf_path, "rb") as f:
        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
        data = {
            "confidence": 0.25,
            "padding": 20,
            "dpi": 600,
            "enhance": True,
            "model": "gpt-4o",
            "debug_doors": False,
            "use_adaptive_grid": True
        }
        
        response = requests.post(f"{BASE_URL}/analyze", files=files, data=data)
        
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

def test_submit_review(session_id: str, reviews: list):
    """Test submitting human review"""
    data = {
        "session_id": session_id,
        "reviews": reviews
    }
    
    response = requests.post(f"{BASE_URL}/submit-review", json=data)
    
    print(f"Review Status Code: {response.status_code}")
    print(f"Review Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json() if response.status_code == 200 else None

def main():
    """Main test function"""
    print("🧪 Testing Fire Door Detection API")
    print("=" * 50)
    
    # Test health check
    print("\n1. Testing health check...")
    if not test_health():
        print("❌ Health check failed")
        return
    print("✅ Health check passed")
    
    # Test with a sample PDF (you'll need to provide a real PDF path)
    pdf_path = input("\nEnter path to a PDF file to test (or press Enter to skip): ").strip()
    
    if pdf_path and os.path.exists(pdf_path):
        print(f"\n2. Testing analysis with {pdf_path}...")
        result = test_analyze_with_pdf(pdf_path)
        
        if result:
            session_id = result.get("session_id")
            status = result.get("status")
            
            print(f"\nSession ID: {session_id}")
            print(f"Status: {status}")
            
            if status == "needs_review":
                print("\n3. Testing human review workflow...")
                # Simulate human review
                review_items = result.get("data", {}).get("review_items", [])
                print(f"Found {len(review_items)} items needing review")
                
                # Create sample reviews (you can modify these)
                reviews = []
                for item in review_items[:2]:  # Review first 2 items
                    image_name = item.get("image_name")
                    print(f"Reviewing: {image_name}")
                    rating = input(f"Enter rating for {image_name} (or 'skip'): ").strip()
                    reviews.append({
                        "image_name": image_name,
                        "rating": rating if rating else "skip"
                    })
                
                if reviews:
                    final_result = test_submit_review(session_id, reviews)
                    if final_result:
                        print("\n✅ Complete workflow test passed!")
                        print("Final results:")
                        print(json.dumps(final_result.get("data", {}), indent=2))
            elif status == "completed":
                print("\n✅ Analysis completed without human review needed!")
                print("Final results:")
                print(json.dumps(result.get("data", {}), indent=2))
    else:
        print("\n⚠️  Skipping PDF test (no file provided)")
    
    print("\n🎉 API testing completed!")

if __name__ == "__main__":
    main()
