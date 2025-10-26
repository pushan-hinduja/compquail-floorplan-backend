"""
Fire Door Detection System - FastAPI Application
Handles PDF/image upload, door detection, fire rating analysis, and human-in-the-loop workflow
"""

import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json

from door_detector import detect_and_save_doors
from fire_rating_detector import process_door_folder, aggregate_ratings

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Initialize FastAPI app
app = FastAPI(
    title="Fire Door Detection API",
    description="API for detecting doors and analyzing fire ratings in floor plans",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (use Redis or database in production)
processing_sessions = {}

# SSE progress tracking
progress_queues = {}  # session_id -> asyncio.Queue for progress updates

def cleanup_old_sessions():
    """Clean up any old session directories on startup"""
    sessions_dir = Path("sessions")
    if sessions_dir.exists():
        try:
            shutil.rmtree(sessions_dir)
            print(f"🧹 Cleaned up old session directories: {sessions_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up old sessions: {e}")
    
    # Create fresh sessions directory
    sessions_dir.mkdir(exist_ok=True)
    print(f"📁 Created fresh sessions directory: {sessions_dir}")

# Clean up old sessions on startup
cleanup_old_sessions()

# Progress tracking functions
async def send_progress(session_id: str, milestone: str, message: str, data: dict = None):
    """Send progress update to SSE stream"""
    if session_id in progress_queues:
        progress_data = {
            "milestone": milestone,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        await progress_queues[session_id].put(progress_data)

async def create_progress_queue(session_id: str):
    """Create a progress queue for a session"""
    progress_queues[session_id] = asyncio.Queue()
    return progress_queues[session_id]

async def cleanup_progress_queue(session_id: str):
    """Clean up progress queue for a session"""
    if session_id in progress_queues:
        del progress_queues[session_id]

# Pydantic models for request/response
class HumanReviewItem(BaseModel):
    image_name: str
    image_data: str  # Base64 encoded image
    current_rating: Optional[str] = None
    needs_review: bool = True

class HumanReviewRequest(BaseModel):
    session_id: str
    reviews: List[Dict[str, str]]  # List of {"image_name": "rating"} or {"image_name": "skip"}

class ProcessingResponse(BaseModel):
    session_id: str
    status: str  # "processing", "needs_review", "completed"
    message: str
    data: Optional[Dict] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Fire Door Detection API is running", "status": "healthy"}

@app.get("/test-sse/{session_id}")
async def test_sse(session_id: str):
    """Test SSE endpoint for debugging"""
    # Create progress queue if it doesn't exist
    if session_id not in progress_queues:
        await create_progress_queue(session_id)
        print(f"📡 Created test progress queue for session: {session_id}")
    
    # Send a test message
    await send_progress(session_id, "test_message", "SSE test successful", {"test": True})
    
    return {"message": f"Test message sent to session {session_id}"}

@app.get("/stream/{session_id}")
async def stream_progress(session_id: str):
    """Server-Sent Events stream for real-time progress updates"""
    # Create progress queue if it doesn't exist
    if session_id not in progress_queues:
        await create_progress_queue(session_id)
        print(f"📡 Created progress queue for session: {session_id}")
    
    async def event_generator():
        queue = progress_queues[session_id]
        try:
            # Send initial connection message
            initial_message = {
                "milestone": "connection_established",
                "message": "Progress stream connected",
                "timestamp": datetime.now().isoformat(),
                "data": {}
            }
            yield f"data: {json.dumps(initial_message)}\n\n"
            
            while True:
                # Wait for progress update with timeout
                try:
                    progress_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send keepalive message
                    keepalive = {
                        "milestone": "keepalive",
                        "message": "Connection alive",
                        "timestamp": datetime.now().isoformat(),
                        "data": {}
                    }
                    yield f"data: {json.dumps(keepalive)}\n\n"
                    continue
                
                # Format as SSE
                event_data = json.dumps(progress_data)
                yield f"data: {event_data}\n\n"
                
                # Check if analysis is complete
                if progress_data["milestone"] in ["analysis_complete", "analysis_failed"]:
                    break
                    
        except asyncio.CancelledError:
            # Client disconnected
            print(f"📡 SSE connection cancelled for session: {session_id}")
            pass
        except Exception as e:
            print(f"📡 SSE error for session {session_id}: {e}")
        finally:
            # Clean up queue
            await cleanup_progress_queue(session_id)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.post("/analyze", response_model=ProcessingResponse)
async def analyze_floorplan(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    padding: int = 20,
    dpi: int = 600,
    enhance: bool = True,
    model: str = "gpt-4o",
    debug_doors: bool = False,
    use_adaptive_grid: bool = True
):
    """
    Analyze a floor plan for door detection and fire rating analysis.
    
    Returns either:
    - Complete results if no human review needed
    - Human review items if manual review required
    """
    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF, PNG, or JPEG"
        )
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Create progress queue for SSE updates
    await create_progress_queue(session_id)
    
    # Create session directory
    session_dir = Path(f"sessions/{session_id}")
    
    # Clean up any existing session directory (safety measure)
    if session_dir.exists():
        shutil.rmtree(session_dir)
        print(f"🧹 Cleaned up existing session directory: {session_dir}")
    
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created fresh session directory: {session_dir}")
    
    # Send initial progress update
    await send_progress(session_id, "analysis_started", "Starting floor plan analysis...")
    
    # Save uploaded file
    file_path = session_dir / file.filename
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured"
        )
    
    try:
        # Step 1: Detect doors
        await send_progress(session_id, "detecting_doors", "Detecting doors in floor plan...")
        
        output_folder = session_dir / "doors"
        door_results = detect_and_save_doors(
            image_path=str(file_path),
            output_folder=str(output_folder),
            confidence=confidence,
            padding=padding,
            dpi=dpi,
            enhance=enhance,
            debug=debug_doors,
            use_adaptive_grid=use_adaptive_grid
        )
        
        # Send doors detected progress
        await send_progress(session_id, "doors_detected", 
                           f"Found {door_results['door_count']} doors", 
                           {"door_count": door_results["door_count"]})
        
        if door_results["door_count"] == 0:
            return ProcessingResponse(
                session_id=session_id,
                status="completed",
                message="No doors detected in the floor plan",
                data={
                    "door_count": 0,
                    "fire_ratings": {},
                    "review_items": [],
                    "human_feedback": {}
                }
            )
        
        # Step 2: Analyze fire ratings
        await send_progress(session_id, "analyzing_fire_ratings", "Analyzing fire ratings for detected doors...")
        
        fire_rating_results = process_door_folder(
            folder_path=str(output_folder),
            api_key=api_key,
            model=model,
            debug=False
        )
        
        # Send fire ratings analyzed progress
        await send_progress(session_id, "fire_ratings_analyzed", 
                           "Fire rating analysis complete", 
                           {"doors_analyzed": len(fire_rating_results)})
        
        # Step 3: Aggregate results
        aggregated_results = aggregate_ratings(fire_rating_results)
        
        # Check if human review is needed
        review_items = []
        for result in fire_rating_results:
            if result.get("needs_review", False):
                # Convert image to base64 for frontend
                image_path = result.get("image_path", "")
                if os.path.exists(image_path):
                    import base64
                    with open(image_path, "rb") as img_file:
                        image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    review_items.append(HumanReviewItem(
                        image_name=os.path.basename(image_path),
                        image_data=image_data,
                        current_rating=result.get("rating"),
                        needs_review=True
                    ))
        
        # Store session data
        processing_sessions[session_id] = {
            "status": "needs_review" if review_items else "completed",
            "door_results": door_results,
            "fire_rating_results": fire_rating_results,
            "aggregated_results": aggregated_results,
            "review_items": review_items,
            "session_dir": str(session_dir),
            "created_at": datetime.now().isoformat()
        }
        
        if review_items:
            # Send human review needed progress
            await send_progress(session_id, "human_review_needed", 
                               f"Human review required for {len(review_items)} doors", 
                               {"review_items_count": len(review_items)})
            
            # Return human review items
            return ProcessingResponse(
                session_id=session_id,
                status="needs_review",
                message=f"Human review required for {len(review_items)} doors",
                data={
                    "review_items": [item.dict() for item in review_items],
                    "total_doors": door_results["door_count"],
                    "automatic_ratings": {k: v for k, v in aggregated_results["ratings"].items() if k != "UNKNOWN"}
                }
            )
        else:
            # Send analysis complete progress
            await send_progress(session_id, "analysis_complete", 
                               "Analysis completed successfully", 
                               {"door_count": door_results["door_count"]})
            
            # Return complete results
            final_results = {
                "door_count": door_results["door_count"],
                "fire_ratings": aggregated_results["ratings"],
                "review_items": [],
                "human_feedback": {
                    "total_reviewed": 0,
                    "skipped": 0,
                    "rated": 0,
                    "no_rating_count": 0
                }
            }
            
            # Clean up session
            shutil.rmtree(session_dir)
            del processing_sessions[session_id]
            await cleanup_progress_queue(session_id)
            
            return ProcessingResponse(
                session_id=session_id,
                status="completed",
                message="Analysis completed successfully",
                data=final_results
            )
            
    except Exception as e:
        # Send analysis failed progress
        await send_progress(session_id, "analysis_failed", 
                           f"Analysis failed: {str(e)}", 
                           {"error": str(e)})
        
        # Clean up on error
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        if session_id in processing_sessions:
            del processing_sessions[session_id]
        await cleanup_progress_queue(session_id)
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/submit-review", response_model=ProcessingResponse)
async def submit_human_review(review_request: HumanReviewRequest):
    """
    Submit human review results and complete the analysis.
    """
    session_id = review_request.session_id
    
    if session_id not in processing_sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired"
        )
    
    session_data = processing_sessions[session_id]
    
    if session_data["status"] != "needs_review":
        raise HTTPException(
            status_code=400,
            detail="Session is not in review state"
        )
    
    try:
        # Process human review inputs
        manual_ratings = {}
        for review in review_request.reviews:
            image_name = review.get("image_name")
            rating = review.get("rating")
            
            if rating and rating.lower() != "skip":
                manual_ratings[image_name] = rating
            else:
                manual_ratings[image_name] = None  # Skip
        
        # Update aggregated results with human feedback
        updated_ratings = session_data["aggregated_results"]["ratings"].copy()
        
        # Remove UNKNOWN entries that were reviewed
        unknown_count = updated_ratings.get("UNKNOWN", 0)
        updated_ratings["UNKNOWN"] = max(0, unknown_count - len(manual_ratings))
        
        # Add human-reviewed ratings
        for image_name, rating in manual_ratings.items():
            if rating:
                if rating == "NO_RATING":
                    updated_ratings["NO_RATING"] = updated_ratings.get("NO_RATING", 0) + 1
                else:
                    updated_ratings[rating] = updated_ratings.get(rating, 0) + 1
        
        # Create final results
        final_results = {
            "door_count": session_data["door_results"]["door_count"],
            "fire_ratings": updated_ratings,
            "review_items": [],
            "human_feedback": {
                "total_reviewed": len(manual_ratings),
                "skipped": len([r for r in manual_ratings.values() if r is None]),
                "rated": len([r for r in manual_ratings.values() if r is not None]),
                "no_rating_count": len([r for r in manual_ratings.values() if r == "NO_RATING"])
            },
            "manual_reviews": manual_ratings
        }
        
        # Send final completion progress
        await send_progress(session_id, "analysis_complete", 
                           "Analysis completed with human review", 
                           {"door_count": final_results["door_count"]})
        
        # Clean up session
        session_dir = session_data["session_dir"]
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
        del processing_sessions[session_id]
        await cleanup_progress_queue(session_id)
        
        return ProcessingResponse(
            session_id=session_id,
            status="completed",
            message="Analysis completed with human review",
            data=final_results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process human review: {str(e)}"
        )

@app.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """Get the status of a processing session"""
    if session_id not in processing_sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    session_data = processing_sessions[session_id]
    return {
        "session_id": session_id,
        "status": session_data["status"],
        "created_at": session_data["created_at"],
        "door_count": session_data["door_results"]["door_count"]
    }

@app.delete("/session/{session_id}")
async def cancel_session(session_id: str):
    """Cancel and clean up a processing session"""
    if session_id not in processing_sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    session_data = processing_sessions[session_id]
    session_dir = session_data["session_dir"]
    
    # Clean up files
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    
    # Remove from memory
    del processing_sessions[session_id]
    
    return {"message": "Session cancelled and cleaned up"}

@app.post("/cleanup")
async def manual_cleanup():
    """Manually clean up all session data"""
    try:
        # Clean up all session directories
        sessions_dir = Path("sessions")
        if sessions_dir.exists():
            shutil.rmtree(sessions_dir)
            sessions_dir.mkdir(exist_ok=True)
        
        # Clear in-memory sessions
        processing_sessions.clear()
        
        return {
            "message": "All session data cleaned up successfully",
            "cleaned_sessions": len(processing_sessions)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
