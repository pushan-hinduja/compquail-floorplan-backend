# Fire Door Detection System

A comprehensive system for detecting doors in floor plans and analyzing their fire ratings using AI. The system includes both a local testing interface and a FastAPI web service for frontend integration.

## Features

- **Door Detection**: Uses YOLOv8 to detect doors in PDF floor plans and images
- **Fire Rating Analysis**: Uses OpenAI GPT-4o Vision to analyze fire ratings from door images
- **Human-in-the-Loop**: Flags uncertain detections for manual review
- **Adaptive Processing**: Handles large PDFs with adaptive grid processing
- **REST API**: FastAPI endpoints for frontend integration
- **Session Management**: Tracks processing sessions with cleanup

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd compquail-floorplan-backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Environment Setup

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 3. Local Testing

Test the system locally with a PDF or image:

```bash
python main.py path/to/your/floorplan.pdf
```

### 4. API Server

Start the FastAPI server:

```bash
python run_api.py
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/

## API Endpoints

### POST `/analyze`

Upload a floor plan for analysis.

**Parameters:**
- `file`: PDF or image file (multipart/form-data)
- `confidence`: Detection confidence threshold (default: 0.25)
- `padding`: Padding around detections (default: 20)
- `dpi`: PDF conversion DPI (default: 600)
- `enhance`: Apply image enhancement (default: true)
- `model`: OpenAI model to use (default: "gpt-4o")
- `debug_doors`: Enable door detection debugging (default: false)
- `use_adaptive_grid`: Use adaptive grid for large PDFs (default: true)

**Response:**
```json
{
  "session_id": "uuid",
  "status": "completed" | "needs_review",
  "message": "Analysis completed successfully",
  "data": {
    "door_count": 5,
    "fire_ratings": {
      "30": 2,
      "60": 1,
      "90": 1,
      "NO_RATING": 1
    },
    "review_items": [...],  // Only if needs_review
    "human_feedback": {...}
  }
}
```

### POST `/submit-review`

Submit human review results for uncertain detections.

**Request Body:**
```json
{
  "session_id": "uuid",
  "reviews": [
    {
      "image_name": "door_1_conf_0.85.png",
      "rating": "90"  // or "skip" or "NO_RATING"
    }
  ]
}
```

### GET `/session/{session_id}`

Get the status of a processing session.

### DELETE `/session/{session_id}`

Cancel and clean up a processing session.

## Workflow

### 1. Automatic Processing
```
Upload PDF → Detect Doors → Analyze Fire Ratings → Return Results
```

### 2. Human-in-the-Loop Processing
```
Upload PDF → Detect Doors → Analyze Fire Ratings → Flag Uncertain → 
Human Review → Complete Analysis → Return Final Results
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Configuration

- **Door Detection Model**: `models/door_detection_model.pt`
- **Fire Rating Model**: OpenAI GPT-4o Vision
- **Detection Confidence**: 0.25 (adjustable)
- **PDF DPI**: 600 (adjustable)

## File Structure

```
├── app.py                 # FastAPI application
├── main.py               # Local testing interface
├── door_detector.py      # Door detection logic
├── fire_rating_detector.py # Fire rating analysis
├── run_api.py           # API startup script
├── test_api.py          # API testing script
├── requirements.txt     # Python dependencies
├── env.example          # Environment template
└── README.md           # This file
```

## Testing

### Local Testing
```bash
# Test with a PDF
python main.py floorplan.pdf

# Test with debugging
python main.py floorplan.pdf --debug --debug-api
```

### API Testing
```bash
# Start the API server
python run_api.py

# In another terminal, test the API
python test_api.py
```

## Advanced Features

### Adaptive Grid Processing

For large PDFs, the system automatically:
- Splits the image into overlapping sections
- Processes each section at high resolution
- Merges duplicate detections
- Ensures no doors are missed

### Image Enhancement

Automatic enhancement for better detection:
- Contrast adjustment
- Sharpness enhancement
- Brightness optimization

### Session Management

- Automatic cleanup of temporary files
- Session-based processing state
- Timeout handling
- Memory management

## Error Handling

The system handles various error conditions:
- Invalid file formats
- Missing API keys
- Processing failures
- Session timeouts
- Network issues

## Performance

- **Small PDFs**: ~10-30 seconds
- **Large PDFs**: ~1-3 minutes
- **Memory Usage**: ~500MB-2GB depending on PDF size
- **Concurrent Sessions**: Limited by available memory

## Troubleshooting

### Common Issues

1. **"OpenAI API key not configured"**
   - Set `OPENAI_API_KEY` in your `.env` file

2. **"No doors detected"**
   - Try adjusting confidence threshold
   - Enable debugging to see detection details
   - Check if PDF quality is sufficient

3. **"Image size exceeds limit"**
   - System automatically reduces DPI for large PDFs
   - Consider splitting very large PDFs

4. **Memory issues with large PDFs**
   - Use adaptive grid processing (enabled by default)
   - Consider processing smaller sections

### Debug Mode

Enable debugging for detailed output:

```bash
# Local testing with debug
python main.py floorplan.pdf --debug --debug-api --debug-pdf

# API with debug
# Set debug_doors=true in API request
```

## Production Deployment

For production deployment:

1. **Use a proper database** instead of in-memory session storage
2. **Configure CORS** properly for your frontend domain
3. **Set up proper logging** and monitoring
4. **Use environment variables** for all configuration
5. **Implement rate limiting** and authentication
6. **Use a reverse proxy** like nginx
7. **Set up health checks** and monitoring

## License

[Add your license information here]

## Support

[Add support contact information here]
