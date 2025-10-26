# Frontend API Integration Specifications

## Overview
This document provides complete specifications for integrating a frontend application with the Fire Door Detection API backend.

## Base Configuration
- **API Base URL**: `http://localhost:8000` (development) / `https://your-domain.com` (production)
- **Content-Type**: `multipart/form-data` for file uploads, `application/json` for other requests
- **CORS**: Enabled for all origins (configure properly for production)

## API Endpoints

### 1. Health Check
**GET** `/`

**Purpose**: Verify API is running

**Response**:
```json
{
  "message": "Fire Door Detection API is running",
  "status": "healthy"
}
```

### 2. Analyze Floor Plan
**POST** `/analyze`

**Purpose**: Upload and analyze a floor plan for door detection and fire rating analysis

**Request Format**: `multipart/form-data`

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | PDF or image file (PNG, JPG, JPEG) |
| `confidence` | Float | No | 0.25 | Detection confidence threshold (0.0-1.0) |
| `padding` | Integer | No | 20 | Pixels to add around each detection |
| `dpi` | Integer | No | 600 | DPI for PDF conversion |
| `enhance` | Boolean | No | true | Apply image enhancement |
| `model` | String | No | "gpt-4o" | OpenAI model for fire rating analysis |
| `debug_doors` | Boolean | No | false | Enable door detection debugging |
| `use_adaptive_grid` | Boolean | No | true | Use adaptive grid for large PDFs |

**Response Format**:
```json
{
  "session_id": "uuid-string",
  "status": "completed" | "needs_review",
  "message": "Analysis completed successfully" | "Human review required for X doors",
  "data": {
    "door_count": 5,
    "fire_ratings": {
      "30": 2,
      "60": 1,
      "90": 1,
      "NO_RATING": 1
    },
    "review_items": [
      {
        "image_name": "door_1_conf_0.85.png",
        "image_data": "base64-encoded-image-data",
        "current_rating": null,
        "needs_review": true
      }
    ],
    "human_feedback": {
      "total_reviewed": 0,
      "skipped": 0,
      "rated": 0,
      "no_rating_count": 0
    }
  }
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid file type or parameters
- `500`: Server error

### 3. Submit Human Review
**POST** `/submit-review`

**Purpose**: Submit human review results for uncertain detections

**Request Format**: `application/json`

**Request Body**:
```json
{
  "session_id": "uuid-string",
  "reviews": [
    {
      "image_name": "door_1_conf_0.85.png",
      "rating": "90"  // or "skip" or "NO_RATING" or null
    },
    {
      "image_name": "door_2_conf_0.72.png",
      "rating": "skip"
    }
  ]
}
```

**Response Format**:
```json
{
  "session_id": "uuid-string",
  "status": "completed",
  "message": "Analysis completed with human review",
  "data": {
    "door_count": 5,
    "fire_ratings": {
      "30": 2,
      "60": 1,
      "90": 2,
      "NO_RATING": 0
    },
    "review_items": [],
    "human_feedback": {
      "total_reviewed": 2,
      "skipped": 1,
      "rated": 1,
      "no_rating_count": 0
    },
    "manual_reviews": {
      "door_1_conf_0.85.png": "90",
      "door_2_conf_0.72.png": null
    }
  }
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid session or review data
- `404`: Session not found
- `500`: Server error

### 4. Get Session Status
**GET** `/session/{session_id}`

**Purpose**: Check the status of a processing session

**Response Format**:
```json
{
  "session_id": "uuid-string",
  "status": "processing" | "needs_review" | "completed",
  "created_at": "2024-01-01T12:00:00Z",
  "door_count": 5
}
```

**Status Codes**:
- `200`: Success
- `404`: Session not found

### 5. Cancel Session
**DELETE** `/session/{session_id}`

**Purpose**: Cancel and clean up a processing session

**Response Format**:
```json
{
  "message": "Session cancelled and cleaned up"
}
```

**Status Codes**:
- `200`: Success
- `404`: Session not found

## Frontend Implementation Guide

### 1. File Upload Component

```javascript
// Example file upload implementation
const uploadFile = async (file, options = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  
  // Add optional parameters
  if (options.confidence) formData.append('confidence', options.confidence);
  if (options.padding) formData.append('padding', options.padding);
  if (options.dpi) formData.append('dpi', options.dpi);
  if (options.enhance !== undefined) formData.append('enhance', options.enhance);
  if (options.model) formData.append('model', options.model);
  if (options.debug_doors !== undefined) formData.append('debug_doors', options.debug_doors);
  if (options.use_adaptive_grid !== undefined) formData.append('use_adaptive_grid', options.use_adaptive_grid);
  
  const response = await fetch('/analyze', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }
  
  return await response.json();
};
```

### 2. Human Review Interface

```javascript
// Example human review submission
const submitHumanReview = async (sessionId, reviews) => {
  const response = await fetch('/submit-review', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      session_id: sessionId,
      reviews: reviews
    })
  });
  
  if (!response.ok) {
    throw new Error(`Review submission failed: ${response.statusText}`);
  }
  
  return await response.json();
};
```

### 3. Complete Workflow Example

```javascript
class FireDoorAnalyzer {
  constructor(apiBaseUrl = 'http://localhost:8000') {
    this.apiBaseUrl = apiBaseUrl;
  }
  
  async analyzeFloorPlan(file, options = {}) {
    try {
      // Step 1: Upload and analyze
      const formData = new FormData();
      formData.append('file', file);
      
      // Add options
      Object.entries(options).forEach(([key, value]) => {
        formData.append(key, value);
      });
      
      const response = await fetch(`${this.apiBaseUrl}/analyze`, {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (result.status === 'completed') {
        // No human review needed
        return {
          success: true,
          data: result.data,
          needsReview: false
        };
      } else if (result.status === 'needs_review') {
        // Human review needed
        return {
          success: true,
          data: result.data,
          needsReview: true,
          sessionId: result.session_id,
          reviewItems: result.data.review_items
        };
      }
      
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }
  
  async submitReview(sessionId, reviews) {
    try {
      const response = await fetch(`${this.apiBaseUrl}/submit-review`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          session_id: sessionId,
          reviews: reviews
        })
      });
      
      const result = await response.json();
      
      return {
        success: true,
        data: result.data
      };
      
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }
  
  async getSessionStatus(sessionId) {
    try {
      const response = await fetch(`${this.apiBaseUrl}/session/${sessionId}`);
      return await response.json();
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }
  
  async cancelSession(sessionId) {
    try {
      const response = await fetch(`${this.apiBaseUrl}/session/${sessionId}`, {
        method: 'DELETE'
      });
      return await response.json();
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }
}

// Usage example
const analyzer = new FireDoorAnalyzer();

// Analyze a floor plan
const result = await analyzer.analyzeFloorPlan(file, {
  confidence: 0.25,
  padding: 20,
  dpi: 600
});

if (result.success && result.needsReview) {
  // Show human review interface
  const reviewItems = result.reviewItems;
  
  // After user provides reviews
  const reviews = [
    { image_name: "door_1.png", rating: "90" },
    { image_name: "door_2.png", rating: "skip" }
  ];
  
  const finalResult = await analyzer.submitReview(result.sessionId, reviews);
  console.log('Final results:', finalResult.data);
}
```

## Error Handling

### Common Error Scenarios

1. **File Upload Errors**:
   - Invalid file type (not PDF/PNG/JPG)
   - File too large
   - Network timeout

2. **Processing Errors**:
   - No doors detected
   - API key not configured
   - Memory issues with large files

3. **Session Errors**:
   - Session expired
   - Invalid session ID
   - Session already completed

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

## UI/UX Recommendations

### 1. File Upload
- Support drag-and-drop
- Show file validation (type, size)
- Display upload progress
- Show file preview if possible

### 2. Processing State
- Show loading spinner during analysis
- Display progress messages
- Handle long processing times (large PDFs can take 1-3 minutes)

### 3. Human Review Interface
- Display door images clearly
- Provide rating options: "30", "60", "90", "180", "NO_RATING", "Skip"
- Show current AI-suggested rating
- Allow batch review of multiple doors
- Preview final results before submission

### 4. Results Display
- Show door count summary
- Display fire rating breakdown
- Highlight any manual reviews
- Export results as JSON/CSV

## Testing

### Test Scenarios

1. **Happy Path**: Upload PDF → No review needed → Get results
2. **Human Review**: Upload PDF → Review needed → Submit reviews → Get results
3. **Error Cases**: Invalid file, network error, session timeout
4. **Edge Cases**: Very large PDF, no doors detected, all doors need review

### Test Files

Create test files for different scenarios:
- Small PDF with clear doors
- Large PDF requiring adaptive grid
- PDF with no doors
- PDF with unclear door ratings
- Invalid file types

## Security Considerations

1. **File Validation**: Validate file types and sizes on frontend
2. **Session Management**: Implement session timeouts
3. **Error Handling**: Don't expose sensitive error details
4. **CORS Configuration**: Configure properly for production
5. **Rate Limiting**: Implement if needed for production

## Performance Considerations

1. **File Size Limits**: Set reasonable limits (e.g., 50MB max)
2. **Processing Time**: Large PDFs can take 1-3 minutes
3. **Memory Usage**: Monitor for large file processing
4. **Concurrent Sessions**: Limit based on server capacity
5. **Caching**: Consider caching results for repeated analysis

## Production Deployment

1. **Environment Variables**: Use proper configuration
2. **Database**: Replace in-memory sessions with database
3. **Logging**: Implement comprehensive logging
4. **Monitoring**: Set up health checks and metrics
5. **Scaling**: Consider horizontal scaling for high load
