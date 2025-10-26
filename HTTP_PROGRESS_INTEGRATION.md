# HTTP Requests with SSE Progress Integration

## Overview
This guide shows how to implement real-time progress updates using direct HTTP requests and Server-Sent Events (SSE) without the SDK.

## 🔧 Implementation Approach

### Step 1: Upload File and Get Session ID

```javascript
async function uploadAndAnalyze(file, options = {}) {
  try {
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Add options
    Object.entries(options).forEach(([key, value]) => {
      formData.append(key, value);
    });
    
    // Upload file and start analysis
    const response = await fetch('/analyze', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    // Set up progress stream if we got a session ID
    if (result.session_id) {
      setupProgressStream(result.session_id);
    }
    
    return result;
}
```

### Step 2: Set Up Server-Sent Events Stream

```javascript
function setupProgressStream(sessionId) {
  // Create EventSource connection
  const eventSource = new EventSource(`/stream/${sessionId}`);
  
  eventSource.onmessage = function(event) {
    try {
      const progressData = JSON.parse(event.data);
      handleProgressUpdate(progressData);
    } catch (error) {
      console.error('Error parsing progress data:', error);
    }
  };
  
  eventSource.onerror = function(event) {
    console.error('SSE connection error:', event);
    eventSource.close();
  };
  
  // Store for cleanup
  window.currentEventSource = eventSource;
}

function handleProgressUpdate(progressData) {
  const { milestone, message, data } = progressData;
  
  // Update your UI based on milestone
  switch(milestone) {
    case 'analysis_started':
      updateProgressUI('started', 'Starting analysis...', 10);
      break;
      
    case 'detecting_doors':
      updateProgressUI('detecting_doors', 'Detecting doors in floor plan...', 30);
      break;
      
    case 'doors_detected':
      updateProgressUI('doors_detected', `Found ${data.door_count} doors`, 50);
      break;
      
    case 'analyzing_fire_ratings':
      updateProgressUI('analyzing_fire_ratings', 'Analyzing fire ratings...', 70);
      break;
      
    case 'fire_ratings_analyzed':
      updateProgressUI('fire_ratings_analyzed', 'Fire rating analysis complete', 90);
      break;
      
    case 'human_review_needed':
      updateProgressUI('human_review_needed', `Human review required for ${data.review_items_count} doors`, 95);
      break;
      
    case 'analysis_complete':
      updateProgressUI('complete', 'Analysis complete!', 100);
      closeProgressStream();
      break;
      
    case 'analysis_failed':
      updateProgressUI('error', `Analysis failed: ${data.error}`, 0);
      closeProgressStream();
      break;
  }
}
```

### Step 3: Create Progress UI Functions

```javascript
function updateProgressUI(step, message, percentage) {
  // Update progress bar
  const progressBar = document.getElementById('progressBar');
  if (progressBar) {
    progressBar.style.width = percentage + '%';
  }
  
  // Update progress text
  const progressText = document.getElementById('progressText');
  if (progressText) {
    progressText.textContent = message;
  }
  
  // Update step indicator
  const stepIndicator = document.getElementById('stepIndicator');
  if (stepIndicator) {
    stepIndicator.textContent = getStepIcon(step);
  }
  
  // Log progress (for debugging)
  console.log(`Progress: ${step} - ${message} (${percentage}%)`);
}

function getStepIcon(step) {
  const icons = {
    'started': '🚀',
    'detecting_doors': '🔍',
    'doors_detected': '🚪',
    'analyzing_fire_ratings': '🔥',
    'fire_ratings_analyzed': '✅',
    'human_review_needed': '👤',
    'complete': '🎉',
    'error': '❌'
  };
  return icons[step] || '⏳';
}

function closeProgressStream() {
  if (window.currentEventSource) {
    window.currentEventSource.close();
    window.currentEventSource = null;
  }
}
```

### Step 4: Handle Human Review Workflow

```javascript
async function submitHumanReview(sessionId, reviews) {
  try {
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
    
    const result = await response.json();
    
    // Set up progress stream for review completion
    if (result.session_id) {
      setupProgressStream(result.session_id);
    }
    
    return result;
    
  } catch (error) {
    console.error('Error submitting review:', error);
    throw error;
  }
}
```

### Step 5: Complete HTML Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fire Door Detection - HTTP Progress</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
        }
        
        .progress-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 24px;
            margin: 20px 0;
            color: white;
            display: none;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background-color: rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background-color: #fff;
            transition: width 0.5s ease;
        }
        
        .progress-text {
            font-size: 18px;
            font-weight: 500;
            margin-bottom: 10px;
        }
        
        .step-icon {
            font-size: 24px;
            margin-right: 10px;
        }
        
        .btn {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        
        .btn:hover {
            background-color: #0056b3;
        }
        
        .btn:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        
        .results {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 20px;
            margin: 20px 0;
            display: none;
        }
        
        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 15px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>Fire Door Detection - Real-Time Progress</h1>
    
    <div class="upload-area" id="uploadArea">
        <p>Drag and drop a PDF or image file here, or click to select</p>
        <input type="file" id="fileInput" style="display: none;" accept=".pdf,.png,.jpg,.jpeg">
        <button class="btn" onclick="document.getElementById('fileInput').click()">Select File</button>
    </div>
    
    <div id="fileInfo" style="display: none;">
        <h3>Selected File:</h3>
        <p id="fileName"></p>
        <p id="fileSize"></p>
        <button class="btn" id="analyzeBtn" onclick="analyzeFile()">Analyze Floor Plan</button>
    </div>

    <div id="progressContainer" class="progress-container">
        <div class="progress-text">
            <span id="stepIndicator" class="step-icon">⏳</span>
            <span id="progressText">Starting analysis...</span>
        </div>
        <div class="progress-bar">
            <div id="progressBar" class="progress-fill" style="width: 0%"></div>
        </div>
    </div>

    <div id="results" class="results">
        <h3>Analysis Results</h3>
        <div id="resultsContent"></div>
    </div>

    <div id="error" class="error" style="display: none;">
        <h3>Error</h3>
        <p id="errorMessage"></p>
    </div>

    <script>
        let currentFile = null;
        let currentSessionId = null;

        // File upload handling
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const progressContainer = document.getElementById('progressContainer');
        const results = document.getElementById('results');
        const error = document.getElementById('error');

        // Drag and drop handlers
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#007bff';
            uploadArea.style.backgroundColor = '#f8f9fa';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '#ccc';
            uploadArea.style.backgroundColor = 'transparent';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#ccc';
            uploadArea.style.backgroundColor = 'transparent';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            // Validate file type
            const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];
            if (!allowedTypes.includes(file.type)) {
                showError('Invalid file type. Please upload a PDF, PNG, or JPEG file.');
                return;
            }

            // Validate file size (50MB limit)
            const maxSize = 50 * 1024 * 1024; // 50MB
            if (file.size > maxSize) {
                showError('File too large. Maximum size is 50MB.');
                return;
            }

            currentFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = `Size: ${(file.size / 1024 / 1024).toFixed(2)} MB`;
            fileInfo.style.display = 'block';
            hideError();
        }

        async function analyzeFile() {
            if (!currentFile) {
                showError('Please select a file first');
                return;
            }

            try {
                // Show progress container
                progressContainer.style.display = 'block';
                results.style.display = 'none';
                hideError();

                // Start analysis
                const result = await uploadAndAnalyze(currentFile, {
                    confidence: 0.25,
                    padding: 20,
                    dpi: 600
                });

                if (result.success) {
                    if (result.needsReview) {
                        // Handle human review workflow
                        handleHumanReview(result);
                    } else {
                        // Show results
                        showResults(result.data);
                    }
                } else {
                    showError(result.error);
                }

            } catch (error) {
                showError(`Analysis failed: ${error.message}`);
            }
        }

        async function uploadAndAnalyze(file, options = {}) {
            try {
                // Create form data
                const formData = new FormData();
                formData.append('file', file);
                
                // Add options
                Object.entries(options).forEach(([key, value]) => {
                    formData.append(key, value);
                });
                
                // Upload file and start analysis
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
                }
                
                const result = await response.json();
                
                // Set up progress stream if we got a session ID
                if (result.session_id) {
                    currentSessionId = result.session_id;
                    setupProgressStream(result.session_id);
                }
                
                return result;
                
            } catch (error) {
                throw new Error(`Upload failed: ${error.message}`);
            }
        }

        function setupProgressStream(sessionId) {
            // Create EventSource connection
            const eventSource = new EventSource(`/stream/${sessionId}`);
            
            eventSource.onmessage = function(event) {
                try {
                    const progressData = JSON.parse(event.data);
                    handleProgressUpdate(progressData);
                } catch (error) {
                    console.error('Error parsing progress data:', error);
                }
            };
            
            eventSource.onerror = function(event) {
                console.error('SSE connection error:', event);
                eventSource.close();
            };
            
            // Store for cleanup
            window.currentEventSource = eventSource;
        }

        function handleProgressUpdate(progressData) {
            const { milestone, message, data } = progressData;
            
            // Update your UI based on milestone
            switch(milestone) {
                case 'analysis_started':
                    updateProgressUI('started', 'Starting analysis...', 10);
                    break;
                    
                case 'detecting_doors':
                    updateProgressUI('detecting_doors', 'Detecting doors in floor plan...', 30);
                    break;
                    
                case 'doors_detected':
                    updateProgressUI('doors_detected', `Found ${data.door_count} doors`, 50);
                    break;
                    
                case 'analyzing_fire_ratings':
                    updateProgressUI('analyzing_fire_ratings', 'Analyzing fire ratings...', 70);
                    break;
                    
                case 'fire_ratings_analyzed':
                    updateProgressUI('fire_ratings_analyzed', 'Fire rating analysis complete', 90);
                    break;
                    
                case 'human_review_needed':
                    updateProgressUI('human_review_needed', `Human review required for ${data.review_items_count} doors`, 95);
                    break;
                    
                case 'analysis_complete':
                    updateProgressUI('complete', 'Analysis complete!', 100);
                    closeProgressStream();
                    break;
                    
                case 'analysis_failed':
                    updateProgressUI('error', `Analysis failed: ${data.error}`, 0);
                    closeProgressStream();
                    break;
            }
        }

        function updateProgressUI(step, message, percentage) {
            // Update progress bar
            const progressBar = document.getElementById('progressBar');
            if (progressBar) {
                progressBar.style.width = percentage + '%';
            }
            
            // Update progress text
            const progressText = document.getElementById('progressText');
            if (progressText) {
                progressText.textContent = message;
            }
            
            // Update step indicator
            const stepIndicator = document.getElementById('stepIndicator');
            if (stepIndicator) {
                stepIndicator.textContent = getStepIcon(step);
            }
            
            // Log progress (for debugging)
            console.log(`Progress: ${step} - ${message} (${percentage}%)`);
        }

        function getStepIcon(step) {
            const icons = {
                'started': '🚀',
                'detecting_doors': '🔍',
                'doors_detected': '🚪',
                'analyzing_fire_ratings': '🔥',
                'fire_ratings_analyzed': '✅',
                'human_review_needed': '👤',
                'complete': '🎉',
                'error': '❌'
            };
            return icons[step] || '⏳';
        }

        function closeProgressStream() {
            if (window.currentEventSource) {
                window.currentEventSource.close();
                window.currentEventSource = null;
            }
        }

        function showResults(data) {
            progressContainer.style.display = 'none';
            results.style.display = 'block';
            
            const resultsContent = document.getElementById('resultsContent');
            resultsContent.innerHTML = `
                <div class="status success">
                    <h4>Analysis Complete!</h4>
                    <p>Total doors detected: ${data.door_count}</p>
                </div>
                <h4>Fire Rating Breakdown:</h4>
                <ul>
                    ${Object.entries(data.fire_ratings).map(([rating, count]) => 
                        `<li>${rating === 'NO_RATING' ? 'No Fire Rating' : rating + '-minute rating'}: ${count}</li>`
                    ).join('')}
                </ul>
                ${data.human_feedback ? `
                    <h4>Human Review Summary:</h4>
                    <ul>
                        <li>Total reviewed: ${data.human_feedback.total_reviewed}</li>
                        <li>Rated: ${data.human_feedback.rated}</li>
                        <li>Skipped: ${data.human_feedback.skipped}</li>
                        <li>No rating: ${data.human_feedback.no_rating_count}</li>
                    </ul>
                ` : ''}
            `;
        }

        function showError(message) {
            error.style.display = 'block';
            document.getElementById('errorMessage').textContent = message;
            progressContainer.style.display = 'none';
            results.style.display = 'none';
        }

        function hideError() {
            error.style.display = 'none';
        }

        function handleHumanReview(result) {
            // Simplified human review handling
            alert('Human review required - this would show the review interface');
            // In a real implementation, you would show a review interface here
        }

        // Check API health on page load
        window.addEventListener('load', async () => {
            try {
                const response = await fetch('/');
                if (!response.ok) {
                    showError('API is not responding. Please make sure the server is running.');
                }
            } catch (error) {
                showError('Cannot connect to API server.');
            }
        });
    </script>
</body>
</html>
```

## 🔧 Key Differences from SDK Approach

### 1. **Manual EventSource Management**
```javascript
// Instead of SDK handling it automatically
const eventSource = new EventSource(`/stream/${sessionId}`);
```

### 2. **Manual Progress Handling**
```javascript
// You handle progress updates manually
eventSource.onmessage = function(event) {
  const progressData = JSON.parse(event.data);
  handleProgressUpdate(progressData);
};
```

### 3. **Manual Cleanup**
```javascript
// You need to close the connection manually
function closeProgressStream() {
  if (window.currentEventSource) {
    window.currentEventSource.close();
    window.currentEventSource = null;
  }
}
```

## 🎯 Benefits of HTTP Approach

- ✅ **Full Control** - You control exactly how progress is handled
- ✅ **No Dependencies** - No SDK required
- ✅ **Customizable** - Easy to customize progress UI
- ✅ **Lightweight** - Smaller bundle size
- ✅ **Framework Agnostic** - Works with any frontend framework

## 🚀 Usage Summary

1. **Upload file** with `fetch('/analyze', ...)`
2. **Get session ID** from response
3. **Set up SSE** with `new EventSource('/stream/{sessionId}')`
4. **Handle progress** in `onmessage` callback
5. **Clean up** when complete

This approach gives you complete control over the progress updates while still getting real-time feedback from the backend! 🎉
