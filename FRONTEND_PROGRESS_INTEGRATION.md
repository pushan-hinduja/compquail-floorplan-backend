# Frontend Progress Integration Guide

## Overview
This guide provides step-by-step instructions for integrating real-time progress updates into your frontend application using Server-Sent Events (SSE).

## 🎯 What You'll Implement

Real-time progress updates that show users exactly what's happening during floor plan analysis:

1. **"Detecting doors..."** - When YOLO model is processing
2. **"Found X doors"** - When door detection completes
3. **"Analyzing fire ratings..."** - When OpenAI API is processing
4. **"Analysis complete!"** - When everything is finished

## 📋 Implementation Steps

### Step 1: Update Your SDK Usage

**Before (without progress):**
```javascript
const result = await sdk.analyzeFloorPlan(file, options);
```

**After (with progress):**
```javascript
// Progress callback function
const onProgress = (progressData) => {
  console.log('Progress:', progressData.message);
  updateLoadingState(progressData);
};

const result = await sdk.analyzeFloorPlan(file, options, onProgress);
```

### Step 2: Create Progress State Management

Add these to your component state:

```javascript
const [analysisProgress, setAnalysisProgress] = useState({
  isAnalyzing: false,
  currentStep: null,
  message: '',
  doorCount: 0,
  error: null
});
```

### Step 3: Implement Progress Handler

```javascript
const updateLoadingState = (progressData) => {
  const { milestone, message, data } = progressData;
  
  switch(milestone) {
    case 'analysis_started':
      setAnalysisProgress({
        isAnalyzing: true,
        currentStep: 'started',
        message: 'Starting analysis...',
        doorCount: 0,
        error: null
      });
      break;
      
    case 'detecting_doors':
      setAnalysisProgress(prev => ({
        ...prev,
        currentStep: 'detecting_doors',
        message: 'Detecting doors in floor plan...'
      }));
      break;
      
    case 'doors_detected':
      setAnalysisProgress(prev => ({
        ...prev,
        currentStep: 'doors_detected',
        message: `Found ${data.door_count} doors`,
        doorCount: data.door_count
      }));
      break;
      
    case 'analyzing_fire_ratings':
      setAnalysisProgress(prev => ({
        ...prev,
        currentStep: 'analyzing_fire_ratings',
        message: 'Analyzing fire ratings...'
      }));
      break;
      
    case 'fire_ratings_analyzed':
      setAnalysisProgress(prev => ({
        ...prev,
        currentStep: 'fire_ratings_analyzed',
        message: 'Fire rating analysis complete'
      }));
      break;
      
    case 'human_review_needed':
      setAnalysisProgress(prev => ({
        ...prev,
        currentStep: 'human_review_needed',
        message: `Human review required for ${data.review_items_count} doors`
      }));
      break;
      
    case 'analysis_complete':
      setAnalysisProgress(prev => ({
        ...prev,
        currentStep: 'complete',
        message: 'Analysis complete!',
        isAnalyzing: false
      }));
      break;
      
    case 'analysis_failed':
      setAnalysisProgress(prev => ({
        ...prev,
        currentStep: 'error',
        message: `Analysis failed: ${data.error}`,
        error: data.error,
        isAnalyzing: false
      }));
      break;
  }
};
```

### Step 4: Create Progress UI Component

```jsx
const ProgressIndicator = ({ progress }) => {
  if (!progress.isAnalyzing && !progress.error) return null;
  
  const getStepIcon = (step) => {
    switch(step) {
      case 'started': return '🚀';
      case 'detecting_doors': return '🔍';
      case 'doors_detected': return '🚪';
      case 'analyzing_fire_ratings': return '🔥';
      case 'fire_ratings_analyzed': return '✅';
      case 'human_review_needed': return '👤';
      case 'complete': return '🎉';
      case 'error': return '❌';
      default: return '⏳';
    }
  };
  
  const getStepColor = (step) => {
    switch(step) {
      case 'error': return 'text-red-600';
      case 'complete': return 'text-green-600';
      default: return 'text-blue-600';
    }
  };
  
  return (
    <div className="progress-container">
      <div className="flex items-center space-x-3">
        <span className="text-2xl">{getStepIcon(progress.currentStep)}</span>
        <div>
          <p className={`font-medium ${getStepColor(progress.currentStep)}`}>
            {progress.message}
          </p>
          {progress.doorCount > 0 && (
            <p className="text-sm text-gray-500">
              {progress.doorCount} doors detected
            </p>
          )}
        </div>
      </div>
      
      {/* Progress bar */}
      <div className="mt-3">
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-600 h-2 rounded-full transition-all duration-500"
            style={{ 
              width: getProgressPercentage(progress.currentStep) + '%' 
            }}
          />
        </div>
      </div>
    </div>
  );
};

const getProgressPercentage = (step) => {
  const steps = {
    'started': 10,
    'detecting_doors': 30,
    'doors_detected': 50,
    'analyzing_fire_ratings': 70,
    'fire_ratings_analyzed': 90,
    'complete': 100,
    'error': 0
  };
  return steps[step] || 0;
};
```

### Step 5: Update Your Analysis Function

```javascript
const analyzeFloorPlan = async (file, options = {}) => {
  try {
    // Reset progress state
    setAnalysisProgress({
      isAnalyzing: false,
      currentStep: null,
      message: '',
      doorCount: 0,
      error: null
    });
    
    // Start analysis with progress callback
    const result = await sdk.analyzeFloorPlan(file, options, updateLoadingState);
    
    if (result.success) {
      if (result.needsReview) {
        // Handle human review workflow
        setShowReviewInterface(true);
        setReviewItems(result.data.review_items);
      } else {
        // Show final results
        setAnalysisResults(result.data);
        setShowResults(true);
      }
    } else {
      setAnalysisProgress(prev => ({
        ...prev,
        error: result.error,
        isAnalyzing: false
      }));
    }
    
  } catch (error) {
    setAnalysisProgress(prev => ({
      ...prev,
      error: error.message,
      isAnalyzing: false
    }));
  }
};
```

### Step 6: Add CSS Styles

```css
.progress-container {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  padding: 24px;
  margin: 20px 0;
  color: white;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.progress-container .flex {
  display: flex;
  align-items: center;
}

.progress-container .space-x-3 > * + * {
  margin-left: 0.75rem;
}

.progress-container .text-2xl {
  font-size: 1.5rem;
  line-height: 2rem;
}

.progress-container .font-medium {
  font-weight: 500;
}

.progress-container .text-sm {
  font-size: 0.875rem;
  line-height: 1.25rem;
}

.progress-container .text-gray-500 {
  color: rgba(156, 163, 175, 1);
}

.progress-container .mt-3 {
  margin-top: 0.75rem;
}

.progress-container .w-full {
  width: 100%;
}

.progress-container .bg-gray-200 {
  background-color: rgba(229, 231, 235, 1);
}

.progress-container .rounded-full {
  border-radius: 9999px;
}

.progress-container .h-2 {
  height: 0.5rem;
}

.progress-container .bg-blue-600 {
  background-color: rgba(37, 99, 235, 1);
}

.progress-container .transition-all {
  transition-property: all;
  transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
  transition-duration: 500ms;
}
```

## 🎨 UI/UX Recommendations

### Loading States
- **Show progress bar** with percentage
- **Display current step** with descriptive text
- **Use icons** to make steps more visual
- **Show door count** when available
- **Handle errors** gracefully

### User Experience
- **Disable upload** during analysis
- **Show estimated time** if possible
- **Allow cancellation** if needed
- **Provide clear feedback** for each step

### Error Handling
- **Show error messages** clearly
- **Allow retry** on failure
- **Log errors** for debugging
- **Graceful degradation** if SSE fails

## 🔧 Advanced Features

### Custom Progress Messages
```javascript
const getCustomMessage = (milestone, data) => {
  const messages = {
    'detecting_doors': '🔍 Scanning floor plan for doors...',
    'doors_detected': `🚪 Found ${data.door_count} doors!`,
    'analyzing_fire_ratings': '🔥 Analyzing fire ratings...',
    'fire_ratings_analyzed': '✅ Fire rating analysis complete!'
  };
  return messages[milestone] || 'Processing...';
};
```

### Progress Persistence
```javascript
// Save progress to localStorage
const saveProgress = (progress) => {
  localStorage.setItem('analysisProgress', JSON.stringify(progress));
};

// Load progress on page refresh
const loadProgress = () => {
  const saved = localStorage.getItem('analysisProgress');
  return saved ? JSON.parse(saved) : null;
};
```

### Multiple File Support
```javascript
const analyzeMultipleFiles = async (files) => {
  const results = [];
  
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    setAnalysisProgress(prev => ({
      ...prev,
      message: `Processing file ${i + 1} of ${files.length}: ${file.name}`
    }));
    
    const result = await sdk.analyzeFloorPlan(file, options, updateLoadingState);
    results.push(result);
  }
  
  return results;
};
```

## 🧪 Testing

### Test Progress Updates
```javascript
// Test with different file types
const testFiles = [
  { name: 'small.pdf', type: 'application/pdf' },
  { name: 'large.pdf', type: 'application/pdf' },
  { name: 'floorplan.png', type: 'image/png' }
];

testFiles.forEach(file => {
  console.log(`Testing with ${file.name}`);
  // Your analysis code here
});
```

### Test Error Handling
```javascript
// Test with invalid files
const invalidFiles = [
  { name: 'text.txt', type: 'text/plain' },
  { name: 'corrupted.pdf', type: 'application/pdf' }
];
```

## 📱 Mobile Considerations

- **Touch-friendly** progress indicators
- **Responsive design** for different screen sizes
- **Battery optimization** for long-running analyses
- **Offline handling** if connection drops

## 🚀 Performance Tips

- **Debounce progress updates** to avoid too many re-renders
- **Use React.memo** for progress components
- **Lazy load** heavy components
- **Optimize images** before upload

## 🔍 Debugging

### Console Logging
```javascript
const onProgress = (progressData) => {
  console.log('Progress Update:', {
    milestone: progressData.milestone,
    message: progressData.message,
    timestamp: progressData.timestamp,
    data: progressData.data
  });
  
  updateLoadingState(progressData);
};
```

### Network Monitoring
- Check browser DevTools → Network tab
- Look for SSE connection to `/stream/{session_id}`
- Monitor for connection drops or errors

## 📋 Checklist

- [ ] Update SDK usage with progress callback
- [ ] Add progress state management
- [ ] Create progress UI component
- [ ] Implement progress handler function
- [ ] Add CSS styles for progress indicator
- [ ] Test with different file types
- [ ] Handle error states
- [ ] Add mobile responsiveness
- [ ] Test SSE connection stability
- [ ] Add loading state persistence

## 🎯 Expected Result

Your users will see real-time progress updates like:

1. **"🚀 Starting analysis..."** (10%)
2. **"🔍 Detecting doors..."** (30%)
3. **"🚪 Found 5 doors"** (50%)
4. **"🔥 Analyzing fire ratings..."** (70%)
5. **"✅ Analysis complete!"** (100%)

This creates a much more engaging and informative user experience! 🎉
