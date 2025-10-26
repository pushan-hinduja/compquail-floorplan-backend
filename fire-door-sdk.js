/**
 * Fire Door Detection API SDK
 * Complete JavaScript SDK for integrating with the Fire Door Detection API
 */

class FireDoorSDK {
  constructor(apiBaseUrl = 'http://localhost:8000') {
    this.apiBaseUrl = apiBaseUrl;
    this.currentSession = null;
  }

  /**
   * Analyze a floor plan file
   * @param {File} file - PDF or image file
   * @param {Object} options - Analysis options
   * @param {Function} onProgress - Optional progress callback function
   * @returns {Promise<Object>} Analysis result
   */
  async analyzeFloorPlan(file, options = {}, onProgress = null) {
    try {
      // Validate file type
      const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];
      if (!allowedTypes.includes(file.type)) {
        throw new Error('Invalid file type. Please upload a PDF, PNG, or JPEG file.');
      }

      // Validate file size (50MB limit)
      const maxSize = 50 * 1024 * 1024; // 50MB
      if (file.size > maxSize) {
        throw new Error('File too large. Maximum size is 50MB.');
      }

      // Prepare form data
      const formData = new FormData();
      formData.append('file', file);
      
      // Add options with defaults
      const defaultOptions = {
        confidence: 0.25,
        padding: 20,
        dpi: 600,
        enhance: true,
        model: 'gpt-4o',
        debug_doors: false,
        use_adaptive_grid: true
      };
      
      const finalOptions = { ...defaultOptions, ...options };
      Object.entries(finalOptions).forEach(([key, value]) => {
        formData.append(key, value);
      });

      // Make API call
      const response = await fetch(`${this.apiBaseUrl}/analyze`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Store session info if review is needed
      if (result.status === 'needs_review') {
        this.currentSession = {
          sessionId: result.session_id,
          reviewItems: result.data.review_items,
          doorCount: result.data.door_count,
          automaticRatings: result.data.automatic_ratings || {}
        };
      }

      // Set up SSE connection for progress updates if callback provided
      if (onProgress && result.session_id) {
        this.setupProgressStream(result.session_id, onProgress);
      }

      return {
        success: true,
        status: result.status,
        sessionId: result.session_id,
        message: result.message,
        data: result.data,
        needsReview: result.status === 'needs_review'
      };

    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Submit human review results
   * @param {string} sessionId - Session ID
   * @param {Array} reviews - Array of review objects
   * @returns {Promise<Object>} Final analysis result
   */
  async submitReview(sessionId, reviews) {
    try {
      // Validate reviews format
      if (!Array.isArray(reviews)) {
        throw new Error('Reviews must be an array');
      }

      const validRatings = ['30', '60', '90', '180', 'NO_RATING', 'skip'];
      
      for (const review of reviews) {
        if (!review.image_name) {
          throw new Error('Each review must have an image_name');
        }
        if (review.rating && !validRatings.includes(review.rating)) {
          throw new Error(`Invalid rating: ${review.rating}. Valid options: ${validRatings.join(', ')}`);
        }
      }

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

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Clear current session
      this.currentSession = null;

      return {
        success: true,
        status: result.status,
        message: result.message,
        data: result.data
      };

    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get current session status
   * @param {string} sessionId - Session ID
   * @returns {Promise<Object>} Session status
   */
  async getSessionStatus(sessionId) {
    try {
      const response = await fetch(`${this.apiBaseUrl}/session/${sessionId}`);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();

    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Cancel current session
   * @param {string} sessionId - Session ID
   * @returns {Promise<Object>} Cancellation result
   */
  async cancelSession(sessionId) {
    try {
      const response = await fetch(`${this.apiBaseUrl}/session/${sessionId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Clear current session if it matches
      if (this.currentSession && this.currentSession.sessionId === sessionId) {
        this.currentSession = null;
      }

      return {
        success: true,
        message: result.message
      };

    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get current session info
   * @returns {Object|null} Current session info
   */
  getCurrentSession() {
    return this.currentSession;
  }

  /**
   * Check if API is healthy
   * @returns {Promise<boolean>} API health status
   */
  async checkHealth() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/`);
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  /**
   * Manually clean up all session data on the server
   * @returns {Promise<Object>} Cleanup result
   */
  async cleanup() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/cleanup`, {
        method: 'POST'
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();

    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Set up Server-Sent Events stream for progress updates
   * @param {string} sessionId - Session ID
   * @param {Function} onProgress - Progress callback function
   */
  setupProgressStream(sessionId, onProgress) {
    const eventSource = new EventSource(`${this.apiBaseUrl}/stream/${sessionId}`);
    
    eventSource.onmessage = function(event) {
      try {
        const progressData = JSON.parse(event.data);
        onProgress(progressData);
      } catch (error) {
        console.error('Error parsing progress data:', error);
      }
    };
    
    eventSource.onerror = function(event) {
      console.error('SSE connection error:', event);
      eventSource.close();
    };
    
    // Store event source for cleanup
    this.currentEventSource = eventSource;
  }

  /**
   * Close the current progress stream
   */
  closeProgressStream() {
    if (this.currentEventSource) {
      this.currentEventSource.close();
      this.currentEventSource = null;
    }
  }

  /**
   * Helper method to create review objects
   * @param {string} imageName - Image filename
   * @param {string|null} rating - Rating or null for skip
   * @returns {Object} Review object
   */
  createReview(imageName, rating) {
    return {
      image_name: imageName,
      rating: rating
    };
  }

  /**
   * Helper method to batch create reviews
   * @param {Array} reviewData - Array of {imageName, rating} objects
   * @returns {Array} Array of review objects
   */
  createReviews(reviewData) {
    return reviewData.map(({ imageName, rating }) => 
      this.createReview(imageName, rating)
    );
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = FireDoorSDK;
}

// Example usage and helper functions
const FireDoorHelpers = {
  /**
   * Display analysis results in a formatted way
   * @param {Object} data - Analysis result data
   * @returns {string} Formatted results
   */
  formatResults(data) {
    if (!data || !data.fire_ratings) {
      return 'No results available';
    }

    const ratings = data.fire_ratings;
    const totalDoors = data.door_count || 0;
    
    let result = `Total Doors: ${totalDoors}\n\nFire Rating Breakdown:\n`;
    
    Object.entries(ratings).forEach(([rating, count]) => {
      if (rating === 'NO_RATING') {
        result += `• No Fire Rating: ${count}\n`;
      } else {
        result += `• ${rating}-minute rating: ${count}\n`;
      }
    });

    if (data.human_feedback) {
      const feedback = data.human_feedback;
      result += `\nHuman Review Summary:\n`;
      result += `• Total reviewed: ${feedback.total_reviewed}\n`;
      result += `• Rated: ${feedback.rated}\n`;
      result += `• Skipped: ${feedback.skipped}\n`;
      result += `• No rating: ${feedback.no_rating_count}\n`;
    }

    return result;
  },

  /**
   * Validate file before upload
   * @param {File} file - File to validate
   * @returns {Object} Validation result
   */
  validateFile(file) {
    const errors = [];
    
    // Check file type
    const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
      errors.push('Invalid file type. Please upload a PDF, PNG, or JPEG file.');
    }
    
    // Check file size (50MB limit)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      errors.push('File too large. Maximum size is 50MB.');
    }
    
    // Check if file is empty
    if (file.size === 0) {
      errors.push('File is empty.');
    }
    
    return {
      valid: errors.length === 0,
      errors: errors
    };
  },

  /**
   * Get rating options for UI
   * @returns {Array} Array of rating options
   */
  getRatingOptions() {
    return [
      { value: '30', label: '30-minute rating' },
      { value: '60', label: '60-minute rating' },
      { value: '90', label: '90-minute rating' },
      { value: '180', label: '180-minute rating' },
      { value: 'NO_RATING', label: 'No fire rating' },
      { value: 'skip', label: 'Skip this door' }
    ];
  }
};

// Example usage:
/*
// Initialize SDK
const sdk = new FireDoorSDK('http://localhost:8000');

// Check API health
const isHealthy = await sdk.checkHealth();
console.log('API Health:', isHealthy);

// Analyze a floor plan with progress updates
const fileInput = document.getElementById('fileInput');
const file = fileInput.files[0];

// Progress callback function
const onProgress = (progressData) => {
  console.log('Progress:', progressData.message);
  
  switch(progressData.milestone) {
    case 'detecting_doors':
      updateUI('Detecting doors...');
      break;
    case 'doors_detected':
      updateUI(`Found ${progressData.data.door_count} doors. Analyzing fire ratings...`);
      break;
    case 'fire_ratings_analyzed':
      updateUI('Fire rating analysis complete!');
      break;
    case 'human_review_needed':
      updateUI('Human review required');
      break;
    case 'analysis_complete':
      updateUI('Analysis complete!');
      sdk.closeProgressStream();
      break;
  }
};

const result = await sdk.analyzeFloorPlan(file, {
  confidence: 0.25,
  padding: 20,
  dpi: 600
}, onProgress);

if (result.success && result.needsReview) {
  // Show human review interface
  const reviewItems = result.data.review_items;
  
  // After user provides reviews
  const reviews = sdk.createReviews([
    { imageName: 'door_1.png', rating: '90' },
    { imageName: 'door_2.png', rating: 'skip' }
  ]);
  
  const finalResult = await sdk.submitReview(result.sessionId, reviews);
  console.log('Final results:', FireDoorHelpers.formatResults(finalResult.data));
}
*/
