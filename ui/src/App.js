import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [model, setModel] = useState('MobileNet');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [folderImages, setFolderImages] = useState([]);
  const [folderPreviews, setFolderPreviews] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResults([]);
      setFolderImages([]);
      setFolderPreviews([]);
      setError(null);
    }
  };

  const handleFolderUpload = (event) => {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
      setFolderImages(files);
      setFolderPreviews(files.map(file => URL.createObjectURL(file)));
      setImage(null);
      setImagePreview(null);
      setResults([]);
      setError(null);
    }
  };

  const handleDetect = async () => {
    if (!image && folderImages.length === 0) {
      alert('Please upload an image or folder first.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      if (model === 'All') {
        if (image) {
          // Single image, all models
          const formData = new FormData();
          formData.append('image', image);
          const response = await axios.post('http://localhost:5000/api/detect/all', formData);
          
          // Convert the response to array format to match existing UI logic
          const allResults = [];
          Object.keys(response.data).forEach(modelName => {
            allResults.push(response.data[modelName]);
          });
          setResults(allResults);
        } else {
          // Folder images, all models
          const formData = new FormData();
          folderImages.forEach(file => formData.append('images', file));
          const response = await axios.post('http://localhost:5000/api/detect/batch/all', formData);
          
          // Convert the response to format that shows all models for each image
          const organizedResults = [];
          const modelNames = Object.keys(response.data);
          const numImages = folderImages.length;
          
          
          for (let imageIndex = 0; imageIndex < numImages; imageIndex++) {
            const imageResults = [];
            modelNames.forEach(modelName => {
              if (response.data[modelName] && response.data[modelName][imageIndex]) {
                imageResults.push(response.data[modelName][imageIndex]);
              }
            });
            organizedResults.push(imageResults);
          }
          
          setResults(organizedResults);
        }
      } else {
    
        const formData = new FormData();
        formData.append('model_name', model);
        
        if (image) {
          formData.append('image', image);
          const response = await axios.post('http://localhost:5000/api/detect', formData);
          setResults([response.data]);
        } else {
          folderImages.forEach(file => formData.append('images', file));
          const response = await axios.post('http://localhost:5000/api/detect/batch', formData);
          setResults(response.data);
        }
      }
    } catch (error) {
      console.error('Error during detection:', error);
      setError(`Detection failed: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-wrapper">
      <header className="main-header">
        <h1>HPE Laptop Defect Detection AI</h1>
      </header>
      <div className="sidebar">
        <label className="label">Select Model:</label>
        <select className="dropdown" value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="ResNet-50">ResNet-50</option>
          <option value="YOLO">YOLOv8</option>
          <option value="MobileNet">MobileNetV2</option>
          <option value="All">All Models</option>
        </select>

        <label className="upload-label">Upload Image:</label>
        <input type="file" accept="image/*" onChange={handleImageUpload} className="upload-input" />

        <label className="upload-label">Upload Folder:</label>
        <input type="file" webkitdirectory="true" directory="true" multiple onChange={handleFolderUpload} className="upload-input" />

        <button 
          className="detect-button" 
          onClick={handleDetect} 
          disabled={loading || (!image && folderImages.length === 0)}
        >
          {loading ? 'Processing...' : 'Detect Defects'}
        </button>

        {error && <div className="error-message">{error}</div>}
      </div>

      <div className="results-section">
        <div className="preview-grid">
          {imagePreview && results.length > 0 && (
            <div className="preview-with-result">
              <img
                src={
                  results.find(r => r.model && r.model.toLowerCase() === "yolo")?.annotated_image_base64
                    ? `data:image/jpeg;base64,${results.find(r => r.model && r.model.toLowerCase() === "yolo").annotated_image_base64}`
                    : imagePreview
                }
                alt="Uploaded"
                className="preview-img"
              />
              {results.map((res, idx) => (
                <div key={idx} className={`result-tag ${res.hasDefect ? 'defect' : 'no-defect'}`}>
                  <strong>Model:</strong> {res.model}<br />
                  <strong>Status:</strong> {res.hasDefect ? 'Defect Found' : 'No Defect Found'}<br />
                  {res.hasDefect && res.predicted_class && (
                    <>
                      <strong>Defect Type:</strong> {res.predicted_class}<br />
                      <strong>Confidence:</strong> {(results[idx].confidence * 100).toFixed(2)}%<br />
                  
                    </>
                  )}
                  <strong>Processed At:</strong> {new Date().toLocaleString()}<br />
                </div>
              ))}
            </div>
          )}

          {folderPreviews.length > 0 && results.length === folderPreviews.length && folderPreviews.map((src, idx) => (
            <div key={idx} className="preview-with-result">
              <img
                src={
                 
                  model === 'All' && Array.isArray(results[idx])
                    ? (results[idx].find(r => r.model && r.model.toLowerCase() === "yolo" && r.annotated_image_base64)?.annotated_image_base64
                        ? `data:image/jpeg;base64,${results[idx].find(r => r.model && r.model.toLowerCase() === "yolo").annotated_image_base64}`
                        : src)
                    : (results[idx].model && results[idx].model.toLowerCase() === "yolo" && results[idx].annotated_image_base64
                        ? `data:image/jpeg;base64,${results[idx].annotated_image_base64}`
                        : src)
                }
                alt={`Folder img ${idx}`}
                className="preview-img"
              />
              
             
              {model === 'All' && Array.isArray(results[idx]) ? (
                results[idx].map((modelResult, modelIdx) => (
                  <div key={modelIdx} className={`result-tag ${modelResult.hasDefect ? 'defect' : 'no-defect'}`}>
                    <strong>Model:</strong> {modelResult.model}<br />
                    <strong>Status:</strong> {modelResult.hasDefect ? 'Defect Found' : 'No Defect Found'}<br />
                    {modelResult.hasDefect && modelResult.predicted_class && (
                      <>
                        <strong>Defect Type:</strong>{' '}
                        {Array.isArray(modelResult.predicted_class)
                          ? [...new Set(modelResult.predicted_class)].join(', ')
                          : [...new Set(modelResult.predicted_class.split(',').map(s => s.trim()))].join(', ')
                        }
                        <br />

                      </>
                    )}
                    {typeof modelResult.confidence === 'number' && modelResult.hasDefect && (
                    <>
                      <strong>Confidence:</strong> {(modelResult.confidence * 100).toFixed(2)}%<br />
                    </>
                    )}

                    <strong>Processed At:</strong> {new Date().toLocaleString()}<br />
                  </div>
                ))
              ) : (
                
                <div className={`result-tag ${results[idx].hasDefect ? 'defect' : 'no-defect'}`}>
                  <strong>Status:</strong> {results[idx].hasDefect ? 'Defect Found' : 'No Defect Found'}<br />
                  {results[idx].hasDefect && results[idx].predicted_class && (
                    <>
                      <strong>Defect Type:</strong>{' '}
                      {Array.isArray(results[idx].predicted_class)
                        ? [...new Set(results[idx].predicted_class)].join(', ')
                        : [...new Set(results[idx].predicted_class.split(',').map(s => s.trim()))].join(', ')
                      }
                      <br />
                    </>
                  )}
                  {typeof results[idx].confidence === 'number' && results[idx].hasDefect && (
                    <>
                      <strong>Confidence:</strong> {(results[idx].confidence * 100).toFixed(2)}%<br />
                    </>
                    )}
                  <strong>Processed At:</strong> {new Date().toLocaleString()}<br />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
