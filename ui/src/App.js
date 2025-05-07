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

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResults([]);
      setFolderImages([]);
      setFolderPreviews([]);
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
    }
  };

  const handleDetect = async () => {
    if (!image && folderImages.length === 0) {
      alert('Please upload an image or folder first.');
      return;
    }

    const formData = new FormData();
    formData.append('model_name', model);

    if (image) {
      formData.append('image', image);
    } else {
      folderImages.forEach(file => {
        formData.append('images', file);
      });
    }

    try {
      const endpoint = image ? '/api/detect' : '/api/detect/batch';
      const response = await axios.post(`https://laptopdefectdetection-80r3.onrender.com${endpoint}`, formData);
      const data = image ? [response.data] : response.data;
      setResults(data);
    } catch (error) {
      console.error('Error during detection:', error);
      alert('Detection failed.');
    }
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <div className="app-container">
      <div className="card">
        <h2>Laptop Defect Detection</h2>

        <label className="label">Select Model:</label>
        <select
          className="dropdown"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="ResNet-50">ResNet-50</option>
          <option value="YOLO">YOLOv8</option>
          <option value="MobileNet">MobileNetV2</option>
        </select>

        <div className="upload-section">
          <label className="upload-label">Upload Image:</label>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="upload-input"
          />

          <label className="upload-label">Upload Folder:</label>
          <input
            type="file"
            webkitdirectory="true"
            directory="true"
            multiple
            onChange={handleFolderUpload}
            className="upload-input"
          />
        </div>

        <button className="detect-button" onClick={handleDetect}>
          Detect Defects
        </button>

        <div className="preview-grid">
          
          {imagePreview && results.length > 0 && (
            <div className="preview-with-result">
              <img src={imagePreview} alt="Uploaded" className="preview-img" />
              <div className={`result-tag ${results[0].hasDefect ? 'defect' : 'no-defect'}`}>
                <strong>Status:</strong> {results[0].hasDefect ? 'Defect Found' : 'No Defect Found'}<br />
                {(results[0].hasDefect === false && results[0].model === "YOLO") ? null:(
                  <>
                    <strong>Confidence:</strong> {(results[0].confidence * 100).toFixed(2)}%<br />
                  </>
                )}
                <strong>Processed At:</strong> {new Date().toLocaleString()}<br />
              </div>
            </div>
          )}
          {folderPreviews.length > 0 && results.length === folderPreviews.length && folderPreviews.map((src, idx) => (
            <div key={idx} className="preview-with-result">
              <img src={src} alt={`Folder img ${idx}`} className="preview-img" />
              <div className={`result-tag ${results[idx].hasDefect ? 'defect' : 'no-defect'}`}>
                <strong>Status:</strong> {results[idx].hasDefect ? 'Defect Found' : 'No Defect Found'}<br />
                {(!results[idx].hasDefect && results[idx].model === "YOLO") ? null:(
                  <>
                    <strong>Confidence:</strong> {(results[idx].confidence * 100).toFixed(2)}%<br />
                  </>
                )}
                <strong>Processed At:</strong> {new Date().toLocaleString()}<br />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;

