import React, { useState } from 'react';
import { 
  Button, 
  Box, 
  LinearProgress, 
  Typography, 
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Paper
} from '@mui/material';
import { Delete as DeleteIcon, CheckCircle as CheckCircleIcon } from '@mui/icons-material';
import axios from 'axios';

const BACKEND_URL = 'http://127.0.0.1:8000';
const MAX_FILE_SIZE = 3 * 1024 * 1024; // 3MB in bytes
const MAX_FILES = 3;

const PDFUploadComponent = ({ onUploadComplete }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState({});
  const [processedFiles, setProcessedFiles] = useState([]);

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    const errors = [];
    
    // Check if adding new files would exceed the maximum count
    if (files.length + selectedFiles.length > MAX_FILES) {
      setError(`Maximum ${MAX_FILES} PDFs allowed`);
      return;
    }

    const validFiles = files.filter(file => {
      // Check file size
      if (file.size > MAX_FILE_SIZE) {
        errors.push(`${file.name} is larger than 3MB`);
        return false;
      }
      // Check file type
      if (!file.type.includes('pdf')) {
        errors.push(`${file.name} is not a PDF file`);
        return false;
      }
      return true;
    });

    // If there are any errors, show them all
    if (errors.length > 0) {
      setError(errors.join('. '));
      return;
    }

    // Check if total files would exceed limit
    if (validFiles.length + selectedFiles.length > MAX_FILES) {
      setError(`Cannot add more files. Maximum ${MAX_FILES} PDFs allowed`);
      return;
    }

    setSelectedFiles(prev => [...prev, ...validFiles]);
    setError(null);
  };

  const handleRemoveFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
    setError(null);
  };

  const handleUpload = async () => {
    setUploading(true);
    setError(null);
    const uploadedFiles = [];

    try {
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        const formData = new FormData();
        formData.append('file', file);
        formData.append('theme_id', 'default');

        setUploadProgress(prev => ({
          ...prev,
          [file.name]: 0
        }));

        const response = await axios.post(`${BACKEND_URL}/upload`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          onUploadProgress: (progressEvent) => {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(prev => ({
              ...prev,
              [file.name]: progress
            }));
          }
        });

        if (response.data.success) {
          uploadedFiles.push({
            name: file.name,
            url: `${BACKEND_URL}${response.data.file_url}`,
            pages: response.data.pages_processed
          });
          
          setProcessedFiles(prev => [...prev, {
            name: file.name,
            pages: response.data.pages_processed
          }]);
        }
      }

      onUploadComplete(uploadedFiles);
      setSelectedFiles([]);
      setUploadProgress({});
    } catch (err) {
      console.error('Upload error:', err);
      const errorMessage = typeof err === 'object' ? 
        (err.response?.data?.detail || err.message || 'Upload failed') : 
        String(err);
      setError(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  return (
    <Paper sx={{ p: 3, maxWidth: 800, margin: 'auto', mt: 3 }}>
      <Typography variant="h5" gutterBottom>
        Upload PDFs for Analysis
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Upload up to {MAX_FILES} PDF files (max 3MB each)
        </Typography>
        
        <Button
          variant="contained"
          component="label"
          disabled={uploading || selectedFiles.length >= MAX_FILES}
          sx={{ mr: 2 }}
        >
          Select PDFs
          <input
            type="file"
            hidden
            multiple
            accept=".pdf"
            onChange={handleFileSelect}
          />
        </Button>

        <Button
          variant="contained"
          color="primary"
          onClick={handleUpload}
          disabled={selectedFiles.length === 0 || uploading}
        >
          Upload & Process
        </Button>
      </Box>

      {error && (
        <Alert 
          severity="error" 
          sx={{ mb: 2 }}
          onClose={() => setError(null)}
        >
          {error}
        </Alert>
      )}

      {selectedFiles.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Selected Files: ({selectedFiles.length}/{MAX_FILES})
          </Typography>
          <List>
            {selectedFiles.map((file, index) => (
              <ListItem key={index}>
                <ListItemText 
                  primary={file.name}
                  secondary={`${(file.size / (1024 * 1024)).toFixed(2)} MB`}
                />
                <ListItemSecondaryAction>
                  {uploading && uploadProgress[file.name] ? (
                    <Box sx={{ width: 100 }}>
                      <LinearProgress 
                        variant="determinate" 
                        value={uploadProgress[file.name]} 
                      />
                    </Box>
                  ) : (
                    <IconButton 
                      edge="end" 
                      onClick={() => handleRemoveFile(index)}
                      disabled={uploading}
                    >
                      <DeleteIcon />
                    </IconButton>
                  )}
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </Box>
      )}

      {processedFiles.length > 0 && (
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            Processed Files:
          </Typography>
          <List>
            {processedFiles.map((file, index) => (
              <ListItem key={index}>
                <ListItemText 
                  primary={file.name}
                  secondary={`${file.pages} pages processed`}
                />
                <ListItemSecondaryAction>
                  <CheckCircleIcon color="success" />
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </Box>
      )}
    </Paper>
  );
};

export default PDFUploadComponent;