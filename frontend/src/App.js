// App.js
import React, { useState } from 'react';
import { Container, Box, Paper, Typography } from '@mui/material';
import PDFUploadComponent from './components/PDFUpload';
import ChatInterfaceComponent from './components/ChatInterface';

function App() {
  const [uploadedPdfs, setUploadedPdfs] = useState([]);

  const handleUploadComplete = (files) => {
    setUploadedPdfs(prev => [...prev, ...files]);
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          PDF Chat Assistant
        </Typography>
        
        <PDFUploadComponent onUploadComplete={handleUploadComplete} />
        
        {uploadedPdfs.length > 0 && (
          <Box sx={{ mt: 4 }}>
            <ChatInterfaceComponent pdfFiles={uploadedPdfs} />
          </Box>
        )}
      </Box>
    </Container>
  );
}

export default App;