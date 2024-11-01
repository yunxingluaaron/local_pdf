import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { 
  Box, 
  Paper, 
  TextField, 
  Button, 
  Typography,
  List,
  ListItem,
  CircularProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Tooltip,
  Alert,
  IconButton
} from '@mui/material';
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = 'http://127.0.0.1:8000';
const INITIAL_ZOOM = 1;
const ZOOM_STEP = 0.1;
const WINDOW_HEIGHT = 'calc(100vh - 200px)';



const ChatInterfaceComponent = ({ pdfFiles }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentPdfUrl, setCurrentPdfUrl] = useState(null);
  const [selectedPdfFile, setSelectedPdfFile] = useState('');
  const [error, setError] = useState(null);
  const [pdfDrawerOpen, setPdfDrawerOpen] = useState(true);
  const [zoom, setZoom] = useState(INITIAL_ZOOM);
  
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handlePdfSelect = async (event) => {
    setLoading(true);
    try {
      const selectedUrl = event.target.value;
      if (selectedUrl && !pdfFiles.some(pdf => pdf.url === selectedUrl)) {
        throw new Error('Invalid PDF selection');
      }
      setSelectedPdfFile(selectedUrl);
      setCurrentPdfUrl(selectedUrl);
      setError(null);
    } catch (err) {
      setError('Error selecting PDF file');
      console.error('PDF selection error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const handleSendMessage = async () => {
    if (!newMessage.trim()) return;

    const userMessage = {
      text: newMessage,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setNewMessage('');
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${BACKEND_URL}/chat`, {
        message: newMessage,
        theme_id: "default"
      });

      if (!response.data || !response.data.response) {
        throw new Error('Invalid response from server');
      }

      const botMessage = {
        text: response.data.response,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        sources: response.data.sources || []
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = typeof error === 'object' ? 
        (error.response?.data?.detail || error.message || 'An error occurred') : 
        String(error);
      
      setError(errorMessage);
      setMessages(prev => [...prev, {
        text: errorMessage,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        error: true
      }]);
    } finally {
      setLoading(false);
    }
  };

  const renderMessage = (message) => {
    return (
      <Paper
        sx={{
          p: 2,
          maxWidth: '70%',
          backgroundColor: message.error ? '#ffebee' : 
                          message.sender === 'user' ? '#e3f2fd' : '#f5f5f5',
        }}
      >
        <Box 
          sx={{
            '& .markdown': {
              '& pre': {
                backgroundColor: '#f5f5f5',
                padding: '8px',
                borderRadius: '4px',
                overflowX: 'auto',
                '& code': {
                  backgroundColor: 'transparent',
                  padding: 0
                }
              },
              '& code': {
                backgroundColor: '#f5f5f5',
                padding: '2px 4px',
                borderRadius: '4px',
                fontSize: '0.875em',
                fontFamily: 'monospace'
              },
              '& p': {
                marginTop: '8px',
                marginBottom: '8px',
                '&:first-of-type': {
                  marginTop: 0
                },
                '&:last-child': {
                  marginBottom: 0
                }
              },
              '& ul, & ol': {
                paddingLeft: '20px',
                marginTop: '8px',
                marginBottom: '8px'
              },
              '& blockquote': {
                borderLeft: '4px solid #ddd',
                margin: '16px 0',
                padding: '0 16px',
                color: '#666'
              },
              '& table': {
                borderCollapse: 'collapse',
                width: '100%',
                marginTop: '8px',
                marginBottom: '8px',
                '& th, & td': {
                  border: '1px solid #ddd',
                  padding: '8px',
                  textAlign: 'left'
                },
                '& th': {
                  backgroundColor: '#f5f5f5'
                }
              }
            }
          }}
        >
          <ReactMarkdown className="markdown">
            {String(message.text)}
          </ReactMarkdown>
        </Box>
        
        {message.sender === 'bot' && message.sources && message.sources.length > 0 && (
          <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {message.sources.map((source, idx) => {
              const fileName = source.patent_id.split('/').pop();
              return (
                <Tooltip 
                  key={idx}
                  title={`Relevance score: ${(source.final_score * 100).toFixed(1)}%`}
                  placement="top"
                >
                  <Chip
                    label={`${fileName} - Chunk ${source.chunk_index}`}
                    size="small"
                    color="primary"
                    variant="outlined"
                    onClick={() => {
                      const pdfFile = pdfFiles.find(pdf => 
                        pdf.name === source.patent_id || 
                        pdf.name === fileName
                      );
                      if (pdfFile) {
                        setSelectedPdfFile(pdfFile.url);
                        setCurrentPdfUrl(pdfFile.url);
                      } else {
                        setError(`Could not find PDF file: ${fileName}`);
                      }
                    }}
                    disabled={loading}
                    sx={{ cursor: 'pointer' }}
                  />
                </Tooltip>
              );
            })}
          </Box>
        )}
      </Paper>
    );
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + ZOOM_STEP, 2));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - ZOOM_STEP, 0.5));
  };

  const togglePdfDrawer = () => {
    setPdfDrawerOpen(prev => !prev);
  };

  // ... Rest of the JSX remains the same ...
  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column',
      height: '100vh', 
      gap: 2, 
      p: 2,
      overflow: 'hidden'
    }}>
      {error && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* PDF Select Control */}
      <Box sx={{ width: '100%', flexShrink: 0 }}>
        <FormControl fullWidth>
          <InputLabel id="pdf-select-label">Select PDF Document</InputLabel>
          <Select
            labelId="pdf-select-label"
            value={selectedPdfFile}
            onChange={handlePdfSelect}
            label="Select PDF Document"
            sx={{ backgroundColor: 'background.paper' }}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            {pdfFiles?.map((pdf, index) => (
              <MenuItem key={index} value={pdf.url}>
                {pdf.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {/* Main Content Area - 50/50 Split */}
      <Box sx={{ 
        display: 'flex', 
        gap: 2,
        position: 'relative',
        height: WINDOW_HEIGHT,
        flexShrink: 0
      }}>
        {/* Chat Section - Fixed 50% */}
        <Box sx={{ 
          width: '50%',
          display: 'flex', 
          flexDirection: 'column',
          height: '100%',
          minWidth: 0 // Allow flex shrinking
        }}>
          {/* Messages Container */}
          <Paper 
            ref={chatContainerRef}
            sx={{ 
              flex: 1,
              mb: 2, 
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              '& > .scroll-container': {
                flex: 1,
                overflowY: 'auto',
                overflowX: 'hidden',
                '&::-webkit-scrollbar': {
                  width: '8px',
                },
                '&::-webkit-scrollbar-track': {
                  background: '#f1f1f1',
                  borderRadius: '4px',
                },
                '&::-webkit-scrollbar-thumb': {
                  background: '#888',
                  borderRadius: '4px',
                },
                '&::-webkit-scrollbar-thumb:hover': {
                  background: '#555',
                },
              }
            }}
          >
            <Box className="scroll-container" sx={{ p: 2 }}>
              <List>
                {messages.map((message, index) => (
                  <ListItem
                    key={index}
                    sx={{
                      justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
                      mb: 1,
                      px: 1
                    }}
                  >
                    {renderMessage(message)}
                  </ListItem>
                ))}
                <div ref={messagesEndRef} />
              </List>
            </Box>
          </Paper>

          {/* Input Area */}
          <Box sx={{ 
            display: 'flex', 
            gap: 1,
            flexShrink: 0
          }}>
            <TextField
              fullWidth
              variant="outlined"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              disabled={loading}
              multiline
              maxRows={4}
              sx={{
                backgroundColor: 'background.paper',
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2
                }
              }}
            />
            <Button 
              variant="contained"
              onClick={handleSendMessage}
              disabled={loading || !newMessage.trim()}
              sx={{ 
                minWidth: '100px',
                borderRadius: 2
              }}
            >
              {loading ? <CircularProgress size={24} /> : 'Send'}
            </Button>
          </Box>
        </Box>

        {/* PDF Viewer Section - Fixed 50% */}
        <Paper sx={{ 
          width: '50%',
          height: '100%',
          overflow: 'hidden',
          borderRadius: 2,
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'background.paper'
        }}>
          {/* PDF Controls */}
          <Box sx={{
            position: 'absolute',
            top: 8,
            right: 8,
            zIndex: 2,
            bgcolor: 'background.paper',
            borderRadius: 1,
            boxShadow: 1,
            display: 'flex',
            gap: 1,
            p: 1
          }}>
            <IconButton onClick={handleZoomOut} size="small">
              <ZoomOut />
            </IconButton>
            <IconButton onClick={handleZoomIn} size="small">
              <ZoomIn />
            </IconButton>
          </Box>

          {/* PDF Content */}
          {loading ? (
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              flex: 1
            }}>
              <CircularProgress />
            </Box>
          ) : currentPdfUrl ? (
            <Box sx={{
              flex: 1,
              overflow: 'auto',
              '&::-webkit-scrollbar': {
                width: '8px',
                height: '8px'
              },
              '&::-webkit-scrollbar-track': {
                background: '#f1f1f1',
                borderRadius: '4px',
              },
              '&::-webkit-scrollbar-thumb': {
                background: '#888',
                borderRadius: '4px',
              },
              '&::-webkit-scrollbar-thumb:hover': {
                background: '#555',
              },
            }}>
              <iframe
                src={`${currentPdfUrl}#zoom=${zoom}`}
                style={{
                  width: '100%',
                  height: '100%',
                  border: 'none',
                  transform: `scale(${zoom})`,
                  transformOrigin: 'top left',
                  transition: 'transform 0.2s ease'
                }}
                title="PDF Viewer"
                onLoad={() => setLoading(false)}
              />
            </Box>
          ) : (
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              flex: 1
            }}>
              <Typography color="text.secondary">
                Select a PDF document to view its content
              </Typography>
            </Box>
          )}
        </Paper>
      </Box>
    </Box>
  );
};

export default ChatInterfaceComponent;

