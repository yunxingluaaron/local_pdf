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
import InfoIcon from '@mui/icons-material/Info';
import axios from 'axios';

const BACKEND_URL = 'http://127.0.0.1:8000';
const WINDOW_HEIGHT = 'calc(100vh - 200px)';

const ChatInterfaceComponent = ({ pdfFiles }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedPdfFile, setSelectedPdfFile] = useState('all');
  const [error, setError] = useState(null);
  const [patentCount, setPatentCount] = useState(5);
  
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handlePdfSelect = (event) => {
    const selectedValue = event.target.value;
    setSelectedPdfFile(selectedValue);
    setError(null);
  };

  const handlePatentCountChange = (event) => {
    const value = parseInt(event.target.value, 10);
    setPatentCount(value);
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
      const pdfSelection = selectedPdfFile === 'all' 
        ? 'all'
        : selectedPdfFile;

      const requestPayload = {
        message: newMessage,
        selected_pdf: pdfSelection,
        theme_id: "default",
        patent_count: parseInt(patentCount, 10),
        mode: "default",
        search_type: "quick"
      };

      const response = await axios.post(`${BACKEND_URL}/chat`, requestPayload);
      
      if (!response.data) {
        throw new Error('Empty response from server');
      }

      if (response.data.error) {
        throw new Error(response.data.error);
      }

      // Handle the response text, ensuring it's a string
      const responseText = typeof response.data.response === 'string' 
        ? response.data.response 
        : JSON.stringify(response.data.response);

      // Process sources array if it exists
      const sources = Array.isArray(response.data.sources) 
        ? response.data.sources 
        : [];

      const botMessage = {
        text: responseText,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        sources: sources.map(source => ({
          ...source,
          patent_id: String(source.patent_id),
          chunk_index: Number(source.chunk_index),
          final_score: Number(source.final_score || source.score || 0)
        }))
      };

      setMessages(prev => [...prev, botMessage]);
      
    } catch (error) {
      console.error('Error sending message:', error);
      
      let errorMessage = 'An unknown error occurred';
      
      if (error.response?.status === 422) {
        const validationErrors = error.response.data;
        if (Array.isArray(validationErrors)) {
          errorMessage = validationErrors
            .map(err => err.msg)
            .join('\n');
        } else if (typeof validationErrors === 'object') {
          errorMessage = validationErrors.detail || validationErrors.msg || JSON.stringify(validationErrors);
        }
      } else if (error.response?.data) {
        const errorData = error.response.data;
        if (typeof errorData === 'string') {
          errorMessage = errorData;
        } else if (typeof errorData === 'object') {
          if (errorData.msg) {
            errorMessage = typeof errorData.msg === 'string' ? errorData.msg : JSON.stringify(errorData.msg);
          } else if (errorData.detail) {
            errorMessage = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
          } else if (errorData.error) {
            errorMessage = typeof errorData.error === 'string' ? errorData.error : JSON.stringify(errorData.error);
          }
        }
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setMessages(prev => [...prev, {
        text: errorMessage,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        error: true
      }]);
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const getMessageContent = (message) => {
    if (!message) return '';
    
    let content = message.text;
  
    if (typeof content === 'string') {
      return content;
    }
  
    if (content && typeof content === 'object') {
      if (content.msg) {
        return typeof content.msg === 'string' ? content.msg : JSON.stringify(content.msg, null, 2);
      }
      if (content.detail) {
        return typeof content.detail === 'string' ? content.detail : JSON.stringify(content.detail, null, 2);
      }
      if (content.type && content.loc && content.msg) {
        const errorMsg = typeof content.msg === 'string' ? content.msg : JSON.stringify(content.msg, null, 2);
        return `Error: ${errorMsg}`;
      }
  
      try {
        return JSON.stringify(content, null, 2);
      } catch {
        return 'Unable to display message content';
      }
    }
  
    return content?.toString() || '';
  };

  const renderMessage = (message) => {
    const messageContent = getMessageContent(message);

    return (
      <Paper
        sx={{
          p: 2,
          maxWidth: '70%',
          backgroundColor: message.error ? '#ffebee' : 
                          message.sender === 'user' ? '#e3f2fd' : '#f5f5f5',
          wordBreak: 'break-word'
        }}
      >
        <Box>
          {message.error ? (
            <Alert severity="error" sx={{ mb: 1 }}>
              {messageContent}
            </Alert>
          ) : (
            <Box sx={{
              '& .markdown': {
                '& pre': {
                  backgroundColor: '#f8f9fa',
                  padding: '16px',
                  borderRadius: '4px',
                  overflow: 'auto',
                  '& code': {
                    backgroundColor: 'transparent',
                    padding: 0
                  }
                },
                '& code': {
                  backgroundColor: '#f8f9fa',
                  padding: '2px 4px',
                  borderRadius: '4px',
                  fontSize: '0.875em'
                },
                '& p': {
                  marginTop: '8px',
                  marginBottom: '8px',
                  lineHeight: 1.6,
                  '&:first-of-type': {
                    marginTop: 0
                  },
                  '&:last-child': {
                    marginBottom: 0
                  }
                },
                '& ul, & ol': {
                  paddingLeft: '24px',
                  marginTop: '8px',
                  marginBottom: '8px'
                },
                '& li': {
                  marginBottom: '4px'
                },
                '& blockquote': {
                  borderLeft: '4px solid #e0e0e0',
                  margin: '16px 0',
                  padding: '8px 16px',
                  color: '#666'
                },
                '& h1, & h2, & h3, & h4, & h5, & h6': {
                  margin: '16px 0 8px 0',
                  lineHeight: 1.4,
                  fontWeight: 600
                },
                '& h1': { fontSize: '1.5em' },
                '& h2': { fontSize: '1.3em' },
                '& h3': { fontSize: '1.2em' },
                '& a': {
                  color: '#2196f3',
                  textDecoration: 'none',
                  '&:hover': {
                    textDecoration: 'underline'
                  }
                },
                '& table': {
                  borderCollapse: 'collapse',
                  width: '100%',
                  margin: '16px 0',
                  '& th, & td': {
                    border: '1px solid #e0e0e0',
                    padding: '8px 12px',
                    textAlign: 'left'
                  },
                  '& th': {
                    backgroundColor: '#f8f9fa',
                    fontWeight: 600
                  }
                }
              }
            }}>
              <ReactMarkdown className="markdown">
                {messageContent}
              </ReactMarkdown>
            </Box>
          )}
        </Box>
{/*         
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
                    sx={{ cursor: 'pointer' }}
                  />
                </Tooltip>
              );
            })}
          </Box>
        )} */}
      </Paper>
    );
  };

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

      <Box sx={{ 
        display: 'flex', 
        gap: 2, 
        width: '100%', 
        flexShrink: 0,
        alignItems: 'center'
      }}>
        <FormControl sx={{ flex: 1 }}>
          <InputLabel id="pdf-select-label">Select PDF Document</InputLabel>
          <Select
            labelId="pdf-select-label"
            value={selectedPdfFile}
            onChange={handlePdfSelect}
            label="Select PDF Document"
            sx={{ backgroundColor: 'background.paper' }}
          >
            <MenuItem value="all">All PDFs</MenuItem>
            {pdfFiles?.map((pdf, index) => (
              <MenuItem key={index} value={pdf.name}>
                {pdf.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl sx={{ width: 200 }}>
          <InputLabel id="patent-count-label">Number of Patents</InputLabel>
          <Select
            labelId="patent-count-label"
            value={patentCount}
            onChange={handlePatentCountChange}
            label="Number of Patents"
            sx={{ backgroundColor: 'background.paper' }}
          >
            {[...Array(10)].map((_, i) => (
              <MenuItem key={i + 1} value={i + 1}>
                {i + 1} Patent{i !== 0 ? 's' : ''}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Tooltip title="Select 'All PDFs' to search across all documents, or choose a specific PDF to focus your search.">
          <IconButton size="small" color="primary">
            <InfoIcon />
          </IconButton>
        </Tooltip>
      </Box>

      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column',
        height: WINDOW_HEIGHT,
        width: '100%'
      }}>
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
</Box>
);
};

export default ChatInterfaceComponent;