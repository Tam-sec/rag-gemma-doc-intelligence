# Happy Life Document Intelligence

Advanced AI-powered document analysis system with natural language querying capabilities.

## 🌟 Features

### 📄 Document Processing
- PDF document upload and processing
- Automatic text extraction and segmentation
- Chunk-based processing with semantic understanding
- Document statistics generation (pages, characters, words)

### 🤖 AI-Powered Analysis
- Natural language question answering about document content
- Retrieval-Augmented Generation (RAG) architecture
- Multiple AI model support (Gemma, Llama, Mistral)
- Adjustable creativity/temperature settings

### 📊 Results & Insights
- Confidence scoring for AI responses
- Document segment relevance assessment
- Processing time metrics
- Document history tracking

### 🎨 User Interface
- Modern, intuitive web interface
- Real-time progress indicators
- Responsive design
- Customizable settings

## ⚙️ Installation

1. **Prerequisites**:
   - Python 3.8+
   - Ollama installed and running (for local AI models)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt```
3. Download AI models (if not already present):
   ```
   ollama pull ollamaced/gemma3_1b_spiders
ollama pull llama3
ollama pull mistral
   ```
## 🚀 Usage
1. Run the application :
   
   ```
   python app.py
   ```
2. Access the web interface at the provided local URL (typically http://localhost:7860 )
3. Basic workflow :
   
   - Upload a PDF document
   - View document statistics
   - Ask questions about the document content
   - Explore previous documents in the history
## 🛠️ Advanced Configuration
### Environment Variables
- OLLAMA_HOST : Set custom Ollama host (default: http://localhost:11434 )
- CHROMA_DB_PATH : Customize vector database storage location
### Model Selection
Choose from available AI models in the "AI Model Selection" dropdown:

- ollamaced/gemma3_1b_spiders (default)
- llama3
- mistral
### Temperature Control
Adjust the "Creativity Level" slider to control response creativity:

- Lower values (0.1-0.- 3): More factual, conservative responses
- Medium values (0.4-0.7): Balanced responses (default)
- Higher values (0.8-1.0): More creative, varied responses
## 📂 File Structure
```
──app.py               # Main application code
├──README.md            # This documentation
├──requirements.txt     # Python dependencies
├──chroma_db/           # Vector database storage
└──document_history/    # Processed document metadata
```
## 🤝 Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch ( git checkout -b feature/your-feature )
3. Commit your changes ( git commit -am 'Add some feature' )
4. Push to the branch ( git push origin feature/your-feature )
5. Open a Pull Request
## 📜 License
This project is licensed under the MIT License.

## 📞 Support
For issues or questions, please open an issue on the GitHub repository.

## 🔮 Future Enhancements
- Multi-document comparison
- Voice interface
- Enhanced visualization
- Export capabilities
- Multi-language support