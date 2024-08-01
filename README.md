Unlocking Corporate Insights with AI: Your Interactive Financial Analyst

Imagine having an AI-powered analyst that instantly distills the most critical information from a company's financial statements, news, and market trends. That's exactly what we're building today!

•	In this codelab, you'll harness the power of Google Cloud and cutting-edge AI to create GC CorpSpectrum, an interactive tool that transforms complex financial data into actionable insights. We'll guide you through:

•	Data Fusion: Seamlessly gather and process information from diverse sources like SEC filings, news articles, and financial databases.

•	AI-Driven Analysis: Extract key financial metrics, uncover market trends, and gauge sentiment using state-of-the-art natural language processing models.

•	Interactive Chatbot: Engage in a conversation with your data, asking questions and receiving insightful summaries tailored to your needs.

•	By the end, you'll have a powerful tool to navigate the corporate landscape, all while gaining hands-on experience with Google Cloud's most innovative technologies. Let's dive in!


Getting Started
 * Prerequisites:
   * A Google Cloud Project with enabled billing.
   * API keys for Google Custom Search Engine and Alpha Vantage.
   * (Optional) A Hugging Face API token if you are not using the default Sentence Transformer model.
 * Installation:
   git clone <repository_url>
cd GC_CorpSpectrum
pip install -r requirements.txt 

 * Setup:
   * Set up your Google Cloud credentials and project ID.
   * Replace placeholders for API keys in the code.
   * Upload your annual statement PDF to the designated Google Cloud Storage bucket.
 * Run the Chatbot:
   python main.py

   Follow the prompts to interact with the chatbot.
Example Usage
Enter Company Name: Tesla
Upload Annual Statement (pdf file): /path/to/your/annual_statement.pdf
Enter query: What are Tesla's key financial highlights?

Future Enhancements
 * Integration of alternative data sources (social media, satellite imagery).
 * Enhanced risk analysis capabilities.
 * Personalized investment recommendation engine.

About Code

Data Flow:
•	User inputs company name and uploads PDF to Google Cloud Storage.
•	Gemini Pro extracts company details.
•	Google Custom Search Engine fetches industry news.
•	Alpha Vantage API fetches market data and news sentiment.
•	TextBlob analyzes sentiment of news.
•	PDF is processed into chunks and embeddings.
•	Embeddings are stored in ChromaDB.
•	User query is combined with context from company details, news, sentiment, and relevant PDF chunks.
•	Combined context is passed to Gemini Pro.
•	Gemini Pro generates a response based on the context.
•	Response is displayed to the user.

Libraries:
a.	Google Cloud:
•	google-cloud-storage: For interacting with Google Cloud Storage.
b.	Vertex AI:
•	vertexai: The core library for accessing Vertex AI services.
•	vertexai.preview.generative_models: Specific tools for working with generative AI models.
•	vertexai.preview.language_models: Tools for text generation models.
c.	Language Processing & Embeddings:
•	langchain: Framework for building LLM applications.
•	langchain-community: Additional components for LangChain.
•	langchain-google-genai: Integration for Google's generative AI.
•	sentence-transformers: Creating sentence embeddings.
•	HuggingFaceEmbeddings: Integrating Hugging Face models into LangChain.
d.	Data Storage:
•	chromadb: Vector database for storing and searching embeddings.
e.	PDF Processing:
•	PyPDF2: For loading and extracting text from PDF files.
f.	Google APIs:
google-api-python-client: Client library for interacting with Google APIs.
g.	Financial Data:
•	alpha_vantage: API for accessing financial market data.
•	yfinance: Library for fetching financial data from Yahoo Finance.
h.	Sentiment Analysis:
•	textblob: Library for performing sentiment analysis.
i.	Other:
•	smart_open: For working with various file-like objects.
•	tiktoken: BPE tokenizer often used with OpenAI models.

Tools:
•	Google Colab: The development environment where you're running the code.
•	Vertex AI Workbench: (Implicitly used) For managing and deploying models.
•	Google Cloud Storage: For storing input files and model data.
•	Hugging Face Hub: Repository of pre-trained models.

APIs:
•	Google Custom Search Engine API: For fetching news and trends.
•	Alpha Vantage API: For fetching financial market data and news sentiment.

Google Cloud Elements:
•	Vertex AI: The platform for building, deploying, and managing machine learning models.
•	Generative AI models: Gemini Pro is used for text generation and understanding.
•	Google Cloud Storage: Cloud-based storage for your data.
•	Authentication: Colab's authentication mechanism is used to access your Google Cloud resources securely.



Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.
License
This project is licensed under the [LICENSE NAME].
 * https://github.com/SheetalPrasad23
