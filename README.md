# Activity Essentials Finder

![License](https://img.shields.io/badge that helps users identify essential items needed for various activities by combining computer vision with AI-powered recommendations.

## Overview

Activity Essentials Finder is a Flask-based application that uses advanced AI technologies to:

1. Analyze uploaded images to detect objects
2. Understand the context of planned activities
3. Recommend essential items based on the activity and context
4. Show which recommended items are already present in the uploaded image

The system leverages a knowledge graph to store relationships between activities and required items, while using computer vision to identify objects in user-uploaded images.

## Features

- **Image Analysis**: Uses YOLOv8 object detection to identify items in uploaded images
- **Activity Context Extraction**: Analyzes user input to understand activity type and context
- **Intelligent Recommendations**: Suggests essential items based on activity type and contextual factors
- **Knowledge Graph Storage**: Stores relationships between activities, contexts, and items in Neo4j
- **Automatic Learning**: Adds new activities and item relationships to the database when encountered
- **Voice Input Support**: Allows users to describe activities using speech recognition
- **Dark/Light Theme Toggle**: Supports user preference for interface appearance
- **Responsive Design**: Works across various device sizes and orientations

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **Computer Vision**: YOLOv8 (with worldv2 weights), OpenCV, Supervision
- **Natural Language Processing**: OpenAI API via OpenRouter
- **Database**: Neo4j Graph Database
- **Styling**: CSS with animations, glassmorphism UI, and responsive design

## Installation

### Prerequisites

- Python 3.8+
- Neo4j Database
- OpenRouter API key
- YOLOv8 model weights

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/activity-essentials-finder.git
   cd activity-essentials-finder
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following variables:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=your_neo4j_username
   NEO4J_PASSWORD=your_neo4j_password
   OPENROUTER_API_KEY=your_openrouter_api_key
   OPENROUTER_MODEL=mistralai/mistral-small-3.1-24b-instruct:free
   YOLO_MODEL_PATH=path_to_yolov8x-worldv2.pt
   ```

4. Start the Neo4j database

5. Run the application:
   ```bash
   python app.py
   ```

6. Open your browser and navigate to `http://localhost:5050`

## Usage

1. Upload an image related to your planned activity
2. Enter a description of what you're planning to do (e.g., "Hiking in winter")
3. Click "Find Essentials" to process your request
4. View the recommended essential items for your activity
5. See which items were detected in your uploaded image
6. Examine the annotated image showing detected objects

## How It Works

1. The system extracts the activity name and context from the user's description
2. It queries the Neo4j database to find essential items for that activity and context
3. If the activity isn't in the database, it uses an LLM to generate recommendations and stores them
4. The YOLOv8 model is configured to detect only the essential items in the user's image
5. Results are displayed showing both recommended items and detected items
6. An annotated version of the image is shown with bounding boxes around detected items

## Project Structure

```
activity-essentials-finder/
├── app.py                # Main Flask application
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates
│   └── interactive-assist.html
├── .env                  # Environment variables
└── README.md             # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 for object detection capabilities
- Neo4j for graph database functionality
- OpenRouter for API access to language models
- Supervision library for annotation tools

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/5324675/68d1f5ac-a241-4794-be9e-15f15fd62c73/paste.txt

---
Answer from Perplexity: pplx.ai/share
