# CNN-RNN: Multi-Label Image Classification

A full-stack Deep Learning application that detects multiple objects in a single image using a hybrid **CNN-RNN architecture** (based on Wang et al., CVPR 2016). 

This project bridges the gap between complex machine learning architecture and a clean, user-facing web product.

---

## 📁 Repository Structure

    cnn-rnn-portfolio/
    │
    ├── backend/                  # PyTorch Model & FastAPI Server
    │   ├── app.py                # Main API routing and CORS configuration
    │   ├── inference.py          # Image transformation and prediction logic
    │   ├── model.py              # CNN-RNN architecture classes (VGG16 + LSTM)
    │   ├── requirements.txt      # Python dependencies
    │   └── weights/              
    │       └── finalmodel.pth  # Trained model weights (Git LFS recommended)
    │
    ├── frontend/                
    │   ├── index.html            # Webpage structure
    │   ├── script.js             # API integration and DOM manipulation
    │   └── style.css             # UI styling
    │
    └── README.md                 # Project documentation

---

## 💻 How to Run Locally (Development)

If you want to clone this repository and run it on your local machine, follow these steps:

### 1. Backend Setup
Open a terminal, navigate to the `backend` folder, and install the required Python dependencies:

    cd backend
    pip install -r requirements.txt

*Note: Ensure your trained model weights (`finalmodel.pth`) are placed in the `backend/weights/` directory.*

Start the FastAPI server:

    uvicorn app:app --reload

*The API is now active at `http://localhost:8000`.*

### 2. Frontend Setup
Open a **new** terminal, navigate to the `frontend` folder, and start a lightweight local web server:

    cd frontend
    python -m http.server 3000

*Navigate your browser to `http://localhost:3000` to interact with the application.*

---

## 🧠 Architecture Details
- **Vision Backbone:** Frozen VGG-16 (extracts 4096-dimensional spatial features while preserving ImageNet semantics).
- **Sequence Modeler:** LSTM (Hidden Size: 512, Joint Embedding Space: 64).
- **Decoding Mechanism:** Autoregressive Beam Search (Beam Width = 2) for dynamic sequence termination.
- **Dataset:** PASCAL VOC 2007 (20 object classes).

## 🛠️ Tech Stack
- **Machine Learning:** PyTorch, Scikit-Learn
- **Backend API:** FastAPI, Uvicorn
- **Frontend UI:** HTML5, CSS3, Vanilla JavaScript