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
    │       └── finalmodel.pth    # Trained model weights (Get downloaded from Release notes )
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

---

## 📐 The Mathematical Foundation (Research Implementation)

Standard multi-label classification often assumes object independence (Binary Relevance), which ignores real-world context (e.g., a "person" is more likely to appear with a "bicycle" than a "boat"). This project implements the CNN-RNN framework to capture **label dependency** by modeling image classification as a sequential prediction task.

### 1. Joint Image-Label Embedding
Let the image feature vector be $I \in \mathbb{R}^{4096}$ (extracted via VGG-16) and the sequence of true labels be $Y = (y_1, y_2, ..., y_n)$. Both the image and the text labels are projected into a shared latent space.

### 2. Sequential Probability Modeling
Instead of predicting all classes simultaneously, the model factors the joint probability using the chain rule. The probability of detecting object $y_t$ relies on both the global image features and the history of previously detected objects:
$$P(Y|I) = \prod_{t=1}^{n} P(y_t | y_1, ..., y_{t-1}, I)$$

### 3. LSTM State Updates
The Long Short-Term Memory (LSTM) network maintains a hidden state $h_t$ that acts as a structural memory. At each time step $t$, the state is updated based on the previously emitted label $y_{t-1}$:
$$h_t = f_{LSTM}(E[y_{t-1}], h_{t-1})$$
Where $E$ represents the label embedding matrix.

### 4. The Scoring Function
To predict the next label in the sequence, the network computes a logit score $s_{t,c}$ for every class $c$ in the vocabulary. The score combines the static image projection and the dynamic LSTM hidden state:
$$s_{t,c} = W_c^\top (U I + V h_t)$$
* $U$: Image projection matrix.
* $V$: Hidden state projection matrix.
* $W_c$: The embedding weight for class $c$.

The sequence is generated autoregressively until the network outputs a dedicated `[END]` token, allowing for a dynamic, variable-length number of predictions per image.

---

## 🛠️ Tech Stack
- **Machine Learning:** PyTorch, Scikit-Learn
- **Backend API:** FastAPI, Uvicorn
- **Frontend UI:** HTML5, CSS3, Vanilla JavaScript
