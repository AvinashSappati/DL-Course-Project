# CNN-RNN: Multi-Label Image Classification

A full-stack Deep Learning application that detects multiple objects in a single image using a hybrid **CNN-RNN architecture**. This repository bridges the gap between complex machine learning architecture ([based on Wang et al., CVPR 2016](https://arxiv.org/pdf/1604.04573)) and a clean, user-facing web interface .

---

## 📊 Dataset: PASCAL VOC 2007
The model is trained and evaluated on the **PASCAL Visual Object Classes (VOC) 2007** dataset, a benchmark in visual object category recognition.
* **Total Images:** 9,963 (Train/Val/Test splits)
* **Classes:** 20 distinct object categories.
* **Categories Include:** * *Vehicles:* Aeroplane, Bicycle, Boat, Bus, Car, Motorbike, Train
  * *Household:* Bottle, Chair, Dining Table, Potted Plant, Sofa, TV/Monitor
  * *Animals/People:* Bird, Cat, Cow, Dog, Horse, Sheep, Person
* **Challenge:** Images frequently contain multiple overlapping classes (e.g., a person riding a horse next to a dog), requiring the model to understand complex semantic context.

---

## 🧠 Architecture Breakdown
The system relies on an encoder-decoder structure to capture both spatial features and label dependencies.

1. **Vision Encoder (CNN):** * **Backbone:** VGG-16 (Pre-trained on ImageNet).
   * **Mechanism:** The final classification layers are removed. The network extracts a high-dimensional spatial feature vector ($4096$-d) from the input image. This acts as the visual "memory" for the model.
2. **Sequence Decoder (RNN):**
   * **Backbone:** Long Short-Term Memory (LSTM) network.
   * **Mechanism:** Takes the CNN feature vector and maps it into a shared joint-embedding space alongside the text labels. It predicts objects sequentially, using its hidden state to remember what objects have already been found.
3. **Inference Engine:**
   * **Beam Search (Width = 2):** Instead of greedy decoding (picking the highest probability word step-by-step), Beam Search explores multiple sequence paths simultaneously to find the highest overall sequence probability before hitting the `<END>` token.

---

## 📐 The Mathematical Foundation

Standard multi-label classification assumes object independence (Binary Relevance). This architecture fundamentally rejects that assumption, modeling image classification as a joint probability distribution where the presence of one object directly influences the probability of another.

### 1. Joint Image-Label Embedding
Let the image feature vector be $I \in \mathbb{R}^{4096}$ and the sequence of true labels be $Y = (y_1, y_2, ..., y_n)$. Both the image and the text labels are projected into a shared latent space of dimension $d = 64$.

### 2. Sequential Probability Modeling
The model factors the joint probability of the label sequence using the chain rule. The probability of detecting object $y_t$ relies on the global image features and the history of previously detected objects:
$$P(Y|I) = \prod_{t=1}^{n} P(y_t | y_1, ..., y_{t-1}, I)$$

### 3. LSTM State Updates
At each time step $t$, the LSTM maintains a hidden state $h_t \in \mathbb{R}^{512}$ that acts as structural memory. The state updates based on the previously emitted label $y_{t-1}$:
$$h_t = f_{\text{LSTM}}(E[y_{t-1}], h_{t-1})$$
Where $E$ represents the learned label embedding matrix.

### 4. The Scoring Function
To predict the next label, the network computes a logit score $s_{t,c}$ for every class $c$ in the vocabulary. The score merges the static image projection and the dynamic LSTM state:
$$s_{t,c} = W_c^\top (U I + V h_t)$$
Where:
* $U$: Image projection matrix.
* $V$: Hidden state projection matrix.
* $W_c$: The embedding weight for class $c$.

---

## 📈 Model Performance & Metrics
* **Evaluation Metric:** Mean Average Precision (mAP).
* **Implementation Result:** 65.07% mAP.
* **Highlights:** Achieved highly correlated scores with the original CVPR paper on high-frequency classes such as *Aeroplane* (95.3% AP) and *Person* (92.6% AP).

---

## 📁 Repository Structure

    cnn-rnn-portfolio/
    │
    ├── backend/                  # PyTorch Model & FastAPI Server
    │   ├── app.py                # Main API routing and CORS configuration
    │   ├── inference.py          # Image transformation and batch prediction logic
    │   ├── model.py              # CNN-RNN architecture classes
    │   ├── requirements.txt      # Python dependencies
    │   └── weights/              
    │       └── finalmodel.pth  # Downloaded from GitHub Releases
    │
    ├── frontend/                 # UI Interface
    │   ├── index.html            # Webpage structure
    │   ├── script.js             # API integration and DOM manipulation
    │   └── style.css             # UI styling
    │
    ├── results/                  # Benchmark comparisons and charts
    │   ├── comparison_benchmark.png
    │   └── paper_vs_implementation.csv
    │
    └── README.md                 # Project documentation

---

## 🚀 How to Run Locally

### 1. Weights Installation
Due to file size limits, the 500MB+ model weights are hosted in the **Releases** section of this repository.
1. Go to the **Releases** tab.
2. Download `finalmodel.pth`.
3. Place the file inside the `backend/weights/` directory.

### 2. Backend Setup
Open a terminal, navigate to the `backend` folder, and install the required dependencies:

    cd backend
    pip install -r requirements.txt

Start the FastAPI server:

    uvicorn app:app --reload

*The API is now active at `http://127.0.0.1:8000`.*

### 3. Frontend Setup
Open a **new** terminal, navigate to the `frontend` folder, and start a local web server:

    cd frontend
    python -m http.server 3000

*Navigate your browser to `http://localhost:3000` to interact with the batch-processing UI.*

---

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, Torchvision
- **Backend API:** FastAPI, Uvicorn, Python-Multipart
- **Frontend UI:** HTML5, CSS3, Vanilla JavaScript
- **Data Processing:** PIL, NumPy, Pandas, Matplotlib
