Markdown
# 🫁 Clinical Diagnostic Portal

An AI-powered web application built to assist with clinical diagnostics using deep learning. This portal provides a secure, user-friendly interface for processing medical data using a custom-trained PyTorch model, complete with user authentication and database management.

## 🚀 Live Demo
**[Insert your Streamlit Cloud URL here]**

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend & Framework** | Streamlit |
| **Machine Learning** | PyTorch, Torchvision |
| **Authentication & Database**| Supabase |
| **Data & Image Processing** | NumPy, Pillow, OpenCV (Headless) |
| **Data Visualization** | Plotly, Matplotlib |

## ✨ Features
* **Secure User Access:** Full user authentication (Login/Register) handled securely via Supabase.
* **AI Diagnostics:** Integrates a lightweight, fine-tuned PyTorch model (`best_student_model_aligned_final.pth`) for rapid inference.
* **Cloud Deployment:** Optimized for and deployed on Streamlit Community Cloud.
* **Headless Image Processing:** Utilizes `opencv-python-headless` for seamless cloud compatibility without missing system dependencies.

## 💻 Local Installation & Setup

If you want to run this project locally on your machine, follow these steps:

# 1. Clone the repository

```
git clone https://github.com/adi13apr/Clinical-Diagnostics--.git
cd Clinical-Diagnostics-
```

## 2. Create and activate a Virtual Environment

### Windows

```
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

Install the required packages from the `requirements.txt` file:

```
pip install -r requirements.txt
```

## 4. Configure Secrets (Supabase)

Create a `.streamlit` folder in the root directory, and inside it, create a file named `secrets.toml`. Add your Supabase credentials:

```
[supabase]
url = "YOUR_SUPABASE_URL"
key = "YOUR_SUPABASE_ANON_KEY"
```
## 5. Run the Application
streamlit run app.py
📝 License
[Specify your license here, e.g., MIT License]


