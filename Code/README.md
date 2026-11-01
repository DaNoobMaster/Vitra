## 1. Software Requirements

Make sure you have the following installed:

| Software | Purpose |
|---|---|
| Python 3.10 or later | Run the classification program |
| pip | Install Python libraries |
| Git | Download the MobileCLIP repository (if needed) |

Download Python from: https://www.python.org/downloads/

Download Git from: https://git-scm.com/downloads

During Python installation on Windows, make sure to select **Add Python to PATH**.

---

## 2. Required Python Libraries

The following libraries are required:

| Library | Purpose |
|---|---|
| torch | PyTorch deep learning framework |
| open_clip_torch | Load and run the MobileCLIP model |
| Pillow | Image processing |
| mobileclip | MobileCLIP model architecture |
| google-genai | Google Gemini API |
| openai | OpenAI API client (imported in the code) |
| groq | Groq API |
| openrouter | OpenRouter API client |
| torchvision | Image transformations and model dependencies |

Other dependencies may be installed automatically alongside these packages.

---

## 3. Installation

### Step 1: Create a virtual environment

Open a terminal in your project directory.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / Raspberry Pi:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install PyTorch

For Windows or Linux systems with an NVIDIA GPU, install the appropriate PyTorch version from:

https://pytorch.org/get-started/locally/

For CPU-only systems, run:

```bash
pip install torch torchvision
```

For Raspberry Pi, use a compatible ARM64 PyTorch installation. Availability depends on your Raspberry Pi OS and Python version.

### Step 3: Install the remaining libraries

Run:

```bash
pip install open_clip_torch Pillow google-genai openai groq
```

Install OpenRouter's Python SDK:

```bash
pip install openrouter
```

### Step 4: Install MobileCLIP

Clone the official MobileCLIP repository:

```bash
git clone https://github.com/apple/ml-mobileclip.git
```

Enter the downloaded directory:

```bash
cd ml-mobileclip
```

Install its dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Return to your project directory after installation.

Official repository: https://github.com/apple/ml-mobileclip

---

## 4. Downloading the MobileCLIP Model

The local classification system requires a pretrained MobileCLIP2-S0 model.

Download the appropriate pretrained checkpoint from the official MobileCLIP repository:

https://github.com/apple/ml-mobileclip

Place the downloaded `.pt` checkpoint in your project directory.

Update the following line in the Python script to match its location:

```python
model_path = "path/to/mobileclip2_s0.pt"
```

For example:

```python
model_path = "models/mobileclip2_s0.pt"
```

Make sure the model name matches the checkpoint:

```python
model_name = "MobileCLIP2-S0"
```

---

## 5. API Key Configuration

Online classification requires API keys from the following providers.

### Google Gemini

1. Visit https://aistudio.google.com/apikey
2. Sign in with your Google account.
3. Generate an API key.
4. Copy the key into the Python script.

### Groq

1. Visit https://console.groq.com/keys
2. Create an account.
3. Generate an API key.
4. Copy the key into the Python script.

### OpenRouter

1. Visit https://openrouter.ai/keys
2. Create an account.
3. Generate an API key.
4. Copy the key into the Python script.

Add the keys to the following variables:

```python
google_api_key = "YOUR_GOOGLE_API_KEY"
grok_api_key = "YOUR_GROQ_API_KEY"
router_api_key = "YOUR_OPENROUTER_API_KEY"
```

**Important:** Never upload your API keys to a public GitHub repository.

---

## 6. Project Structure

The recommended project directory structure is:

```text
Vitra/
│
├── main.py
├── trash.jpg
│
├── models/
│   └── mobileclip2_s0.pt
│
└── README.md
```

- `main.py` – Main Python classification program.
- `trash.jpg` – Image of the waste item to classify.
- `models/mobileclip2_s0.pt` – Pretrained MobileCLIP2-S0 model.
- `README.md` – Project documentation.

If you use the recommended directory structure, update the model path accordingly.

---

## 7. Running the Program

First, activate your virtual environment.

Then execute:

```bash
python main.py
```

The program will:

1. Load the input image (`trash.jpg`).
2. Send the image to Google Gemini, Groq, and OpenRouter if online mode is enabled.
3. Collect the classification results from all available models.
4. Combine their confidence scores.
5. Determine the final waste category.
6. Switch to the local MobileCLIP model if all online models fail.

---

## 8. Configuration

The following variables control the classification system:

```python
online = True
multiple = True
```

| Setting | Description |
|---|---|
| `online = True` | Enables online AI classification. |
| `online = False` | Uses the local MobileCLIP model. |
| `multiple = True` | Uses multiple online AI models and combines their results. |
| `multiple = False` | Intended to disable ensemble classification (single-model handling must be implemented). |

### AI Models Used

| Provider | Model |
|---|---|
| Google | Gemini 2.5 Flash |
| Groq | Llama 4 Scout 17B |
| OpenRouter | Qwen 2.5 VL 72B |
| Local | MobileCLIP2-S0 |

---

## 9. Troubleshooting

**PyTorch installation errors**

Visit https://pytorch.org/get-started/locally/ and select the correct operating system, package manager, Python version, and compute platform.

**ModuleNotFoundError**

Make sure the virtual environment is activated and install the missing package using:

```bash
pip install package_name
```

**MobileCLIP model loading errors**

Check that:
- The checkpoint exists.
- The model path is correct.
- The checkpoint matches the selected MobileCLIP architecture.

**API errors**

Check that:
- Your API keys are valid.
- Your API accounts have access to the selected models.
- You have sufficient API credits or quota.
- The selected models are still available through their respective providers.

---

## 10. Important Notes

- An internet connection is required for online classification.
- Local classification does not require an internet connection once the model and dependencies have been installed.
- The local model requires sufficient RAM and processing power, issues may arise if different microcontrollers are used.
- Online classification may take longer depending on API response times.
- API usage may incur charges.
