## 🏃‍♂️ How to Run the Application

You will need **two separate terminal windows** to run both the backend API and the frontend UI simultaneously. 

**⚠️ Important:** Make sure your virtual environment (`venv`) is activated in **both** terminals before running the commands!
* **Windows:** `.\venv\Scripts\activate`
* **Mac/Linux:** `source venv/bin/activate`

### 1. Start the Backend API
In your first terminal, start the FastAPI server:
```bash
python main.py
```
### 2. Start the Frontend UI:
In your second terminal, launch the Streamlit chat interface:
```
streamlit run app.py
```
### 3. Start the app:
1. Open the web interface in your browser.
2. Open the left sidebar and upload your PDF files.
3. Click the "Process PDFs" button to extract the text and embed it into the local LanceDB database.
4. Wait for the success message.
