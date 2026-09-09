# FARMIQ — SIH26131 Crop Health Intelligence

Streamlit MVP for SIH26131.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Deploy `app.py` from this repository.

In App Settings -> Secrets, add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Never commit the API key to GitHub.
