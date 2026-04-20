# Assignment Bundle Startup

```bash
cd "your own folder path"
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python3 -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add:

```env
OPENAI_API_KEY=your_key_here
```

Run the cached frontend:

```bash
PYTHONPATH=src .venv/bin/python3 app.py
```

Open:

```text
http://localhost:7860
```
