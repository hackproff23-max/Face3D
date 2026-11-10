# Rosto 3D (GLB com esqueleto)

## 1. Servidor (pasta server/)
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
(ou use o Dockerfile em Render / Railway / Hugging Face Spaces, porta 7860)

## 2. App (pasta android/)
Suba tudo num repo no GitHub: o workflow gera o app-debug.apk em Actions > Artifacts.
No app, cole a URL do servidor (https://... se for na nuvem).

## Fotos
Frente, perfil esquerdo e perfil direito (60° a 90°). Luz uniforme, cabelo fora do rosto, sem óculos.
