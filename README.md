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

## Molde do mod (importante)
Coloque a cabeça exportada do seu mod em server/template.glb (é o arquivo CR720_head_v6b_rigid_jaw.glb renomeado).
O servidor mantém os ossos, a malha e o atlas desse molde e só ajusta o formato e a pele ao seu rosto.
Esse arquivo é do jogo: deixe-o fora de repo público (já está no .gitignore). No Render/Space, envie-o por fora
ou use um repo privado e a variável TEMPLATE_GLB.
Sem o template.glb, o servidor cai no modo genérico (malha própria).
