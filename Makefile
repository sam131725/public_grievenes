.PHONY: install data eda baseline evaluate train-muril train-indicbert train-mbert compare test api dashboard docker

install:        ; uv sync
data:           ; python scripts/prepare_data.py
eda:            ; python scripts/eda.py
baseline:       ; python scripts/train_baseline.py
evaluate:       ; python scripts/evaluate.py
train-muril:    ; python scripts/train_transformer.py --config configs/muril.yaml
train-indicbert:; python scripts/train_transformer.py --config configs/indicbert.yaml
train-mbert:    ; python scripts/train_transformer.py --config configs/mbert.yaml
compare:        ; python scripts/compare_models.py
test:           ; PYTHONPATH=src python -m pytest -q
api:            ; uvicorn grievance.api.main:app --app-dir src --reload
dashboard:      ; streamlit run src/grievance/dashboard/app.py
docker:         ; docker build -t grievance-api . && docker run --rm -p 8000:8000 grievance-api
