# Licensed-data pipeline (needs data/raw/DataCube_v6_Q126.xlsx and configs/masking.yaml)
all: prep eda backtest evaluate analysis hbnn final
prep:
	python src/prep.py
eda:
	python src/eda.py
backtest:
	python src/backtest.py
evaluate:
	python src/evaluate.py
analysis:
	python src/analysis.py
hbnn:
	python src/backtest_hbnn.py
final:
	python src/evaluate_final.py
	python src/final_analysis.py
validate:
	python src/validate_svi.py
test:
	python tests/test_kalman.py

# Public demo: synthetic data with known elasticities, no licensed data needed
demo:
	python src/synthetic.py
	python src/recovery.py
