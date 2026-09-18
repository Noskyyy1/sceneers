# IDX Analytics Pro

Professional Streamlit dashboard for IDX stock screening.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## GitHub / Streamlit

Push `app.py`, `requirements.txt`, and the `data/` folder to GitHub. Select `app.py` as the Streamlit entrypoint.

## Universe

`data/idx_universe.csv` is an editable starter universe. It is intentionally not described as an always-current official list of every IDX issuer. Update it periodically from the IDX universe you want to screen.

## Notes

Yahoo Finance data availability can vary by ticker. Missing fundamental fields are shown as `N/A`. ATR levels are mechanical analytical references, not guarantees or investment recommendations.
