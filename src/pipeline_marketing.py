import time
import logging
from pathlib import Path

from data_clean import phase_1_ingestion, phase_2_wrangling
from utils_viz import phase_3_eda_and_viz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_pipeline(filepath: str, output_path: str):
    start = time.perf_counter()

    df_raw = phase_1_ingestion(filepath)
    df_clean = phase_2_wrangling(df_raw)
    phase_3_eda_and_viz(df_clean)

    df_clean.to_parquet(output_path, engine="pyarrow")
    logging.info(f"Export : {output_path} ({time.perf_counter() - start:.3f}s)")

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    process_pipeline(
        str(base / "data" / "raw" / "marketing_campaign.csv"),
        str(base / "data" / "processed" / "marketing_clean.parquet"),
    )
