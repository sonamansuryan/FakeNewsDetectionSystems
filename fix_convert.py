import os
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

root_folder = "."

def read_arrow_file(path):
    try:
        with pa.input_stream(path) as source:
            reader = ipc.open_stream(source)
            return reader.read_all().to_pandas()
    except Exception:
        pass

    try:
        with pa.input_stream(path) as source:
            reader = ipc.open_file(source)
            return reader.read_all().to_pandas()
    except Exception:
        pass

    try:
        return pd.read_feather(path)
    except Exception:
        return None


def convert_all():
    count = 0

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".arrow"):
                arrow_path = os.path.join(root, file)
                csv_path = arrow_path.replace(".arrow", ".csv")


                df = read_arrow_file(arrow_path)

                if df is not None:
                    try:
                        df.to_csv(csv_path, index=False)
                        print(f"[OK] -> {os.path.basename(csv_path)}")
                        count += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to write CSV. {e}")
                else:
                    print(f"[FAIL] Unknown format or file is corrupt.")

    print(f"\nCompleted. {count} files converted.")


if __name__ == "__main__":
    convert_all()
