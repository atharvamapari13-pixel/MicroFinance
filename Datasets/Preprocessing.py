import pandas as pd
import zipfile
from pathlib import Path
from IPython.display import display

zip_path = Path(
    r"E:\Projects\MicroFinance\Datasets\API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_206.zip"
)

if not zip_path.exists():
    raise FileNotFoundError(f"ZIP file not found: {zip_path}")

with zipfile.ZipFile(zip_path, "r") as z:
    files = z.namelist()

    print("Files inside ZIP:")
    for file in files:
        print(file)

    gdp_file = next(
        (
            file for file in files
            if Path(file).name.startswith("API_NY.GDP.MKTP.KD.ZG")
            and "Metadata_" not in Path(file).name
        ),
        None
    )

    if gdp_file is None:
        raise FileNotFoundError(
            "Main GDP dataset not found. "
            "Download the complete World Bank GDP dataset ZIP."
        )

    gdp = pd.read_csv(z.open(gdp_file), skiprows=4)

print("\nGDP dataset loaded successfully!")
print("Shape:", gdp.shape)
display(gdp.head())


print("GDP columns:")
print(gdp.columns.tolist())

print("\nMissing values in GDP data:")
print(gdp.isnull().sum())

# Remove the extra unnamed column, if present
gdp = gdp.loc[
    :, ~gdp.columns.astype(str).str.startswith("Unnamed")
]

# Identify columns containing years
year_cols = [
    col for col in gdp.columns
    if str(col).isdigit() and len(str(col)) == 4
]

# Convert year columns into rows
gdp_long = gdp.melt(
    id_vars=[
        "Country Name",
        "Country Code",
        "Indicator Name",
        "Indicator Code"
    ],
    value_vars=year_cols,
    var_name="Year",
    value_name="GDP_Growth"
)

gdp_long["Year"] = gdp_long["Year"].astype(int)

print("New shape:", gdp_long.shape)
display(gdp_long.head(10))

country_info = country_meta[
    ["Country Code", "Region", "IncomeGroup", "TableName"]
].drop_duplicates(subset=["Country Code"])

gdp_long = gdp_long.merge(
    country_info,
    on="Country Code",
    how="left",
    validate="many_to_one"
)

display(gdp_long.head())

output_dir = Path("datasets/processed")
output_dir.mkdir(parents=True, exist_ok=True)

gdp_long.to_csv(
    output_dir / "gdp_growth_long.csv",
    index=False
)

print("GDP preprocessing completed.")
print("Saved to:", output_dir / "gdp_growth_long.csv")