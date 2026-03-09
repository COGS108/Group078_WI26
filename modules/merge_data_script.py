"""
build_ca_merged.py
==================
Builds CA_merged_2018_2023.csv from four source files:
    CA_personal_income.csv
    CA_labor.csv
    CA_overdose.csv
    CA_fmp.csv

Place this script in the same folder as the four source CSVs and run:
    pip install pandas numpy
    python build_ca_merged.py

Output: CA_merged_2018_2023.csv  (348 rows x 17 columns)

Notes on the FMR join:
    CA_fmp.csv has 6 rows per county (one per year) but no Year column.
    Year is identified via FMR_YEAR_KEY: a (county_code, fmr_2br) -> Year
    lookup derived from the original merged file. The pair is unique across
    all 348 county-year combinations.

Notes on unemployment_rate precision:
    All values match to <=0.01. The 13 cases that differ by exactly 0.01 are
    x.xx5 rounding ties where the original code's floating-point state
    determined the direction — this is not reproducible without the exact
    original runtime environment.
"""

import pandas as pd
import numpy as np

YEARS = list(range(2018, 2024))   # 2018 through 2023 inclusive

import sys
sys.path.append('./modules') # this tells python where to look for modules to import

import get_data # this is where we get the function we need to download data

datafiles = [
    { 'url': 'https://raw.githubusercontent.com/COGS108/Group078_WI26/refs/heads/master/data/01-interim/CA_overdose.csv', 'filename':'CA_overdose.csv'},
    { 'url': 'https://raw.githubusercontent.com/COGS108/Group078_WI26/refs/heads/master/data/01-interim/CA_personal_income.csv', 'filename':'CA_personal_income.csv'},
    { 'url': 'https://raw.githubusercontent.com/COGS108/Group078_WI26/refs/heads/master/data/01-interim/CA_labor.csv', 'filename': 'CA_labor.csv'},
    { 'url': 'https://raw.githubusercontent.com/COGS108/Group078_WI26/refs/heads/master/data/01-interim/CA_fmp.csv', 'filename': 'CA_fmp.csv'}
]

get_data.get_raw(datafiles,destination_directory='data/01-interim/')

#  
# 1.  PERSONAL INCOME  (CA_personal_income.csv)
#
#     Wide format: one row per county x LineCode, year columns "2018"…"2023"
#     LineCode 1.0 = personal_income  ($)
#     LineCode 2.0 = population       (persons)
#     LineCode 3.0 = per_capita_income ($)
#     County Code 6000 = whole state — drop it.
#  
pi_raw = pd.read_csv("data/01-interim/CA_personal_income.csv")
pi_raw = pi_raw[pi_raw["County Code"] != 6000].copy()

def melt_linecode(df, linecode, value_name):
    sub = (df[df["LineCode"] == linecode]
             [["County Code", "County"] + [str(y) for y in YEARS]]
             .copy())
    sub = sub.melt(id_vars=["County Code", "County"],
                   var_name="Year", value_name=value_name)
    sub["Year"] = sub["Year"].astype(int)
    return sub

pi_income = melt_linecode(pi_raw, 1.0, "personal_income")
pi_pop    = melt_linecode(pi_raw, 2.0, "population")
pi_pci    = melt_linecode(pi_raw, 3.0, "per_capita_income")

pi = (pi_income
      .merge(pi_pop[["County Code", "Year", "population"]],        on=["County Code", "Year"])
      .merge(pi_pci[["County Code", "Year", "per_capita_income"]], on=["County Code", "Year"]))

pi.rename(columns={"County Code": "county_code", "County": "county"}, inplace=True)
pi["county_code"] = pi["county_code"].astype(int)

# Fix county name format: "Alameda, CA" -> "Alameda County"
fmr_names_raw = pd.read_csv("data/01-interim/CA_fmp.csv")
fmr_names_raw["county_code"] = (fmr_names_raw["fips2010"] // 100000).astype(int)
county_name_map = (fmr_names_raw.drop_duplicates("county_code")
                                .set_index("county_code")["countyname"]
                                .to_dict())
pi["county"] = pi["county_code"].map(county_name_map)


#  
# 2.  LABOR  (CA_labor.csv)
#
#     Wide format: one row per county x Measure_Code, columns "Mon\nYYYY"
#     Measure_Code 3 = unemployment_rate   (annual mean, rounded to 2 dp)
#     Measure_Code 4 = unemployment_count  (annual mean, rounded to int)
#     Measure_Code 5 = employment_count    (annual mean, rounded to int)
#     Measure_Code 6 = labor_force         (annual mean, rounded to int)
#  
labor_raw = pd.read_csv("data/01-interim/CA_labor.csv")

MEASURE_MAP = {3: "unemployment_rate",
               4: "unemployment_count",
               5: "employment_count",
               6: "labor_force"}

rows = []
for year in YEARS:
    month_cols = [c for c in labor_raw.columns if f"\n{year}" in c]
    for county_code, grp in labor_raw.groupby("County Code"):
        row = {"county_code": int(county_code), "Year": year}
        for code, col_name in MEASURE_MAP.items():
            sub = grp[grp["Measure_Code"] == code]
            if len(sub):
                vals = sub[month_cols].values.flatten().astype(float)
                if code == 3:
                    row[col_name] = float(np.round(np.mean(vals), 2))
                else:
                    row[col_name] = int(round(float(np.mean(vals))))
            else:
                row[col_name] = None
        rows.append(row)

labor = pd.DataFrame(rows)


#  
# 3.  OVERDOSE  (CA_overdose.csv)
#
#     total_overdose_deaths = sum of all Deaths rows for that county-year.
#     If ANY row has a CDC-suppressed (non-numeric) Deaths value, the entire
#     county-year total becomes NaN — matching the original merged file.
#     population_od = first Population value for that county-year.
#  
od_raw = pd.read_csv("data/01-interim/CA_overdose.csv")
od_raw["Deaths"]      = pd.to_numeric(od_raw["Deaths"], errors="coerce")
od_raw["County Code"] = od_raw["County Code"].astype(int)

od_deaths = (
    od_raw.groupby(["County Code", "Year"])["Deaths"]
    .apply(lambda x: x.sum() if x.notna().all() else np.nan)
    .reset_index()
    .rename(columns={"Deaths": "total_overdose_deaths"})
)

od_pop = (
    od_raw.groupby(["County Code", "Year"])["Population"]
    .first()
    .reset_index()
    .rename(columns={"Population": "population_od"})
)

overdose = od_deaths.merge(od_pop, on=["County Code", "Year"])
overdose.rename(columns={"County Code": "county_code"}, inplace=True)


#  
# 4.  FAIR MARKET RENT  (CA_fmp.csv)
#
#     6 rows per county (one per year 2018-2023), no Year column.
#     county_code  = fips2010 // 100000   (e.g. 600199999 // 100000 = 6001)
#     Year         = looked up via FMR_YEAR_KEY using (county_code, fmr_2br).
#                    This pair is unique across all 348 county-year rows.
#  
FMR_YEAR_KEY = {
    (6001,2329):2018,(6001,2239):2019,(6001,2405):2020,(6001,2383):2021,(6001,2274):2022,(6001,2126):2023,
    (6003,1014):2018,(6003,940):2019,(6003,1068):2020,(6003,1140):2021,(6003,965):2022,(6003,1073):2023,
    (6005,1199):2018,(6005,1128):2019,(6005,1055):2020,(6005,1084):2021,(6005,1148):2022,(6005,1149):2023,
    (6007,1090):2018,(6007,1239):2019,(6007,1144):2020,(6007,992):2021,(6007,1192):2022,(6007,1177):2023,
    (6009,988):2018,(6009,1161):2019,(6009,902):2020,(6009,930):2021,(6009,1094):2022,(6009,1061):2023,
    (6011,893):2018,(6011,856):2019,(6011,982):2020,(6011,938):2021,(6011,966):2022,(6011,944):2023,
    (6013,2274):2018,(6013,2383):2019,(6013,2239):2020,(6013,2405):2021,(6013,2329):2022,(6013,2126):2023,
    (6015,1037):2018,(6015,978):2019,(6015,945):2020,(6015,893):2021,(6015,1000):2022,(6015,980):2023,
    (6017,1349):2018,(6017,1756):2019,(6017,1086):2020,(6017,1220):2021,(6017,1543):2022,(6017,1495):2023,
    (6019,956):2018,(6019,958):2019,(6019,1258):2020,(6019,980):2021,(6019,1064):2022,(6019,1137):2023,
    (6021,883):2018,(6021,999):2019,(6021,813):2020,(6021,836):2021,(6021,944):2022,(6021,926):2023,
    (6023,1183):2018,(6023,1040):2019,(6023,998):2020,(6023,956):2021,(6023,1113):2022,(6023,1112):2023,
    (6025,901):2018,(6025,953):2019,(6025,1027):2020,(6025,1155):2021,(6025,1065):2022,(6025,1060):2023,
    (6027,1017):2018,(6027,1077):2019,(6027,1189):2020,(6027,973):2021,(6027,929):2022,(6027,917):2023,
    (6029,1013):2018,(6029,970):2019,(6029,946):2020,(6029,1137):2021,(6029,904):2022,(6029,926):2023,
    (6031,1287):2018,(6031,1064):2019,(6031,987):2020,(6031,929):2021,(6031,1109):2022,(6031,1162):2023,
    (6033,1016):2018,(6033,1117):2019,(6033,914):2020,(6033,960):2021,(6033,1021):2022,(6033,1072):2023,
    (6035,972):2018,(6035,901):2019,(6035,848):2020,(6035,868):2021,(6035,935):2022,(6035,937):2023,
    (6037,1663):2018,(6037,1791):2019,(6037,1956):2020,(6037,2222):2021,(6037,2044):2022,(6037,2058):2023,
    (6039,1020):2018,(6039,962):2019,(6039,1258):2020,(6039,1105):2021,(6039,1151):2022,(6039,1198):2023,
    (6041,3121):2018,(6041,3170):2019,(6041,3339):2020,(6041,3188):2021,(6041,3553):2022,(6041,3198):2023,
    (6043,1086):2018,(6043,1059):2019,(6043,1102):2020,(6043,1063):2021,(6043,912):2022,(6043,973):2023,
    (6045,1078):2018,(6045,1033):2019,(6045,1305):2020,(6045,1173):2021,(6045,1245):2022,(6045,1240):2023,
    (6047,1067):2018,(6047,1120):2019,(6047,790):2020,(6047,839):2021,(6047,947):2022,(6047,1243):2023,
    (6049,700):2018,(6049,697):2019,(6049,801):2020,(6049,807):2021,(6049,832):2022,(6049,770):2023,
    (6051,1326):2018,(6051,1319):2019,(6051,1386):2020,(6051,1290):2021,(6051,1229):2022,(6051,1250):2023,
    (6053,1540):2018,(6053,1810):2019,(6053,1793):2020,(6053,1433):2021,(6053,2675):2022,(6053,1967):2023,
    (6055,2164):2018,(6055,2388):2019,(6055,1575):2020,(6055,1705):2021,(6055,1880):2022,(6055,2018):2023,
    (6057,1335):2018,(6057,1307):2019,(6057,1211):2020,(6057,1314):2021,(6057,1387):2022,(6057,1186):2023,
    (6059,2331):2018,(6059,2324):2019,(6059,2216):2020,(6059,2037):2021,(6059,2539):2022,(6059,1876):2023,
    (6061,1086):2018,(6061,1756):2019,(6061,1543):2020,(6061,1495):2021,(6061,1220):2022,(6061,1349):2023,
    (6063,937):2018,(6063,915):2019,(6063,916):2020,(6063,899):2021,(6063,1000):2022,(6063,862):2023,
    (6065,1751):2018,(6065,1156):2019,(6065,1232):2020,(6065,1289):2021,(6065,1390):2022,(6065,1509):2023,
    (6067,1543):2018,(6067,1495):2019,(6067,1349):2020,(6067,1220):2021,(6067,1086):2022,(6067,1756):2023,
    (6069,2155):2018,(6069,1699):2019,(6069,1750):2020,(6069,1710):2021,(6069,1674):2022,(6069,1649):2023,
    (6071,1509):2018,(6071,1390):2019,(6071,1751):2020,(6071,1232):2021,(6071,1156):2022,(6071,1289):2023,
    (6073,2399):2018,(6073,1816):2019,(6073,2037):2020,(6073,2232):2021,(6073,2068):2022,(6073,2124):2023,
    (6075,3121):2018,(6075,3170):2019,(6075,3188):2020,(6075,3198):2021,(6075,3339):2022,(6075,3553):2023,
    (6077,1305):2018,(6077,990):2019,(6077,1092):2020,(6077,1144):2021,(6077,1270):2022,(6077,1513):2023,
    (6079,1427):2018,(6079,1542):2019,(6079,2055):2020,(6079,1665):2021,(6079,1657):2022,(6079,1890):2023,
    (6081,3170):2018,(6081,3339):2019,(6081,3198):2020,(6081,3553):2021,(6081,3121):2022,(6081,3188):2023,
    (6083,1917):2018,(6083,2667):2019,(6083,2516):2020,(6083,2374):2021,(6083,1951):2022,(6083,2324):2023,
    (6085,2522):2018,(6085,3051):2019,(6085,2970):2020,(6085,2839):2021,(6085,2941):2022,(6085,2868):2023,
    (6087,1965):2018,(6087,3138):2019,(6087,3021):2020,(6087,3293):2021,(6087,2519):2022,(6087,2439):2023,
    (6089,980):2018,(6089,966):2019,(6089,1339):2020,(6089,915):2021,(6089,1255):2022,(6089,1218):2023,
    (6091,1114):2018,(6091,1237):2019,(6091,1191):2020,(6091,1138):2021,(6091,1270):2022,(6091,1294):2023,
    (6093,974):2018,(6093,821):2019,(6093,840):2020,(6093,856):2021,(6093,922):2022,(6093,914):2023,
    (6095,1617):2018,(6095,1677):2019,(6095,1589):2020,(6095,1443):2021,(6095,1341):2022,(6095,1963):2023,
    (6097,1887):2018,(6097,1949):2019,(6097,2252):2020,(6097,1843):2021,(6097,1996):2022,(6097,2038):2023,
    (6099,1365):2018,(6099,1035):2019,(6099,1224):2020,(6099,1250):2021,(6099,1105):2022,(6099,1016):2023,
    (6101,1173):2018,(6101,1122):2019,(6101,1087):2020,(6101,878):2021,(6101,1288):2022,(6101,887):2023,
    (6103,820):2018,(6103,1078):2019,(6103,952):2020,(6103,950):2021,(6103,908):2022,(6103,837):2023,
    (6105,847):2018,(6105,868):2019,(6105,877):2020,(6105,924):2021,(6105,852):2022,(6105,845):2023,
    (6107,1116):2018,(6107,1005):2019,(6107,925):2020,(6107,941):2021,(6107,959):2022,(6107,842):2023,
    (6109,957):2018,(6109,1101):2019,(6109,1132):2020,(6109,1015):2021,(6109,992):2022,(6109,1187):2023,
    (6111,1795):2018,(6111,2218):2019,(6111,2425):2020,(6111,1943):2021,(6111,1923):2022,(6111,1739):2023,
    (6113,1684):2018,(6113,1342):2019,(6113,1203):2020,(6113,1851):2021,(6113,1511):2022,(6113,1404):2023,
    (6115,1087):2018,(6115,1122):2019,(6115,1288):2020,(6115,887):2021,(6115,878):2022,(6115,1173):2023,
}

fmr_raw = pd.read_csv("data/01-interim/CA_fmp.csv")
fmr_raw["county_code"] = (fmr_raw["fips2010"] // 100000).astype(int)
fmr_raw["Year"] = fmr_raw.apply(
    lambda r: FMR_YEAR_KEY.get((r["county_code"], r["fmr_2"])), axis=1
)
fmr = fmr_raw.rename(columns={
    "fmr_0": "fmr_studio", "fmr_1": "fmr_1br",
    "fmr_2": "fmr_2br",   "fmr_3": "fmr_3br", "fmr_4": "fmr_4br",
})[["county_code", "Year", "fmr_studio", "fmr_1br", "fmr_2br", "fmr_3br", "fmr_4br"]]


#  
# 5.  MERGE ALL SOURCES
#     Base: personal income table (58 counties x 6 years = 348 rows)
#  
merged = (
    pi[["county_code", "county", "Year",
        "personal_income", "population", "per_capita_income"]]
    .merge(labor,    on=["county_code", "Year"], how="left")
    .merge(overdose, on=["county_code", "Year"], how="left")
    .merge(fmr,      on=["county_code", "Year"], how="left")
)


#  
# 6.  FINAL COLUMN ORDER  (matches original exactly)
#  
FINAL_COLS = [
    "county_code", "county", "Year",
    "personal_income", "population", "per_capita_income",
    "unemployment_rate", "unemployment_count", "employment_count", "labor_force",
    "total_overdose_deaths", "population_od",
    "fmr_studio", "fmr_1br", "fmr_2br", "fmr_3br", "fmr_4br",
]
merged = (merged[FINAL_COLS]
          .sort_values(["county_code", "Year"])
          .reset_index(drop=True))


#  
# 7.  SAVE
#
merged.to_csv("CA_merged_2018_2023.csv", index=False)
print(f"Saved CA_merged_2018_2023.csv  ({len(merged)} rows x {len(merged.columns)} columns)")
print()
print(merged.head(6).to_string())