"""FAO Penman-Monteith ET0 计算模块

严格遵循 FAO Irrigation and Drainage Paper 56 (1998)
中文版：《作物需水量计算指南FAO-56》
所有公式编号对应 FAO-56 原文。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Stefan-Boltzmann常数: 4.903×10⁻⁹ MJ·K⁻⁴·m⁻²·day⁻¹ (FAO-56 §Eq.39)
STEFAN_BOLTZMANN = 4.903e-9


def saturation_vapor_pressure(temp_c: pd.Series | np.ndarray) -> pd.Series:
    """饱和水汽压 e°(T) —— FAO-56 Eq.11

    e°(T) = 0.6108 × exp[17.27·T / (T + 237.3)]  [kPa]
    """
    values = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    return pd.Series(values, index=getattr(temp_c, "index", None))


def actual_vapor_pressure_from_dewpoint(
    dewpoint_c: pd.Series | np.ndarray,
) -> pd.Series:
    """由露点温度计算实际水汽压ea —— FAO-56 Eq.14 (优先级1)

    ea = e°(Tdew)  即把露点温度代入饱和水汽压公式
    这是FAO-56推荐的三种ea算法中最准确的一种
    因为我们有ERA5-Land的露点温度数据，走这个路径
    """
    values = 0.6108 * np.exp((17.27 * dewpoint_c) / (dewpoint_c + 237.3))
    return pd.Series(values, index=getattr(dewpoint_c, "index", None))


def slope_svp_curve(temp_c: pd.Series | np.ndarray) -> pd.Series:
    """饱和水汽压曲线斜率Δ —— FAO-56 Eq.13

    Δ = 4098 × e°(T) / (T + 237.3)²  [kPa/°C]
    """
    es = saturation_vapor_pressure(temp_c)
    values = 4098 * es / np.power(temp_c + 237.3, 2)
    return pd.Series(values, index=getattr(temp_c, "index", None))


def atmospheric_pressure_kpa(
    elevation_m: pd.Series | np.ndarray,
) -> pd.Series:
    """大气压P —— FAO-56 Eq.7

    P = 101.3 × [(293 − 0.0065·Z) / 293]^5.26  [kPa]
    仅在缺少实测气压时使用；我们优先用ERA5-Land的surface_pressure
    """
    values = 101.3 * np.power((293.0 - 0.0065 * elevation_m) / 293.0, 5.26)
    return pd.Series(values, index=getattr(elevation_m, "index", None))


def psychrometric_constant_kpa_c(
    pressure_kpa: pd.Series | np.ndarray,
) -> pd.Series:
    """湿度计常数γ —— FAO-56 Eq.8

    γ = 0.000665 × P  [kPa/°C]
    """
    values = 0.000665 * pressure_kpa
    return pd.Series(values, index=getattr(pressure_kpa, "index", None))


def extraterrestrial_radiation_mj_m2_d(
    latitude_deg: pd.Series | np.ndarray,
    doy: pd.Series | np.ndarray,
) -> pd.Series:
    """地外辐射Ra —— FAO-56 Eq.21, Eq.23, Eq.24, Eq.25

    dr   = 1 + 0.033 × cos(2π·J/365)                              Eq.23
    δ    = 0.409 × sin(2π·J/365 − 1.39)                            Eq.24
    ωs   = arccos(−tan(φ)·tan(δ))                                  Eq.25
    Ra   = (24×60/π) × 0.0820 × dr × [ωs·sinφ·sinδ + cosφ·cosδ·sinωs]
                                                                    Eq.21
    其中:
      J   = DOY (1月1日=1)
      φ   = 纬度 (弧度)
      dr  = 日地距离倒数
      δ   = 太阳赤纬 (弧度)
      ωs  = 日落时角 (弧度)
      0.0820 = 太阳常数 Gsc (MJ/m²/min) × 转换系数
    """
    lat_rad = np.deg2rad(latitude_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    solar_declination = 0.409 * np.sin((2 * np.pi * doy / 365.0) - 1.39)
    sunset_hour_angle = np.arccos(-np.tan(lat_rad) * np.tan(solar_declination))
    values = (
        (24 * 60 / np.pi)
        * 0.0820
        * dr
        * (
            sunset_hour_angle * np.sin(lat_rad) * np.sin(solar_declination)
            + np.cos(lat_rad)
            * np.cos(solar_declination)
            * np.sin(sunset_hour_angle)
        )
    )
    return pd.Series(values, index=getattr(latitude_deg, "index", None))


def clear_sky_radiation_mj_m2_d(
    extraterrestrial_radiation: pd.Series | np.ndarray,
    elevation_m: pd.Series | np.ndarray,
) -> pd.Series:
    """晴空太阳辐射Rso —— FAO-56 Eq.37

    Rso = (0.75 + 2×10⁻⁵ × Z) × Ra  [MJ/m²/d]
    Z = 海拔高度 (m)
    """
    values = (0.75 + 2e-5 * elevation_m) * extraterrestrial_radiation
    return pd.Series(values, index=getattr(extraterrestrial_radiation, "index", None))


def convert_wind_10m_to_2m(
    wind_10m_m_s: pd.Series | np.ndarray,
) -> pd.Series:
    """风速高度转换 —— FAO-56 Eq.47

    u₂ = u₁₀ × 4.87 / ln(67.8×10 − 5.42)
    将10m高度风速转换为标准2m高度风速
    如果数据已经是2m风速，则不调用此函数
    """
    factor = 4.87 / np.log(67.8 * 10 - 5.42)
    values = wind_10m_m_s * factor
    return pd.Series(values, index=getattr(wind_10m_m_s, "index", None))


def compute_et0_fao56(weather_df: pd.DataFrame) -> pd.DataFrame:
    """计算逐日参照腾发量ET0 —— FAO-56 Penman-Monteith (Eq.6)

           0.408·Δ·(Rn−G) + γ·[900/(T+273.16)]·u₂·(es−ea)
    ET0 = ———————————————————————————————————————————    [mm/d]
                    Δ + γ·(1 + 0.34·u₂)

    参数说明:
      Δ    = 饱和水汽压曲线斜率 [kPa/°C]
      Rn   = 净辐射 [MJ/m²/d]
      G    = 土壤热通量 [MJ/m²/d] (逐日计算取0, FAO-56 §42)
      γ    = 湿度计常数 [kPa/°C]
      T    = 日平均气温 [°C] (按Eq.9 = (Tmax+Tmin)/2)
      u₂   = 2m高度风速 [m/s]
      es   = 饱和水汽压 [kPa]
      ea   = 实际水汽压 [kPa]
      0.408 = 1/2.45 (蒸发潜热λ≈2.45 MJ/kg → mm转换系数)

    必需输入列:
      - date          : 日期 (YYYY-MM-DD)
      - tmin_c        : 日最低气温 [°C]
      - tmax_c        : 日最高气温 [°C]
      - tmean_c       : 日平均气温 [°C] (有Tmax/Tmin时自动覆盖为均值)
      - solar_rad_mj_m2_d : 太阳辐射 [MJ/m²/d]
      - wind_10m_m_s  : 10m风速 [m/s] (或wind_2m_m_s)
      - centroid_lat  : 纬度 [度]
      - elevation_m   : 海拔 [m] (或pressure_kpa)
      - dewpoint_c    : 露点温度 [°C] (或rh_mean)
    """
    df = weather_df.copy()

    # 检查必需列
    if "date" not in df.columns:
        raise ValueError("weather_df must include a 'date' column")

    # 日期处理
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear  # 日序 (1月1日=1)

    # ---- 步骤1: 日平均气温 ----
    # FAO-56 Eq.9: Tmean = (Tmax+Tmin)/2, 不用小时均值
    if "tmax_c" in df.columns and "tmin_c" in df.columns:
        df["tmean_c"] = (df["tmax_c"] + df["tmin_c"]) / 2.0

    # ---- 步骤2: 风速转换（如需）----
    # 优先用2m风速，否则从10m转换 (FAO-56 Eq.47)
    if "wind_2m_m_s" not in df.columns:
        if "wind_10m_m_s" not in df.columns:
            raise ValueError("Need wind_2m_m_s or wind_10m_m_s to compute ET0")
        df["wind_2m_m_s"] = convert_wind_10m_to_2m(df["wind_10m_m_s"])

    # ---- 步骤3: 大气压（如需）----
    # 优先用实测气压(ERA5-Land提供)，否则从高程推算 (FAO-56 Eq.7)
    if "pressure_kpa" not in df.columns:
        if "elevation_m" not in df.columns:
            raise ValueError("Need pressure_kpa or elevation_m to compute ET0")
        df["pressure_kpa"] = atmospheric_pressure_kpa(df["elevation_m"])

    # ---- 步骤4: 饱和水汽压 es ----
    # es = [e°(Tmax) + e°(Tmin)] / 2  (FAO-56 Eq.12)
    # 用(Tmax+Tmin)/2而非日均温，因为饱和水汽压-温度关系是非线性的
    es_tmax = saturation_vapor_pressure(df["tmax_c"])
    es_tmin = saturation_vapor_pressure(df["tmin_c"])
    df["es_kpa"] = (es_tmax + es_tmin) / 2.0

    # ---- 步骤5: 实际水汽压 ea ----
    # 优先级1: 露点温度 → e°(Tdew)  (FAO-56 Eq.14) ← 我们走这条路
    # 优先级2: 干湿球温度计
    # 优先级3: RHmax/RHmin加权 → [e°(Tmin)·RHmax + e°(Tmax)·RHmin]/2
    # 简化版: RHmean → es × RHmean / 100 (FAO-56 §19, 较不准确)
    if "dewpoint_c" in df.columns:
        df["ea_kpa"] = actual_vapor_pressure_from_dewpoint(df["dewpoint_c"])
    elif "rh_mean" in df.columns:
        df["ea_kpa"] = df["es_kpa"] * df["rh_mean"] / 100.0
    else:
        raise ValueError("Need dewpoint_c or rh_mean to compute actual vapor pressure")

    # ---- 步骤6: Δ ----
    # Δ = 4098 × es/(T+237.3)²  (FAO-56 Eq.13)
    df["delta_kpa_c"] = slope_svp_curve(df["tmean_c"])

    # ---- 步骤7: γ ----
    # γ = 0.000665 × P  (FAO-56 Eq.8)
    df["gamma_kpa_c"] = psychrometric_constant_kpa_c(df["pressure_kpa"])

    # ---- 步骤8: 辐射计算 ----
    # 8a. 地外辐射 Ra  (FAO-56 Eq.21)
    df["ra_mj_m2_d"] = extraterrestrial_radiation_mj_m2_d(
        df["centroid_lat"], df["doy"]
    )
    # 8b. 晴空辐射 Rso  (FAO-56 Eq.37)
    df["rso_mj_m2_d"] = clear_sky_radiation_mj_m2_d(
        df["ra_mj_m2_d"], df.get("elevation_m", 0.0)
    )
    # 8c. 净短波辐射 Rns = (1−0.23)×Rs  (FAO-56 Eq.38)
    #     albedo=0.23 是假想草地的标准反照率
    df["rns_mj_m2_d"] = (1.0 - 0.23) * df["solar_rad_mj_m2_d"]

    # 8d. Rs/Rso 比值 (限制在0-1, 用于Rnl计算)
    safe_rso = df["rso_mj_m2_d"].replace(0, np.nan)
    rs_rso = (df["solar_rad_mj_m2_d"] / safe_rso).clip(lower=0.0, upper=1.0).fillna(0.0)

    # 8e. 净长波辐射 Rnl  (FAO-56 Eq.39)
    #     Rnl = σ × [(Tmax_k⁴+Tmin_k⁴)/2] × (0.34−0.14√ea) × (1.35Rs/Rso−0.35)
    tmax_k = df["tmax_c"] + 273.16   # °C → K
    tmin_k = df["tmin_c"] + 273.16
    df["rnl_mj_m2_d"] = (
        STEFAN_BOLTZMANN
        * ((np.power(tmax_k, 4) + np.power(tmin_k, 4)) / 2.0)
        * (0.34 - 0.14 * np.sqrt(df["ea_kpa"].clip(lower=0.0)))
        * (1.35 * rs_rso - 0.35)
    )

    # 8f. 净辐射 Rn = Rns − Rnl  (FAO-56 Eq.40)
    #     逐日计算土壤热通量G≈0 (FAO-56 §42)
    df["rn_mj_m2_d"] = df["rns_mj_m2_d"] - df["rnl_mj_m2_d"]

    # ---- 步骤9: 饱和水汽压差 VPD ----
    df["vpd_kpa"] = (df["es_kpa"] - df["ea_kpa"]).clip(lower=0.0)

    # ---- 步骤10: FAO-56 Penman-Monteith公式 (Eq.6) ----
    numerator = (
        0.408 * df["delta_kpa_c"] * df["rn_mj_m2_d"]           # 辐射项
        + df["gamma_kpa_c"]                                      # 空气动力项
        * (900.0 / (df["tmean_c"] + 273.16))
        * df["wind_2m_m_s"]
        * df["vpd_kpa"]
    )
    denominator = df["delta_kpa_c"] + df["gamma_kpa_c"] * (1.0 + 0.34 * df["wind_2m_m_s"])
    df["et0_pm_mm"] = (numerator / denominator).clip(lower=0.0)

    return df
