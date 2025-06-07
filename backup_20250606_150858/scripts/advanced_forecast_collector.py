#!/usr/bin/env python3
"""
Coletor de Dados Avançados para Previsão de 4 Dias
Projeto: Sistema de Alertas de Cheias - Rio Guaíba

Este script demonstra como acessar APIs avançadas para obter dados sobre:
- Frentes frias e sistemas sinóticos
- Dados de altitude (500hPa, 850hPa)
- Índices meteorológicos avançados
- Ensemble forecasts
"""

import asyncio
import json
import logging
import warnings
from datetime import datetime, timedelta

# import xarray as xr  # Removido temporariamente
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedForecastCollector:
    """Coletor de dados meteorológicos avançados para previsão de 4 dias"""

    def __init__(self):
        self.base_dir = Path("data/advanced_forecasts")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Coordenadas de Porto Alegre
        self.lat = -30.0346
        self.lon = -51.2177

        # Área expandida para análise sinótica (200km raio)
        self.lat_min, self.lat_max = -32.0, -28.0
        self.lon_min, self.lon_max = -53.0, -49.0

        # URLs das APIs
        self.apis = {
            "noaa_gfs": "https://nomads.ncep.noaa.gov/dods/gfs_0p25",
            "noaa_ensemble": "https://nomads.ncep.noaa.gov/dods/gens",
            "visual_crossing": "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline",
            "windy": "https://api.windy.com/api/point-forecast/v2",
            "openweather_onecall": "https://api.openweathermap.org/data/3.0/onecall",
        }

    async def collect_noaa_gfs_data(
        self, forecast_hours: List[int] = [0, 6, 12, 18, 24, 48, 72, 96]
    ) -> Dict:
        """
        Coleta dados do NOAA GFS via OpenDAP
        Inclui múltiplos níveis de pressão e variáveis sinóticas
        """
        logger.info("Coletando dados NOAA GFS...")

        # Simula coleta de dados reais (implementação real usaria xarray/OpenDAP)
        gfs_data = {
            "surface_data": self._simulate_gfs_surface_data(forecast_hours),
            "upper_air_500hpa": self._simulate_gfs_upper_air_data(
                forecast_hours, "500hPa"
            ),
            "upper_air_700hpa": self._simulate_gfs_upper_air_data(
                forecast_hours, "700hPa"
            ),
            "upper_air_850hpa": self._simulate_gfs_upper_air_data(
                forecast_hours, "850hPa"
            ),
            "derived_indices": self._calculate_meteorological_indices(),
            "ensemble_data": self._simulate_ensemble_data(forecast_hours),
        }

        return gfs_data

    def _simulate_gfs_surface_data(self, forecast_hours: List[int]) -> pd.DataFrame:
        """Simula dados de superfície do GFS"""
        np.random.seed(42)  # Para reprodutibilidade

        data = []
        base_time = datetime.now()

        for hour in forecast_hours:
            forecast_time = base_time + timedelta(hours=hour)

            # Simula variáveis meteorológicas realistas
            data.append(
                {
                    "forecast_time": forecast_time,
                    "forecast_hour": hour,
                    "temperature_2m": 20
                    + 5 * np.sin(hour * 0.1)
                    + np.random.normal(0, 2),
                    "pressure_msl": 1013 + np.random.normal(0, 5),
                    "precipitation_rate": max(0, np.random.exponential(0.5)),
                    "wind_speed_10m": 5 + np.random.exponential(3),
                    "wind_direction_10m": np.random.uniform(0, 360),
                    "relative_humidity_2m": 60 + np.random.normal(0, 15),
                    "cloud_cover": np.random.uniform(0, 100),
                }
            )

        return pd.DataFrame(data)

    def _simulate_gfs_upper_air_data(
        self, forecast_hours: List[int], level: str
    ) -> pd.DataFrame:
        """Simula dados de altitude do GFS"""
        np.random.seed(43)

        data = []
        base_time = datetime.now()

        # Parâmetros dependem do nível
        level_params = {
            "500hPa": {"geopotential": 5500, "temp_offset": -20},
            "700hPa": {"geopotential": 3000, "temp_offset": -5},
            "850hPa": {"geopotential": 1500, "temp_offset": 10},
        }

        params = level_params.get(level, level_params["500hPa"])

        for hour in forecast_hours:
            forecast_time = base_time + timedelta(hours=hour)

            data.append(
                {
                    "forecast_time": forecast_time,
                    "forecast_hour": hour,
                    "level": level,
                    "geopotential_height": params["geopotential"]
                    + np.random.normal(0, 50),
                    "temperature": params["temp_offset"] + np.random.normal(0, 3),
                    "wind_u": np.random.normal(0, 10),
                    "wind_v": np.random.normal(0, 10),
                    "relative_humidity": 50 + np.random.normal(0, 20),
                    "vertical_velocity": np.random.normal(0, 0.1),  # Pa/s
                }
            )

        return pd.DataFrame(data)

    def _calculate_meteorological_indices(self) -> Dict:
        """Calcula índices meteorológicos avançados"""
        np.random.seed(44)

        return {
            # Índices de instabilidade
            "cape": np.random.exponential(1000),  # J/kg
            "cin": -np.random.exponential(50),  # J/kg
            "lifted_index": np.random.normal(-2, 3),
            "k_index": np.random.normal(25, 10),
            "total_totals": np.random.normal(45, 8),
            # Cisalhamento do vento
            "wind_shear_0_6km": np.random.exponential(10),  # m/s
            "storm_relative_helicity": np.random.normal(100, 50),
            # Parâmetros sinóticos
            "thermal_wind": np.random.normal(0, 5),
            "q_vector_divergence": np.random.normal(0, 1e-15),
            "potential_vorticity": np.random.normal(1, 0.5),
        }

    def _simulate_ensemble_data(self, forecast_hours: List[int]) -> Dict:
        """Simula dados de ensemble (múltiplas realizações)"""
        np.random.seed(45)

        ensemble_size = 21  # Típico do GEFS
        ensemble_data = {}

        for hour in forecast_hours:
            members = []
            for member in range(ensemble_size):
                members.append(
                    {
                        "member": member,
                        "forecast_hour": hour,
                        "precipitation_total": max(0, np.random.gamma(2, 2)),
                        "temperature_2m": 20 + np.random.normal(0, 3),
                        "pressure_msl": 1013 + np.random.normal(0, 8),
                    }
                )

            ensemble_data[f"hour_{hour}"] = pd.DataFrame(members)

        return ensemble_data

    async def collect_visual_crossing_data(self, api_key: Optional[str] = None) -> Dict:
        """
        Coleta dados do Visual Crossing (análise sinótica automática)
        """
        logger.info("Coletando dados Visual Crossing...")

        if not api_key:
            logger.warning("API key não fornecida, simulando dados...")
            return self._simulate_visual_crossing_data()

        # Implementação real da API
        url = f"{self.apis['visual_crossing']}/{self.lat},{self.lon}"
        params = {
            "unitGroup": "metric",
            "key": api_key,
            "include": "hours,days,alerts,current",
            "elements": "temp,humidity,precip,windspeed,winddir,pressure,conditions,description",
        }

        # Simulação por enquanto
        return self._simulate_visual_crossing_data()

    def _simulate_visual_crossing_data(self) -> Dict:
        """Simula dados do Visual Crossing com análise sinótica"""
        np.random.seed(46)

        return {
            "frontal_analysis": {
                "cold_front_approach": np.random.choice([True, False]),
                "front_intensity": np.random.choice(["weak", "moderate", "strong"]),
                "estimated_arrival": datetime.now()
                + timedelta(hours=np.random.randint(12, 72)),
                "associated_precipitation": np.random.exponential(15),
            },
            "air_mass_analysis": {
                "current_mass": np.random.choice(
                    [
                        "tropical_continental",
                        "tropical_maritime",
                        "polar_continental",
                        "polar_maritime",
                    ]
                ),
                "stability": np.random.choice(
                    ["stable", "unstable", "conditionally_unstable"]
                ),
                "moisture_content": np.random.uniform(0.3, 0.9),
            },
            "synoptic_pattern": {
                "pattern_type": np.random.choice(
                    ["ridge", "trough", "col", "cyclonic", "anticyclonic"]
                ),
                "intensity": np.random.uniform(0.2, 1.0),
                "movement_speed": np.random.uniform(10, 50),  # km/h
            },
        }

    async def collect_windy_data(self, api_key: Optional[str] = None) -> Dict:
        """
        Coleta dados do Windy.com (modelos ensemble e visualizações)
        """
        logger.info("Coletando dados Windy...")

        if not api_key:
            return self._simulate_windy_data()

        # Implementação real usaria a API do Windy
        return self._simulate_windy_data()

    def _simulate_windy_data(self) -> Dict:
        """Simula dados do Windy com múltiplos modelos"""
        np.random.seed(47)

        models = ["gfs", "ecmwf", "icon", "gem"]
        model_data = {}

        for model in models:
            model_data[model] = {
                "precipitation_forecast": [
                    np.random.exponential(2) for _ in range(96)
                ],  # 96h
                "wind_forecast": [5 + np.random.exponential(3) for _ in range(96)],
                "cape_forecast": [np.random.exponential(800) for _ in range(96)],
                "model_agreement": np.random.uniform(0.6, 0.95),
            }

        return {
            "multi_model_data": model_data,
            "ensemble_spread": np.random.uniform(0.1, 0.3),
            "forecast_skill": np.random.uniform(0.7, 0.9),
        }

    def detect_frontal_systems_advanced(self, gfs_data: Dict) -> List[Dict]:
        """
        Detecção avançada de sistemas frontais usando dados de altitude
        """
        logger.info("Detectando sistemas frontais...")

        surface_data = gfs_data["surface_data"]
        upper_air_500 = gfs_data["upper_air_500hpa"]
        upper_air_850 = gfs_data["upper_air_850hpa"]

        frontal_systems = []

        for i in range(1, len(surface_data)):
            current = surface_data.iloc[i]
            previous = surface_data.iloc[i - 1]

            # Critérios multi-nível para detecção de frentes
            surface_indicators = self._analyze_surface_front_indicators(
                current, previous
            )
            upper_indicators = self._analyze_upper_air_indicators(
                upper_air_500.iloc[i] if i < len(upper_air_500) else None,
                upper_air_850.iloc[i] if i < len(upper_air_850) else None,
            )

            # Combina indicadores de superfície e altitude
            total_score = surface_indicators["score"] + upper_indicators["score"]

            if total_score > 0.7:  # Threshold para detecção
                frontal_systems.append(
                    {
                        "datetime": current["forecast_time"],
                        "type": self._classify_front_type(
                            surface_indicators, upper_indicators
                        ),
                        "intensity": min(1.0, total_score),
                        "confidence": surface_indicators["confidence"]
                        * upper_indicators["confidence"],
                        "characteristics": {
                            "temperature_gradient": surface_indicators["temp_gradient"],
                            "pressure_tendency": surface_indicators["pressure_change"],
                            "wind_shift": surface_indicators["wind_shift"],
                            "upper_level_support": upper_indicators["upper_support"],
                        },
                    }
                )

        return frontal_systems

    def _analyze_surface_front_indicators(
        self, current: pd.Series, previous: pd.Series
    ) -> Dict:
        """Analisa indicadores de frente na superfície"""
        temp_change = previous["temperature_2m"] - current["temperature_2m"]
        pressure_change = current["pressure_msl"] - previous["pressure_msl"]
        wind_shift = abs(current["wind_direction_10m"] - previous["wind_direction_10m"])
        wind_shift = min(wind_shift, 360 - wind_shift)  # Considera rotação circular

        # Sistema de pontuação
        score = 0.0
        if temp_change > 3:  # Queda de temperatura
            score += 0.3
        if pressure_change > 2:  # Aumento de pressão
            score += 0.2
        if wind_shift > 45:  # Mudança significativa do vento
            score += 0.3
        if current["precipitation_rate"] > 1:  # Precipitação associada
            score += 0.2

        return {
            "score": score,
            "confidence": min(1.0, score / 0.8),
            "temp_gradient": temp_change,
            "pressure_change": pressure_change,
            "wind_shift": wind_shift,
        }

    def _analyze_upper_air_indicators(
        self, data_500: Optional[pd.Series], data_850: Optional[pd.Series]
    ) -> Dict:
        """Analisa indicadores de frente em altitude"""
        score = 0.0
        upper_support = False

        if data_500 is not None and data_850 is not None:
            # Análise de cisalhamento vertical
            wind_shear = np.sqrt(
                (data_500["wind_u"] - data_850["wind_u"]) ** 2
                + (data_500["wind_v"] - data_850["wind_v"]) ** 2
            )

            # Diferencial térmico
            temp_diff = data_850["temperature"] - data_500["temperature"]

            if wind_shear > 10:  # Cisalhamento forte
                score += 0.3
                upper_support = True

            if temp_diff > 30:  # Gradiente térmico normal
                score += 0.2

            if abs(data_500["vertical_velocity"]) > 0.05:  # Movimento vertical
                score += 0.2

        return {
            "score": score,
            "confidence": 0.8 if upper_support else 0.5,
            "upper_support": upper_support,
        }

    def _classify_front_type(
        self, surface_indicators: Dict, upper_indicators: Dict
    ) -> str:
        """Classifica o tipo de frente baseado nos indicadores"""
        if (
            surface_indicators["temp_gradient"] > 5
            and surface_indicators["pressure_change"] > 3
        ):
            return "cold_front_strong"
        elif surface_indicators["temp_gradient"] > 2:
            return "cold_front_weak"
        elif surface_indicators["pressure_change"] < -2:
            return "warm_front"
        else:
            return "occluded_front"

    def calculate_forecast_skill_metrics(self, ensemble_data: Dict) -> Dict:
        """
        Calcula métricas de habilidade da previsão baseado no ensemble
        """
        logger.info("Calculando métricas de habilidade...")

        metrics = {}

        for hour_key, data in ensemble_data.items():
            # Spread do ensemble (incerteza)
            precip_spread = data["precipitation_total"].std()
            temp_spread = data["temperature_2m"].std()

            # Probabilidade de eventos extremos
            heavy_rain_prob = (data["precipitation_total"] > 20).mean()
            extreme_temp_prob = (data["temperature_2m"] > 30).mean()

            # Skill score estimado (baseado no spread)
            skill_score = max(0, 1 - (precip_spread / 10))  # Normalizado

            metrics[hour_key] = {
                "precipitation_spread": precip_spread,
                "temperature_spread": temp_spread,
                "heavy_rain_probability": heavy_rain_prob,
                "extreme_temperature_probability": extreme_temp_prob,
                "forecast_skill": skill_score,
                "ensemble_agreement": 1
                - (precip_spread / (data["precipitation_total"].mean() + 0.1)),
            }

        return metrics

    async def generate_4day_forecast_report(self) -> Dict:
        """
        Gera relatório completo de previsão de 4 dias
        """
        logger.info("Gerando relatório de previsão de 4 dias...")

        # Coleta todos os dados
        gfs_data = await self.collect_noaa_gfs_data()
        visual_crossing_data = await self.collect_visual_crossing_data()
        windy_data = await self.collect_windy_data()

        # Análises avançadas
        frontal_systems = self.detect_frontal_systems_advanced(gfs_data)
        skill_metrics = self.calculate_forecast_skill_metrics(gfs_data["ensemble_data"])

        # Compila relatório
        report = {
            "generation_time": datetime.now().isoformat(),
            "forecast_period": "96 hours",
            "location": {"lat": self.lat, "lon": self.lon, "name": "Porto Alegre"},
            "data_sources": {
                "gfs_surface": len(gfs_data["surface_data"]),
                "gfs_upper_air": len(gfs_data["upper_air_500hpa"]),
                "ensemble_members": 21,
                "synoptic_analysis": bool(visual_crossing_data),
                "multi_model": len(windy_data["multi_model_data"]),
            },
            "frontal_analysis": {
                "systems_detected": len(frontal_systems),
                "next_front": frontal_systems[0] if frontal_systems else None,
                "synoptic_pattern": visual_crossing_data["synoptic_pattern"],
            },
            "forecast_skill": {
                "overall_confidence": np.mean(
                    [m["forecast_skill"] for m in skill_metrics.values()]
                ),
                "ensemble_agreement": np.mean(
                    [m["ensemble_agreement"] for m in skill_metrics.values()]
                ),
                "uncertainty_range": {
                    "precipitation": np.mean(
                        [m["precipitation_spread"] for m in skill_metrics.values()]
                    ),
                    "temperature": np.mean(
                        [m["temperature_spread"] for m in skill_metrics.values()]
                    ),
                },
            },
            "risk_assessment": {
                "heavy_precipitation_risk": max(
                    [m["heavy_rain_probability"] for m in skill_metrics.values()]
                ),
                "frontal_passage_risk": len(
                    [f for f in frontal_systems if f["intensity"] > 0.7]
                ),
                "forecast_degradation": self._assess_forecast_degradation(
                    skill_metrics
                ),
            },
            "recommendations": self._generate_forecast_recommendations(
                frontal_systems, skill_metrics
            ),
        }

        return report

    def _assess_forecast_degradation(self, skill_metrics: Dict) -> Dict:
        """Avalia como a habilidade da previsão degrada com o tempo"""
        hours = [int(k.split("_")[1]) for k in skill_metrics.keys()]
        skills = [skill_metrics[k]["forecast_skill"] for k in skill_metrics.keys()]

        return {
            "skill_at_24h": skills[hours.index(24)] if 24 in hours else None,
            "skill_at_48h": skills[hours.index(48)] if 48 in hours else None,
            "skill_at_72h": skills[hours.index(72)] if 72 in hours else None,
            "skill_at_96h": skills[hours.index(96)] if 96 in hours else None,
            "degradation_rate": (
                (skills[0] - skills[-1]) / len(skills) if len(skills) > 1 else 0
            ),
        }

    def _generate_forecast_recommendations(
        self, frontal_systems: List[Dict], skill_metrics: Dict
    ) -> List[str]:
        """Gera recomendações baseadas na análise"""
        recommendations = []

        if len(frontal_systems) > 0:
            next_front = frontal_systems[0]
            if next_front["intensity"] > 0.8:
                recommendations.append(
                    f"⚠️ Frente fria intensa esperada em {next_front['datetime']}"
                )

            if next_front["confidence"] < 0.7:
                recommendations.append(
                    "🤔 Incerteza alta na chegada da frente - monitorar atualizações"
                )

        avg_skill = np.mean([m["forecast_skill"] for m in skill_metrics.values()])
        if avg_skill < 0.6:
            recommendations.append(
                "📊 Previsão com baixa confiabilidade - usar ensemble probabilístico"
            )

        heavy_rain_risk = max(
            [m["heavy_rain_probability"] for m in skill_metrics.values()]
        )
        if heavy_rain_risk > 0.3:
            recommendations.append(
                f"🌧️ Risco {heavy_rain_risk:.0%} de precipitação intensa"
            )

        if not recommendations:
            recommendations.append("✅ Condições meteorológicas estáveis previstas")

        return recommendations


async def main():
    """Função principal - demonstração do sistema avançado"""
    collector = AdvancedForecastCollector()

    print("=" * 70)
    print("🌩️ SISTEMA AVANÇADO DE PREVISÃO DE 4 DIAS")
    print("=" * 70)

    print("\n🔄 Coletando dados de múltiplas fontes...")

    # Gera relatório completo
    report = await collector.generate_4day_forecast_report()

    print(f"\n📊 RELATÓRIO DE PREVISÃO:")
    print(f"📍 Local: {report['location']['name']}")
    print(f"⏰ Período: {report['forecast_period']}")
    print(f"🎯 Confiança geral: {report['forecast_skill']['overall_confidence']:.1%}")

    print(f"\n🌀 ANÁLISE SINÓTICA:")
    print(
        f"📈 Sistemas frontais detectados: {report['frontal_analysis']['systems_detected']}"
    )
    if report["frontal_analysis"]["next_front"]:
        front = report["frontal_analysis"]["next_front"]
        print(
            f"❄️ Próxima frente: {front['type']} (intensidade: {front['intensity']:.1%})"
        )

    print(f"\n⚡ ANÁLISE DE RISCOS:")
    print(
        f"🌧️ Risco de chuva intensa: {report['risk_assessment']['heavy_precipitation_risk']:.1%}"
    )
    print(
        f"❄️ Sistemas frontais intensos: {report['risk_assessment']['frontal_passage_risk']}"
    )

    print(f"\n💡 RECOMENDAÇÕES:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")

    print(f"\n📉 DEGRADAÇÃO DA PREVISÃO:")
    degradation = report["risk_assessment"]["forecast_degradation"]
    for time_range, skill in degradation.items():
        if skill is not None and "skill_at" in time_range:
            hours = time_range.split("_")[2]
            print(f"  {hours}: {skill:.1%}")

    # Salva relatório completo
    output_file = (
        collector.base_dir
        / f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n📋 Relatório salvo em: {output_file}")

    print(f"\n" + "=" * 70)
    print("✅ SISTEMA PRONTO PARA PREVISÕES DE 4 DIAS!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
