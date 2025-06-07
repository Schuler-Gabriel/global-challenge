
import { Sidebar } from "@/components/Sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  TrendingUp, 
  Cloud, 
  Droplets, 
  ThermometerSun,
  Wind,
  Eye,
  Brain,
  RefreshCw
} from "lucide-react";

const Previsoes = () => {
  const predictions = [
    {
      region: "Porto Alegre - Zona Norte",
      risk: "high",
      probability: 85,
      timeframe: "6-12 horas",
      rainfall: "45-60mm",
      temperature: "16-19°C",
      humidity: "85%",
      details: "Frente fria intensa se aproximando. Chuvas torrenciais esperadas na região do Sarandi."
    },
    {
      region: "Porto Alegre - Centro Histórico",
      risk: "medium",
      probability: 65,
      timeframe: "12-24 horas",
      rainfall: "25-40mm",
      temperature: "17-21°C",
      humidity: "75%",
      details: "Instabilidade atmosférica moderada. Chuvas esparsas previstas no centro da cidade."
    },
    {
      region: "Porto Alegre - Zona Sul",
      risk: "low",
      probability: 25,
      timeframe: "24-48 horas",
      rainfall: "5-15mm",
      temperature: "15-20°C",
      humidity: "60%",
      details: "Condições meteorológicas estáveis. Baixa probabilidade de precipitação na região."
    },
    {
      region: "Porto Alegre - Orla do Guaíba",
      risk: "medium",
      probability: 55,
      timeframe: "18-24 horas",
      rainfall: "20-35mm",
      temperature: "18-22°C",
      humidity: "80%",
      details: "Combinação de maré alta no Guaíba e instabilidade. Atenção redobrada na orla."
    },
    {
      region: "Porto Alegre - Zona Leste",
      risk: "high",
      probability: 78,
      timeframe: "8-16 horas",
      rainfall: "35-50mm",
      temperature: "16-20°C",
      humidity: "88%",
      details: "Sistema tropical se intensificando. Chuvas fortes esperadas na Restinga e Lami."
    },
    {
      region: "Porto Alegre - Zona Oeste",
      risk: "low",
      probability: 15,
      timeframe: "48+ horas",
      rainfall: "0-10mm",
      temperature: "14-18°C",
      humidity: "45%",
      details: "Tempo seco predominante. Sem risco significativo de enchentes na região oeste."
    }
  ];

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "high": return "bg-red-100 text-red-800 border-red-200";
      case "medium": return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "low": return "bg-green-100 text-green-800 border-green-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getRiskLabel = (risk: string) => {
    switch (risk) {
      case "high": return "Alto Risco";
      case "medium": return "Risco Moderado";
      case "low": return "Baixo Risco";
      default: return "Indefinido";
    }
  };

  const getProgressColor = (risk: string) => {
    switch (risk) {
      case "high": return "bg-red-500";
      case "medium": return "bg-yellow-500";
      case "low": return "bg-green-500";
      default: return "bg-gray-500";
    }
  };

  const highRiskCount = predictions.filter(p => p.risk === "high").length;
  const mediumRiskCount = predictions.filter(p => p.risk === "medium").length;
  const lowRiskCount = predictions.filter(p => p.risk === "low").length;
  const avgAccuracy = 94.2;

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        <div className="p-6 lg:p-8">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Previsões Inteligentes - Porto Alegre
              </h1>
              <p className="text-gray-600">
                Análises preditivas baseadas em IA para riscos de enchentes
              </p>
            </div>
            <div className="flex space-x-3 mt-4 sm:mt-0">
              <Button variant="outline" size="sm">
                <Eye className="h-4 w-4 mr-2" />
                Visualizar Mapa
              </Button>
              <Button variant="outline" size="sm">
                <RefreshCw className="h-4 w-4 mr-2" />
                Atualizar
              </Button>
            </div>
          </div>

          {/* AI Status and Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <Card className="border-purple-200 bg-purple-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-purple-600">IA Ativa</p>
                    <p className="text-3xl font-bold text-purple-700">24/7</p>
                    <p className="text-xs text-purple-600 mt-1">Processamento contínuo</p>
                  </div>
                  <div className="p-3 bg-purple-100 rounded-full">
                    <Brain className="h-6 w-6 text-purple-600 animate-pulse" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-red-200 bg-red-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-red-600">Alto Risco</p>
                    <p className="text-3xl font-bold text-red-700">{highRiskCount}</p>
                    <p className="text-xs text-red-600 mt-1">Regiões em alerta</p>
                  </div>
                  <div className="p-3 bg-red-100 rounded-full">
                    <TrendingUp className="h-6 w-6 text-red-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-yellow-200 bg-yellow-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-yellow-600">Risco Moderado</p>
                    <p className="text-3xl font-bold text-yellow-700">{mediumRiskCount}</p>
                    <p className="text-xs text-yellow-600 mt-1">Monitoramento ativo</p>
                  </div>
                  <div className="p-3 bg-yellow-100 rounded-full">
                    <Cloud className="h-6 w-6 text-yellow-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-green-200 bg-green-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-green-600">Precisão da IA</p>
                    <p className="text-3xl font-bold text-green-700">{avgAccuracy}%</p>
                    <p className="text-xs text-green-600 mt-1">Última avaliação</p>
                  </div>
                  <div className="p-3 bg-green-100 rounded-full">
                    <TrendingUp className="h-6 w-6 text-green-600" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* AI Processing Status */}
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Brain className="h-5 w-5 text-purple-600" />
                <span>Status do Processamento IA</span>
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-6">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Análise Meteorológica</span>
                    <span>96%</span>
                  </div>
                  <Progress value={96} className="h-2 mb-2" />
                  <p className="text-xs text-gray-600">Dados de 45 estações processados</p>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Modelagem Hidrológica</span>
                    <span>91%</span>
                  </div>
                  <Progress value={91} className="h-2 mb-2" />
                  <p className="text-xs text-gray-600">Simulações de 15 bacias hídricas</p>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Geração de Alertas</span>
                    <span>94%</span>
                  </div>
                  <Progress value={94} className="h-2 mb-2" />
                  <p className="text-xs text-gray-600">Próxima atualização em 2h 15min</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Predictions Grid */}
          <div className="grid gap-6">
            {predictions.map((prediction, index) => (
              <Card key={index} className="hover:shadow-lg transition-all duration-300">
                <CardContent className="p-6">
                  <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-4">
                    <div className="flex items-center space-x-4 mb-4 lg:mb-0">
                      <h3 className="text-lg font-semibold text-gray-900">
                        {prediction.region}
                      </h3>
                      <Badge className={getRiskColor(prediction.risk)}>
                        {getRiskLabel(prediction.risk)}
                      </Badge>
                    </div>
                    <div className="text-sm text-gray-600">
                      Previsão para: {prediction.timeframe}
                    </div>
                  </div>

                  <div className="mb-4">
                    <div className="flex justify-between text-sm mb-2">
                      <span>Probabilidade de Enchente</span>
                      <span className="font-semibold">{prediction.probability}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${getProgressColor(prediction.risk)}`}
                        style={{ width: `${prediction.probability}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div className="flex items-center space-x-2">
                      <Droplets className="h-4 w-4 text-blue-600" />
                      <div>
                        <p className="text-xs text-gray-600">Precipitação</p>
                        <p className="text-sm font-semibold">{prediction.rainfall}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <ThermometerSun className="h-4 w-4 text-orange-600" />
                      <div>
                        <p className="text-xs text-gray-600">Temperatura</p>
                        <p className="text-sm font-semibold">{prediction.temperature}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Wind className="h-4 w-4 text-gray-600" />
                      <div>
                        <p className="text-xs text-gray-600">Umidade</p>
                        <p className="text-sm font-semibold">{prediction.humidity}</p>
                      </div>
                    </div>
                    <div className="flex items-center justify-end">
                      <Button variant="outline" size="sm">
                        Ver Detalhes
                      </Button>
                    </div>
                  </div>

                  <div className="pt-4 border-t">
                    <p className="text-sm text-gray-700">
                      <strong>Análise da IA:</strong> {prediction.details}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Previsoes;
