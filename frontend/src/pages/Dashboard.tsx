
import { Sidebar } from "@/components/Sidebar";
import { CountdownTimer } from "@/components/CountdownTimer";
import { AlertCard } from "@/components/AlertCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  MapPin, 
  Droplets, 
  ThermometerSun, 
  Wind,
  TrendingUp,
  TrendingDown,
  Users
} from "lucide-react";

const Dashboard = () => {
  const mockAlerts = [
    {
      level: "high" as const,
      location: "Porto Alegre - Zona Norte",
      message: "Previsão de chuvas intensas nas próximas 6 horas. Risco elevado de alagamentos.",
      time: "há 2 min"
    },
    {
      level: "medium" as const,
      location: "Porto Alegre - Centro Histórico",
      message: "Nível do Guaíba acima do normal. Monitoramento contínuo ativado.",
      time: "há 15 min"
    },
    {
      level: "low" as const,
      location: "Porto Alegre - Zona Sul",
      message: "Condições meteorológicas estáveis. Baixo risco de enchentes.",
      time: "há 1 hora"
    }
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        <div className="p-6 lg:p-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              Dashboard de Monitoramento - Porto Alegre
            </h1>
            <p className="text-gray-600">
              Sistema inteligente de previsão e alerta de enchentes
            </p>
          </div>

          {/* Timer Section */}
          <div className="mb-8">
            <CountdownTimer />
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Bairros Monitorados</p>
                    <p className="text-2xl font-bold text-blue-600">82</p>
                    <p className="text-xs text-green-600 flex items-center mt-1">
                      <TrendingUp className="h-3 w-3 mr-1" />
                      +8% este mês
                    </p>
                  </div>
                  <div className="p-3 bg-blue-100 rounded-full">
                    <MapPin className="h-6 w-6 text-blue-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Precipitação Média</p>
                    <p className="text-2xl font-bold text-cyan-600">32mm</p>
                    <p className="text-xs text-red-600 flex items-center mt-1">
                      <TrendingDown className="h-3 w-3 mr-1" />
                      -12% hoje
                    </p>
                  </div>
                  <div className="p-3 bg-cyan-100 rounded-full">
                    <Droplets className="h-6 w-6 text-cyan-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Temperatura</p>
                    <p className="text-2xl font-bold text-orange-600">18°C</p>
                    <p className="text-xs text-gray-500 mt-1">
                      Sensação: 16°C
                    </p>
                  </div>
                  <div className="p-3 bg-orange-100 rounded-full">
                    <ThermometerSun className="h-6 w-6 text-orange-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="hover:shadow-lg transition-shadow duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Alertas Ativos</p>
                    <p className="text-2xl font-bold text-red-600">5</p>
                    <p className="text-xs text-green-600 flex items-center mt-1">
                      <TrendingDown className="h-3 w-3 mr-1" />
                      -2 resolvidos
                    </p>
                  </div>
                  <div className="p-3 bg-red-100 rounded-full">
                    <Users className="h-6 w-6 text-red-600" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Main Content Area */}
            <div className="lg:col-span-2 space-y-6">
              {/* IA Status */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                    <span>Status da IA Preditiva</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Processamento de Dados</span>
                        <span>92%</span>
                      </div>
                      <Progress value={92} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Análise Meteorológica</span>
                        <span>88%</span>
                      </div>
                      <Progress value={88} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Geração de Previsões</span>
                        <span>85%</span>
                      </div>
                      <Progress value={85} className="h-2" />
                    </div>
                    
                    <div className="flex items-center justify-between pt-4 border-t">
                      <span className="text-sm font-medium">Sistema Operacional</span>
                      <Badge className="bg-green-100 text-green-800">
                        99.8% Uptime
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Map Placeholder */}
              <Card>
                <CardHeader>
                  <CardTitle>Mapa de Riscos - Porto Alegre</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-80 bg-gradient-to-br from-blue-100 to-green-100 rounded-lg flex items-center justify-center relative overflow-hidden">
                    <div className="absolute inset-0 opacity-20">
                      <div className="absolute top-1/4 left-1/4 w-8 h-8 bg-red-500 rounded-full animate-ping"></div>
                      <div className="absolute top-1/2 right-1/3 w-6 h-6 bg-yellow-500 rounded-full animate-pulse"></div>
                      <div className="absolute bottom-1/3 left-1/2 w-4 h-4 bg-blue-500 rounded-full"></div>
                    </div>
                    <div className="text-center z-10">
                      <MapPin className="h-16 w-16 text-blue-600 mx-auto mb-4 animate-bounce" />
                      <p className="text-xl font-semibold text-gray-700">Mapa Interativo</p>
                      <p className="text-gray-500">Visualização de riscos por bairro</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Sidebar Content */}
            <div className="space-y-6">
              {/* Current Weather */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Condições Atuais - POA</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Droplets className="h-4 w-4 text-blue-600" />
                      <span className="text-sm">Umidade</span>
                    </div>
                    <span className="font-semibold">72%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Wind className="h-4 w-4 text-gray-600" />
                      <span className="text-sm">Vento</span>
                    </div>
                    <span className="font-semibold">15 km/h</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <ThermometerSun className="h-4 w-4 text-orange-600" />
                      <span className="text-sm">Pressão</span>
                    </div>
                    <span className="font-semibold">1015 hPa</span>
                  </div>
                </CardContent>
              </Card>

              {/* Recent Alerts */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Alertas Recentes</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {mockAlerts.map((alert, index) => (
                    <AlertCard key={index} {...alert} />
                  ))}
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
