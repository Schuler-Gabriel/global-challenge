
import { Sidebar } from "@/components/Sidebar";
import { AlertCard } from "@/components/AlertCard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bell, Filter, RefreshCw, MapPin } from "lucide-react";

const Alertas = () => {
  const alerts = [
    {
      level: "high" as const,
      location: "Porto Alegre - Zona Norte",
      message: "Previsão de chuvas intensas nas próximas 6 horas. Risco elevado de alagamentos nos bairros Sarandi e Rubem Berta.",
      time: "há 2 min"
    },
    {
      level: "high" as const,
      location: "Porto Alegre - Centro Histórico",
      message: "Nível do Guaíba atingiu 95% da capacidade. Evacuação preventiva recomendada na orla.",
      time: "há 8 min"
    },
    {
      level: "medium" as const,
      location: "Porto Alegre - Zona Leste",
      message: "Nível dos arroios acima do normal. Monitoramento contínuo ativado para Restinga e Lami.",
      time: "há 15 min"
    },
    {
      level: "medium" as const,
      location: "Porto Alegre - Bairro Menino Deus",
      message: "Acúmulo de água em vias principais. Trânsito reduzido na Av. Praia de Belas.",
      time: "há 22 min"
    },
    {
      level: "high" as const,
      location: "Porto Alegre - Zona Sul",
      message: "Combinação de maré alta no Guaíba com chuvas fortes. Risco de inundação elevado em Cristal.",
      time: "há 35 min"
    },
    {
      level: "low" as const,
      location: "Porto Alegre - Moinhos de Vento",
      message: "Condições meteorológicas estáveis. Baixo risco de enchentes previsto para as próximas 24h.",
      time: "há 1 hora"
    },
    {
      level: "medium" as const,
      location: "Porto Alegre - Bairro Cidade Baixa",
      message: "Chuvas moderadas previstas. Atenção especial às áreas históricas de alagamento.",
      time: "há 1 hora"
    },
    {
      level: "low" as const,
      location: "Porto Alegre - Zona Oeste",
      message: "Sistema meteorológico estável. Condições favoráveis sem risco de enchentes.",
      time: "há 2 horas"
    }
  ];

  const highAlerts = alerts.filter(alert => alert.level === "high").length;
  const mediumAlerts = alerts.filter(alert => alert.level === "medium").length;
  const lowAlerts = alerts.filter(alert => alert.level === "low").length;

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        <div className="p-6 lg:p-8">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Central de Alertas - Porto Alegre
              </h1>
              <p className="text-gray-600">
                Monitoramento em tempo real de riscos de enchentes
              </p>
            </div>
            <div className="flex space-x-3 mt-4 sm:mt-0">
              <Button variant="outline" size="sm">
                <Filter className="h-4 w-4 mr-2" />
                Filtrar
              </Button>
              <Button variant="outline" size="sm">
                <RefreshCw className="h-4 w-4 mr-2" />
                Atualizar
              </Button>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <Card className="border-red-200 bg-red-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-red-600">Alto Risco</p>
                    <p className="text-3xl font-bold text-red-700">{highAlerts}</p>
                  </div>
                  <div className="p-3 bg-red-100 rounded-full">
                    <Bell className="h-6 w-6 text-red-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-yellow-200 bg-yellow-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-yellow-600">Risco Moderado</p>
                    <p className="text-3xl font-bold text-yellow-700">{mediumAlerts}</p>
                  </div>
                  <div className="p-3 bg-yellow-100 rounded-full">
                    <Bell className="h-6 w-6 text-yellow-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-blue-200 bg-blue-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-blue-600">Baixo Risco</p>
                    <p className="text-3xl font-bold text-blue-700">{lowAlerts}</p>
                  </div>
                  <div className="p-3 bg-blue-100 rounded-full">
                    <Bell className="h-6 w-6 text-blue-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-green-200 bg-green-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-green-600">Total de Bairros</p>
                    <p className="text-3xl font-bold text-green-700">82</p>
                  </div>
                  <div className="p-3 bg-green-100 rounded-full">
                    <MapPin className="h-6 w-6 text-green-600" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Active Alerts */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  <Bell className="h-5 w-5 text-red-600" />
                  <span>Alertas Ativos</span>
                  <Badge variant="destructive">{alerts.length}</Badge>
                </CardTitle>
                <div className="text-sm text-gray-500">
                  Última atualização: há 2 minutos
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {alerts.map((alert, index) => (
                  <AlertCard key={index} {...alert} />
                ))}
              </div>
              
              {alerts.length === 0 && (
                <div className="text-center py-12">
                  <Bell className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Nenhum alerta ativo
                  </h3>
                  <p className="text-gray-500">
                    Todos os bairros de Porto Alegre estão em condições normais.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default Alertas;
