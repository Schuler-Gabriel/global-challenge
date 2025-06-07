
import { Sidebar } from "@/components/Sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Calendar, Download, TrendingDown, TrendingUp, History } from "lucide-react";

const Historico = () => {
  const historicalEvents = [
    {
      date: "15 Mar 2024",
      location: "Porto Alegre - Zona Norte",
      severity: "high",
      description: "Enchente severa causada por chuvas torrenciais de 120mm em 3 horas no bairro Sarandi",
      impact: "1.200 pessoas afetadas",
      duration: "6 horas",
      status: "Resolvido"
    },
    {
      date: "08 Mar 2024", 
      location: "Porto Alegre - Centro Histórico",
      severity: "medium",
      description: "Alagamentos pontuais em vias principais durante temporal",
      impact: "450 pessoas afetadas",
      duration: "3 horas",
      status: "Resolvido"
    },
    {
      date: "22 Fev 2024",
      location: "Porto Alegre - Zona Sul",
      severity: "high",
      description: "Transbordamento do Arroio Dilúvio após chuvas intensas",
      impact: "2.100 pessoas afetadas",
      duration: "8 horas",
      status: "Resolvido"
    },
    {
      date: "14 Fev 2024",
      location: "Porto Alegre - Orla do Guaíba",
      severity: "medium",
      description: "Maré alta do Guaíba combinada com chuvas causou inundações costeiras",
      impact: "800 pessoas afetadas",
      duration: "4 horas",
      status: "Resolvido"
    },
    {
      date: "05 Fev 2024",
      location: "Porto Alegre - Moinhos de Vento",
      severity: "low",
      description: "Acúmulo de água em pontos isolados do bairro",
      impact: "85 pessoas afetadas",
      duration: "2 horas",
      status: "Resolvido"
    },
    {
      date: "28 Jan 2024",
      location: "Porto Alegre - Zona Leste",
      severity: "high",
      description: "Enchente do Arroio Cavalhada afetou bairros residenciais",
      impact: "1.800 pessoas afetadas",
      duration: "12 horas",
      status: "Resolvido"
    }
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high": return "bg-red-100 text-red-800 border-red-200";
      case "medium": return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "low": return "bg-blue-100 text-blue-800 border-blue-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getSeverityLabel = (severity: string) => {
    switch (severity) {
      case "high": return "Alto";
      case "medium": return "Moderado";
      case "low": return "Baixo";
      default: return "Desconhecido";
    }
  };

  const totalEvents = historicalEvents.length;
  const highSeverityEvents = historicalEvents.filter(e => e.severity === "high").length;
  const totalAffected = historicalEvents.reduce((acc, event) => {
    const num = parseInt(event.impact.replace(/[^\d]/g, ''));
    return acc + num;
  }, 0);

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      
      <main className="flex-1 overflow-auto">
        <div className="p-6 lg:p-8">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">
                Histórico de Eventos - Porto Alegre
              </h1>
              <p className="text-gray-600">
                Registro completo de enchentes e inundações monitoradas
              </p>
            </div>
            <div className="flex space-x-3 mt-4 sm:mt-0">
              <Button variant="outline" size="sm">
                <Calendar className="h-4 w-4 mr-2" />
                Filtrar Período
              </Button>
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Exportar
              </Button>
            </div>
          </div>

          {/* Statistics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <Card className="border-blue-200 bg-blue-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-blue-600">Total de Eventos</p>
                    <p className="text-3xl font-bold text-blue-700">{totalEvents}</p>
                    <p className="text-xs text-gray-600 mt-1">Últimos 3 meses</p>
                  </div>
                  <div className="p-3 bg-blue-100 rounded-full">
                    <History className="h-6 w-6 text-blue-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-red-200 bg-red-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-red-600">Eventos Críticos</p>
                    <p className="text-3xl font-bold text-red-700">{highSeverityEvents}</p>
                    <p className="text-xs text-red-600 flex items-center mt-1">
                      <TrendingDown className="h-3 w-3 mr-1" />
                      -30% vs mês anterior
                    </p>
                  </div>
                  <div className="p-3 bg-red-100 rounded-full">
                    <TrendingDown className="h-6 w-6 text-red-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-orange-200 bg-orange-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-orange-600">Pessoas Afetadas</p>
                    <p className="text-3xl font-bold text-orange-700">{totalAffected.toLocaleString()}</p>
                    <p className="text-xs text-orange-600 flex items-center mt-1">
                      <TrendingUp className="h-3 w-3 mr-1" />
                      Total acumulado
                    </p>
                  </div>
                  <div className="p-3 bg-orange-100 rounded-full">
                    <TrendingUp className="h-6 w-6 text-orange-600" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-green-200 bg-green-50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-green-600">Taxa de Resolução</p>
                    <p className="text-3xl font-bold text-green-700">100%</p>
                    <p className="text-xs text-green-600 mt-1">Todos resolvidos</p>
                  </div>
                  <div className="p-3 bg-green-100 rounded-full">
                    <TrendingUp className="h-6 w-6 text-green-600" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Historical Events List */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <History className="h-5 w-5 text-gray-600" />
                <span>Eventos Registrados</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {historicalEvents.map((event, index) => (
                  <div key={index} className="border rounded-lg p-6 hover:shadow-md transition-shadow duration-300">
                    <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-4">
                      <div className="flex items-center space-x-4 mb-2 lg:mb-0">
                        <div className="text-sm font-medium text-gray-600">
                          {event.date}
                        </div>
                        <Badge className={getSeverityColor(event.severity)}>
                          Severidade {getSeverityLabel(event.severity)}
                        </Badge>
                        <Badge variant="secondary">
                          {event.status}
                        </Badge>
                      </div>
                      <div className="text-sm text-gray-500">
                        Duração: {event.duration}
                      </div>
                    </div>
                    
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      {event.location}
                    </h3>
                    
                    <p className="text-gray-700 mb-3">
                      {event.description}
                    </p>
                    
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">
                        <strong>Impacto:</strong> {event.impact}
                      </span>
                      <Button variant="ghost" size="sm">
                        Ver Detalhes
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-8 text-center">
                <Button variant="outline">
                  Carregar Mais Eventos
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default Historico;
