
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Link } from "react-router-dom";
import { ArrowRight, Shield, TrendingUp, Bell, MapPin } from "lucide-react";

const Index = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0 water-gradient"></div>
        <div className="absolute inset-0 bg-black/30"></div>
        
        {/* Animated background elements */}
        <div className="absolute top-20 left-10 w-32 h-32 bg-white/10 rounded-full animate-float"></div>
        <div className="absolute bottom-20 right-10 w-24 h-24 bg-white/10 rounded-full animate-float" style={{animationDelay: '2s'}}></div>
        <div className="absolute top-1/2 right-1/4 w-16 h-16 bg-white/10 rounded-full animate-float" style={{animationDelay: '4s'}}></div>
        
        <div className="relative z-10 text-center max-w-6xl mx-auto px-6">
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 animate-fade-in">
            Sistema de Controle de
            <span className="block text-white drop-shadow-lg">
              Enchentes em Porto Alegre
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-white/90 mb-8 max-w-3xl mx-auto animate-fade-in drop-shadow-md" style={{animationDelay: '0.5s'}}>
            Previsões inteligentes, alertas em tempo real e monitoramento contínuo 
            para proteger comunidades de Porto Alegre contra enchentes e inundações.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center animate-fade-in" style={{animationDelay: '1s'}}>
            <Link to="/dashboard">
              <Button size="lg" className="bg-white text-blue-600 hover:bg-gray-100 text-lg px-8 py-4">
                Acessar Sistema
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
            <Button variant="outline" size="lg" className="bg-white/10 text-white border-white hover:bg-white hover:text-blue-600 backdrop-blur-sm text-lg px-8 py-4">
              Saiba Mais
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Tecnologia Avançada para Prevenção
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Nossa plataforma utiliza inteligência artificial e dados meteorológicos 
              em tempo real para oferecer previsões precisas e alertas antecipados para Porto Alegre.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <Card className="text-center hover:shadow-xl transition-all duration-300 hover:scale-105">
              <CardContent className="p-8">
                <div className="w-16 h-16 water-gradient rounded-full flex items-center justify-center mx-auto mb-4">
                  <TrendingUp className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3">IA Preditiva</h3>
                <p className="text-gray-600">
                  Previsões diárias automatizadas usando machine learning e dados meteorológicos
                </p>
              </CardContent>
            </Card>
            
            <Card className="text-center hover:shadow-xl transition-all duration-300 hover:scale-105">
              <CardContent className="p-8">
                <div className="w-16 h-16 alert-gradient rounded-full flex items-center justify-center mx-auto mb-4">
                  <Bell className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Alertas em Tempo Real</h3>
                <p className="text-gray-600">
                  Notificações instantâneas sobre riscos de enchentes em Porto Alegre
                </p>
              </CardContent>
            </Card>
            
            <Card className="text-center hover:shadow-xl transition-all duration-300 hover:scale-105">
              <CardContent className="p-8">
                <div className="w-16 h-16 nature-gradient rounded-full flex items-center justify-center mx-auto mb-4">
                  <MapPin className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Monitoramento Regional</h3>
                <p className="text-gray-600">
                  Cobertura completa da região metropolitana de Porto Alegre
                </p>
              </CardContent>
            </Card>
            
            <Card className="text-center hover:shadow-xl transition-all duration-300 hover:scale-105">
              <CardContent className="p-8">
                <div className="w-16 h-16 water-gradient rounded-full flex items-center justify-center mx-auto mb-4">
                  <Shield className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3">Prevenção Eficaz</h3>
                <p className="text-gray-600">
                  Histórico detalhado e análises para melhor preparação e resposta
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 nature-gradient">
        <div className="max-w-4xl mx-auto text-center px-6">
          <h2 className="text-4xl font-bold text-white mb-6">
            Comece a Monitorar Agora
          </h2>
          <p className="text-xl text-white/90 mb-8">
            Acesso gratuito ao sistema de previsão de enchentes de Porto Alegre
          </p>
          <Link to="/dashboard">
            <Button size="lg" className="bg-white text-green-600 hover:bg-gray-100 text-lg px-12 py-4">
              Entrar no Dashboard
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </Link>
        </div>
      </section>
    </div>
  );
};

export default Index;
