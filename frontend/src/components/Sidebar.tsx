
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Link, useLocation } from "react-router-dom";
import { 
  LayoutDashboard, 
  Bell, 
  History, 
  TrendingUp, 
  Menu,
  X
} from "lucide-react";
import { useState } from "react";

const navigation = [
  {
    name: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    name: "Alertas",
    href: "/alertas",
    icon: Bell,
  },
  {
    name: "Histórico",
    href: "/historico",
    icon: History,
  },
  {
    name: "Previsões",
    href: "/previsoes",
    icon: TrendingUp,
  },
];

export function Sidebar() {
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);

  const toggleSidebar = () => setIsOpen(!isOpen);

  return (
    <>
      {/* Mobile menu button */}
      <Button
        variant="outline"
        size="icon"
        className="fixed top-4 left-4 z-50 lg:hidden"
        onClick={toggleSidebar}
      >
        {isOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
      </Button>

      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={toggleSidebar}
        />
      )}

      {/* Sidebar */}
      <div
        className={cn(
          "fixed left-0 top-0 z-40 h-full w-64 transform bg-white shadow-xl transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0",
          isOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex h-full flex-col">
          {/* Header */}
          <div className="flex h-20 items-center justify-center border-b water-gradient">
            <h1 className="text-xl font-bold text-white">
              FloodWatch Brasil
            </h1>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-2 p-4">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setIsOpen(false)}
                  className={cn(
                    "flex items-center space-x-3 rounded-lg px-3 py-3 text-sm font-medium transition-all duration-200 hover:scale-105",
                    isActive
                      ? "water-gradient text-white shadow-lg"
                      : "text-gray-700 hover:bg-blue-50 hover:text-blue-700"
                  )}
                >
                  <item.icon className={cn("h-5 w-5", isActive && "animate-pulse")} />
                  <span>{item.name}</span>
                  {isActive && (
                    <div className="ml-auto h-2 w-2 rounded-full bg-white animate-pulse" />
                  )}
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="border-t p-4">
            <div className="glass-effect rounded-lg p-3 text-center">
              <p className="text-xs text-gray-600">
                Sistema de Monitoramento
              </p>
              <p className="text-xs font-semibold text-blue-600">
                24h Ativo
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
