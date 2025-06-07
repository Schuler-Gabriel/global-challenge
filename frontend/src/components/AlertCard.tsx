
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Info, Shield } from "lucide-react";

interface AlertProps {
  level: "low" | "medium" | "high";
  location: string;
  message: string;
  time: string;
}

const alertConfig = {
  low: {
    icon: Info,
    bgColor: "bg-blue-50 border-blue-200",
    iconColor: "text-blue-600",
    badgeColor: "bg-blue-100 text-blue-800",
    label: "Baixo Risco"
  },
  medium: {
    icon: AlertTriangle,
    bgColor: "bg-yellow-50 border-yellow-200",
    iconColor: "text-yellow-600",
    badgeColor: "bg-yellow-100 text-yellow-800",
    label: "Risco Moderado"
  },
  high: {
    icon: Shield,
    bgColor: "bg-red-50 border-red-200",
    iconColor: "text-red-600",
    badgeColor: "bg-red-100 text-red-800",
    label: "Alto Risco"
  }
};

export function AlertCard({ level, location, message, time }: AlertProps) {
  const config = alertConfig[level];
  const Icon = config.icon;

  return (
    <Card className={`${config.bgColor} hover:shadow-lg transition-all duration-300`}>
      <CardContent className="p-4">
        <div className="flex items-start space-x-3">
          <div className={`p-2 rounded-full bg-white ${config.iconColor}`}>
            <Icon className="h-4 w-4" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-2">
              <Badge className={config.badgeColor}>
                {config.label}
              </Badge>
              <span className="text-xs text-gray-500">{time}</span>
            </div>
            <h4 className="text-sm font-semibold text-gray-900 mb-1">
              {location}
            </h4>
            <p className="text-sm text-gray-600">
              {message}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
