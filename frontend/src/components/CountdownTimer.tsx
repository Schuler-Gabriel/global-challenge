
import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Clock } from "lucide-react";

export function CountdownTimer() {
  const [timeLeft, setTimeLeft] = useState({
    hours: 0,
    minutes: 0,
    seconds: 0,
  });

  useEffect(() => {
    const updateTimer = () => {
      const now = new Date();
      const nextPrediction = new Date();
      nextPrediction.setHours(24, 0, 0, 0); // Next day at midnight
      
      if (now.getHours() >= 9) {
        // If it's past 9 AM, next prediction is tomorrow at 9 AM
        nextPrediction.setDate(nextPrediction.getDate() + 1);
        nextPrediction.setHours(9, 0, 0, 0);
      } else {
        // If it's before 9 AM, next prediction is today at 9 AM
        nextPrediction.setHours(9, 0, 0, 0);
      }

      const diff = nextPrediction.getTime() - now.getTime();
      
      if (diff > 0) {
        const hours = Math.floor(diff / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((diff % (1000 * 60)) / 1000);
        
        setTimeLeft({ hours, minutes, seconds });
      }
    };

    updateTimer();
    const interval = setInterval(updateTimer, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Card className="water-gradient text-white">
      <CardContent className="p-6">
        <div className="flex items-center space-x-4">
          <div className="bg-white/20 p-3 rounded-full">
            <Clock className="h-6 w-6 animate-pulse" />
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-1">Próxima Previsão IA</h3>
            <div className="flex items-center space-x-2 text-2xl font-bold">
              <span className="bg-white/20 px-3 py-1 rounded-lg">
                {String(timeLeft.hours).padStart(2, '0')}
              </span>
              <span>:</span>
              <span className="bg-white/20 px-3 py-1 rounded-lg">
                {String(timeLeft.minutes).padStart(2, '0')}
              </span>
              <span>:</span>
              <span className="bg-white/20 px-3 py-1 rounded-lg">
                {String(timeLeft.seconds).padStart(2, '0')}
              </span>
            </div>
            <p className="text-sm text-white/80 mt-1">
              Atualização automática diária às 09:00
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
