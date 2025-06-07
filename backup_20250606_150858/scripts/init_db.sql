-- Inicialização do banco de dados para Sistema de Alertas de Cheias
-- Este script é executado automaticamente quando o container PostgreSQL é criado

-- Extensões necessárias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Schema principal
CREATE SCHEMA IF NOT EXISTS alerta_cheias;

-- Tabela de dados meteorológicos históricos
CREATE TABLE IF NOT EXISTS alerta_cheias.weather_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    temperature DECIMAL(5,2),
    humidity DECIMAL(5,2),
    pressure DECIMAL(7,2),
    wind_speed DECIMAL(5,2),
    wind_direction INTEGER,
    precipitation DECIMAL(6,2),
    river_level DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tabela de previsões geradas
CREATE TABLE IF NOT EXISTS alerta_cheias.forecasts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    forecast_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version VARCHAR(50) NOT NULL,
    rain_probability DECIMAL(5,2),
    precipitation_forecast DECIMAL(6,2),
    confidence_score DECIMAL(5,4),
    input_features JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tabela de alertas emitidos
CREATE TABLE IF NOT EXISTS alerta_cheias.alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_level VARCHAR(20) NOT NULL CHECK (alert_level IN ('Baixo', 'Moderado', 'Alto', 'Crítico')),
    action_required VARCHAR(50) NOT NULL,
    river_level DECIMAL(5,2) NOT NULL,
    rain_prediction DECIMAL(5,2) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- Tabela de métricas do modelo
CREATE TABLE IF NOT EXISTS alerta_cheias.model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    evaluation_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    dataset_type VARCHAR(20) CHECK (dataset_type IN ('train', 'validation', 'test')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tabela de logs de API externa
CREATE TABLE IF NOT EXISTS alerta_cheias.external_api_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_name VARCHAR(50) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    success BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_data JSONB
);

-- Índices para performance
CREATE INDEX IF NOT EXISTS idx_weather_data_timestamp ON alerta_cheias.weather_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_weather_data_created_at ON alerta_cheias.weather_data(created_at);
CREATE INDEX IF NOT EXISTS idx_forecasts_timestamp ON alerta_cheias.forecasts(forecast_timestamp);
CREATE INDEX IF NOT EXISTS idx_forecasts_generated_at ON alerta_cheias.forecasts(generated_at);
CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerta_cheias.alerts(alert_level);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerta_cheias.alerts(is_active);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerta_cheias.alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_model_metrics_version ON alerta_cheias.model_metrics(model_version);
CREATE INDEX IF NOT EXISTS idx_api_logs_timestamp ON alerta_cheias.external_api_logs(request_timestamp);
CREATE INDEX IF NOT EXISTS idx_api_logs_api_name ON alerta_cheias.external_api_logs(api_name);

-- Triggers para updated_at automático
CREATE OR REPLACE FUNCTION alerta_cheias.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_weather_data_updated_at 
    BEFORE UPDATE ON alerta_cheias.weather_data 
    FOR EACH ROW EXECUTE FUNCTION alerta_cheias.update_updated_at_column();

-- View para últimas condições meteorológicas
CREATE OR REPLACE VIEW alerta_cheias.latest_conditions AS
SELECT 
    w1.*
FROM alerta_cheias.weather_data w1
INNER JOIN (
    SELECT MAX(timestamp) as max_timestamp
    FROM alerta_cheias.weather_data
    WHERE timestamp <= NOW()
) w2 ON w1.timestamp = w2.max_timestamp;

-- View para alertas ativos
CREATE OR REPLACE VIEW alerta_cheias.active_alerts AS
SELECT 
    *
FROM alerta_cheias.alerts
WHERE is_active = TRUE
ORDER BY created_at DESC;

-- View para métricas de performance da API
CREATE OR REPLACE VIEW alerta_cheias.api_performance AS
SELECT 
    api_name,
    DATE_TRUNC('hour', request_timestamp) as hour_bucket,
    COUNT(*) as total_requests,
    COUNT(*) FILTER (WHERE success = TRUE) as successful_requests,
    AVG(response_time_ms) as avg_response_time,
    MAX(response_time_ms) as max_response_time,
    (COUNT(*) FILTER (WHERE success = TRUE) * 100.0 / COUNT(*)) as success_rate
FROM alerta_cheias.external_api_logs
WHERE request_timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY api_name, DATE_TRUNC('hour', request_timestamp)
ORDER BY hour_bucket DESC;

-- Função para limpeza automática de dados antigos
CREATE OR REPLACE FUNCTION alerta_cheias.cleanup_old_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Limpar logs de API com mais de 30 dias
    DELETE FROM alerta_cheias.external_api_logs 
    WHERE request_timestamp < NOW() - INTERVAL '30 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Limpar alertas resolvidos com mais de 90 dias
    DELETE FROM alerta_cheias.alerts 
    WHERE is_active = FALSE 
    AND resolved_at < NOW() - INTERVAL '90 days';
    
    -- Limpar previsões antigas (manter apenas 1 ano)
    DELETE FROM alerta_cheias.forecasts 
    WHERE generated_at < NOW() - INTERVAL '1 year';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Inserir dados iniciais de exemplo (opcional)
INSERT INTO alerta_cheias.weather_data (
    timestamp, temperature, humidity, pressure, wind_speed, 
    wind_direction, precipitation, river_level
) VALUES
    (NOW() - INTERVAL '1 hour', 22.5, 75.2, 1013.2, 5.4, 180, 0.0, 1.85),
    (NOW() - INTERVAL '2 hours', 23.1, 73.8, 1012.8, 6.2, 175, 0.2, 1.82),
    (NOW() - INTERVAL '3 hours', 24.0, 71.5, 1012.1, 7.1, 185, 0.5, 1.89)
ON CONFLICT DO NOTHING;

-- Comentários para documentação
COMMENT ON TABLE alerta_cheias.weather_data IS 'Dados meteorológicos históricos e em tempo real';
COMMENT ON TABLE alerta_cheias.forecasts IS 'Previsões geradas pelo modelo ML';
COMMENT ON TABLE alerta_cheias.alerts IS 'Alertas de cheia emitidos pelo sistema';
COMMENT ON TABLE alerta_cheias.model_metrics IS 'Métricas de performance dos modelos ML';
COMMENT ON TABLE alerta_cheias.external_api_logs IS 'Logs de chamadas para APIs externas';

COMMENT ON COLUMN alerta_cheias.weather_data.river_level IS 'Nível do rio em metros';
COMMENT ON COLUMN alerta_cheias.weather_data.precipitation IS 'Precipitação em mm/h';
COMMENT ON COLUMN alerta_cheias.forecasts.confidence_score IS 'Score de confiança da previsão (0-1)';
COMMENT ON COLUMN alerta_cheias.alerts.alert_level IS 'Nível do alerta: Baixo, Moderado, Alto, Crítico';

-- Conceder permissões (ajustar conforme necessário)
GRANT USAGE ON SCHEMA alerta_cheias TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA alerta_cheias TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA alerta_cheias TO postgres;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA alerta_cheias TO postgres; 