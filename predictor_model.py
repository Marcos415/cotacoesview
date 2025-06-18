import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os
import datetime

# REMOVIDO: from app import get_historical_prices_yfinance_cached
# Agora, a função que busca o histórico será passada como argumento para train_and_predict_price

# Número de dias anteriores para usar na previsão (recursos de 'lag')
N_DAYS = 5

def train_and_predict_price(symbol, get_historical_prices_func, N_DAYS=N_DAYS):
    """
    Treina um modelo de regressão linear para prever o preço de fechamento
    de um ativo usando os N dias anteriores de fechamento como features.
    
    Salva o modelo treinado num ficheiro .joblib e retorna a previsão para o próximo dia.

    Args:
        symbol (str): O símbolo do ativo (ticker) a ser previsto (ex: 'PETR4.SA', 'BTC-USD').
        get_historical_prices_func (function): A função a ser usada para obter os dados históricos.
                                                Deve aceitar (symbol, period, interval) e retornar um DataFrame.
        N_DAYS (int): O número de dias de fechamento anteriores a serem usados como features.

    Returns:
        float: O preço de fechamento previsto para o próximo dia, ou None em caso de erro.
    """
    print(f"Iniciando treinamento e previsão para: {symbol} com N_DAYS={N_DAYS}")

    # Usa a função de histórico passada como argumento
    df_hist = get_historical_prices_func(symbol, period="1y", interval="1d") # Período de 1 ano para treinamento

    if df_hist.empty or len(df_hist) < N_DAYS + 1: # Precisamos de pelo menos N_DAYS + 1 para criar uma amostra
        print(f"Erro: Dados históricos insuficientes para treinar o modelo para {symbol}. Requer pelo menos {N_DAYS + 1} dias.")
        return None

    # Garantir que a coluna 'Close' é numérica
    df_hist['Close'] = pd.to_numeric(df_hist['Close'], errors='coerce')
    df_hist.dropna(subset=['Close'], inplace=True)

    if df_hist.empty or len(df_hist) < N_DAYS + 1:
        print(f"Erro: Dados 'Close' insuficientes após limpeza para treinar o modelo para {symbol}.")
        return None

    # Criar features de lag (dias anteriores de fechamento)
    for i in range(1, N_DAYS + 1):
        df_hist[f'Close_lag_{i}'] = df_hist['Close'].shift(i)

    df_hist.dropna(inplace=True) # Remove linhas com valores NaN (devido ao shift)

    if df_hist.empty:
        print(f"Erro: Dados insuficientes para treinamento após criar features de lag para {symbol}.")
        return None

    # Definir features (X) e target (y)
    features = [f'Close_lag_{i}' for i in range(1, N_DAYS + 1)]
    X = df_hist[features]
    y = df_hist['Close']

    # Treinar o modelo de Regressão Linear
    model = LinearRegression()
    model.fit(X, y)
    print(f"Modelo de regressão linear treinado para {symbol}.")

    # Salvar o modelo treinado
    # Sanitiza o nome do símbolo para o nome do ficheiro
    sanitized_symbol = symbol.replace(".SA", "").replace(".", "").replace("^", "").replace("-", "").replace("=", "").replace(" ", "_")
    model_filename = f'model_{sanitized_symbol}.joblib'
    joblib.dump(model, model_filename)
    print(f"Modelo salvo como {model_filename}")

    # Fazer a previsão para o próximo dia
    # Precisamos dos últimos N_DAYS de fechamento do df_hist original (sem as linhas dropadas do shift)
    # Buscamos novamente os últimos N_DAYS de forma independente para garantir que são os mais recentes
    # Isto é crucial porque df_hist.dropna() pode remover os últimos dias se forem NaN.
    # O period 'N_DAYS+5d' é uma margem de segurança caso haja fins de semana/feriados
    df_last_N_days = get_historical_prices_func(symbol, period=f"{N_DAYS+5}d", interval="1d")
    df_last_N_days['Close'] = pd.to_numeric(df_last_N_days['Close'], errors='coerce')
    df_last_N_days.dropna(subset=['Close'], inplace=True)


    if df_last_N_days.empty or len(df_last_N_days) < N_DAYS:
        print(f"Aviso: Não foi possível obter os últimos {N_DAYS} dias de fechamento para previsão para {symbol}.")
        return None

    last_N_closes = df_last_N_days['Close'].tail(N_DAYS).values.astype(float).tolist()

    if len(last_N_closes) < N_DAYS:
        print(f"Erro: Não há dados suficientes (menos de {N_DAYS} dias) para criar as features para previsão para {symbol}.")
        return None
    
    # Criar DataFrame para a previsão
    future_features = {}
    for i in range(1, N_DAYS + 1):
        future_features[f'Close_lag_{i}'] = [last_N_closes[N_DAYS - i]] # O lag 1 é o último dia, lag 2 o penúltimo, etc.
    
    df_future_features = pd.DataFrame(future_features)
    
    predicted_price = model.predict(df_future_features)[0]
    print(f"Previsão para {symbol} para o próximo dia: {predicted_price:.2f}")

    return predicted_price

# Exemplo de uso (apenas para teste, não será executado diretamente pelo Flask)
if __name__ == '__main__':
    # Para testar este ficheiro independentemente, você precisaria de uma mock function
    # para get_historical_prices_func
    def mock_get_historical_prices(symbol, period, interval):
        print(f"Mock: Buscando histórico para {symbol} (period={period}, interval={interval})")
        # Retorna um DataFrame de exemplo
        return pd.DataFrame({
            'Close': [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', 
                                '2023-01-06', '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12',
                                '2022-12-26', '2022-12-27', '2022-12-28', '2022-12-29', '2022-12-30'])
        ).sort_index()

    # Exemplo de chamada da função com a mock function
    # predicted = train_and_predict_price('TESTE', mock_get_historical_prices)
    # if predicted:
    #     print(f"Previsão de teste: {predicted:.2f}")
    # else:
    #     print("Não foi possível gerar previsão de teste.")
    pass