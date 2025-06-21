import datetime
import decimal
import os
import joblib
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

import requests
from bs4 import BeautifulSoup

from fpdf import FPDF

import plotly.graph_objects as go
import plotly.utils
import json

# Importa a biblioteca para carregar variáveis de ambiente do .env
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env (DEVE SER UMA DAS PRIMEIRAS LINHAS)
load_dotenv()

# Importa apenas train_and_predict_price
from predictor_model import train_and_predict_price

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError("A variável de ambiente 'SECRET_KEY' não está definida. Por favor, defina-a no seu arquivo .env.")
else:
    print(f"DEBUG: SECRET_KEY carregada com sucesso (primeiros 5 caracteres): {app.secret_key[:5]}*****")

CORS(app)

# --- Constantes de Mapeamento de Símbolos ---
SYMBOL_MAPPING = {
    'PETROBRAS': 'PETR4.SA',
    'VALE': 'VALE3.SA',
    'ITAU': 'ITUB4.SA',
    'BRADESCO': 'BBDC4.SA',
    'AMBEV': 'ABEV3.SA',
    'B3': 'B3SA3.SA',
    'WEG': 'WEGE3.SA',
    'MAGALU': 'MGLU3.SA',
    'AMERICANAS': 'AMER3.SA',
    'ELETROBRAS': 'ELET3.SA',
    'COSAN': 'CSAN3.SA',
    'IBOVESPA': '^BVSP',
    'BITCOIN': 'BTC-USD',
    'ETHEREUM': 'ETH-USD',
    'OURO_FUTURO': 'GC=F',
    'NASDAQ_COMPOSITE': '^IXIC',
    'RUMO S.A.': 'RAIL3.SA',
    'DOW_JONES': '^DJI'
}

REVERSE_SYMBOL_MAPPING = {}
for name, ticker in SYMBOL_MAPPING.items():
    REVERSE_SYMBOL_MAPPING[ticker] = name
    if '.SA' in ticker:
        REVERSE_SYMBOL_MAPPING[ticker.replace('.SA', '')] = name # Permite buscar PETR4 e mapear para PETROBRAS
    elif '^' in ticker or '-' in ticker or '=' in ticker:
        REVERSE_SYMBOL_MAPPING[ticker] = name
    else:
        REVERSE_SYMBOL_MAPPING[ticker] = name # Se não for .SA, mantem o ticker original (ex: BTC-USD)


# --- REGISTRA VARIÁVEIS GLOBAIS NO AMBIENTE JINJA2 ---
app.jinja_env.globals['now'] = datetime.datetime.now
app.jinja_env.globals['SYMBOL_MAPPING'] = SYMBOL_MAPPING
app.jinja_env.globals['REVERSE_SYMBOL_MAPPING'] = REVERSE_SYMBOL_MAPPING

# --- CONFIGURAÇÃO DO CACHE ---
market_data_cache = {}
news_cache = {}
portfolio_cache = {}
prediction_cache = {}
historical_chart_cache = {}

MARKET_DATA_CACHE_TTL = 300 # 5 minutos
NEWS_CACHE_TTL = 3600      # 1 hora
PORTFOLIO_CACHE_TTL = 120  # 2 minutos
PREDICTION_CACHE_TTL = 600 # 10 minutos
HISTORICAL_CHART_CACHE_TTL = 3600 # 1 hora para gráficos históricos

def is_cache_fresh(cache, key, ttl):
    """Verifica se um item no cache ainda é válido."""
    if key in cache:
        timestamp = cache[key]['timestamp']
        if (datetime.datetime.now() - timestamp).total_seconds() < ttl:
            return True
    return False

# --- FIM DA CONFIGURAÇÃO DO CACHE ---

# --- CONFIGURAÇÃO DA API DE NOTÍCIAS ---
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

# --- Constante para Paginação ---
TRANSACTIONS_PER_PAGE = 6

# --- Registro do filtro 'datetimeformat' para Jinja2 ---
@app.template_filter('datetimeformat')
def datetimeformat(value, format_string='%Y-%m-%d'):
    """
    Formata um objeto datetime ou a string 'now' para o formato desejado.
    """
    if value == 'now':
        dt = datetime.datetime.now()
    elif isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, datetime.date):
        dt = datetime.datetime(value.year, value.month, value.day)
    elif isinstance(value, datetime.time): # Add handling for datetime.time objects
        # For time objects, return a formatted string directly
        return value.strftime(format_string)
    elif value is None:
        return 'N/A'
    else:
        try:
            # Handle psycopg2.time objects which might not be directly datetime.time
            if isinstance(value, datetime.timedelta):
                total_seconds = int(value.total_seconds())
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                # Construct a time object for consistent formatting
                dt = datetime.time(hours, minutes, seconds)
                return dt.strftime(format_string)
            else:
                dt = datetime.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except ValueError:
            return str(value)
    return dt.strftime(format_string)

# --- Registro do filtro 'floatformat' para Jinja2 ---
@app.template_filter('floatformat')
def floatformat(value, precision=2):
    """
    Formata um número float para uma determinada precisão de casas decimais.
    """
    try:
        if value is None:
            return f"0.{'0' * precision}"
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return value

# --- Configurações do Banco de Dados ---
DATABASE_URL = os.getenv('DATABASE_URL')

if DATABASE_URL:
    DB_TYPE = 'postgresql'
    print("DEBUG: Usando conexão PostgreSQL (DATABASE_URL detectada).")
else:
    DB_TYPE = 'mysql'
    print("DEBUG: Usando conexão MySQL (DATABASE_URL não detectada).")
    DB_CONFIG_MYSQL = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_DATABASE'),
        'port': int(os.getenv('DB_PORT', 3306))
    }
    if not all([DB_CONFIG_MYSQL['user'], DB_CONFIG_MYSQL['password'], DB_CONFIG_MYSQL['database']]):
        raise ValueError("Uma ou mais variáveis de ambiente do banco de dados MySQL (DB_USER, DB_PASSWORD, DB_DATABASE) não estão definidas para uso local. Por favor, defina-as no seu arquivo .env.")

# --- Context Manager para Conexões Unificado (MySQL ou PostgreSQL) ---
class DBConnectionManager:
    def __init__(self, dictionary=False, buffered=False):
        self.conn = None
        self.cursor = None
        self.dictionary = dictionary
        self.buffered = buffered
        self.db_type = DB_TYPE

    def __enter__(self):
        print(f"DEBUG: DBConnectionManager - Entrando no contexto ({self.db_type}, buffered={self.buffered})...")
        try:
            if self.db_type == 'mysql':
                import mysql.connector
                self.conn = mysql.connector.connect(**DB_CONFIG_MYSQL)
                self.cursor = self.conn.cursor(dictionary=self.dictionary, buffered=self.buffered)
            elif self.db_type == 'postgresql':
                import psycopg2
                from urllib.parse import urlparse
                result = urlparse(DATABASE_URL)
                username = result.username
                password = result.password
                database = result.path[1:]
                hostname = result.hostname
                port = result.port if result.port else 5432

                self.conn = psycopg2.connect(
                    database=database,
                    user=username,
                    password=password,
                    host=hostname,
                    port=port
                )
                if self.dictionary:
                    import psycopg2.extras
                    self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                else:
                    self.cursor = self.conn.cursor()
            print("DEBUG: DBConnectionManager - Conexão e cursor estabelecidos.")
            return self.cursor
        except Exception as err:
            print(f"ERRO: DBConnectionManager - Erro ao conectar ao {self.db_type}: {err}")
            if self.cursor:
                try: self.cursor.close()
                except Exception as close_err:
                    print(f"ERRO: DBConnectionManager - Erro ao fechar cursor em __enter__ após falha: {close_err}")
            if self.conn:
                try: self.conn.close()
                except Exception as close_err:
                    print(f"ERRO: DBConnectionManager - Erro ao fechar conexão em __enter__ após falha: {close_err}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"DEBUG: DBConnectionManager - Saindo do contexto (exc_type={exc_type})...")
        if self.cursor:
            try:
                self.cursor.close()
                print("DEBUG: DBConnectionManager - Cursor fechado.")
            except Exception as err:
                print(f"ERRO: DBConnectionManager - Erro ao fechar cursor no __exit__: {err}")
        if self.conn:
            try:
                if exc_type is None:
                    self.conn.commit()
                    print("DEBUG: DBConnectionManager - Conexão commitada.")
                else:
                    print(f"DEBUG: DBConnectionManager - Transação será revertida devido a {exc_val}")
                    self.conn.rollback()
            except Exception as err:
                print(f"ERRO: DBConnectionManager - Erro ao comitar/reverter no __exit__: {err}")
            finally:
                try:
                    self.conn.close()
                    print("DEBUG: DBConnectionManager - Conexão fechada.")
                except Exception as err:
                    print(f"ERRO: DBConnectionManager - Erro ao fechar conexão no __exit__: {err}")


# --- Decorators para Autenticação e Autorização ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Você precisa estar logado para acessar esta página.', 'danger')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        print(f"DEBUG: admin_required - user_id na sessão: {session.get('user_id')}, is_admin na sessão: {session.get('is_admin')}")
        if not session.get('is_admin'):
            flash('Acesso negado: Você não tem permissões de administrador.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# --- Funções Auxiliares para Autenticação e Gestão de Utilizadores ---
def get_user_by_id(user_id):
    try:
        with DBConnectionManager(dictionary=True, buffered=True) as cursor_db:
            cursor_db.execute("SELECT id, username, full_name, email, contact_number, is_admin, created_at FROM users WHERE id = %s", (user_id,))
            user_data = cursor_db.fetchone()
            print(f"DEBUG: get_user_by_id({user_id}) - Dados do utilizador: {user_data}")
            return user_data
    except Exception as err:
        print(f"Erro ao buscar utilizador por ID: {err}")
        return None

def get_user_by_username(username):
    try:
        with DBConnectionManager(dictionary=True, buffered=True) as cursor_db:
            cursor_db.execute("SELECT id, username, password_hash, full_name, email, contact_number, is_admin FROM users WHERE username = %s", (username,))
            user_data = cursor_db.fetchone()
            print(f"DEBUG: get_user_by_username({username}) - Dados do utilizador: {user_data}")
            return user_data
    except Exception as err:
        print(f"Erro ao buscar utilizador por nome de utilizador: {err}")
        return None

def get_user_by_email(email):
    try:
        with DBConnectionManager(dictionary=True, buffered=True) as cursor_db:
            cursor_db.execute("SELECT id, username, full_name, email, contact_number, is_admin FROM users WHERE email = %s", (email,))
            user_data = cursor_db.fetchone()
            return user_data
    except Exception as err:
        print(f"Erro ao buscar utilizador por email: {err}")
        return None

def get_all_users():
    users = []
    current_user_id = session.get('user_id')
    print(f"DEBUG: get_all_users() - current_user_id na sessão: {current_user_id}")

    if current_user_id is None:
        print("DEBUG: get_all_users() - user_id não encontrado na sessão. Retornando lista vazia.")
        return []

    try:
        with DBConnectionManager(dictionary=True, buffered=True) as cursor_db:
            sql_query = "SELECT id, username, full_name, email, contact_number, is_admin, created_at FROM users"
            cursor_db.execute(sql_query)
            all_users_from_db = cursor_db.fetchall()
            print(f"DEBUG: get_all_users() - Todos os utilizadores do DB: {all_users_from_db}")
            users = [u for u in all_users_from_db if u['id'] != current_user_id]
            print(f"DEBUG: get_all_users() - Utilizadores após filtrar o admin logado: {users}")
    except Exception as err:
        print(f"Erro ao buscar todos os utilizadores: {err}")
    return users

def get_admin_count():
    try:
        with DBConnectionManager(dictionary=True, buffered=True) as cursor_db:
            cursor_db.execute("SELECT COUNT(*) as admin_count FROM users WHERE is_admin = TRUE")
            result = cursor_db.fetchone()
            count = result['admin_count'] if result else 0
            print(f"DEBUG: get_admin_count() - Total de administradores: {count}")
            return count
    except Exception as e:
        print(f"Erro ao contar administradores: {e}")
        return 0

def delete_user_from_db(user_id):
    print(f"DEBUG: Tentando excluir utilizador com ID: {user_id}")
    try:
        with DBConnectionManager() as cursor_db:
            cursor_db.execute("DELETE FROM users WHERE id = %s", (user_id,))
            rows_affected = cursor_db.rowcount
            print(f"DEBUG: delete_user_from_db - Linhas afetadas na tabela users: {rows_affected}")
            return rows_affected > 0
    except Exception as err:
        print(f"Erro ao excluir utilizador: {err}")
        return False

def update_user_password(user_id, new_password):
    print(f"DEBUG: Tentando redefinir senha para utilizador com ID: {user_id}")
    try:
        hashed_password = generate_password_hash(new_password)
        with DBConnectionManager() as cursor_db:
            cursor_db.execute("UPDATE users SET password_hash = %s WHERE id = %s", (hashed_password, user_id))
            rows_affected = cursor_db.rowcount
            print(f"DEBUG: update_user_password - Linhas afetadas: {rows_affected}")
            return rows_affected > 0
    except Exception as err:
        print(f"Erro ao atualizar palavra-passe: {err}")
        return False

def toggle_user_admin_status(user_id, new_status):
    print(f"DEBUG: Tentando definir is_admin para utilizador {user_id} como {new_status}")
    try:
        with DBConnectionManager() as cursor_db:
            sql = "UPDATE users SET is_admin = %s WHERE id = %s"
            cursor_db.execute(sql, (new_status, user_id))
            rows_affected = cursor_db.rowcount
            print(f"DEBUG: toggle_user_admin_status - Linhas afetadas: {rows_affected}")
            return rows_affected > 0
    except Exception as err:
        print(f"Erro ao alternar status de admin: {err}")
        return False

def update_user_profile_data(user_id, full_name, email, contact_number):
    try:
        with DBConnectionManager() as cursor_db:
            sql = """
            UPDATE users
            SET full_name = %s, email = %s, contact_number = %s
            WHERE id = %s
            """
            cursor_db.execute(sql, (full_name, email, contact_number, user_id))
            rows_affected = cursor_db.rowcount
            print(f"DEBUG: update_user_profile_data - Linhas afetadas: {rows_affected}")
            return rows_affected > 0
    except Exception as e:
        print(f"Erro inesperado ao atualizar dados do perfil: {e}")
        return False

# --- FUNÇÃO: Registar Ações de Administrador ---
def log_admin_action(admin_user_id, action_type, target_user_id=None, details=None):
    try:
        admin_username = session.get('username')
        if not admin_username:
            admin_user_data = get_user_by_id(admin_user_id)
            admin_username = admin_user_data['username'] if admin_user_data else f"ID_Desconhecido_{admin_user_id}"

        target_username = None
        if target_user_id:
            target_user_data = get_user_by_id(target_user_id)
            target_username = target_user_data['username'] if target_user_data else f"ID_Deletado_{target_user_id}"

        with DBConnectionManager() as cursor_db:
            sql = """
            INSERT INTO admin_audit_logs (admin_user_id, admin_username_at_action, action_type, target_user_id, target_username_at_action, details, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            details_to_save = details
            if details is not None and not isinstance(details, str):
                details_to_save = json.dumps(details)

            cursor_db.execute(sql, (admin_user_id, admin_username, action_type, target_user_id, target_username, details_to_save, datetime.datetime.now()))
    except Exception as err:
        print(f"ERRO ao registar ação de admin no log: {err}")

# --- Função Auxiliar para Buscar Transações (agora por utilizador) ---
def buscar_transacoes_filtradas(user_id, data_inicio, data_fim, ordenar_por, ordem, simbolo_filtro=None, page=1, per_page=TRANSACTIONS_PER_PAGE):
    transacoes = []
    total_transacoes = 0
    try:
        print(f"DEBUG FILTERS: user_id={user_id}, data_inicio={data_inicio}, data_fim={data_fim}, ordenar_por={ordenar_por}, ordem={ordem}, simbolo_filtro='{simbolo_filtro}' (raw)")

        with DBConnectionManager(dictionary=True) as cursor_db:
            final_simbolo_to_fetch_for_filter = None

            # Ajuste aqui para pegar o símbolo mapeado se for do SYMBOL_MAPPING, senão usa o que veio
            if simbolo_filtro and simbolo_filtro.lower() != 'none':
                # Primeiro tenta o mapeamento direto (nome amigável -> ticker YF)
                final_simbolo_to_fetch_for_filter = SYMBOL_MAPPING.get(simbolo_filtro.upper(), simbolo_filtro)
                
                # Se não encontrou mapeamento direto E não parece ser um ticker YF padrão, tenta inferir .SA
                if final_simbolo_to_fetch_for_filter == simbolo_filtro and \
                   not any(c in simbolo_filtro for c in ['^', '-', '=']) and \
                   not any(simbolo_filtro.upper().endswith(suf) for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
                    if 4 <= len(simbolo_filtro) <= 6 and simbolo_filtro.isalnum():
                        final_simbolo_to_fetch_for_filter = f"{simbolo_filtro.upper()}.SA"
            
            print(f"DEBUG FILTERS: Simbolo filtro final para busca: '{final_simbolo_to_fetch_for_filter}'")

            params = [user_id]
            where_clauses = ["user_id = %s"]

            if data_inicio:
                where_clauses.append("data_transacao >= %s")
                params.append(data_inicio)

            if data_fim:
                where_clauses.append("data_transacao <= %s")
                params.append(data_fim)

            if final_simbolo_to_fetch_for_filter:
                where_clauses.append("simbolo_ativo = %s")
                params.append(final_simbolo_to_fetch_for_filter)

            where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

            count_query = f"SELECT COUNT(*) AS total FROM transacoes{where_sql}"
            print(f"DEBUG FILTERS: Count Query: {count_query}, Params: {params}")
            cursor_db.execute(count_query, tuple(params))
            total_transacoes = cursor_db.fetchone()['total']
            print(f"DEBUG FILTERS: Total Transações: {total_transacoes}")

            query = f"SELECT * FROM transacoes{where_sql}"

            valid_columns = ['data_transacao', 'simbolo_ativo', 'preco_unitario', 'quantidade', 'tipo_operacao', 'id', 'custos_taxas', 'hora_transacao', 'observacoes']
            if ordenar_por not in valid_columns:
                ordenar_por = 'data_transacao'

            valid_orders = ['ASC', 'DESC']
            if ordem.upper() not in valid_orders:
                ordem = 'DESC'

            query += f" ORDER BY {ordenar_por} {ordem}"

            offset = (page - 1) * per_page
            query += f" LIMIT {per_page} OFFSET {offset}"

            print(f"DEBUG FILTERS: Main Query: {query}, Params: {params}")
            cursor_db.execute(query, tuple(params))
            transacoes = cursor_db.fetchall()

            for transacao in transacoes:
                if isinstance(transacao.get('hora_transacao'), datetime.timedelta):
                    total_seconds = int(transacao['hora_transacao'].total_seconds())
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    transacao['hora_transacao'] = datetime.time(hours, minutes, seconds)
                elif transacao.get('hora_transacao') is None:
                    transacao['hora_transacao'] = None

    except Exception as err:
        print(f"ERRO ao buscar transações: {err}")
        return [], 0
    return transacoes, total_transacoes

# --- Nova Função para Buscar Alertas de Preço ---
def buscar_alertas(user_id):
    alertas = []
    try:
        # Adiciona verificação explícita da existência da tabela antes de consultar
        if not check_table_exists('alertas'):
            print("AVISO: Tabela 'alertas' não existe ao tentar buscar alertas. Retornando lista vazia.")
            return [] # Retorna lista vazia se a tabela não existe
            
        with DBConnectionManager(dictionary=True) as cursor_db:
            # Garante que 'alertas' é acessado em minúsculas
            sql = 'SELECT id, user_id, simbolo_ativo, preco_alvo, tipo_alerta, status, data_criacao, data_disparo FROM alertas WHERE user_id = %s ORDER BY data_criacao DESC'
            cursor_db.execute(sql, (user_id,))
            alertas = cursor_db.fetchall()
    except Exception as err:
        # AVISO: A tabela 'alertas' pode não existir na primeira execução ou após um reset do DB.
        # Imprime o erro, mas não lança, permitindo que a app continue a funcionar sem alertas.
        # Isso é esperado até que o 'create_tables_if_not_exist' garanta a criação.
        print(f"AVISO: Erro ao buscar alertas: {err}")
    return alertas

# --- Função para obter o histórico de preços com Cache (usado por _get_current_price_yfinance e predictor_model) ---
def get_historical_prices_yfinance_cached(simbolo, period, interval):
    final_simbolo_to_fetch = SYMBOL_MAPPING.get(simbolo.upper(), simbolo)
    if final_simbolo_to_fetch == simbolo and \
       not any(c in simbolo for c in ['^', '-', '=']) and \
       not any(simbolo.upper().endswith(suf)for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
        if 4 <= len(simbolo) <= 6 and simbolo.isalnum():
            final_simbolo_to_fetch = f"{simbolo.upper()}.SA"

    cache_key = (final_simbolo_to_fetch, period, interval)

    if is_cache_fresh(market_data_cache, cache_key, MARKET_DATA_CACHE_TTL):
        return market_data_cache[cache_key]['data'].copy()

    try:
        ticker = yf.Ticker(final_simbolo_to_fetch)
        df_hist = ticker.history(period=period, interval=interval)
        if not df_hist.empty:
            market_data_cache[cache_key] = {'data': df_hist, 'timestamp': datetime.datetime.now()}
            return df_hist.copy()
        return pd.DataFrame()
    except Exception as e:
        print(f"Erro ao buscar dados históricos para {final_simbolo_to_fetch}: {e}")
        return pd.DataFrame()


# --- Função Auxiliar para Buscar Cotação Atual (Yfinance) com Cache ---
def _get_current_price_yfinance(simbolo):
    final_simbolo_to_fetch = SYMBOL_MAPPING.get(simbolo.upper(), simbolo)
    if final_simbolo_to_fetch == simbolo and \
       not any(c in simbolo for c in ['^', '-', '=']) and \
       not any(simbolo.upper().endswith(suf)for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
        if 4 <= len(simbolo) <= 6 and simbolo.isalnum():
            final_simbolo_to_fetch = f"{simbolo.upper()}.SA"

    cache_key = (final_simbolo_to_fetch, "1d", "1m")

    if is_cache_fresh(market_data_cache, cache_key, MARKET_DATA_CACHE_TTL):
        return market_data_cache[cache_key]['data']

    try:
        ticker = yf.Ticker(final_simbolo_to_fetch)
        hist = ticker.history(period="1d", interval="1m")

        if hist.empty:
            hist = ticker.history(period="5d", interval="1d")

        if not hist.empty:
            latest_price = None
            if 'Adj Close' in hist.columns:
                latest_price = hist['Adj Close'].iloc[-1]
            elif 'Close' in hist.columns:
                latest_price = hist['Close'].iloc[-1]
            else:
                print(f"DEBUG: Nenhuma coluna 'Adj Close' ou 'Close' encontrada para {final_simbolo_to_fetch} no DataFrame histórico. Colunas presentes: {hist.columns.tolist()}")
                return None

            if pd.isna(latest_price):
                print(f"DEBUG: Preço encontrado para {final_simbolo_to_fetch} é NaN. Retornando None.")
                return None

            market_data_cache[cache_key] = {'data': float(latest_price), 'timestamp': datetime.datetime.now()}
            return float(latest_price)
        else:
            print(f"DEBUG: Cotação para {final_simbolo_to_fetch} não encontrada ou dados indisponíveis após múltiplas tentativas.")
            return None

    except Exception as e:
        print(f"DEBUG: Erro ao buscar cotação atual para {final_simbolo_to_fetch}: {e}")
        return None


# --- Função para obter a previsão do modelo com Cache ---
def get_predicted_price_for_display(simbolo):
    simbolo_yf_for_prediction = SYMBOL_MAPPING.get(simbolo.upper(), simbolo)
    if simbolo_yf_for_prediction == simbolo and \
       not any(c in simbolo for c in ['^', '-', '=']) and \
       not any(simbolo.upper().endswith(suf)for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
        if 4 <= len(simbolo) <= 6 and simbolo.isalnum():
            simbolo_yf_for_prediction = f"{simbolo.upper()}.SA"

    cache_key = simbolo_yf_for_prediction

    if is_cache_fresh(prediction_cache, cache_key, PREDICTION_CACHE_TTL):
        return prediction_cache[cache_key]['data']

    predicted_price = None

    predicted_price = train_and_predict_price(simbolo_yf_for_prediction, get_historical_prices_yfinance_cached)

    if predicted_price is not None:
        prediction_cache[cache_key] = {'data': float(predicted_price), 'timestamp': datetime.datetime.now()}
        return float(predicted_price)
    return None

# --- Função para Calcular Preço Médio e Quantidade Total na Carteira (com Cache) ---
def calcular_posicoes_carteira(user_id):
    cache_key = user_id

    if is_cache_fresh(portfolio_cache, cache_key, PORTFOLIO_CACHE_TTL):
        return portfolio_cache[cache_key]['data']

    posicoes = {}
    total_valor_carteira = 0.0
    total_lucro_nao_realizado = 0.0
    total_prejuizo_nao_realizado = 0.0

    try:
        with DBConnectionManager(dictionary=True) as cursor_db:
            sql_transacoes = """
            SELECT
                simbolo_ativo,
                tipo_operacao,
                quantidade,
                preco_unitario,
                custos_taxas
            FROM
                transacoes
            WHERE
                user_id = %s
            ORDER BY
                simbolo_ativo, data_transacao ASC, hora_transacao ASC;
            """
            cursor_db.execute(sql_transacoes, (user_id,))
            todas_transacoes = cursor_db.fetchall()

            estado_ativo = {}

            for transacao in todas_transacoes:
                simbolo = transacao['simbolo_ativo']
                tipo = transacao['tipo_operacao']

                # Convert Decimal to float for calculations if necessary, or use Decimal consistently
                quantidade = float(transacao['quantidade'])
                preco_unitario = float(transacao['preco_unitario'])
                custos_taxas = float(transacao['custos_taxas'])

                if simbolo not in estado_ativo:
                    estado_ativo[simbolo] = {'quantidade': 0.0, 'custo_acumulado': 0.0}

                if tipo == 'COMPRA':
                    estado_ativo[simbolo]['quantidade'] += quantidade
                    estado_ativo[simbolo]['custo_acumulado'] += (quantidade * preco_unitario) + custos_taxas
                elif tipo == 'VENDA':
                    if estado_ativo[simbolo]['quantidade'] > 0:
                        if quantidade <= estado_ativo[simbolo]['quantidade']:
                            # Calcula o custo médio antes da venda
                            custo_medio_atual = estado_ativo[simbolo]['custo_acumulado'] / estado_ativo[simbolo]['quantidade']
                            custo_das_vendidas = quantidade * custo_medio_atual

                            estado_ativo[simbolo]['quantidade'] -= quantidade
                            estado_ativo[simbolo]['custo_acumulado'] -= (custo_das_vendidas + custos_taxas)

                            if estado_ativo[simbolo]['quantidade'] <= 0.00001: # Lida com imprecisões de float
                                estado_ativo[simbolo]['quantidade'] = 0.0
                                estado_ativo[simbolo]['custo_acumulado'] = 0.0
                        else: # Venda de mais do que o que se tem, zera a posição
                            estado_ativo[simbolo]['quantidade'] = 0.0
                            estado_ativo[simbolo]['custo_acumulado'] = 0.0
                    else:
                        pass # Ignora vendas se não houver quantidade para vender

            for simbolo, dados_posicao in estado_ativo.items():
                if dados_posicao['quantidade'] > 0:
                    preco_medio = float(dados_posicao['custo_acumulado']) / float(dados_posicao['quantidade'])

                    preco_atual = _get_current_price_yfinance(simbolo)
                    preco_previsto = get_predicted_price_for_display(simbolo)

                    lucro_prejuizo_nao_realizado_individual = None
                    if preco_atual is not None:
                        lucro_prejuizo_nao_realizado_individual = (preco_atual - preco_medio) * float(dados_posicao['quantidade'])
                    else:
                        lucro_prejuizo_nao_realizado_individual = 0.0
                    
                    valor_total_ativo = (preco_atual * dados_posicao['quantidade']) if preco_atual is not None else 0.0

                    posicoes[simbolo] = {
                        'quantidade': dados_posicao['quantidade'],
                        'preco_medio': preco_medio,
                        'preco_atual': preco_atual,
                        'preco_previsto': preco_previsto,
                        'valor_atual': valor_total_ativo, # Used by pie chart
                        'lucro_prejuizo_nao_realizado': lucro_prejuizo_nao_realizado_individual,
                        'nome_popular': REVERSE_SYMBOL_MAPPING.get(simbolo, simbolo.replace('.SA', '').upper()), # Add nome_popular
                        'quantidade_total': dados_posicao['quantidade'], # For clarity in the template (already exists as 'quantidade')
                        'valor_total_ativo': valor_total_ativo # For clarity in the template
                    }
                    total_valor_carteira += posicoes[simbolo]['valor_atual']

                    if lucro_prejuizo_nao_realizado_individual is not None:
                        if lucro_prejuizo_nao_realizado_individual > 0:
                            total_lucro_nao_realizado += lucro_prejuizo_nao_realizado_individual
                        else:
                            total_prejuizo_nao_realizado += lucro_prejuizo_nao_realizado_individual

    except Exception as e:
        print(f"ERRO inesperado ao calcular posições da carteira: {e}")
        return {}, 0.0, 0.0, 0.0

    portfolio_cache[cache_key] = {
        'data': (posicoes, total_valor_carteira, total_lucro_nao_realizado, total_prejuizo_nao_realizado),
        'timestamp': datetime.datetime.now()
    }
    return posicoes, total_valor_carteira, total_lucro_nao_realizado, total_prejuizo_nao_realizado

# --- Funções para Notícias ---
def fetch_news(query):
    cache_key = query
    if is_cache_fresh(news_cache, cache_key, NEWS_CACHE_TTL):
        return news_cache[cache_key]['data']

    if not NEWS_API_KEY:
        print("NEWS_API_KEY não está configurada. Não será possível buscar notícias.")
        return []

    params = {
        'q': query,
        'language': 'pt',
        'sortBy': 'relevancy',
        'apiKey': NEWS_API_KEY
    }
    try:
        response = requests.get(NEWS_API_BASE_URL, params=params)
        response.raise_for_status()
        news_data = response.json()
        articles = news_data.get('articles', [])
        filtered_articles = [
            {
                'title': art.get('title'),
                'description': art.get('description'),
                'url': art.get('url'),
                'source': art.get('source', {}).get('name'),
                'publishedAt': datetime.datetime.fromisoformat(art['publishedAt'].replace('Z', '+00:00')) if 'publishedAt' in art else None
            }
            for art in articles if art.get('title') and art.get('description') and art.get('url')
        ]
        news_cache[cache_key] = {'data': filtered_articles, 'timestamp': datetime.datetime.now()}
        return filtered_articles
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar notícias: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON da API de notícias: {e}. Resposta: {response.text}")
        return []

# --- FUNÇÃO: Verificar se uma tabela existe ---
def check_table_exists(table_name):
    print(f"DEBUG: Verificando se a tabela '{table_name}' existe...")
    try:
        with DBConnectionManager(dictionary=True) as cursor_db:
            if DB_TYPE == 'postgresql':
                # Verifica se a tabela existe em minúsculas (comportamento padrão)
                cursor_db.execute(f"SELECT to_regclass('{table_name}') IS NOT NULL AS exists_table;")
                result_default_case = cursor_db.fetchone()
                exists_default = result_default_case['exists_table'] if result_default_case else False

                # Verifica se a tabela existe com o nome exato (se foi criada com aspas duplas)
                cursor_db.execute(f"SELECT to_regclass('\"{table_name}\"') IS NOT NULL AS exists_table_quoted;")
                result_quoted_case = cursor_db.fetchone()
                exists_quoted = result_quoted_case['exists_table_quoted'] if result_quoted_case else False

                exists = exists_default or exists_quoted
                print(f"DEBUG: Tabela '{table_name}' existe (padrão/minúsculas): {exists_default}, (com aspas/exato): {exists_quoted}, Resultado final: {exists}")
                return exists
            else: # MySQL (mantém como estava)
                cursor_db.execute(f"SELECT COUNT(*) AS exists_table FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = '{table_name}';")
                result = cursor_db.fetchone()
                exists = result['exists_table']
                print(f"DEBUG: Tabela '{table_name}' existe: {exists}")
                return exists
    except Exception as e:
        print(f"ERRO: Falha ao verificar a existência da tabela '{table_name}': {e}")
        return False

# --- NOVA FUNÇÃO: Criar tabelas se não existirem ---
def create_tables_if_not_exist():
    print("DEBUG: Iniciando verificação e criação de tabelas...")
    try:
        with DBConnectionManager() as cursor_db:
            if DB_TYPE == 'postgresql':
                # Tabela 'users'
                print("DEBUG: Executando SQL: CREATE TABLE IF NOT EXISTS users (PostgreSQL)...")
                cursor_db.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(80) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        full_name VARCHAR(255) NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        contact_number VARCHAR(20),
                        is_admin BOOLEAN DEFAULT FALSE NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                print("DEBUG: Comando SQL para 'users' executado.")
                cursor_db.connection.commit() # Commit explícito

                # Tabela 'transacoes' (depende de 'users')
                print("DEBUG: Executando SQL: CREATE TABLE IF NOT EXISTS transacoes (PostgreSQL)...")
                cursor_db.execute("""
                    CREATE TABLE IF NOT EXISTS transacoes (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        data_transacao DATE NOT NULL,
                        hora_transacao TIME,
                        simbolo_ativo VARCHAR(20) NOT NULL,
                        quantidade DECIMAL(18, 8) NOT NULL,
                        preco_unitario DECIMAL(18, 8) NOT NULL,
                        tipo_operacao VARCHAR(10) NOT NULL,
                        custos_taxas DECIMAL(10, 2) DEFAULT 0.00,
                        observacoes TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                """)
                print("DEBUG: Comando SQL para 'transacoes' executado.")
                cursor_db.connection.commit() # Commit explícito

                # Tabela 'alertas' (depende de 'users') -- Melhorando a resiliência para PostgreSQL
                alertas_table_name = "alertas" # Nome da tabela que queremos que o PG veja em minúsculas
                print(f"DEBUG: Verificando e garantindo tabela '{alertas_table_name}' (PostgreSQL) com drops robustos...")
                
                # Tenta dropar versões maiúsculas ou com aspas que podem ter sido criadas anteriormente por engano
                try:
                    cursor_db.execute(f'DROP TABLE IF EXISTS "{alertas_table_name.upper()}" CASCADE;')
                    cursor_db.connection.commit() # Commit explícito após DROP
                    print(f"DEBUG: DROP TABLE IF EXISTS \"{alertas_table_name.upper()}\" CASCADE executado e commitado (ignora se não existir).")
                except Exception as drop_err:
                    print(f"AVISO: Erro ao tentar dropar tabela maiúscula '{alertas_table_name.upper()}': {drop_err}. Ignorado se não existir.")

                try:
                    cursor_db.execute(f'DROP TABLE IF EXISTS {alertas_table_name} CASCADE;') # Dropar a versão minúscula/padrão
                    cursor_db.connection.commit() # Commit explícito após DROP
                    print(f"DEBUG: DROP TABLE IF EXISTS {alertas_table_name} CASCADE executado e commitado (ignora se não existir).")
                except Exception as drop_err:
                    print(f"AVISO: Erro ao tentar dropar tabela minúscula '{alertas_table_name}': {drop_err}. Ignorado se não existir.")


                sql_create_alertas = f"""
                    CREATE TABLE {alertas_table_name} ( -- SEM ASPAS para garantir que seja minúscula por padrão
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        simbolo_ativo VARCHAR(20) NOT NULL,
                        preco_alvo DECIMAL(10, 2) NOT NULL,
                        tipo_alerta VARCHAR(10) NOT NULL, -- 'ACIMA' ou 'ABAIXO'
                        status VARCHAR(20) DEFAULT 'ATIVO' NOT NULL, -- 'ATIVO', 'DISPARADO', 'CANCELADO'
                        data_criacao TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        data_disparo TIMESTAMP WITH TIME ZONE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                """
                print(f"DEBUG: Executando SQL CREATE TABLE para '{alertas_table_name}':\n{sql_create_alertas}")
                cursor_db.execute(sql_create_alertas)
                cursor_db.connection.commit() # Commit explícito após CREATE
                print(f"DEBUG: Comando SQL para '{alertas_table_name}' executado e commitado.")


                # Tabela 'admin_audit_logs'
                print("DEBUG: Executando SQL: CREATE TABLE IF NOT EXISTS admin_audit_logs (PostgreSQL)...")
                cursor_db.execute("""
                    CREATE TABLE IF NOT EXISTS admin_audit_logs (
                        id SERIAL PRIMARY KEY,
                        admin_user_id INTEGER NOT NULL,
                        admin_username_at_action VARCHAR(80) NOT NULL,
                        action_type VARCHAR(50) NOT NULL,
                        target_user_id INTEGER,
                        target_username_at_action VARCHAR(80),
                        details JSONB, -- PostgreSQL uses JSONB for JSON data
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                print("DEBUG: Comando SQL para 'admin_audit_logs' executado.")
                cursor_db.connection.commit() # Commit explícito

            else: # MySQL (Keep as is as MySQL isn't the problem here)
                # Tabela 'users'
                print("DEBUG: Executando SQL: CREATE TABLE IF NOT EXISTS users (MySQL)...")
                cursor_db.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(80) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        full_name VARCHAR(255) NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        contact_number VARCHAR(20),
                        is_admin BOOLEAN DEFAULT FALSE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                print("DEBUG: Comando SQL para 'users' executado.")
                cursor_db.connection.commit() # Commit explícito

                print("DEBUG: Executando SQL: CREATE TABLE IF NOT EXISTS transacoes (MySQL)...")
                cursor_db.execute("""
                    CREATE TABLE IF NOT EXISTS transacoes (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT NOT NULL,
                        data_transacao DATE NOT NULL,
                        hora_transacao TIME,
                        simbolo_ativo VARCHAR(20) NOT NULL,
                        quantidade DECIMAL(18, 8) NOT NULL,
                        preco_unitario DECIMAL(18, 8) NOT NULL,
                        tipo_operacao VARCHAR(10) NOT NULL,
                        custos_taxas DECIMAL(10, 2) DEFAULT 0.00,
                        observacoes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                """)
                print("DEBUG: Comando SQL para 'transacoes' executado.")
                cursor_db.connection.commit() # Commit explícito

                print("DEBUG: Executando SQL: CREATE TABLE IF NOT EXISTS alertas (MySQL)...") # NOME AGORA É 'alertas'
                cursor_db.execute("""
                    CREATE TABLE IF NOT EXISTS alertas (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT NOT NULL,
                        simbolo_ativo VARCHAR(20) NOT NULL,
                        preco_alvo DECIMAL(10, 2) NOT NULL,
                        tipo_alerta VARCHAR(10) NOT NULL,
                        status VARCHAR(20) DEFAULT 'ATIVO' NOT NULL,
                        data_criacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        data_disparo TIMESTAMP NULL,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                """)
                print("DEBUG: Comando SQL para 'alertas' executado.")
                cursor_db.connection.commit() # Commit explícito

                print("DEBUG: Executando SQL: CREATE TABLE IF NOT EXISTS admin_audit_logs (MySQL)...")
                cursor_db.execute("""
                    CREATE TABLE IF NOT EXISTS admin_audit_logs (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        admin_user_id INT NOT NULL,
                        admin_username_at_action VARCHAR(80) NOT NULL,
                        action_type VARCHAR(50) NOT NULL,
                        target_user_id INT,
                        target_username_at_action VARCHAR(80),
                        details JSON,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                print("DEBUG: Comando SQL para 'admin_audit_logs' executado.")
                cursor_db.connection.commit() # Commit explícito
            print("DEBUG: Todas as operações CREATE TABLE IF NOT EXISTS foram enviadas ao banco de dados.")
    except Exception as e:
        print(f"ERRO CRÍTICO: Falha ao tentar verificar/criar tabelas: {e}")
        raise # Re-lança a exceção para que a Render saiba que a inicialização falhou

# --- ROTAS DA APLICAÇÃO ---

@app.route('/')
@login_required
def index():
    user_id = session.get('user_id')
    user_data = get_user_by_id(user_id)
    if not user_data:
        flash('Sua sessão expirou ou o usuário não foi encontrado.', 'danger')
        return redirect(url_for('logout'))

    user_name = user_data['username']

    # Dados da Carteira
    posicoes, total_valor_carteira, total_lucro_nao_realizado, total_prejuizo_nao_realizado = calcular_posicoes_carteira(user_id)
    print(f"DEBUG: index route - Posições da carteira: {posicoes}")
    print(f"DEBUG: index route - Total valor carteira: {total_valor_carteira}")

    # Gráfico de Pizza
    chart_labels = []
    chart_values = []
    for simbolo, dados in posicoes.items():
        if dados['valor_atual'] > 0:
            chart_labels.append(dados['nome_popular']) # Use nome amigável para o label
            chart_values.append(dados['valor_atual'])

    pie_chart_json = None
    if chart_labels:
        fig_pie = go.Figure(data=[go.Pie(labels=chart_labels, values=chart_values, hole=.3)])
        fig_pie.update_layout(
            title_text='Alocação da Carteira por Ativo',
            title_font_size=20, showlegend=True, margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#333"), height=350
        )
        pie_chart_json = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
        print(f"DEBUG: index route - Pie chart JSON gerado.")
    else:
        print(f"DEBUG: index route - Nenhum dado para o gráfico de pizza. Labels: {chart_labels}, Values: {chart_values}")


    # Gráfico de Barras Lucro/Prejuízo Não Realizado
    profit_loss_labels = []
    profit_loss_values = []
    profit_loss_colors = []

    for simbolo, dados in posicoes.items():
        if dados['lucro_prejuizo_nao_realizado'] is not None:
            profit_loss_labels.append(dados['nome_popular']) # Use nome amigável para o label
            profit_loss_values.append(dados['lucro_prejuizo_nao_realizado'])
            profit_loss_colors.append('green' if dados['lucro_prejuizo_nao_realizado'] >= 0 else 'red')

    bar_chart_json = None
    if profit_loss_labels:
        bar_fig = go.Figure(data=[
            go.Bar(x=profit_loss_labels, y=profit_loss_values, marker_color=profit_loss_colors)
        ])
        bar_fig.update_layout(
            title_text='Lucro/Prejuízo Não Realizado por Ativo',
            title_font_size=20, xaxis_title='Ativo', yaxis_title='Lucro/Prejuízo',
            margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#333"), height=350
        )
        bar_chart_json = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)
        print(f"DEBUG: index route - Bar chart JSON gerado.")
    else:
        print(f"DEBUG: index route - Nenhum dado para o gráfico de barras. Labels: {profit_loss_labels}, Values: {profit_loss_values}")


    # Notícias
    all_news = []
    # Fetch general news
    general_news_articles = fetch_news("Mercado Financeiro Brasil")
    for article in general_news_articles:
        all_news.append({
            'title': article['title'],
            'description': article['description'],
            'url': article['url'],
            'source': article['source'],
            'publishedAt': article['publishedAt'],
            'simbolo_display': 'Geral' # Add a flag for general news
        })

    # Fetch news for top 3 assets in portfolio
    if posicoes:
        # Sort positions by current value to get top assets
        sorted_posicoes = sorted(posicoes.items(), key=lambda item: item[1]['valor_atual'], reverse=True)
        top_symbols_for_news = [s for s, _ in sorted_posicoes][:3] # Get only symbols

        for simbolo_yf in top_symbols_for_news:
            news_query_for_asset = REVERSE_SYMBOL_MAPPING.get(simbolo_yf, simbolo_yf.replace('.SA', '').upper())
            asset_news = fetch_news(news_query_for_asset)
            for article in asset_news:
                all_news.append({
                    'title': article['title'],
                    'description': article['description'],
                    'url': article['url'],
                    'source': article['source'],
                    'publishedAt': article['publishedAt'],
                    'simbolo_display': REVERSE_SYMBOL_MAPPING.get(simbolo_yf, simbolo_yf.replace('.SA', '').upper())
                })
    
    # Sort all news by published date, newest first
    all_news.sort(key=lambda x: x['publishedAt'] if x['publishedAt'] else datetime.datetime.min, reverse=True)
    # Format date for display in template
    for news_item in all_news:
        news_item['date'] = news_item['publishedAt'].strftime('%d/%m/%Y %H:%M') if news_item['publishedAt'] else 'N/A'
    
    print(f"DEBUG: index route - Total de notícias coletadas: {len(all_news)}")


    # Histórico de Transações para o Dashboard
    data_inicio = request.args.get('data_inicio')
    data_fim = request.args.get('data_fim')
    ordenar_por = request.args.get('ordenar_por', 'data_transacao')
    ordem = request.args.get('ordem', 'DESC')
    simbolo_filtro_raw = request.args.get('simbolo_filtro', '').strip()
    page = request.args.get('page', 1, type=int)

    simbolo_filtro_mapeado = SYMBOL_MAPPING.get(simbolo_filtro_raw.upper(), simbolo_filtro_raw)
    transacoes, total_transacoes = buscar_transacoes_filtradas(
        user_id, data_inicio, data_fim, ordenar_por, ordem, simbolo_filtro=simbolo_filtro_mapeado, page=page, per_page=TRANSACTIONS_PER_PAGE
    )
    total_pages = (total_transacoes + TRANSACTIONS_PER_PAGE - 1) // TRANSACTIONS_PER_PAGE
    print(f"DEBUG: index route - Transações: {len(transacoes)}, Total: {total_transacoes}")


    # Alertas de Preço
    alertas = buscar_alertas(user_id) # Esta função agora lida com a ausência da tabela
    print(f"DEBUG: index route - Alertas encontrados: {len(alertas)}")


    return render_template('index.html',
                           user_name=user_name,
                           posicoes_carteira=posicoes, # Changed to posicoes_carteira for template match
                           total_valor_carteira=total_valor_carteira,
                           total_lucro_nao_realizado=total_lucro_nao_realizado,
                           total_prejuizo_nao_realizado=total_prejuizo_nao_realizado,
                           all_news=all_news, # Changed to all_news for template match
                           pie_chart_json=pie_chart_json,
                           bar_chart_json=bar_chart_json,
                           transacoes=transacoes,
                           total_transacoes=total_transacoes,
                           data_inicio=data_inicio,
                           data_fim=data_fim,
                           ordenar_por=ordenar_por,
                           ordem=ordem,
                           simbolo_filtro=simbolo_filtro_raw,
                           page=page,
                           total_pages=total_pages,
                           alertas=alertas)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            print(f"DEBUG: ROTA /REGISTER (POST) - Headers da Requisição: {request.headers}")
            print(f"DEBUG: ROTA /REGISTER (POST) - Dados crus da requisição (request.get_data()): {request.get_data(as_text=True)}")
            print(f"DEBUG: ROTA /REGISTER (POST) - Conteúdo do Formulário (request.form): {request.form}")

            username = request.form['username']
            password = request.form['password']
            confirm_password = request.form.get('confirm_password')
            
            full_name = request.form['full_name']
            email = request.form['email']
            contact_number = request.form.get('contact_number')

            if not username or not password or not confirm_password or not full_name or not email:
                flash('Por favor, preencha todos os campos obrigatórios (incluindo a confirmação de palavra-passe).', 'danger')
                return render_template('register.html', username=username, full_name=full_name, email=email, contact_number=contact_number)

            if password != confirm_password:
                flash('As palavras-passe não coincidem.', 'danger')
                return render_template('register.html', username=username, full_name=full_name, email=email, contact_number=contact_number)

            if len(password) < 6:
                flash('A palavra-passe deve ter pelo menos 6 caracteres.', 'danger')
                return render_template('register.html', username=username, full_name=full_name, email=email, contact_number=contact_number)

            if get_user_by_username(username):
                flash('Nome de utilizador já registado. Por favor, escolha outro.', 'danger')
                return render_template('register.html', username=username, full_name=full_name, email=email, contact_number=contact_number)

            if get_user_by_email(email):
                flash('Endereço de email já registado. Por favor, use outro ou faça login.', 'danger')
                return render_template('register.html', username=username, full_name=full_name, email=email, contact_number=contact_number)

            hashed_password = generate_password_hash(password)

            is_admin = False
            if get_admin_count() == 0:
                is_admin = True

            with DBConnectionManager() as cursor_db:
                if DB_TYPE == 'postgresql':
                    sql = """
                    INSERT INTO users (username, password_hash, full_name, email, contact_number, is_admin, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id
                    """
                    cursor_db.execute(sql, (username, hashed_password, full_name, email, contact_number, is_admin, datetime.datetime.now()))
                    new_user_id = cursor_db.fetchone()[0]
                else: # MySQL
                    sql = """
                    INSERT INTO users (username, password_hash, full_name, email, contact_number, is_admin, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor_db.execute(sql, (username, hashed_password, full_name, email, contact_number, is_admin, datetime.datetime.now()))
                    new_user_id = cursor_db.lastrowid

                flash('Registo bem-sucedido! Pode agora fazer login.', 'success')

                if is_admin:
                    try:
                        log_admin_action(admin_user_id=new_user_id,
                                         action_type='PRIMEIRO_ADMIN_REGISTADO',
                                         target_user_id=new_user_id,
                                         details={'username': username, 'email': email})
                    except Exception as log_e:
                        print(f"ATENÇÃO: Erro ao logar ação de admin: {log_e}")

            return redirect(url_for('login'))
        except Exception as e:
            print(f"ERRO INESPERADO NA ROTA /REGISTER (POST): {e}")
            flash(f'Ocorreu um erro ao registar. Por favor, tente novamente mais tarde. Detalhes técnicos: {e}', 'danger')
            return render_template('register.html', username=request.form.get('username'), full_name=request.form.get('full_name'), email=request.form.get('email'), contact_number=request.form.get('contact_number'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        print(f"DEBUG: ROTA /LOGIN (POST) - Tentativa de login para username: '{username}'")
        print(f"DEBUG: ROTA /LOGIN (POST) - Senha recebida (parcial): '{password[:2]}*****'")

        user = get_user_by_username(username)

        if user:
            print(f"DEBUG: ROTA /LOGIN (POST) - Usuário '{username}' encontrado no DB. ID: {user['id']}, É Admin: {user['is_admin']}")
            print(f"DEBUG: ROTA /LOGIN (POST) - Hash da senha no DB: '{user['password_hash'][:5]}*****'")
        else:
            print(f"DEBUG: ROTA /LOGIN (POST) - Usuário '{username}' NÃO encontrado no DB.")

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])

            print(f"DEBUG: ROTA /LOGIN (POST) - Login BEM-SUCEDIDO para '{username}'.")

            flash('Login bem-sucedido!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            print(f"DEBUG: ROTA /LOGIN (POST) - Login FALHOU para '{username}'. Credenciais inválidas ou senha não corresponde ao hash.")
            flash('Nome de utilizador ou palavra-passe inválidos.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('is_admin', None)
    flash('Sessão encerrada com sucesso.', 'info')
    return redirect(url_for('login'))

@app.route('/transactions', methods=['GET', 'POST'])
@login_required
def transactions_list():
    user_id = session.get('user_id')
    user_name = session.get('username')

    data_inicio = request.args.get('data_inicio')
    data_fim = request.args.get('data_fim')
    ordenar_por = request.args.get('ordenar_por', 'data_transacao')
    ordem = request.args.get('ordem', 'DESC')
    simbolo_filtro_raw = request.args.get('simbolo_filtro', '').strip()

    page = request.args.get('page', 1, type=int)

    simbolo_filtro_mapeado = SYMBOL_MAPPING.get(simbolo_filtro_raw.upper(), simbolo_filtro_raw)

    transacoes, total_transacoes = buscar_transacoes_filtradas(
        user_id,
        data_inicio,
        data_fim,
        ordenar_por,
        ordem,
        simbolo_filtro=simbolo_filtro_mapeado,
        page=page,
        per_page=TRANSACTIONS_PER_PAGE
    )

    total_pages = (total_transacoes + TRANSACTIONS_PER_PAGE - 1) // TRANSACTIONS_PER_PAGE

    return render_template('transactions_list.html',
                           user_name=user_name,
                           transacoes=transacoes,
                           data_inicio=data_inicio,
                           data_fim=data_fim,
                           ordenar_por=ordenar_por,
                           ordem=ordem,
                           simbolo_filtro=simbolo_filtro_raw,
                           page=page,
                           total_pages=total_pages)


@app.route('/add_transaction', methods=['GET', 'POST'])
@login_required
def add_transaction():
    user_id = session.get('user_id')
    user_name = session.get('username')
    symbols = sorted(list(SYMBOL_MAPPING.keys()))

    print(f"DEBUG: add_transaction (GET/POST) - Symbols carregados para o template: {symbols}")

    if request.method == 'POST':
        print(f"DEBUG: add_transaction (POST) - request.form: {request.form}")

        # --- Lógica para obter o símbolo ativo (Simplificada) ---
        simbolo_ativo = request.form.get('simbolo_ativo') # Este campo agora virá do select ou do input manual
        simbolo_ativo_select_value = request.form.get('simbolo_ativo_select_hidden') # Valor do select, se não for OUTRO
        simbolo_ativo_manual_value = request.form.get('simbolo_ativo_manual_hidden') # Valor do input manual, se OUTRO

        # DEBUG: Adiciona logs para ver o que realmente está sendo recebido
        print(f"DEBUG: Simbolo ativo recebido (geral): '{simbolo_ativo}'")
        print(f"DEBUG: Simbolo ativo recebido (select hidden): '{simbolo_ativo_select_value}'")
        print(f"DEBUG: Simbolo ativo recebido (manual hidden): '{simbolo_ativo_manual_value}'")


        if not simbolo_ativo or simbolo_ativo.strip() == '':
            flash('Por favor, selecione ou digite um símbolo para o ativo.', 'danger')
            # Passa os valores de volta para o template para que os campos selem preenchidos
            return render_template('add_transaction.html', symbols=symbols,
                                   data_transacao=request.form.get('data_transacao'),
                                   hora_transacao=request.form.get('hora_transacao'),
                                   quantidade=request.form.get('quantidade'),
                                   preco_unitario=request.form.get('preco_unitario'),
                                   tipo_operacao=request.form.get('tipo_operacao'),
                                   custos_taxas=request.form.get('custos_taxas', '0.00'),
                                   observacoes=request.form.get('observacoes'),
                                   # Passa o simbolo_ativo_manual para manter o valor no campo
                                   simbolo_ativo_manual=simbolo_ativo_manual_value,
                                   simbolo_ativo_select=simbolo_ativo_select_value) # E o valor do select

        # Se o simbolo_ativo veio do campo manual (selecionou "OUTRO"), usamos ele.
        # Caso contrário (select normal ou já um ticker YF), usamos o que veio direto.
        final_simbolo_para_processamento = simbolo_ativo.strip().upper()


        data_transacao_str = request.form['data_transacao']
        try:
            data_transacao = datetime.datetime.strptime(data_transacao_str, '%Y-%m-%d').date()
        except ValueError as e:
            flash(f'Formato de data inválido. Use AAAA-MM-DD. Erro: {e}', 'danger')
            print(f"ERRO: Data inválida: {e}")
            return render_template('add_transaction.html', symbols=symbols,
                                   data_transacao=data_transacao_str,
                                   hora_transacao=request.form.get('hora_transacao'),
                                   simbolo_ativo_manual=simbolo_ativo_manual_value,
                                   simbolo_ativo_select=simbolo_ativo_select_value,
                                   quantidade=request.form.get('quantidade'),
                                   preco_unitario=request.form.get('preco_unitario'),
                                   tipo_operacao=request.form.get('tipo_operacao'),
                                   custos_taxas=request.form.get('custos_taxas', '0.00'),
                                   observacoes=request.form.get('observacoes'))

        hora_transacao_str = request.form.get('hora_transacao')
        hora_transacao = None
        if hora_transacao_str:
            try:
                hora_transacao = datetime.datetime.strptime(hora_transacao_str, '%H:%M').time()
            except ValueError as e:
                flash(f'Formato de hora inválido. Use HH:MM. Erro: {e}', 'danger')
                print(f"ERRO: Hora inválida: {e}")
                return render_template('add_transaction.html', symbols=symbols,
                                       data_transacao=data_transacao_str,
                                       hora_transacao=hora_transacao_str,
                                       simbolo_ativo_manual=simbolo_ativo_manual_value,
                                       simbolo_ativo_select=simbolo_ativo_select_value,
                                       quantidade=request.form.get('quantidade'),
                                       preco_unitario=request.form.get('preco_unitario'),
                                       tipo_operacao=request.form.get('tipo_operacao'),
                                       custos_taxas=request.form.get('custos_taxas', '0.00'),
                                       observacoes=request.form.get('observacoes'))

        quantidade_str = request.form['quantidade']
        preco_unitario_str = request.form['preco_unitario']
        tipo_operacao = request.form['tipo_operacao']
        custos_taxas_str = request.form.get('custos_taxas', '0.00')
        observacoes = request.form.get('observacoes')

        try:
            quantidade = decimal.Decimal(quantidade_str)
            preco_unitario = decimal.Decimal(preco_unitario_str)
            custos_taxas = decimal.Decimal(custos_taxas_str)
        except decimal.InvalidOperation as e:
            flash(f'Quantidade, Preço Unitário ou Custos/Taxas devem ser números válidos. Erro: {e}', 'danger')
            print(f"ERRO: Conversão decimal inválida: {e}")
            return render_template('add_transaction.html', symbols=symbols,
                                   data_transacao=data_transacao_str,
                                   hora_transacao=hora_transacao_str,
                                   simbolo_ativo_manual=simbolo_ativo_manual_value,
                                   simbolo_ativo_select=simbolo_ativo_select_value,
                                   quantidade=quantidade_str,
                                   preco_unitario=preco_unitario_str,
                                   tipo_operacao=tipo_operacao,
                                   custos_taxas=custos_taxas_str,
                                   observacoes=observacoes)

        if quantidade <= 0:
            flash('Quantidade deve ser maior que zero.', 'danger')
            return render_template('add_transaction.html', symbols=symbols,
                                   data_transacao=data_transacao_str,
                                   hora_transacao=hora_transacao_str,
                                   simbolo_ativo_manual=simbolo_ativo_manual_value,
                                   simbolo_ativo_select=simbolo_ativo_select_value,
                                   quantidade=quantidade_str,
                                   preco_unitario=preco_unitario_str,
                                   tipo_operacao=tipo_operacao,
                                   custos_taxas=custos_taxas_str,
                                   observacoes=observacoes)
        
        if preco_unitario <= 0:
            flash('Preço Unitário deve ser maior que zero.', 'danger')
            return render_template('add_transaction.html', symbols=symbols,
                                   data_transacao=data_transacao_str,
                                   hora_transacao=hora_transacao_str,
                                   simbolo_ativo_manual=simbolo_ativo_manual_value,
                                   simbolo_ativo_select=simbolo_ativo_select_value,
                                   quantidade=quantidade_str,
                                   preco_unitario=preco_unitario_str,
                                   tipo_operacao=tipo_operacao,
                                   custos_taxas=custos_taxas_str,
                                   observacoes=observacoes)
        
        # Mapeamento do símbolo ativo para o formato YFinance para salvar no DB
        # Usa o final_simbolo_para_processamento que já foi validado e capitalizado/stripado
        simbolo_ativo_yf = SYMBOL_MAPPING.get(final_simbolo_para_processamento, final_simbolo_para_processamento)
        if simbolo_ativo_yf == final_simbolo_para_processamento and \
           not any(c in final_simbolo_para_processamento for c in ['^', '-', '=']) and \
           not any(final_simbolo_para_processamento.upper().endswith(suf) for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
            if 4 <= len(final_simbolo_para_processamento) <= 6 and final_simbolo_para_processamento.isalnum():
                simbolo_ativo_yf = f"{final_simbolo_para_processamento.upper()}.SA"
        simbolo_ativo_yf = simbolo_ativo_yf.upper() # Garante que o símbolo final esteja em maiúsculas
        
        print(f"DEBUG: Simbolo ativo YF final para DB: {simbolo_ativo_yf}")

        try:
            with DBConnectionManager() as cursor_db:
                sql = """
                INSERT INTO transacoes (user_id, data_transacao, hora_transacao, simbolo_ativo, quantidade, preco_unitario, tipo_operacao, custos_taxas, observacoes, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor_db.execute(sql, (
                    user_id,
                    data_transacao,
                    hora_transacao,
                    simbolo_ativo_yf, # Salva o símbolo padronizado ou o que o usuário digitou
                    quantidade,
                    preco_unitario,
                    tipo_operacao,
                    custos_taxas,
                    observacoes,
                    datetime.datetime.now()
                ))
                flash('Transação adicionada com sucesso!', 'success')
                return redirect(url_for('index'))
        except Exception as e:
            flash(f'Ocorreu um erro ao adicionar a transação. Verifique se todos os campos estão corretos. Erro: {e}', 'danger')
            print(f"Erro ao adicionar transação no DB: {e}")
            return render_template('add_transaction.html', symbols=symbols,
                                   data_transacao=data_transacao_str,
                                   hora_transacao=hora_transacao_str,
                                   simbolo_ativo_manual=simbolo_ativo_manual_value,
                                   simbolo_ativo_select=simbolo_ativo_select_value,
                                   quantidade=quantidade_str,
                                   preco_unitario=preco_unitario_str,
                                   tipo_operacao=tipo_operacao,
                                   custos_taxas=custos_taxas_str,
                                   observacoes=observacoes)

    # Para GET request ou quando há erro na submissão
    # Adiciona valores iniciais para simbolo_ativo_manual e simbolo_ativo_select para o caso de um POST com erro.
    initial_simbolo_ativo_manual = request.args.get('simbolo_ativo_manual', '')
    initial_simbolo_ativo_select = request.args.get('simbolo_ativo_select', '')
    
    return render_template('add_transaction.html', 
                           user_name=user_name, 
                           symbols=symbols,
                           simbolo_ativo_manual=initial_simbolo_ativo_manual,
                           simbolo_ativo_select=initial_simbolo_ativo_select)


@app.route('/edit_transaction/<int:transaction_id>', methods=['GET', 'POST'])
@login_required
def edit_transaction(transaction_id):
    user_id = session.get('user_id')
    user_name = session.get('username')
    transaction = None
    symbols = sorted(list(SYMBOL_MAPPING.keys()))

    try:
        with DBConnectionManager(dictionary=True) as cursor_db:
            cursor_db.execute("SELECT * FROM transacoes WHERE id = %s AND user_id = %s", (transaction_id, user_id))
            transaction = cursor_db.fetchone()
    except Exception as e:
        flash(f"Erro ao buscar transação para edição: {e}", "danger")
        print(f"ERRO: edit_transaction - Erro ao buscar transação: {e}")
        return redirect(url_for('transactions_list'))

    if not transaction:
        flash('Transação não encontrada ou você não tem permissão para editá-la.', 'danger')
        print(f"DEBUG: edit_transaction - Transação {transaction_id} não encontrada ou não pertence ao user {user_id}.")
        return redirect(url_for('transactions_list'))

    # DEBUG: Imprime o objeto transaction ANTES da formatação para ver o estado inicial
    print(f"DEBUG: edit_transaction - Transaction object antes da formatação: {transaction}")


    if request.method == 'POST':
        data_transacao_str = request.form['data_transacao']
        try:
            data_transacao = datetime.datetime.strptime(data_transacao_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Formato de data inválido. Use AAAA-MM-DD.', 'danger')
            # Garante que 'transaction' tem os valores formatados mesmo em caso de erro de validação
            transaction['data_transacao_formatted'] = data_transacao_str
            if request.form.get('hora_transacao'):
                transaction['hora_transacao_formatted'] = request.form.get('hora_transacao')
            else:
                transaction['hora_transacao_formatted'] = None
            # Tenta obter o nome amigável para exibição em caso de erro
            # Prioriza o valor do campo manual se presente, senão do select
            if request.form.get('simbolo_ativo_manual_edit'):
                transaction['simbolo_ativo_display'] = request.form.get('simbolo_ativo_manual_edit')
                transaction['simbolo_ativo'] = request.form.get('simbolo_ativo_manual_edit') # Update the actual symbol for consistency
            else:
                transaction['simbolo_ativo_display'] = REVERSE_SYMBOL_MAPPING.get(request.form.get('simbolo_ativo'), request.form.get('simbolo_ativo'))
                transaction['simbolo_ativo'] = request.form.get('simbolo_ativo')


            return render_template('editar_transacao.html', user_name=user_name, transaction=transaction, symbols=symbols)

        hora_transacao_str = request.form.get('hora_transacao')
        hora_transacao = None
        if hora_transacao_str:
            try:
                hora_transacao = datetime.datetime.strptime(hora_transacao_str, '%H:%M').time()
            except ValueError:
                flash('Formato de hora inválido. Use HH:MM.', 'danger')
                transaction['data_transacao_formatted'] = data_transacao_str
                transaction['hora_transacao_formatted'] = hora_transacao_str
                if request.form.get('simbolo_ativo_manual_edit'):
                    transaction['simbolo_ativo_display'] = request.form.get('simbolo_ativo_manual_edit')
                    transaction['simbolo_ativo'] = request.form.get('simbolo_ativo_manual_edit')
                else:
                    transaction['simbolo_ativo_display'] = REVERSE_SYMBOL_MAPPING.get(request.form.get('simbolo_ativo'), request.form.get('simbolo_ativo'))
                    transaction['simbolo_ativo'] = request.form.get('simbolo_ativo')
                return render_template('editar_transacao.html', user_name=user_name, transaction=transaction, symbols=symbols)

        # Lógica para obter o símbolo ativo do formulário de edição
        simbolo_ativo = request.form.get('simbolo_ativo') # Do select principal
        simbolo_ativo_manual_edit = request.form.get('simbolo_ativo_manual_edit') # Do campo manual, se visível

        final_simbolo_para_processamento = simbolo_ativo
        if simbolo_ativo == 'OUTRO_EDIT' and simbolo_ativo_manual_edit:
            final_simbolo_para_processamento = simbolo_ativo_manual_edit

        if not final_simbolo_para_processamento or final_simbolo_para_processamento.strip() == '':
            flash('Por favor, selecione ou digite um símbolo para o ativo.', 'danger')
            transaction['data_transacao_formatted'] = data_transacao_str
            transaction['hora_transacao_formatted'] = hora_transacao_str
            transaction['simbolo_ativo_display'] = REVERSE_SYMBOL_MAPPING.get(final_simbolo_para_processamento, final_simbolo_para_processamento) # Tentativa de preencher
            return render_template('editar_transacao.html', user_name=user_name, transaction=transaction, symbols=symbols)


        quantidade = request.form['quantidade']
        preco_unitario = request.form['preco_unitario']
        tipo_operacao = request.form['tipo_operacao']
        custos_taxas = request.form.get('custos_taxas', '0.00')
        observacoes = request.form.get('observacoes')

        try:
            quantidade = decimal.Decimal(quantidade)
            preco_unitario = decimal.Decimal(preco_unitario)
            custos_taxas = decimal.Decimal(custos_taxas)
        except decimal.InvalidOperation:
            flash('Quantidade, Preço Unitário ou Custos/Taxas devem ser números válidos.', 'danger')
            transaction['data_transacao_formatted'] = data_transacao_str
            transaction['hora_transacao_formatted'] = hora_transacao_str
            transaction['simbolo_ativo_display'] = REVERSE_SYMBOL_MAPPING.get(final_simbolo_para_processamento, final_simbolo_para_processamento)
            # Passa os valores do formulário para o template para preencher novamente
            transaction['quantidade'] = quantidade
            transaction['preco_unitario'] = preco_unitario
            transaction['custos_taxas'] = custos_taxas
            transaction['observacoes'] = observacoes
            return render_template('editar_transacao.html', user_name=user_name, transaction=transaction, symbols=symbols)

        if quantidade <= 0 or preco_unitario <= 0:
            flash('Quantidade e Preço Unitário devem ser maiores que zero.', 'danger')
            transaction['data_transacao_formatted'] = data_transacao_str
            transaction['hora_transacao_formatted'] = hora_transacao_str
            transaction['simbolo_ativo_display'] = REVERSE_SYMBOL_MAPPING.get(final_simbolo_para_processamento, final_simbolo_para_processamento)
            transaction['quantidade'] = quantidade
            transaction['preco_unitario'] = preco_unitario
            transaction['custos_taxas'] = custos_taxas
            transaction['observacoes'] = observacoes
            return render_template('editar_transacao.html', user_name=user_name, transaction=transaction, symbols=symbols)

        simbolo_ativo_yf = SYMBOL_MAPPING.get(final_simbolo_para_processamento.upper(), final_simbolo_para_processamento)
        if simbolo_ativo_yf == final_simbolo_para_processamento and \
           not any(c in final_simbolo_para_processamento for c in ['^', '-', '=']) and \
           not any(final_simbolo_para_processamento.upper().endswith(suf) for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
            if 4 <= len(final_simbolo_para_processamento) <= 6 and final_simbolo_para_processamento.isalnum():
                simbolo_ativo_yf = f"{final_simbolo_para_processamento.upper()}.SA"
        simbolo_ativo_yf = simbolo_ativo_yf.upper()

        try:
            with DBConnectionManager() as cursor_db:
                sql = """
                UPDATE transacoes
                SET data_transacao = %s, hora_transacao = %s, simbolo_ativo = %s, quantidade = %s,
                    preco_unitario = %s, tipo_operacao = %s, custos_taxas = %s, observacoes = %s
                WHERE id = %s AND user_id = %s
                """
                cursor_db.execute(sql, (
                    data_transacao, hora_transacao, simbolo_ativo_yf, quantidade,
                    preco_unitario, tipo_operacao, custos_taxas, observacoes,
                    transaction_id, user_id
                ))
                flash('Transação atualizada com sucesso!', 'success')
                return redirect(url_for('index'))
        except Exception as e:
            flash(f'Ocorreu um erro ao atualizar a transação. Erro: {e}', 'danger')
            print(f"Erro ao atualizar transação: {e}")
            transaction['data_transacao_formatted'] = data_transacao_str
            transaction['hora_transacao_formatted'] = hora_transacao_str
            transaction['simbolo_ativo_display'] = REVERSE_SYMBOL_MAPPING.get(final_simbolo_para_processamento, final_simbolo_para_processamento)
            transaction['quantidade'] = quantidade
            transaction['preco_unitario'] = preco_unitario
            transaction['custos_taxas'] = custos_taxas
            transaction['observacoes'] = observacoes
            return render_template('editar_transacao.html', user_name=user_name, transaction=transaction, symbols=symbols)


    # Esta parte do código é executada para GET requests (quando a página é carregada)
    # ou se houve um POST com erro de validação que levou ao re-render.
    if 'data_transacao' in transaction and transaction['data_transacao']:
        transaction['data_transacao_formatted'] = transaction['data_transacao'].strftime('%Y-%m-%d')
    else:
        transaction['data_transacao_formatted'] = None # Garante que a chave existe

    if 'hora_transacao' in transaction and transaction['hora_transacao']:
        if isinstance(transaction['hora_transacao'], datetime.timedelta):
            total_seconds = int(transaction['hora_transacao'].total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            transaction['hora_transacao_formatted'] = f"{hours:02d}:{minutes:02d}"
        elif isinstance(transaction['hora_transacao'], datetime.time):
            transaction['hora_transacao_formatted'] = transaction['hora_transacao'].strftime('%H:%M')
        else:
            transaction['hora_transacao_formatted'] = None
    else:
        transaction['hora_transacao_formatted'] = None # Garante que a chave existe

    # Define simbolo_ativo_display para o template
    transaction['simbolo_ativo_display'] = REVERSE_SYMBOL_MAPPING.get(transaction['simbolo_ativo'], transaction['simbolo_ativo'])
    
    print(f"DEBUG: edit_transaction - Transaction object antes de renderizar template: {transaction}")
    print(f"DEBUG: Símbolo de exibição no template: {transaction.get('simbolo_ativo_display')}")
    print(f"DEBUG: Símbolo YF no template: {transaction.get('simbolo_ativo')}")


    return render_template('editar_transacao.html',
                           user_name=user_name,
                           transaction=transaction,
                           symbols=symbols)

@app.route('/delete_transaction/<int:transaction_id>', methods=['POST'])
@login_required
def delete_transaction(transaction_id):
    user_id = session.get('user_id')
    try:
        with DBConnectionManager() as cursor_db:
            cursor_db.execute("DELETE FROM transacoes WHERE id = %s AND user_id = %s", (transaction_id, user_id))
            if cursor_db.rowcount > 0:
                flash('Transação excluída com sucesso.', 'success')
            else:
                flash('Transação não encontrada ou você não tem permissão para excluí-la.', 'danger')
    except Exception as e:
        flash(f'Ocorreu um erro ao excluir a transação. Erro: {e}', 'danger')
        print(f"Erro ao excluir transação: {e}")
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    users = get_all_users()
    admin_count = get_admin_count()
    return render_template('admin_dashboard.html', users=users, admin_count=admin_count)

@app.route('/admin/toggle_admin/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def toggle_admin(user_id):
    if user_id == session.get('user_id'):
        flash('Você não pode alterar seu próprio status de administrador.', 'danger')
        return redirect(url_for('admin_dashboard'))

    target_user = get_user_by_id(user_id)
    if not target_user:
        flash('Utilizador não encontrado.', 'danger')
        return redirect(url_for('admin_dashboard'))

    if target_user['is_admin'] and get_admin_count() <= 1:
        flash('Não é possível remover o último administrador do sistema.', 'danger')
        return redirect(url_for('admin_dashboard'))

    new_status = not target_user['is_admin']

    if toggle_user_admin_status(user_id, new_status):
        action = 'PROMOVIDO_A_ADMIN' if new_status else 'REMOVIDO_DE_ADMIN'
        admin_user_id = session.get('user_id')
        log_admin_action(admin_user_id, action, target_user_id=user_id, details={'new_status': new_status})
        flash(f"Status de administrador de '{target_user['username']}' alterado para {'Admin' if new_status else 'Utilizador Comum'}.", 'success')
    else:
        flash('Ocorreu um erro ao alterar o status de administrador.', 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    if user_id == session.get('user_id'):
        flash('Você não pode excluir sua própria conta.', 'danger')
        return redirect(url_for('admin_dashboard'))

    target_user = get_user_by_id(user_id)
    if not target_user:
        flash('Utilizador não encontrado.', 'danger')
        return redirect(url_for('admin_dashboard'))

    if target_user['is_admin'] and get_admin_count() <= 1:
        flash('Não é possível excluir o último administrador do sistema.', 'danger')
        return redirect(url_for('admin_dashboard'))

    admin_user_id = session.get('user_id')
    log_admin_action(admin_user_id, 'EXCLUSAO_UTILIZADOR', target_user_id=user_id, details={'username_alvo': target_user['username']})

    if delete_user_from_db(user_id):
        flash(f"Utilizador '{target_user['username']}' e todas as suas transações foram excluídos com sucesso.", 'success')
    else:
        flash('Ocorreu um erro ao excluir o utilizador.', 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/reset_password/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def reset_password(user_id):
    if user_id == session.get('user_id'):
        flash('Você não pode redefinir sua própria palavra-passe por aqui.', 'danger')
        return redirect(url_for('admin_dashboard'))

    new_password = request.form.get('new_password')
    if not new_password or len(new_password) < 6:
        flash('A nova palavra-passe deve ter pelo menos 6 caracteres.', 'danger')
        return redirect(url_for('admin_dashboard'))

    target_user = get_user_by_id(user_id)
    if not target_user:
        flash('Utilizador não encontrado.', 'danger')
        return redirect(url_for('admin_dashboard'))

    if update_user_password(user_id, new_password):
        admin_user_id = session.get('user_id')
        log_admin_action(admin_user_id, 'REDEFINICAO_SENHA_UTILIZADOR', target_user_id=user_id, details={'username_alvo': target_user['username']})
        flash(f"Palavra-passe de '{target_user['username']}' redefinida com sucesso.", 'success')
    else:
        flash('Ocorreu um erro ao redefinir a palavra-passe.', 'danger')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/audit_logs')
@login_required
@admin_required
def admin_audit_logs():
    logs = []
    try:
        with DBConnectionManager(dictionary=True) as cursor_db:
            cursor_db.execute("SELECT * FROM admin_audit_logs ORDER BY timestamp DESC")
            logs = cursor_db.fetchall()
            for log in logs:
                if log['details']:
                    if isinstance(log['details'], dict):
                        log['details_parsed'] = log['details']
                    else:
                        try:
                            log['details_parsed'] = json.loads(log['details'])
                        except json.JSONDecodeError:
                            log['details_parsed'] = {}
                else:
                    log['details_parsed'] = {}
    except Exception as e:
        flash(f"Erro ao carregar logs de auditoria: {e}", "danger")
        print(f"Erro ao carregar logs de auditoria: {e}")
    return render_template('admin_audit_logs.html', logs=logs)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_id = session.get('user_id')
    user = get_user_by_id(user_id)

    if not user:
        flash('Usuário não encontrado.', 'danger')
        return redirect(url_for('index'))

    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        contact_number = request.form.get('contact_number')

        if not full_name or not email:
            flash('Nome completo e email são obrigatórios.', 'danger')
            return render_template('profile.html', user=user)

        user_with_same_email = get_user_by_email(email)
        if user_with_same_email and user_with_same_email['id'] != user_id:
            flash('Este endereço de email já está registado por outro utilizador.', 'danger')
            return render_template('profile.html', user=user)

        if update_user_profile_data(user_id, full_name, email, contact_number):
            flash('Perfil atualizado com sucesso!', 'success')
            return redirect(url_for('profile'))
        else:
            flash('Ocorreu um erro ao atualizar o perfil.', 'danger')

    return render_template('profile.html', user=user)

@app.route('/profile/change_password', methods=['POST'])
@login_required
def change_password():
    user_id = session.get('user_id')
    current_password = request.form['current_password']
    new_password = request.form['new_password']
    confirm_new_password = request.form['confirm_new_password']

    user = get_user_by_id(user_id)
    if not user or not check_password_hash(user['password_hash'], current_password):
        flash('Palavra-passe atual incorreta.', 'danger')
        return redirect(url_for('profile'))

    if new_password != confirm_new_password:
        flash('A nova palavra-passe e a confirmação não coincidem.', 'danger')
        return redirect(url_for('profile'))

    if len(new_password) < 6:
        flash('A nova palavra-passe deve ter pelo menos 6 caracteres.', 'danger')
        return redirect(url_for('profile'))

    if update_user_password(user_id, new_password):
        flash('Palavra-passe atualizada com sucesso!', 'success')
    else:
        flash('Ocorreu um erro ao atualizar a palavra-passe.', 'danger')

    return redirect(url_for('profile'))

# --- ROTAS E FUNÇÕES PARA ALERTAS DE PREÇO ---
@app.route('/adicionar_alerta', methods=['POST'])
@login_required
def adicionar_alerta():
    user_id = session.get('user_id')
    simbolo_ativo = request.form['simbolo_ativo'].upper()
    preco_alvo_str = request.form['preco_alvo']
    tipo_alerta = request.form['tipo_alerta']

    try:
        preco_alvo = decimal.Decimal(preco_alvo_str)
        if preco_alvo <= 0:
            flash('O preço alvo deve ser maior que zero.', 'danger')
            return redirect(url_for('index'))
    except decimal.InvalidOperation:
        flash('Preço Alvo deve ser um número válido.', 'danger')
        return redirect(url_for('index'))

    simbolo_ativo_yf = SYMBOL_MAPPING.get(simbolo_ativo, simbolo_ativo)
    if simbolo_ativo_yf == simbolo_ativo and \
       not any(c in simbolo_ativo for c in ['^', '-', '=']) and \
       not any(simbolo_ativo.upper().endswith(suf) for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
        if 4 <= len(simbolo_ativo) <= 6 and simbolo_ativo.isalnum():
            simbolo_ativo_yf = f"{simbolo_ativo.upper()}.SA"
    simbolo_ativo_yf = simbolo_ativo_yf.upper()

    try:
        with DBConnectionManager() as cursor_db:
            sql = """
            INSERT INTO alertas (user_id, simbolo_ativo, preco_alvo, tipo_alerta, status, data_criacao)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            # Certifique-se de que o nome da tabela no SQL é 'alertas' (em minúsculas)
            cursor_db.execute(sql, (user_id, simbolo_ativo_yf, preco_alvo, tipo_alerta, 'ATIVO', datetime.datetime.now()))
            flash('Alerta de preço adicionado com sucesso!', 'success')
    except Exception as e:
        flash(f'Ocorreu um erro ao adicionar o alerta de preço. Erro: {e}', 'danger')
        print(f"Erro ao adicionar alerta de preço: {e}")
    return redirect(url_for('index'))

@app.route('/excluir_alerta', methods=['POST'])
@login_required
def excluir_alerta():
    user_id = session.get('user_id')
    alert_id = request.form.get('id', type=int)

    if not alert_id:
        flash('ID do alerta não fornecido.', 'danger')
        return redirect(url_for('index'))

    try:
        with DBConnectionManager() as cursor_db:
            # Garante que 'alertas' é acessado em minúsculas
            cursor_db.execute("DELETE FROM alertas WHERE id = %s AND user_id = %s", (alert_id, user_id))
            if cursor_db.rowcount > 0:
                flash('Alerta de preço excluído com sucesso.', 'success')
            else:
                flash('Alerta não encontrado ou você não tem permissão para excluí-lo.', 'danger')
    except Exception as e:
        flash(f'Ocorreu um erro ao excluir o alerta de preço. Erro: {e}', 'danger')
        print(f"Erro ao excluir alerta de preço: {e}")
    return redirect(url_for('index'))

# --- ROTA PARA O GRÁFICO HISTÓRICO NO MODAL ---
@app.route('/get_historical_chart_data/<simbolo>')
@login_required
def get_historical_chart_data(simbolo):
    simbolo_yf = SYMBOL_MAPPING.get(simbolo.upper(), simbolo)
    if simbolo_yf == simbolo and \
       not any(c in simbolo for c in ['^', '-', '=']) and \
       not any(simbolo.upper().endswith(suf) for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
        if 4 <= len(simbolo) <= 6 and simbolo.isalnum():
            simbolo_yf = f"{simbolo.upper()}.SA"

    cache_key = (simbolo_yf, "1y", "1d")

    if is_cache_fresh(historical_chart_cache, cache_key, HISTORICAL_CHART_CACHE_TTL):
        print(f"DEBUG: Dados históricos para {simbolo_yf} encontrados no cache.")
        return jsonify(historical_chart_cache[cache_key]['data'])

    try:
        df_hist = get_historical_prices_yfinance_cached(simbolo_yf, period="1y", interval="1d")
        
        if df_hist.empty:
            print(f"DEBUG: Não foi possível obter dados históricos para {simbolo_yf}.")
            return jsonify({'error': 'Não foi possível obter dados históricos para este símbolo.'}), 404

        fig = go.Figure(data=[go.Candlestick(x=df_hist.index,
                        open=df_hist['Open'],
                        high=df_hist['High'],
                        low=df_hist['Low'],
                        close=df_hist['Close'])])

        fig.update_layout(
            title=f'Preços Históricos de {REVERSE_SYMBOL_MAPPING.get(simbolo_yf, simbolo_yf)} ({simbolo_yf})',
            yaxis_title='Preço (R$)',
            xaxis_title='Data',
            xaxis_rangeslider_visible=False,
            height=450,
            margin=dict(l=40, r=40, t=80, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#333")
        )

        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        response_data = {'simbolo': simbolo, 'plot_json': plot_json}
        historical_chart_cache[cache_key] = {'data': response_data, 'timestamp': datetime.datetime.now()}
        print(f"DEBUG: Dados históricos para {simbolo_yf} obtidos e cacheados.")
        return jsonify(response_data)

    except Exception as e:
        print(f"ERRO ao gerar gráfico histórico para {simbolo_yf}: {e}")
        return jsonify({'error': f'Erro interno ao gerar o gráfico: {e}'}), 500


@app.route('/predict_price/<simbolo>')
@login_required
def predict_price_api(simbolo):
    predicted_price = get_predicted_price_for_display(simbolo)
    if predicted_price is not None:
        return jsonify({'simbolo': simbolo, 'predicted_price': predicted_price})
    else:
        return jsonify({'error': 'Não foi possível obter previsão para este símbolo.', 'simbolo': simbolo}), 404

print("DEBUG: Entrando no bloco if __name__ == '__main__':")
if __name__ == '__main__':
    # Chama a função para criar tabelas no início
    try:
        create_tables_if_not_exist()
        print("DEBUG: create_tables_if_not_exist() finalizada. Verificando a existência das tabelas agora...")
        
        # Verificações pós-criação
        if check_table_exists('users'):
            print("DEBUG: Tabela 'users' confirmada após a tentativa de criação.")
        else:
            print("ERRO: Tabela 'users' NÃO EXISTE após a tentativa de criação!")
        
        if check_table_exists('transacoes'):
            print("DEBUG: Tabela 'transacoes' confirmada após a tentativa de criação.")
        else:
            print("ERRO: Tabela 'transacoes' NÃO EXISTE após a tentativa de criação!")

        if check_table_exists('alertas'): # Mudado para 'alertas'
            print("DEBUG: Tabela 'alertas' confirmada após a tentativa de criação.")
        else:
            print("ERRO: Tabela 'alertas' NÃO EXISTE após a tentativa de criação!")
        
        if check_table_exists('admin_audit_logs'):
            print("DEBUG: Tabela 'admin_audit_logs' confirmada após a tentativa de criação.")
        else:
            print("ERRO: Tabela 'admin_audit_logs' NÃO EXISTE após a tentativa de criação!")

    except Exception as startup_error:
        print(f"ERRO CRÍTICO NA INICIALIZAÇÃO: A aplicação falhou ao iniciar devido a problemas no banco de dados: {startup_error}")
        import sys
        sys.exit(1) # Força a saída se a criação das tabelas falhar

    port = int(os.environ.get("PORT", 5000))
    print(f"DEBUG: Iniciando Flask app em host 0.0.0.0, porta {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
