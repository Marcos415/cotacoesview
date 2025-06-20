import datetime
import decimal
import os
import joblib
import pandas as pd
import yfinance as yf
# import mysql.connector # REMOVIDO: Será importado condicionalmente dentro do DBConnectionManager
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

import requests
from bs4 import BeautifulSoup

from fpdf import FPDF

import plotly.graph_objects as go
import json

# Importa a biblioteca para carregar variáveis de ambiente do .env
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env (DEVE SER UMA DAS PRIMEIRAS LINHAS)
load_dotenv()

# Importa apenas train_and_predict_price
from predictor_model import train_and_predict_price

app = Flask(__name__)
# MUITO IMPORTANTE: Mude esta chave para uma string aleatória complexa e secreta em produção!
# Esta chave é usada para proteger as sessões dos utilizadores.
# Agora pega a chave da variável de ambiente
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    # Levanta um erro se a chave secreta não for encontrada, o que indica um problema no .env
    raise ValueError("A variável de ambiente 'SECRET_KEY' não está definida. Por favor, defina-a no seu arquivo .env.")
else:
    print(f"DEBUG: SECRET_KEY carregada com sucesso (primeiros 5 caracteres): {app.secret_key[:5]}*****") # Linha de debug

CORS(app)

# Passamos a função datetime.datetime.now (sem parênteses)
app.jinja_env.globals['now'] = datetime.datetime.now

# --- CONFIGURAÇÃO DO CACHE ---
# Caching Dictionaries (globais)
market_data_cache = {} # Key: (symbol, period, interval), Value: {'data': df or price, 'timestamp': datetime.datetime}
news_cache = {}        # Key: query_term, Value: {'data': list_of_news_items, 'timestamp': datetime.datetime}
portfolio_cache = {}   # Key: user_id, Value: {'data': (posicoes, total_valor, total_lucro, total_prejuizo), 'timestamp': datetime.datetime}
prediction_cache = {}  # Key: symbol, Value: {'data': predicted_price, 'timestamp': datetime.datetime}

# Cache TTL (Time To Live) em segundos
MARKET_DATA_CACHE_TTL = 300  # 5 minutos para dados de mercado (cotações, históricos para modelos)
NEWS_CACHE_TTL = 3600        # 1 hora para notícias
PORTFOLIO_CACHE_TTL = 120    # 2 minutos para cálculos de portfólio (depende dos dados de mercado)
PREDICTION_CACHE_TTL = 600   # 10 minutos para previsões (geralmente mais estáveis)

def is_cache_fresh(cache, key, ttl):
    """Verifica se um item no cache ainda é válido."""
    if key in cache:
        timestamp = cache[key]['timestamp']
        if (datetime.datetime.now() - timestamp).total_seconds() < ttl:
            return True
    return False

# --- FIM DA CONFIGURAÇÃO DO CACHE ---

# --- CONFIGURAÇÃO DA API DE NOTÍCIAS ---
# IMPORTANTE: Sua chave de API real da NewsAPI.org
# Você pode obter uma chave gratuita em: https://newsapi.org/
# Agora pega a chave da variável de ambiente
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"
# --- FIM DA CONFIGURAÇÃO DA API DE NOTÍCIAS ---

# --- Constante para Paginação ---
TRANSACTIONS_PER_PAGE = 6 # Número de transações a serem exibidas por página. Ajuste conforme necessário.

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
    else:
        try:
            dt = datetime.datetime.fromisoformat(str(value))
        except ValueError:
            return value
    return dt.strftime(format_string)

# --- NOVO: Registro do filtro 'floatformat' para Jinja2 ---
@app.template_filter('floatformat')
def floatformat(value, precision=2):
    """
    Formata um número float para uma determinada precisão de casas decimais.
    """
    try:
        # Garante que o valor é um float antes de formatar
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return value # Retorna o valor original se não puder ser convertido para float

# --- Configurações do Banco de Dados ---
# Tenta obter a URL do banco de dados para PostgreSQL (Render)
DATABASE_URL = os.getenv('DATABASE_URL')

# Se DATABASE_URL não estiver definida, usa as configurações do MySQL para desenvolvimento local
if DATABASE_URL:
    DB_TYPE = 'postgresql'
    print("DEBUG: Usando conexão PostgreSQL (DATABASE_URL detectada).")
else:
    DB_TYPE = 'mysql'
    print("DEBUG: Usando conexão MySQL (DATABASE_URL não detectada).")
    DB_CONFIG_MYSQL = {
        'host': os.getenv('DB_HOST', 'localhost'), # 'localhost' como fallback para desenvolvimento
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_DATABASE'),
        'port': int(os.getenv('DB_PORT', 3306)) # Porta padrão do MySQL é 3306
    }
    # Verifica se as variáveis de ambiente essenciais para o MySQL foram carregadas
    if not all([DB_CONFIG_MYSQL['user'], DB_CONFIG_MYSQL['password'], DB_CONFIG_MYSQL['database']]):
        # Esta exceção só deve ocorrer em desenvolvimento local se o .env estiver mal configurado.
        raise ValueError("Uma ou mais variáveis de ambiente do banco de dados MySQL (DB_USER, DB_PASSWORD, DB_DATABASE) não estão definidas para uso local. Por favor, defina-as no seu arquivo .env.")

# --- Context Manager para Conexões Unificado (MySQL ou PostgreSQL) ---
class DBConnectionManager:
    def __init__(self, dictionary=False, buffered=False):
        self.conn = None
        self.cursor = None
        self.dictionary = dictionary
        self.buffered = buffered
        self.db_type = DB_TYPE # Armazena o tipo de DB selecionado

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
                # Analisa a DATABASE_URL para extrair credenciais para psycopg2
                result = urlparse(DATABASE_URL)
                username = result.username
                password = result.password
                database = result.path[1:]
                hostname = result.hostname
                port = result.port if result.port else 5432 # Default PostgreSQL port

                self.conn = psycopg2.connect(
                    database=database,
                    user=username,
                    password=password,
                    host=hostname,
                    port=port
                )
                # Para PostgreSQL, o cursor tipo 'dictionary' é mais complexo,
                # normalmente psycopg2 retorna tuplas ou você usa dictcursor.
                # psycopg2.extras.RealDictCursor é o equivalente a dictionary=True
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
                    print("DEBUG: DBConnectionManager - Conexão revertida.")
            except Exception as err:
                print(f"ERRO: DBConnectionManager - Erro ao comitar/reverter no __exit__: {err}")
            finally:
                try:
                    self.conn.close()
                    print("DEBUG: DBConnectionManager - Conexão fechada.")
                except Exception as err:
                    print(f"ERRO: DBConnectionManager - Erro ao fechar conexão no __exit__: {err}")

# --- Dicionário de Mapeamento de Símbolos ---
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
    # Adicionar variações sem ".SA" para busca no filtro
    if '.SA' in ticker:
        REVERSE_SYMBOL_MAPPING[ticker.replace('.SA', '')] = name
    elif '^' in ticker or '-' in ticker or '=' in ticker:
        # Para símbolos como ^BVSP, BTC-USD, GC=F, manter como está
        REVERSE_SYMBOL_MAPPING[ticker] = name
    else:
        REVERSE_SYMBOL_MAPPING[ticker] = name # Fallback, se não for nenhum dos anteriores

# --- Decorators para Autenticação e Autorização ---
def login_required(f):
    """
    Decorator que verifica se um utilizador está logado.
    Se não estiver, redireciona para a página de login.
    """
    @wraps(f) # ESSENCIAL para Flask
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Você precisa estar logado para acessar esta página.', 'danger')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """
    Decorator que verifica se o utilizador logado é um administrador.
    Se não for, redireciona para a página principal com uma mensagem de erro.
    """
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
        # Usar buffered=True aqui para garantir que os resultados sejam lidos imediatamente
        with DBConnectionManager(dictionary=True, buffered=True) as cursor_db:
            # Inclui 'full_name', 'email', 'contact_number' na seleção
            cursor_db.execute("SELECT id, username, full_name, email, contact_number, is_admin, created_at FROM users WHERE id = %s", (user_id,))
            user_data = cursor_db.fetchone()
            print(f"DEBUG: get_user_by_id({user_id}) - Dados do utilizador: {user_data}")
            return user_data
    except Exception as err:
        print(f"Erro ao buscar utilizador por ID: {err}")
        return None

def get_user_by_username(username):
    try:
        # Usar buffered=True aqui para garantir que os resultados sejam lidos imediatamente
        with DBConnectionManager(dictionary=True, buffered=True) as cursor_db:
            # Inclui 'full_name', 'email', 'contact_number' na seleção
            cursor_db.execute("SELECT id, username, password_hash, full_name, email, contact_number, is_admin FROM users WHERE username = %s", (username,))
            user_data = cursor_db.fetchone()
            print(f"DEBUG: get_user_by_username({username}) - Dados do utilizador: {user_data}")
            return user_data
    except Exception as err:
        print(f"Erro ao buscar utilizador por nome de utilizador: {err}")
        return None

# Nova função para buscar utilizador por email
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
    """Busca todos os utilizadores (exceto o próprio admin logado) para exibir no painel de admin."""
    users = []
    current_user_id = session.get('user_id')
    print(f"DEBUG: get_all_users() - current_user_id na sessão: {current_user_id}")

    if current_user_id is None:
        print("DEBUG: get_all_users() - user_id não encontrado na sessão. Retornando lista vazia.")
        return []

    try:
        # Usar buffered=True aqui para garantir que os resultados sejam lidos imediatamente
        with DBConnectionManager(dictionary=True, buffered=True) as cursor_db:
            # Inclui 'full_name', 'email', 'contact_number' na seleção
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
    """Retorna o número total de utilizadores com is_admin = TRUE."""
    try:
        # Usar buffered=True aqui para garantir que os resultados sejam lidos imediatamente
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
    """Exclui um utilizador do banco de dados e todas as suas transações/alertas."""
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
    """Atualiza a palavra-passe de um utilizador."""
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
    """
    Alterna o status de administrador de um utilizador.
    new_status deve ser um booleano (True/False).
    """
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

# Nova função para atualizar dados do perfil do utilizador
def update_user_profile_data(user_id, full_name, email, contact_number):
    """Atualiza o nome completo, email e número de contacto de um utilizador."""
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
    """
    Regista uma ação realizada por um administrador na tabela admin_audit_logs,
    armazenando os nomes de utilizador no momento da ação para persistência.
    """
    try:
        # Obter o nome de utilizador do admin (do user_id na sessão)
        admin_username = session.get('username')
        if not admin_username:
            admin_user_data = get_user_by_id(admin_user_id)
            admin_username = admin_user_data['username'] if admin_user_data else f"ID_Desconhecido_{admin_user_id}"

        # Obter o nome de utilizador do target (se houver)
        target_username = None
        if target_user_id:
            target_user_data = get_user_by_id(target_user_id)
            target_username = target_user_data['username'] if target_user_data else f"ID_Deletado_{target_user_id}"

        with DBConnectionManager() as cursor_db:
            sql = """
            INSERT INTO admin_audit_logs (admin_user_id, admin_username_at_action, action_type, target_user_id, target_username_at_action, details, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            details_json = json.dumps(details) if details is not None else None

            cursor_db.execute(sql, (admin_user_id, admin_username, action_type, target_user_id, target_username, details_json, datetime.datetime.now()))
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

            # CORREÇÃO AQUI: Verifica se o simbolo_filtro não é vazio E não é a string "None" (ignorando maiúsculas/minúsculas)
            if simbolo_filtro and simbolo_filtro.lower() != 'none':
                # Tenta mapear ou ajustar o símbolo de filtro
                final_simbolo_to_fetch_for_filter = SYMBOL_MAPPING.get(simbolo_filtro.upper(), simbolo_filtro)
                # Adiciona .SA se for um ticker sem sufixo e alfanumérico com 4-6 caracteres
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

            if final_simbolo_to_fetch_for_filter: # Só adiciona esta cláusula se houver um símbolo válido
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
                # Converte timedelta para time se necessário
                if isinstance(transacao.get('hora_transacao'), datetime.timedelta):
                    total_seconds = int(transacao['hora_transacao'].total_seconds())
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    transacao['hora_transacao'] = datetime.time(hours, minutes, seconds)
                elif transacao.get('hora_transacao') is None:
                    transacao['hora_transacao'] = None

    except Exception as err:
        print(f"ERRO ao buscar transações: {err}")
        raise
    return transacoes, total_transacoes

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
                            custo_medio_atual = estado_ativo[simbolo]['custo_acumulado'] / estado_ativo[simbolo]['quantidade']
                            custo_das_vendidas = quantidade * custo_medio_atual

                            estado_ativo[simbolo]['quantidade'] -= quantidade
                            estado_ativo[simbolo]['custo_acumulado'] -= (custo_das_vendidas + custos_taxas)

                            if estado_ativo[simbolo]['quantidade'] <= 0.00001:
                                estado_ativo[simbolo]['quantidade'] = 0.0
                                estado_ativo[simbolo]['custo_acumulado'] = 0.0
                        else:
                            estado_ativo[simbolo]['quantidade'] = 0.0
                            estado_ativo[simbolo]['custo_acumulado'] = 0.0
                    else:
                        pass

            for simbolo, dados_posicao in estado_ativo.items():
                if dados_posicao['quantidade'] > 0:
                    preco_medio = float(dados_posicao['custo_acumulado']) / float(dados_posicao['quantidade'])

                    preco_atual = _get_current_price_yfinance(simbolo)
                    preco_previsto = get_predicted_price_for_display(simbolo)

                    lucro_prejuizo_nao_realizado_individual = None
                    if preco_atual is not None:
                        lucro_prejuizo_nao_realizado_individual = (preco_atual - preco_medio) * float(dados_posicao['quantidade'])
                    else:
                        # Se não há preço atual, ainda assim podemos calcular o valor atual com o preço médio
                        lucro_prejuizo_nao_realizado_individual = 0.0

                    posicoes[simbolo] = {
                        'quantidade': dados_posicao['quantidade'],
                        'preco_medio': preco_medio,
                        'preco_atual': preco_atual,
                        'preco_previsto': preco_previsto,
                        'valor_atual': (preco_atual * dados_posicao['quantidade']) if preco_atual is not None else 0.0,
                        'lucro_prejuizo_nao_realizado': lucro_prejuizo_nao_realizado_individual
                    }
                    total_valor_carteira += posicoes[simbolo]['valor_atual']

                    if lucro_prejuizo_nao_realizado_individual is not None:
                        if lucro_prejuizo_nao_realizado_individual > 0:
                            total_lucro_nao_realizado += lucro_prejuizo_nao_realizado_individual
                        else:
                            total_prejuizo_nao_realizado += lucro_prejuizo_nao_realizado_individual

    except Exception as e:
        print(f"ERRO inesperado ao calcular posições da carteira: {e}")
        # Retorna valores padrão em caso de erro para evitar que a página quebre
        return {}, 0.0, 0.0, 0.0

    portfolio_cache[cache_key] = {
        'data': (posicoes, total_valor_carteira, total_lucro_nao_realizado, total_prejuizo_nao_realizado),
        'timestamp': datetime.datetime.now()
    }
    return posicoes, total_valor_carteira, total_lucro_nao_realizado, total_prejuizo_nao_realizado

# --- Funções para Notícias (mantidas como estavam) ---
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
        response.raise_for_status() # Lança um erro para status de resposta HTTP ruins (4xx ou 5xx)
        news_data = response.json()
        articles = news_data.get('articles', [])
        # Filtra artigos para ter URL, title e description
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


# --- ROTAS DA APLICAÇÃO ---

@app.route('/')
@login_required
def index():
    user_id = session.get('user_id')
    user_data = get_user_by_id(user_id)
    if not user_data:
        flash('Sua sessão expirou ou o usuário não foi encontrado.', 'danger')
        return redirect(url_for('logout'))

    user_name = user_data['username'] # Pega o nome de usuário para exibir

    posicoes, total_valor_carteira, total_lucro_nao_realizado, total_prejuizo_nao_realizado = calcular_posicoes_carteira(user_id)

    # Buscar notícias sobre o mercado financeiro ou as principais ações na carteira
    news_query = "Mercado Financeiro Brasil"
    if posicoes:
        top_symbols = list(posicoes.keys())[:3] # Pega até 3 símbolos das posições
        news_query = ", ".join([REVERSE_SYMBOL_MAPPING.get(s, s) for s in top_symbols]) + ", Mercado Financeiro Brasil"

    news_articles = fetch_news(news_query)


    # Preparar dados para o gráfico de pizza de alocação da carteira
    chart_labels = []
    chart_values = []
    for simbolo, dados in posicoes.items():
        if dados['valor_atual'] > 0:
            chart_labels.append(simbolo)
            chart_values.append(dados['valor_atual'])

    # Criar o gráfico de pizza
    if chart_labels:
        fig = go.Figure(data=[go.Pie(labels=chart_labels, values=chart_values, hole=.3)])
        fig.update_layout(
            title_text='Alocação da Carteira por Ativo',
            title_font_size=20,
            showlegend=True,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#333"),
            height=350
        )
        # Convertendo para JSON para passar ao template
        pie_chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        pie_chart_json = None


    # Preparar dados para o gráfico de barras Lucro/Prejuízo Não Realizado
    profit_loss_labels = []
    profit_loss_values = []
    profit_loss_colors = []

    for simbolo, dados in posicoes.items():
        if dados['lucro_prejuizo_nao_realizado'] is not None:
            profit_loss_labels.append(simbolo)
            profit_loss_values.append(dados['lucro_prejuizo_nao_realizado'])
            profit_loss_colors.append('green' if dados['lucro_prejuizo_nao_realizado'] >= 0 else 'red')

    bar_chart_json = None
    if profit_loss_labels:
        bar_fig = go.Figure(data=[
            go.Bar(
                x=profit_loss_labels,
                y=profit_loss_values,
                marker_color=profit_loss_colors
            )
        ])
        bar_fig.update_layout(
            title_text='Lucro/Prejuízo Não Realizado por Ativo',
            title_font_size=20,
            xaxis_title='Ativo',
            yaxis_title='Lucro/Prejuízo',
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#333"),
            height=350
        )
        bar_chart_json = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)


    return render_template('index.html',
                           user_name=user_name,
                           posicoes=posicoes,
                           total_valor_carteira=total_valor_carteira,
                           total_lucro_nao_realizado=total_lucro_nao_realizado,
                           total_prejuizo_nao_realizado=total_prejuizo_nao_realizado,
                           news_articles=news_articles,
                           pie_chart_json=pie_chart_json,
                           bar_chart_json=bar_chart_json)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try: # Adicionado bloco try-except para capturar erros
            # --- NOVAS LINHAS DE DEBUG AQUI ---
            print(f"DEBUG: ROTA /REGISTER (POST) - Headers da Requisição: {request.headers}")
            print(f"DEBUG: ROTA /REGISTER (POST) - Dados crus da requisição (request.get_data()): {request.get_data(as_text=True)}")
            print(f"DEBUG: ROTA /REGISTER (POST) - Conteúdo do Formulário (request.form): {request.form}")
            # --- FIM DAS NOVAS LINHAS DE DEBUG ---

            username = request.form['username']
            password = request.form['password']
            # ALTERADO AQUI: Usar .get() para evitar KeyError se o campo estiver ausente
            confirm_password = request.form.get('confirm_password') 
            
            full_name = request.form['full_name']
            email = request.form['email']
            contact_number = request.form.get('contact_number') # Opcional

            # Agora, a validação de `confirm_password` será alcançada
            if not username or not password or not confirm_password or not full_name or not email:
                flash('Por favor, preencha todos os campos obrigatórios (incluindo a confirmação de palavra-passe).', 'danger')
                return render_template('register.html', username=username, full_name=full_name, email=email, contact_number=contact_number)

            if password != confirm_password:
                flash('As palavras-passe não coincidem.', 'danger')
                return render_template('register.html', username=username, full_name=full_name, email=email, contact_number=contact_number)

            if len(password) < 6:
                flash('A palavra-passe deve ter pelo menos 6 caracteres.', 'danger')
                return render_template('register.html', username=username, full_name=full_name, email=email, contact_number=contact_number)

            # Verificar se o username já existe
            if get_user_by_username(username):
                flash('Nome de utilizador já registado. Por favor, escolha outro.', 'danger')
                return render_template('register.html', username=username, full_name=full_name, email=email, contact_number=contact_number)

            # Verificar se o email já existe
            if get_user_by_email(email):
                flash('Endereço de email já registado. Por favor, use outro ou faça login.', 'danger')
                return render_template('register.html', username=username, full_name=full_name, email=email, contact_number=contact_number)

            # Hash da senha
            hashed_password = generate_password_hash(password)

            # Definir o primeiro utilizador como admin
            is_admin = False
            if get_admin_count() == 0:
                is_admin = True

            with DBConnectionManager() as cursor_db:
                # Modificado para PostgreSQL para obter o ID inserido via RETURNING
                if DB_TYPE == 'postgresql':
                    sql = """
                    INSERT INTO users (username, password_hash, full_name, email, contact_number, is_admin, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id
                    """
                    cursor_db.execute(sql, (username, hashed_password, full_name, email, contact_number, is_admin, datetime.datetime.now()))
                    new_user_id = cursor_db.fetchone()[0] # Pega o ID retornado
                else: # MySQL
                    sql = """
                    INSERT INTO users (username, password_hash, full_name, email, contact_number, is_admin, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor_db.execute(sql, (username, hashed_password, full_name, email, contact_number, is_admin, datetime.datetime.now()))
                    new_user_id = cursor_db.lastrowid # Pega o ID para MySQL


                flash('Registo bem-sucedido! Pode agora fazer login.', 'success')

                # Logar a criação do primeiro admin, se for o caso
                if is_admin:
                    try:
                        log_admin_action(admin_user_id=new_user_id, # Agora passamos o ID do novo usuário
                                         action_type='PRIMEIRO_ADMIN_REGISTADO',
                                         target_user_id=new_user_id,
                                         details={'username': username, 'email': email})
                    except Exception as log_e:
                        print(f"ATENÇÃO: Erro ao logar ação de admin: {log_e}")

            return redirect(url_for('login'))
        except Exception as e: # Captura exceções mais gerais agora
            print(f"ERRO INESPERADO NA ROTA /REGISTER (POST): {e}") # Loga o erro real
            # Adiciona o erro real à mensagem flash para depuração
            flash(f'Ocorreu um erro ao registar. Por favor, tente novamente mais tarde. Detalhes técnicos: {e}', 'danger')
            return render_template('register.html', username=request.form.get('username'), full_name=request.form.get('full_name'), email=request.form.get('email'), contact_number=request.form.get('contact_number'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = get_user_by_username(username)
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin']) # Converte para booleano

            flash('Login bem-sucedido!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
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
    simbolo_filtro_raw = request.args.get('simbolo_filtro', '').strip() # Pega o filtro

    page = request.args.get('page', 1, type=int)

    # Mapeia símbolos comuns para os tickers do YFinance se o usuário digitar o nome
    simbolo_filtro_mapeado = SYMBOL_MAPPING.get(simbolo_filtro_raw.upper(), simbolo_filtro_raw)

    transacoes, total_transacoes = buscar_transacoes_filtradas(
        user_id,
        data_inicio,
        data_fim,
        ordenar_por,
        ordem,
        simbolo_filtro=simbolo_filtro_mapeado, # Passa o símbolo mapeado
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
                           simbolo_filtro=simbolo_filtro_raw, # Mantém o valor original para o campo de input
                           page=page,
                           total_pages=total_pages,
                           symbols_for_filter=sorted(list(SYMBOL_MAPPING.keys())))


@app.route('/add_transaction', methods=['GET', 'POST'])
@login_required
def add_transaction():
    user_id = session.get('user_id')
    user_name = session.get('username')
    symbols = sorted(list(SYMBOL_MAPPING.keys())) # Lista de símbolos para o dropdown

    if request.method == 'POST':
        # Converta a data de string para objeto date
        data_transacao_str = request.form['data_transacao']
        try:
            data_transacao = datetime.datetime.strptime(data_transacao_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Formato de data inválido. Use AAAA-MM-DD.', 'danger')
            return render_template('add_transaction.html', symbols=symbols)

        hora_transacao_str = request.form.get('hora_transacao') # Pode ser opcional
        hora_transacao = None
        if hora_transacao_str:
            try:
                hora_transacao = datetime.datetime.strptime(hora_transacao_str, '%H:%M').time()
            except ValueError:
                flash('Formato de hora inválido. Use HH:MM.', 'danger')
                return render_template('add_transaction.html', symbols=symbols)

        simbolo_ativo = request.form['simbolo_ativo']
        quantidade = request.form['quantidade']
        preco_unitario = request.form['preco_unitario']
        tipo_operacao = request.form['tipo_operacao']
        custos_taxas = request.form.get('custos_taxas', '0.00') # Pega com valor padrão '0.00'
        observacoes = request.form.get('observacoes') # Opcional

        # Converte para float/decimal e trata erros
        try:
            quantidade = decimal.Decimal(quantidade)
            preco_unitario = decimal.Decimal(preco_unitario)
            custos_taxas = decimal.Decimal(custos_taxas)
        except decimal.InvalidOperation:
            flash('Quantidade, Preço Unitário ou Custos/Taxas devem ser números válidos.', 'danger')
            return render_template('add_transaction.html', symbols=symbols)

        if quantidade <= 0 or preco_unitario <= 0:
            flash('Quantidade e Preço Unitário devem ser maiores que zero.', 'danger')
            return render_template('add_transaction.html', symbols=symbols)
        
        # Mapeia o símbolo escolhido pelo usuário para o ticker do YFinance
        simbolo_ativo_yf = SYMBOL_MAPPING.get(simbolo_ativo.upper(), simbolo_ativo)
        # Se não for um mapeamento direto e não tiver .SA, tenta adicionar .SA para ações brasileiras
        if simbolo_ativo_yf == simbolo_ativo and \
           not any(c in simbolo_ativo for c in ['^', '-', '=']) and \
           not any(simbolo_ativo.upper().endswith(suf) for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
            if 4 <= len(simbolo_ativo) <= 6 and simbolo_ativo.isalnum():
                simbolo_ativo_yf = f"{simbolo_ativo.upper()}.SA"
        # Garante que o símbolo final é capitalizado para consistência
        simbolo_ativo_yf = simbolo_ativo_yf.upper()

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
                    simbolo_ativo_yf, # Usa o símbolo mapeado
                    quantidade,
                    preco_unitario,
                    tipo_operacao,
                    custos_taxas,
                    observacoes,
                    datetime.datetime.now()
                ))
                flash('Transação adicionada com sucesso!', 'success')
                return redirect(url_for('transactions_list'))
        except Exception as e:
            flash(f'Ocorreu um erro ao adicionar a transação. Erro: {e}', 'danger')
            print(f"Erro ao adicionar transação: {e}")

    return render_template('add_transaction.html', user_name=user_name, symbols=symbols)


@app.route('/edit_transaction/<int:transaction_id>', methods=['GET', 'POST'])
@login_required
def edit_transaction(transaction_id):
    user_id = session.get('user_id')
    user_name = session.get('username')
    transaction = None
    symbols = sorted(list(SYMBOL_MAPPING.keys())) # Lista de símbolos para o dropdown

    try:
        with DBConnectionManager(dictionary=True) as cursor_db:
            cursor_db.execute("SELECT * FROM transacoes WHERE id = %s AND user_id = %s", (transaction_id, user_id))
            transaction = cursor_db.fetchone()
    except Exception as e:
        flash(f"Erro ao buscar transação para edição: {e}", "danger")
        return redirect(url_for('transactions_list'))

    if not transaction:
        flash('Transação não encontrada ou você não tem permissão para editá-la.', 'danger')
        return redirect(url_for('transactions_list'))

    if request.method == 'POST':
        data_transacao_str = request.form['data_transacao']
        try:
            data_transacao = datetime.datetime.strptime(data_transacao_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Formato de data inválido. Use AAAA-MM-DD.', 'danger')
            return render_template('editar_transacao.html', transaction=transaction, symbols=symbols)

        hora_transacao_str = request.form.get('hora_transacao')
        hora_transacao = None
        if hora_transacao_str:
            try:
                hora_transacao = datetime.datetime.strptime(hora_transacao_str, '%H:%M').time()
            except ValueError:
                flash('Formato de hora inválido. Use HH:MM.', 'danger')
                return render_template('editar_transacao.html', transaction=transaction, symbols=symbols)

        simbolo_ativo = request.form['simbolo_ativo']
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
            return render_template('editar_transacao.html', transaction=transaction, symbols=symbols)

        if quantidade <= 0 or preco_unitario <= 0:
            flash('Quantidade e Preço Unitário devem ser maiores que zero.', 'danger')
            return render_template('editar_transacao.html', transaction=transaction, symbols=symbols)

        # Mapeia o símbolo escolhido pelo usuário para o ticker do YFinance
        simbolo_ativo_yf = SYMBOL_MAPPING.get(simbolo_ativo.upper(), simbolo_ativo)
        # Se não for um mapeamento direto e não tiver .SA, tenta adicionar .SA para ações brasileiras
        if simbolo_ativo_yf == simbolo_ativo and \
           not any(c in simbolo_ativo for c in ['^', '-', '=']) and \
           not any(simbolo_ativo.upper().endswith(suf) for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
            if 4 <= len(simbolo_ativo) <= 6 and simbolo_ativo.isalnum():
                simbolo_ativo_yf = f"{simbolo_ativo.upper()}.SA"
        # Garante que o símbolo final é capitalizado para consistência
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
                return redirect(url_for('transactions_list'))
        except Exception as e:
            flash(f'Ocorreu um erro ao atualizar a transação. Erro: {e}', 'danger')
            print(f"Erro ao atualizar transação: {e}")

    # Para GET request, formatar data e hora para exibição no formulário
    if transaction['data_transacao']:
        transaction['data_transacao_formatted'] = transaction['data_transacao'].strftime('%Y-%m-%d')
    if transaction['hora_transacao']:
        # Converte timedelta para time se necessário (retorno do MySQL)
        if isinstance(transaction['hora_transacao'], datetime.timedelta):
            total_seconds = int(transaction['hora_transacao'].total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            transaction['hora_transacao_formatted'] = datetime.time(hours, minutes, seconds).strftime('%H:%M')
        elif isinstance(transaction['hora_transacao'], datetime.time):
            transaction['hora_transacao_formatted'] = transaction['hora_transacao'].strftime('%H:%M')
        else:
            transaction['hora_transacao_formatted'] = None
    else:
        transaction['hora_transacao_formatted'] = None

    # Mapear o ticker de volta para o nome "amigável" se existir
    original_simbolo_ativo_display = REVERSE_SYMBOL_MAPPING.get(transaction['simbolo_ativo'], transaction['simbolo_ativo'])
    transaction['simbolo_ativo_display'] = original_simbolo_ativo_display


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
            # Verifique se a transação pertence ao utilizador logado antes de excluir
            cursor_db.execute("DELETE FROM transacoes WHERE id = %s AND user_id = %s", (transaction_id, user_id))
            if cursor_db.rowcount > 0:
                flash('Transação excluída com sucesso.', 'success')
            else:
                flash('Transação não encontrada ou você não tem permissão para excluí-la.', 'danger')
    except Exception as e:
        flash(f'Ocorreu um erro ao excluir a transação. Erro: {e}', 'danger')
        print(f"Erro ao excluir transação: {e}")
    return redirect(url_for('transactions_list'))

@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    users = get_all_users()
    admin_count = get_admin_count() # Adicionado para exibir a contagem de admins
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

    # Se o alvo for um admin e for o último admin, não permitir a desativação
    if target_user['is_admin'] and get_admin_count() <= 1:
        flash('Não é possível remover o último administrador do sistema.', 'danger')
        return redirect(url_for('admin_dashboard'))

    new_status = not target_user['is_admin'] # Inverte o status atual

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

    # Prevenir exclusão do último admin
    if target_user['is_admin'] and get_admin_count() <= 1:
        flash('Não é possível excluir o último administrador do sistema.', 'danger')
        return redirect(url_for('admin_dashboard'))

    # Log antes de tentar excluir (para ter o username do target)
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
            # Tenta fazer o parse do JSON na coluna 'details'
            for log in logs:
                if log['details']:
                    try:
                        log['details_parsed'] = json.loads(log['details'])
                    except json.JSONDecodeError:
                        log['details_parsed'] = {'error': 'Invalid JSON'}
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
        contact_number = request.form.get('contact_number') # Pode ser None

        if not full_name or not email:
            flash('Nome completo e email são obrigatórios.', 'danger')
            return render_template('profile.html', user=user)

        # Verifica se o novo email já está em uso por outro usuário
        user_with_same_email = get_user_by_email(email)
        if user_with_same_email and user_with_same_email['id'] != user_id:
            flash('Este endereço de email já está registado por outro utilizador.', 'danger')
            return render_template('profile.html', user=user)

        if update_user_profile_data(user_id, full_name, email, contact_number):
            flash('Perfil atualizado com sucesso!', 'success')
            # Atualiza os dados do usuário na sessão se necessário
            # user_data['email'] = email
            # user_data['full_name'] = full_name
            # user_data['contact_number'] = contact_number
            # Redireciona para GET para evitar reenvio de formulário
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


@app.route('/predict_price/<simbolo>')
@login_required
def predict_price_api(simbolo):
    predicted_price = get_predicted_price_for_display(simbolo)
    if predicted_price is not None:
        return jsonify({'simbolo': simbolo, 'predicted_price': predicted_price})
    else:
        return jsonify({'error': 'Não foi possível obter previsão para este símbolo.', 'simbolo': simbolo}), 404

if __name__ == '__main__':
    # A porta será fornecida pelo Render na variável de ambiente PORT
    # Use 0.0.0.0 para que o servidor seja acessível externamente
    port = int(os.environ.get("PORT", 5000)) # 5000 é um fallback para desenvolvimento local
    app.run(debug=True, host='0.0.0.0', port=port)