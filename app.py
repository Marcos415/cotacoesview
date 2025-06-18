import datetime
import decimal
import os
import joblib
import pandas as pd
import yfinance as yf
import mysql.connector
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
# Agora pega as credenciais do banco de dados das variáveis de ambiente
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'), # 'localhost' como fallback para desenvolvimento
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE'),
    'port': int(os.getenv('DB_PORT', 3307)) # Converte para int, com 3307 como fallback
}

# Verifica se as variáveis de ambiente essenciais para o DB foram carregadas
if not all([DB_CONFIG['user'], DB_CONFIG['password'], DB_CONFIG['database']]):
    raise ValueError("Uma ou mais variáveis de ambiente do banco de dados (DB_USER, DB_PASSWORD, DB_DATABASE) não estão definidas. Por favor, defina-as no seu arquivo .env.")


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

# --- Context Manager para Conexões MySQL ---
class MySQLConnectionManager:
    def __init__(self, db_config, dictionary=False, buffered=False): 
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.dictionary = dictionary
        self.buffered = buffered 

    def __enter__(self):
        print(f"DEBUG: MySQLConnectionManager - Entrando no contexto (buffered={self.buffered})...")
        try:
            self.conn = mysql.connector.connect(**self.db_config)
            self.cursor = self.conn.cursor(dictionary=self.dictionary, buffered=self.buffered) 
            print("DEBUG: MySQLConnectionManager - Conexão e cursor estabelecidos.")
            return self.cursor
        except mysql.connector.Error as err:
            print(f"ERRO: MySQLConnectionManager - Erro ao conectar: {err}")
            if self.cursor:
                try: self.cursor.close()
                except mysql.connector.Error as close_err:
                    print(f"ERRO: MySQLConnectionManager - Erro ao fechar cursor em __enter__ após falha de conexão: {close_err}")
            if self.conn:
                try: self.conn.close()
                except mysql.connector.Error as close_err:
                    print(f"ERRO: MySQLConnectionManager - Erro ao fechar conexão em __enter__ após falha de conexão: {close_err}")
            raise 
        except Exception as e:
            print(f"ERRO: MySQLConnectionManager - Erro inesperado no __enter__: {e}")
            if self.cursor:
                try: self.cursor.close()
                except mysql.connector.Error as close_err:
                    print(f"ERRO: MySQLConnectionManager - Erro ao fechar cursor em __enter__ após erro inesperado: {close_err}")
            if self.conn:
                try: self.conn.close()
                except mysql.connector.Error as close_err:
                    print(f"ERRO: MySQLConnectionManager - Erro ao fechar conexão em __enter__ após erro inesperado: {close_err}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"DEBUG: MySQLConnectionManager - Saindo do contexto (exc_type={exc_type})...")
        if self.cursor:
            try:
                self.cursor.close()
                print("DEBUG: MySQLConnectionManager - Cursor fechado.")
            except mysql.connector.Error as err:
                print(f"ERRO: MySQLConnectionManager - Erro ao fechar cursor no __exit__: {err}")
        if self.conn:
            try:
                if exc_type is None: 
                    self.conn.commit()
                    print("DEBUG: MySQLConnectionManager - Conexão commitada.")
                else: 
                    print(f"DEBUG: MySQLConnectionManager - Transação será revertida devido a {exc_val}")
                    self.conn.rollback()
                    print("DEBUG: MySQLConnectionManager - Conexão revertida.")
            except mysql.connector.Error as err:
                print(f"ERRO: MySQLConnectionManager - Erro ao comitar/reverter no __exit__: {err}")
            finally: 
                try:
                    self.conn.close()
                    print("DEBUG: MySQLConnectionManager - Conexão fechada.")
                except mysql.connector.Error as err:
                    print(f"ERRO: MySQLConnectionManager - Erro ao fechar conexão no __exit__: {err}")

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
        with MySQLConnectionManager(DB_CONFIG, dictionary=True, buffered=True) as cursor_db:
            # Inclui 'full_name', 'email', 'contact_number' na seleção
            cursor_db.execute("SELECT id, username, full_name, email, contact_number, is_admin, created_at FROM users WHERE id = %s", (user_id,))
            user_data = cursor_db.fetchone()
            print(f"DEBUG: get_user_by_id({user_id}) - Dados do utilizador: {user_data}")
            return user_data
    except mysql.connector.Error as err:
        print(f"Erro MySQL ao buscar utilizador por ID: {err}")
        return None

def get_user_by_username(username):
    try:
        # Usar buffered=True aqui para garantir que os resultados sejam lidos imediatamente
        with MySQLConnectionManager(DB_CONFIG, dictionary=True, buffered=True) as cursor_db:
            # Inclui 'full_name', 'email', 'contact_number' na seleção
            cursor_db.execute("SELECT id, username, password_hash, full_name, email, contact_number, is_admin FROM users WHERE username = %s", (username,))
            user_data = cursor_db.fetchone()
            print(f"DEBUG: get_user_by_username({username}) - Dados do utilizador: {user_data}")
            return user_data
    except mysql.connector.Error as err:
        print(f"Erro MySQL ao buscar utilizador por nome de utilizador: {err}")
        return None

# Nova função para buscar utilizador por email
def get_user_by_email(email):
    try:
        with MySQLConnectionManager(DB_CONFIG, dictionary=True, buffered=True) as cursor_db:
            cursor_db.execute("SELECT id, username, full_name, email, contact_number, is_admin FROM users WHERE email = %s", (email,))
            user_data = cursor_db.fetchone()
            return user_data
    except mysql.connector.Error as err:
        print(f"Erro MySQL ao buscar utilizador por email: {err}")
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
        with MySQLConnectionManager(DB_CONFIG, dictionary=True, buffered=True) as cursor_db:
            # Inclui 'full_name', 'email', 'contact_number' na seleção
            sql_query = "SELECT id, username, full_name, email, contact_number, is_admin, created_at FROM users"
            cursor_db.execute(sql_query)
            all_users_from_db = cursor_db.fetchall()
            
            print(f"DEBUG: get_all_users() - Todos os utilizadores do DB: {all_users_from_db}")

            users = [u for u in all_users_from_db if u['id'] != current_user_id]
            
            print(f"DEBUG: get_all_users() - Utilizadores após filtrar o admin logado: {users}")

    except mysql.connector.Error as err:
        print(f"Erro MySQL ao buscar todos os utilizadores: {err}")
    except Exception as e:
        print(f"Erro inesperado em get_all_users: {e}")
    return users

def get_admin_count():
    """Retorna o número total de utilizadores com is_admin = TRUE."""
    try:
        # Usar buffered=True aqui para garantir que os resultados sejam lidos imediatamente
        with MySQLConnectionManager(DB_CONFIG, dictionary=True, buffered=True) as cursor_db:
            cursor_db.execute("SELECT COUNT(*) as admin_count FROM users WHERE is_admin = TRUE")
            result = cursor_db.fetchone()
            count = result['admin_count'] if result else 0
            print(f"DEBUG: get_admin_count() - Total de administradores: {count}")
            return count
    except mysql.connector.Error as err:
        print(f"Erro MySQL ao contar administradores: {err}")
        return 0
    except Exception as e:
        print(f"Erro inesperado em get_admin_count: {e}")
        return 0

def delete_user_from_db(user_id):
    """Exclui um utilizador do banco de dados e todas as suas transações/alertas."""
    print(f"DEBUG: Tentando excluir utilizador com ID: {user_id}")
    try:
        with MySQLConnectionManager(DB_CONFIG) as cursor_db:
            cursor_db.execute("DELETE FROM users WHERE id = %s", (user_id,))
            rows_affected = cursor_db.rowcount
            print(f"DEBUG: delete_user_from_db - Linhas afetadas na tabela users: {rows_affected}")
            return rows_affected > 0
    except mysql.connector.Error as err:
        print(f"Erro MySQL ao excluir utilizador: {err}")
        return False

def update_user_password(user_id, new_password):
    """Atualiza a palavra-passe de um utilizador."""
    print(f"DEBUG: Tentando redefinir senha para utilizador com ID: {user_id}")
    try:
        hashed_password = generate_password_hash(new_password)
        with MySQLConnectionManager(DB_CONFIG) as cursor_db:
            cursor_db.execute("UPDATE users SET password_hash = %s WHERE id = %s", (hashed_password, user_id))
            rows_affected = cursor_db.rowcount
            print(f"DEBUG: update_user_password - Linhas afetadas: {rows_affected}")
            return rows_affected > 0
    except mysql.connector.Error as err:
        print(f"Erro MySQL ao atualizar palavra-passe: {err}")
        return False

def toggle_user_admin_status(user_id, new_status):
    """
    Alterna o status de administrador de um utilizador.
    new_status deve ser um booleano (True/False).
    """
    print(f"DEBUG: Tentando definir is_admin para utilizador {user_id} como {new_status}")
    try:
        with MySQLConnectionManager(DB_CONFIG) as cursor_db:
            sql = "UPDATE users SET is_admin = %s WHERE id = %s"
            cursor_db.execute(sql, (new_status, user_id))
            rows_affected = cursor_db.rowcount
            print(f"DEBUG: toggle_user_admin_status - Linhas afetadas: {rows_affected}")
            return rows_affected > 0
    except mysql.connector.Error as err:
        print(f"Erro MySQL ao alternar status de admin: {err}")
        return False

# Nova função para atualizar dados do perfil do utilizador
def update_user_profile_data(user_id, full_name, email, contact_number):
    """Atualiza o nome completo, email e número de contacto de um utilizador."""
    try:
        with MySQLConnectionManager(DB_CONFIG) as cursor_db:
            sql = """
            UPDATE users 
            SET full_name = %s, email = %s, contact_number = %s 
            WHERE id = %s
            """
            cursor_db.execute(sql, (full_name, email, contact_number, user_id))
            rows_affected = cursor_db.rowcount
            print(f"DEBUG: update_user_profile_data - Linhas afetadas: {rows_affected}")
            return rows_affected > 0
    except mysql.connector.Error as err:
        print(f"Erro MySQL ao atualizar dados do perfil: {err}")
        return False
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
            
        with MySQLConnectionManager(DB_CONFIG) as cursor_db:
            sql = """
            INSERT INTO admin_audit_logs (admin_user_id, admin_username_at_action, action_type, target_user_id, target_username_at_action, details, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            details_json = json.dumps(details) if details is not None else None
            
            cursor_db.execute(sql, (admin_user_id, admin_username, action_type, target_user_id, target_username, details_json, datetime.datetime.now()))
    except mysql.connector.Error as err:
        print(f"ERRO ao registar ação de admin no log: {err}")
    except Exception as e:
        print(f"ERRO inesperado ao registar ação de admin: {e}")


# --- Função Auxiliar para Buscar Transações (agora por utilizador) ---
def buscar_transacoes_filtradas(user_id, data_inicio, data_fim, ordenar_por, ordem, simbolo_filtro=None, page=1, per_page=TRANSACTIONS_PER_PAGE):
    transacoes = []
    total_transacoes = 0
    try:
        print(f"DEBUG FILTERS: user_id={user_id}, data_inicio={data_inicio}, data_fim={data_fim}, ordenar_por={ordenar_por}, ordem={ordem}, simbolo_filtro='{simbolo_filtro}' (raw)")

        with MySQLConnectionManager(DB_CONFIG, dictionary=True) as cursor_db:
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

    except mysql.connector.Error as err:
        print(f"ERRO MySQL ao buscar transações: {err}")
        raise
    except Exception as e:
        print(f"ERRO inesperado em buscar_transacoes_filtradas: {e}")
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
        with MySQLConnectionManager(DB_CONFIG, dictionary=True) as cursor_db:
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
                        
                        total_valor_carteira += (preco_atual * float(dados_posicao['quantidade']))
                        
                        if lucro_prejuizo_nao_realizado_individual is not None:
                            if lucro_prejuizo_nao_realizado_individual >= 0:
                                total_lucro_nao_realizado += lucro_prejuizo_nao_realizado_individual
                            else:
                                total_prejuizo_nao_realizado += lucro_prejuizo_nao_realizado_individual

                    posicoes[simbolo] = {
                        'nome_popular': REVERSE_SYMBOL_MAPPING.get(simbolo, simbolo),
                        'quantidade_total': float(dados_posicao['quantidade']),
                        'preco_medio': float(preco_medio),
                        'preco_atual': float(preco_atual) if preco_atual is not None else None,
                        'lucro_prejuizo_nao_realizado': float(lucro_prejuizo_nao_realizado_individual) if lucro_prejuizo_nao_realizado_individual is not None else None,
                        'preco_previsto': float(preco_previsto) if preco_previsto is not None else None,
                        'valor_total_ativo': (float(preco_atual) * float(dados_posicao['quantidade'])) if preco_atual is not None else 0.0
                    }
    except mysql.connector.Error as err:
        print(f"Erro MySQL em calcular_posicoes_carteira: {err}")
        flash(f"Erro ao calcular posições da carteira: {err}", "danger")
    except Exception as e:
        print(f"Erro inesperado em calcular_posicoes_carteira: {e}")
        flash(f"Erro inesperado ao calcular posições da carteira: {e}", "danger")
    
    portfolio_cache[cache_key] = {
        'data': (posicoes, total_valor_carteira, total_lucro_nao_realizado, total_prejuizo_nao_realizado),
        'timestamp': datetime.datetime.now()
    }
    return posicoes, total_valor_carteira, total_lucro_nao_realizado, total_prejuizo_nao_realizado

# --- Funções para Alertas de Preço (agora por utilizador) ---
def checar_alertas_disparados(user_id):
    try:
        alertas_disparados_para_notificar = []
        with MySQLConnectionManager(DB_CONFIG, dictionary=True) as cursor_db:
            cursor_db.execute("SELECT * FROM alertas_preco WHERE user_id = %s AND status = 'ATIVO'", (user_id,))
            alertas_ativos = cursor_db.fetchall()

            for alerta in alertas_ativos:
                simbolo_alerta = alerta['simbolo_ativo']
                preco_alvo = float(alerta['preco_alvo'])
                tipo_alerta = alerta['tipo_alerta']
                
                # Usa a função de cotação com cache
                current_price = _get_current_price_yfinance(simbolo_alerta)

                if current_price is None:
                    continue

                alert_triggered = False
                if tipo_alerta == 'ACIMA' and current_price >= preco_alvo:
                    alert_triggered = True
                elif tipo_alerta == 'ABAIXO' and current_price <= preco_alvo:
                    alert_triggered = True

                if alert_triggered:
                    update_sql = "UPDATE alertas_preco SET status = 'DISPARADO', data_disparo = %s WHERE id = %s AND user_id = %s"
                    cursor_db.execute(update_sql, (datetime.datetime.now(), alerta['id'], user_id))
                    
                    alertas_disparados_para_notificar.append({
                        'simbolo': REVERSE_SYMBOL_MAPPING.get(simbolo_alerta, simbolo_alerta),
                        'preco_alvo': preco_alvo,
                        'tipo_alerta': tipo_alerta,
                        'preco_atual': current_price
                    })
        
        for alerta in alertas_disparados_para_notificar:
            flash(f"ALERTA: {alerta['simbolo']} atingiu {alerta['tipo_alerta']} R$ {alerta['preco_alvo']:.2f}! Preço atual: R$ {alerta['preco_atual']:.2f}", "warning")

    except mysql.connector.Error as err:
        print(f"Erro MySQL ao checar alertas: {err}")
        flash(f"Erro ao checar alertas: {err}", "danger")
    except Exception as e:
        print(f"Erro inesperado ao checar alertas: {e}")
        flash(f"Erro inesperado ao checar alertas: {e}", "danger")

def buscar_alertas(user_id):
    alertas = []
    try:
        with MySQLConnectionManager(DB_CONFIG, dictionary=True) as cursor_db:
            cursor_db.execute("SELECT * FROM alertas_preco WHERE user_id = %s ORDER BY data_criacao DESC", (user_id,))
            alertas = cursor_db.fetchall()
    except mysql.connector.Error as err:
        print(f"Erro MySQL ao buscar alertas: {err}")
        flash(f"Erro ao buscar alertas: {err}", "danger")
    except Exception as e:
        print(f"Erro inesperado ao buscar alertas: {e}")
        flash(f"Erro inesperado ao buscar alertas: {e}", "danger")
    return alertas

# --- Função para buscar notícias financeiras de uma API real (NewsAPI.org exemplo) com Cache ---
def _get_financial_news_from_api(query_term):
    cache_key = query_term

    if is_cache_fresh(news_cache, cache_key, NEWS_CACHE_TTL):
        return news_cache[cache_key]['data']

    news_list = []
    
    # Adicionando uma verificação mais robusta para a chave da API
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWSAPI_ORG_KEY_HERE": # Ajustado para a nova instrução
        print("AVISO: NEWS_API_KEY não configurada ou inválida. Por favor, obtenha uma em https://newsapi.org/ e atualize o app.py.")
        news_list.append({
            'title': f"Notícias para {REVERSE_SYMBOL_MAPPING.get(query_term, query_term)}: Chave NewsAPI.org não configurada!",
            'link': "https://newsapi.org/register",
            'source': "Erro de Configuração (NewsAPI)",
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        })
        news_cache[cache_key] = {'data': news_list, 'timestamp': datetime.datetime.now()}
        return news_list

    search_query_name = REVERSE_SYMBOL_MAPPING.get(query_term, query_term).replace('.SA', '')
    if search_query_name == query_term: 
        search_query_name = query_term.replace('.SA', '')
    
    search_query = f"{search_query_name} OR {query_term}"
    
    params = {
        'q': search_query,
        'language': 'pt',
        'sortBy': 'publishedAt',
        'apiKey': NEWS_API_KEY,
        'pageSize': 5
    }

    try:
        response = requests.get(NEWS_API_BASE_URL, params=params, timeout=10)
        response.raise_for_status() # Lança HTTPError para respostas de erro (4xx ou 5xx)
        data = response.json()

        if data['status'] == 'ok' and data['articles']:
            for article in data['articles']:
                source_name = article['source']['name'] if article['source'] and 'name' in article['source'] else "Desconhecida"
                
                published_at_str = article['publishedAt']
                try:
                    published_dt = datetime.datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                    formatted_date = published_dt.strftime('%Y-%m-%d %H:%M')
                except ValueError:
                    formatted_date = published_at_str

                news_list.append({
                    'title': article['title'],
                    'link': article['url'],
                    'source': source_name,
                    'date': formatted_date
                })
        else:
            news_list.append({
                'title': f"Nenhuma notícia encontrada para {REVERSE_SYMBOL_MAPPING.get(query_term, query_term)} via API.",
                'link': "#",
                'source': "NewsAPI.org",
                'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            })

    except requests.exceptions.HTTPError as http_err:
        print(f"ERRO HTTP ao buscar notícias para {query_term}: {http_err}. Resposta: {response.text}")
        news_list.append({
            'title': f"Erro HTTP ao buscar notícias para {REVERSE_SYMBOL_MAPPING.get(query_term, query_term)}: {http_err}",
            'link': "#",
            'source': "Erro HTTP (NewsAPI)",
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        })
    except requests.exceptions.ConnectionError as conn_err:
        print(f"ERRO de Conexão ao buscar notícias para {query_term}: {conn_err}")
        news_list.append({
            'title': f"Erro de conexão ao NewsAPI para {REVERSE_SYMBOL_MAPPING.get(query_term, query_term)}.",
            'link': "#",
            'source': "Erro de Rede",
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        })
    except requests.exceptions.Timeout as timeout_err:
        print(f"ERRO de Timeout ao buscar notícias para {query_term}: {timeout_err}")
        news_list.append({
            'title': f"Tempo esgotado ao buscar notícias para {REVERSE_SYMBOL_MAPPING.get(query_term, query_term)}.",
            'link': "#",
            'source': "Timeout de Rede",
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        })
    except requests.exceptions.RequestException as e:
        print(f"ERRO geral RequestException ao buscar notícias para {query_term}: {e}")
        news_list.append({
            'title': f"Erro inesperado de requisição à NewsAPI para {REVERSE_SYMBOL_MAPPING.get(query_term, query_term)}.",
            'link': "#",
            'source': "Erro de Requisição",
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        })
    except json.JSONDecodeError as e:
        print(f"ERRO ao decodificar JSON da NewsAPI para {query_term}: {e}. Resposta: {response.text}")
        news_list.append({
            'title': f"Erro ao decodificar JSON da NewsAPI para {REVERSE_SYMBOL_MAPPING.get(query_term, query_term)}.",
            'link': "#",
            'source': "Erro de Processamento",
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        })
    except Exception as e:
        print(f"ERRO inesperado ao buscar notícias para {query_term}: {e}")
        news_list.append({
            'title': f"Erro inesperado ao buscar notícias para {REVERSE_SYMBOL_MAPPING.get(query_term, query_term)}.",
            'link': "#",
            'source': "Erro Geral",
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        })
    
    for news_item in news_list:
        news_item['simbolo_display'] = REVERSE_SYMBOL_MAPPING.get(query_term, query_term)
    
    news_cache[cache_key] = {'data': news_list, 'timestamp': datetime.datetime.now()}
    return news_list

# --- Rotas de Autenticação ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: 
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = get_user_by_username(username)
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin']) 
            print(f"DEBUG: Login bem-sucedido para {user['username']}. is_admin: {session['is_admin']}")
            flash('Login bem-sucedido!', 'success')
            next_url = request.args.get('next') or url_for('index')
            return redirect(next_url)
        else:
            print(f"DEBUG: Tentativa de login falhou para {username}.")
            flash('Nome de utilizador ou palavra-passe inválidos.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session: 
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        full_name = request.form['full_name'] 
        email = request.form['email']         
        contact_number = request.form.get('contact_number') 
        
        if not username or not password or not full_name or not email:
            flash('Nome de utilizador, nome completo, email e palavra-passe são obrigatórios.', 'danger')
            return redirect(url_for('register'))

        existing_user = get_user_by_username(username)
        if existing_user:
            flash('Nome de utilizador já existe. Por favor, escolha outro.', 'danger')
            return redirect(url_for('register'))
            
        if email: 
            existing_email_user = get_user_by_email(email)
            if existing_email_user:
                flash('Este email já está registado. Por favor, use outro.', "danger")
                return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        
        try:
            with MySQLConnectionManager(DB_CONFIG) as cursor_db:
                sql = """
                INSERT INTO users (username, password_hash, full_name, email, contact_number, is_admin) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor_db.execute(sql, (username, hashed_password, full_name, email, contact_number, False)) 
            print(f"DEBUG: Novo utilizador '{username}' registado como não-admin.")
            flash('Registo bem-sucedido! Por favor, faça login.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            print(f"Erro MySQL ao registar utilizador: {err}")
            flash(f"Erro ao registar utilizador: {err}", "danger")
        except Exception as e:
            print(f"Erro inesperado no registo: {e}")
            flash(f"Erro inesperado no registo: {e}", "danger")
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    print(f"DEBUG: A terminar sessão para user_id: {session.get('user_id')}")
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('is_admin', None) 
    flash('Sessão encerrada com sucesso.', 'info')
    return redirect(url_for('login'))

# --- Rotas do Aplicativo (Protegidas) ---
@app.route('/')
@login_required 
def index():
    user_id = session.get('user_id')
        
    checar_alertas_disparados(user_id) 

    data_inicio = request.args.get('data_inicio')
    data_fim = request.args.get('data_fim')
    ordenar_por = request.args.get('ordenar_por', 'data_transacao')
    ordem = request.args.get('ordem', 'DESC')
    simbolo_filtro = request.args.get('simbolo_filtro')
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', TRANSACTIONS_PER_PAGE, type=int)

    logged_in_username = session.get('username', 'Convidado') 
    is_admin = session.get('is_admin', False) 
    print(f"DEBUG: index() - Renderizando para utilizador: {logged_in_username}, is_admin: {is_admin}")

    try:
        transacoes, total_transacoes = buscar_transacoes_filtradas(user_id, data_inicio, data_fim, ordenar_por, ordem, simbolo_filtro, page, per_page)
        
        posicoes_carteira, total_valor_carteira, total_lucro_nao_realizado, total_prejuizo_nao_realizado = calcular_posicoes_carteira(user_id)
        
        alertas = buscar_alertas(user_id)

        all_news = []
        ativos_com_posicao = [simbolo for simbolo, dados in posicoes_carteira.items() if dados['quantidade_total'] > 0]
        ativos_de_alertas = [alerta['simbolo_ativo'] for alerta in alertas if alerta['status'] == 'ATIVO']
        symbols_to_fetch_news = list(set(ativos_com_posicao + ativos_de_alertas))

        if symbols_to_fetch_news:
            for simbolo_ativo in symbols_to_fetch_news[:5]: 
                news_for_symbol = _get_financial_news_from_api(simbolo_ativo)
                for news_item in news_for_symbol:
                    all_news.append(news_item)
        else:
             all_news.append({
                'title': "Adicione transações ou alertas para ver notícias relevantes!",
                'link': "#",
                'source': "Informação do Sistema",
                'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'simbolo_display': "Seu Portfólio'S"
            })
        
        total_pages = (total_transacoes + per_page - 1) // per_page
        
    except Exception as e:
        flash(f"Erro ao carregar dados: {e}", "danger")
        transacoes = []
        posicoes_carteira = {}
        alertas = []
        all_news = [{
            'title': f"Erro geral ao carregar notícias: {e}",
            'link': "#",
            'source': "Erro do Sistema",
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'simbolo_display': "Sistema"
        }]
        total_transacoes = 0
        total_pages = 0
        total_valor_carteira = 0.0
        total_lucro_nao_realizado = 0.0
        total_prejuizo_nao_realizado = 0.0

    return render_template('index.html',
                           transacoes=transacoes,
                           posicoes_carteira=posicoes_carteira,
                           alertas=alertas,
                           all_news=all_news,
                           REVERSE_SYMBOL_MAPPING=REVERSE_SYMBOL_MAPPING,
                           SYMBOL_MAPPING=SYMBOL_MAPPING,
                           data_inicio=data_inicio,
                           data_fim=data_fim,
                           ordenar_por=ordenar_por,
                           ordem=ordem,
                           simbolo_filtro=simbolo_filtro,
                           page=page,
                           per_page=per_page,
                           total_transacoes=total_transacoes,
                           total_pages=total_pages,
                           total_valor_carteira=total_valor_carteira,
                           total_lucro_nao_realizado=total_lucro_nao_realizado,
                           total_prejuizo_nao_realizado=total_prejuizo_nao_realizado,
                           logged_in_username=logged_in_username,
                           is_admin=is_admin) 

# Nova rota para a página de adicionar transação
@app.route('/add_transaction', methods=['GET'])
@login_required
def add_transaction():
    """Renderiza o formulário para adicionar uma nova transação."""
    return render_template('add_transaction.html') # Você precisará criar este template, se já não tiver

@app.route('/adicionar_transacao', methods=['POST'])
@login_required
def adicionar_transacao(): # Endpoint para processar o formulário POST de adição de transação
    user_id = session['user_id'] 
    try:
        simbolo_ativo_input = request.form['simbolo_ativo'].upper()
        simbolo_ativo_yf = SYMBOL_MAPPING.get(simbolo_ativo_input, simbolo_ativo_input)

        if simbolo_ativo_yf == simbolo_ativo_input and \
           not any(c in simbolo_ativo_input for c in ['^', '-', '=']) and \
           not any(simbolo_ativo_input.upper().endswith(suf)for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
            if 4 <= len(simbolo_ativo_input) <= 6 and simbolo_ativo_input.isalnum():
                simbolo_ativo_yf = f"{simbolo_ativo_input}.SA"

        data_transacao = request.form['data_transacao']
        hora_transacao_str = request.form.get('hora_transacao')
        tipo_operacao = request.form['tipo_operacao'].upper()
        preco_unitario = decimal.Decimal(request.form['preco_unitario'])
        quantidade = decimal.Decimal(request.form['quantidade'])
        custos_taxas = decimal.Decimal(request.form.get('custos_taxas', '0.00'))
        observacoes = request.form.get('observacoes')

        hora_transacao = None
        if hora_transacao_str:
            try:
                hora_transacao = datetime.datetime.strptime(hora_transacao_str,'%H:%M').time()
            except ValueError:
                flash("Formato de hora inválido. Use HH:MM.", "danger")
                # Redireciona de volta para a página de adicionar transação ou index
                return redirect(url_for('add_transaction')) 

        with MySQLConnectionManager(DB_CONFIG) as cursor_db:
            sql = """
            INSERT INTO transacoes (user_id, simbolo_ativo, data_transacao, hora_transacao, tipo_operacao, preco_unitario, quantidade, custos_taxas, observacoes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor_db.execute(sql, (user_id, simbolo_ativo_yf, data_transacao, hora_transacao, tipo_operacao, preco_unitario, quantidade, custos_taxas, observacoes))
        flash("Transação adicionada com sucesso!", "success")
        
        if user_id in portfolio_cache:
            del portfolio_cache[user_id]
        for key in list(prediction_cache.keys()):
            if simbolo_ativo_yf in key:
                del prediction_cache[key]
        for key in list(market_data_cache.keys()):
            if simbolo_ativo_yf in key:
                del market_data_cache[key]


    except mysql.connector.Error as err:
        flash(f"Erro ao adicionar transação: {err}", "danger")
    except decimal.InvalidOperation:
        flash("Erro: Preço unitário ou quantidade inválidos.", "danger")
    except Exception as e:
        flash(f"Erro inesperado ao adicionar transação: {e}", "danger")
    return redirect(url_for('index')) # Redireciona para o index após adicionar


@app.route('/edit_transaction/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_transaction(id): # Endpoint renomeado para 'edit_transaction' para consistência
    user_id = session['user_id']
    if request.method == 'POST':
        try:
            simbolo_ativo_input = request.form['simbolo_ativo'].upper()
            simbolo_ativo_yf = SYMBOL_MAPPING.get(simbolo_ativo_input, simbolo_ativo_input)

            if simbolo_ativo_yf == simbolo_ativo_input and \
               not any(c in simbolo_ativo_input for c in ['^', '-', '=']) and \
               not any(simbolo_ativo_input.upper().endswith(suf)for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
                if 4 <= len(simbolo_ativo_input) <= 6 and simbolo_ativo_input.isalnum():
                    simbolo_ativo_yf = f"{simbolo_ativo_input}.SA"

            data_transacao = request.form['data_transacao']
            hora_transacao_str = request.form.get('hora_transacao')
            tipo_operacao = request.form['tipo_operacao'].upper()
            preco_unitario = decimal.Decimal(request.form['preco_unitario'])
            quantidade = decimal.Decimal(request.form['quantidade'])
            custos_taxas = decimal.Decimal(request.form.get('custos_taxas', '0.00'))
            observacoes = request.form.get('observacoes')

            hora_transacao = None
            if hora_transacao_str:
                try:
                    hora_transacao = datetime.datetime.strptime(hora_transacao_str, '%H:%M').time()
                except ValueError:
                    flash("Formato de hora inválido. Use HH:MM.", "danger")
                    return redirect(url_for('edit_transaction', id=id))

            with MySQLConnectionManager(DB_CONFIG) as cursor_db:
                sql = """
                UPDATE transacoes SET simbolo_ativo = %s, data_transacao = %s, hora_transacao = %s, tipo_operacao = %s, preco_unitario = %s, quantidade = %s, custos_taxas = %s, observacoes = %s
                WHERE id = %s AND user_id = %s
                """
                cursor_db.execute(sql, (simbolo_ativo_yf, data_transacao, hora_transacao, tipo_operacao, preco_unitario, quantidade, custos_taxas, observacoes, id, user_id))
                if cursor_db.rowcount == 0:
                    flash("Transação não encontrada ou não pertence ao seu utilizador.", "danger")
                    return redirect(url_for('index'))
            flash("Transação atualizada com sucesso!", "success")

            if user_id in portfolio_cache:
                del portfolio_cache[user_id]
            for key in list(prediction_cache.keys()):
                if simbolo_ativo_yf in key:
                    del prediction_cache[key]
            for key in list(market_data_cache.keys()):
                if simbolo_ativo_yf in key:
                    del market_data_cache[key]

            return redirect(url_for('index'))
        except mysql.connector.Error as err:
            flash(f"Erro ao editar transação: {err}", "danger")
        except decimal.InvalidOperation:
            flash("Erro: Preço unitário ou quantidade inválidos.", "danger")
        except Exception as e:
            flash(f"Erro inesperado ao editar transação: {e}", "danger")
        return redirect(url_for('edit_transaction', id=id))
    else:
        transacao = None
        try:
            with MySQLConnectionManager(DB_CONFIG, dictionary=True) as cursor_db: # Garante dictionary=True
                cursor_db.execute("SELECT * FROM transacoes WHERE id = %s AND user_id = %s", (id, user_id))
                transacao_raw = cursor_db.fetchone() # Busca o resultado
                
                if transacao_raw:
                    # Converte para dicionário se for uma tupla (segurança extra, já que dictionary=True deveria cuidar disso)
                    if isinstance(transacao_raw, tuple):
                        # Isso deve ser redundante se dictionary=True funcionar, mas é uma defesa
                        transacao = {desc[0]: val for desc, val in zip(cursor_db.description, transacao_raw)}
                        print(f"DEBUG: edit_transaction - Transação convertida de tupla para dict: {transacao.keys()}")
                    else:
                        transacao = transacao_raw # Já é um dicionário se dictionary=True funcionou

                    if transacao and isinstance(transacao.get('hora_transacao'), datetime.timedelta):
                        total_seconds = int(transacao['hora_transacao'].total_seconds())
                        hours, remainder = divmod(total_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        transacao['hora_transacao'] = datetime.time(hours, minutes, seconds)
                else:
                    flash("Transação não encontrada ou não pertence ao seu utilizador.", "danger")
                    return redirect(url_for('index'))

        except mysql.connector.Error as err:
            flash(f"Erro ao buscar transação para edição: {err}", "danger")
            print(f"ERRO: MySQL ao buscar transação para edição: {err}")
        except Exception as e:
            flash(f"Erro inesperado ao buscar transação para edição: {e}", "danger")
            print(f"ERRO: Inesperado ao buscar transação para edição: {e}")

        if transacao:
            # print(f"DEBUG: edit_transaction - Tipo final de transacao para render: {type(transacao)}")
            return render_template('editar_transacao.html', transacao=transacao, REVERSE_SYMBOL_MAPPING=REVERSE_SYMBOL_MAPPING)
        # Se chegou aqui sem transacao válida ou com erro, o flash já foi enviado
        return redirect(url_for('index'))


@app.route('/excluir_transacao/<int:id>', methods=['POST'])
@login_required
def excluir_transacao(id):
    user_id = session['user_id']
    try:
        with MySQLConnectionManager(DB_CONFIG) as cursor_db:
            cursor_db.execute("DELETE FROM transacoes WHERE id = %s AND user_id = %s", (id, user_id))
            if cursor_db.rowcount == 0:
                flash("Transação não encontrada ou não pertence ao seu utilizador.", "danger")
        flash("Transação excluída com sucesso!", "success")

        if user_id in portfolio_cache:
            del portfolio_cache[user_id]
        prediction_cache.clear() 
        market_data_cache.clear()

    except mysql.connector.Error as err:
        flash(f"Erro ao excluir transação: {err}", "danger")
    except Exception as e:
        flash(f"Erro inesperado ao excluir transação: {e}", "danger")
    return redirect(url_for('index'))

# --- Rota para API de Gráficos (Plotly JSON) ---
@app.route('/api/historical_prices/<simbolo_display>')
def historical_prices(simbolo_display):
    try:
        chart_type = request.args.get('type', 'line') 

        final_simbolo_to_fetch = SYMBOL_MAPPING.get(simbolo_display.upper(), simbolo_display)

        if final_simbolo_to_fetch == simbolo_display and \
           not any(c in simbolo_display for c in ['^', '-', '=']) and \
           not any(simbolo_display.upper().endswith(suf)for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
            if 4 <= len(simbolo_display) <= 6 and simbolo_display.isalnum():
                final_simbolo_to_fetch = f"{simbolo_display.upper()}.SA"

        df_hist = get_historical_prices_yfinance_cached(final_simbolo_to_fetch, period="1y", interval="1d")

        required_cols_candlestick = ['Open', 'High', 'Low', 'Close']
        has_candlestick_data = all(col in df_hist.columns for col in required_cols_candlestick)

        # Determina a moeda com base no símbolo
        currency_symbol = "R$" # Padrão para BRL (ativos brasileiros)
        if '.SA' not in final_simbolo_to_fetch: # Se não for ativo brasileiro (ex: BTC-USD, ^IXIC)
            if 'USD' in final_simbolo_to_fetch or '^IXIC' in final_simbolo_to_fetch or '^DJI' in final_simbolo_to_fetch: # Dólar para criptos e índices USA
                currency_symbol = "$"
            elif '=F' in final_simbolo_to_fetch: # Futuros (como ouro GC=F)
                currency_symbol = "$" # Ou outra mais específica se soubermos (ex: oz)
            # Adicione mais regras de moeda conforme necessário para outros símbolos

        if df_hist.empty or 'Close' not in df_hist.columns:
            # Retorna uma resposta JSON que o frontend pode usar para exibir uma mensagem
            return jsonify({
                "error": "Dados históricos não encontrados ou insuficientes para o símbolo.",
                "simbolo_buscado": REVERSE_SYMBOL_MAPPING.get(final_simbolo_to_fetch, final_simbolo_to_fetch),
                "graph": { # Estrutura para o Plotly exibir uma mensagem
                    "data": [], # Sem dados para Plotly
                    "layout": {
                        "title": f"Dados não disponíveis para {REVERSE_SYMBOL_MAPPING.get(final_simbolo_to_fetch, final_simbolo_to_fetch)}",
                        "xaxis": {"visible": False}, # Esconde eixos
                        "yaxis": {"visible": False},
                        "annotations": [{ # Adiciona uma anotação no centro do gráfico
                            "text": "Dados de preço histórico não disponíveis ou insuficientes.",
                            "xref": "paper", "yref": "paper",
                            "showarrow": False, "font": {"size": 16, "color": "#dc2626"} # Cor vermelha
                        }]
                    }
                }
            }), 404 # Retorna status 404 (Not Found)

        data_traces = []
        yaxis_title = f'Preço ({currency_symbol})' # Título do eixo Y com moeda
        xaxis_rangeslider_visible = False 

        if chart_type == 'candlestick' and has_candlestick_data:
            data_traces.append(go.Candlestick(
                x=df_hist.index,
                open=df_hist['Open'],
                high=df_hist['High'],
                low=df_hist['Low'],
                close=df_hist['Close'],
                name='Candlestick',
                # Atualiza hovertemplate com o símbolo da moeda
                hovertemplate=f'<b>Data</b>: %{{x|%Y-%m-%d}}<br>' +
                              f'<b>Abertura</b>: {currency_symbol} %{{open:.2f}}<br>' +
                              f'<b>Máxima</b>: {currency_symbol} %{{high:.2f}}<br>' +
                              f'<b>Mínima</b>: {currency_symbol} %{{low:.2f}}<br>' +
                              f'<b>Fechamento</b>: {currency_symbol} %{{close:.2f}}<extra></extra>' 
            ))
            yaxis_title = f'Preço (OHLC) ({currency_symbol})'
            xaxis_rangeslider_visible = True
        else: # Fallback para linha se o tipo não for reconhecido ou dados candlestick não completos
             trace = go.Scatter( 
                x=df_hist.index,
                y=df_hist['Close'],
                mode='lines',
                name='Preço de Fechamento',
                line=dict(color='#1a73e8', width=3), 
                # Atualiza hovertemplate com o símbolo da moeda
                hovertemplate=f'<b>Data</b>: %{{x|%Y-%m-%d}}<br>'+
                              f'<b>Preço</b>: {currency_symbol} %{{y:.2f}}<extra></extra>' 
            )
             data_traces.append(trace)
             yaxis_title = f'Preço de Fechamento ({currency_symbol})'
             xaxis_rangeslider_visible = False
        
        fig = go.Figure(data=data_traces)

        fig.update_layout(
            title_text=f"Preço Histórico de {REVERSE_SYMBOL_MAPPING.get(final_simbolo_to_fetch, final_simbolo_to_fetch)} ({chart_type.capitalize()})",
            xaxis_title='Data',
            yaxis_title=yaxis_title,
            xaxis_rangeslider_visible=xaxis_rangeslider_visible, 
            hovermode="x unified", 
            height=550,
            template="plotly_white",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1a", step="year", stepmode="backward"),
                        dict(count=5, label="5a", step="year", stepmode="backward"),
                        dict(step="all", label="Tudo")
                    ])
                ),
                rangeslider=dict(visible=xaxis_rangeslider_visible), 
                type="date"
            )
        )
        
        fig_dict = json.loads(fig.to_json())

        # Estes ajustes de cor e preenchimento devem estar já no go.Scatter, mas mantemos para garantir
        if chart_type == 'line' or chart_type == 'area':
            if fig_dict['data'] and len(fig_dict['data']) > 0:
                if 'line' not in fig_dict['data'][0]:
                    fig_dict['data'][0]['line'] = {}
                fig_dict['data'][0]['line']['color'] = '#1a73e8'
                fig_dict['data'][0]['line']['width'] = 3
                if chart_type == 'area':
                    fig_dict['data'][0]['fill'] = 'tozeroy'
                    fig_dict['data'][0]['fillcolor'] = 'rgba(26, 115, 232, 0.2)'
                else: 
                    if 'fill' in fig_dict['data'][0]:
                        del fig_dict['data'][0]['fill']
                    if 'fillcolor' in fig_dict['data'][0]:
                        del fig_dict['data'][0]['fillcolor']


        return jsonify(fig_dict) 

    except Exception as e:
        print(f"ERRO ao buscar dados do gráfico para {simbolo_display}: {e}")
        # Retorna um erro genérico com uma estrutura que o frontend possa lidar
        return jsonify({
            "error": f"Erro interno ao buscar dados do gráfico: {str(e)}",
            "graph": {
                "data": [],
                "layout": {
                    "title": f"Erro ao carregar o gráfico de {REVERSE_SYMBOL_MAPPING.get(final_simbolo_to_fetch, final_simbolo_to_fetch)}",
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "annotations": [{
                        "text": f"Ocorreu um erro ao carregar os dados: {str(e)}",
                        "xref": "paper", "yref": "paper",
                        "showarrow": False, "font": {"size": 14, "color": "#dc2626"}
                    }]
                }
            }
        }), 500

# --- ROTAS DE EXPORTAÇÃO (agora por utilizador) ---
@app.route('/export_transactions/<format>')
@login_required
def export_transactions(format):
    user_id = session['user_id']
    try:
        data_inicio = request.args.get('data_inicio')
        data_fim = request.args.get('data_fim')
        ordenar_por = request.args.get('ordenar_por', 'data_transacao')
        ordem = request.args.get('ordem', 'DESC')
        simbolo_filtro = request.args.get('simbolo_filtro')
        
        transacoes_data, _ = buscar_transacoes_filtradas(user_id, data_inicio, data_fim, ordenar_por, ordem, simbolo_filtro, page=1, per_page=999999)

        if not transacoes_data:
            flash("Nenhuma transação encontrada para exportar com os filtros aplicados.", "warning")
            return redirect(url_for('index'))

        processed_data = []
        for t in transacoes_data:
            processed_data.append({
                'ID': t['id'],
                'Data': t['data_transacao'].strftime('%Y-%m-%d') if t['data_transacao'] else '',
                'Hora': t['hora_transacao'].strftime('%H:%M') if t['hora_transacao'] else '',
                'Símbolo Ativo': REVERSE_SYMBOL_MAPPING.get(t['simbolo_ativo'], t['simbolo_ativo']),
                'Tipo Operação': t['tipo_operacao'],
                'Preço Unitário': float(t['preco_unitario']),
                'Quantidade': float(t['quantidade']),
                'Custos/Taxas': float(t['custos_taxas']),
                'Observações': t['observacoes'] if t['observacoes'] else ''
            })
        
        df_export = pd.DataFrame(processed_data)

        if format == 'csv':
            csv_output = df_export.to_csv(index=False, sep=';', encoding='utf-8-sig')
            response = app.make_response(csv_output)
            response.headers["Content-Disposition"] = "attachment; filename=transacoes.csv"
            response.headers["Content-type"] = "text/csv; charset=utf-8-sig"
            return response
        elif format == 'pdf':
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)

            col_widths = [10, 20, 14, 26, 20, 20, 18, 18, 44] 
            headers = ['ID', 'Data', 'Hora', 'Símbolo Ativo', 'Tipo Op.', 'Preço Un.', 'Qtd.', 'Custos', 'Obs.']
            
            base_line_height = 8 

            pdf.set_font("Arial", style='B')
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], base_line_height, header, 1, 0, 'C')
            pdf.ln(base_line_height)

            pdf.set_font("Arial", style='')

            for row_data in processed_data:
                row_start_y = pdf.get_y()
                
                obs_text = row_data['Observações'] if row_data['Observações'] else ''
                
                try:
                    text_width_obs = pdf.get_string_width(obs_text)
                    num_lines_obs = (text_width_obs / (col_widths[8] - 2)) + 0.999 if (col_widths[8] - 2) > 0 else 1
                    num_lines_obs = max(1, int(num_lines_obs))
                except Exception as e:
                    print(f"Erro ao calcular largura da string para PDF: {e}. Texto: '{obs_text}'")
                    num_lines_obs = 1

                actual_row_height = max(base_line_height, num_lines_obs * base_line_height)
                
                if pdf.get_y() + actual_row_height > pdf.page_break_trigger:
                    pdf.add_page()
                    pdf.set_font("Arial", size=10, style='B')
                    for i, header in enumerate(headers):
                        pdf.cell(col_widths[i], base_line_height, header, 1, 0, 'C')
                    pdf.ln(base_line_height)
                    pdf.set_font("Arial", style='')
                    row_start_y = pdf.get_y()

                current_x_pos = pdf.get_x()

                cells_data_and_widths = [
                    (str(row_data['ID']), col_widths[0], 'C'),
                    (str(row_data['Data']), col_widths[1], 'C'),
                    (str(row_data['Hora']), col_widths[2], 'C'),
                    (str(row_data['Símbolo Ativo']), col_widths[3], 'C'),
                    (str(row_data['Tipo Operação']), col_widths[4], 'C'),
                    (f"{row_data['Preço Unitário']:.2f}", col_widths[5], 'R'),
                    (f"{row_data['Quantidade']:.2f}", col_widths[6], 'R'),
                    (f"{row_data['Custos/Taxas']:.2f}", col_widths[7], 'R')
                ]

                for text_val, width, align in cells_data_and_widths:
                    pdf.set_xy(current_x_pos, row_start_y)
                    pdf.cell(width, actual_row_height, text_val, 1, 0, align)
                    current_x_pos += width

                pdf.set_xy(current_x_pos, row_start_y)
                pdf.multi_cell(col_widths[8], base_line_height, obs_text, 1, 'L', False)

                pdf.set_y(row_start_y + actual_row_height)
                pdf.set_x(10)

            pdf_output = pdf.output(dest='S').encode('latin-1')
            response = app.make_response(pdf_output)
            response.headers["Content-Disposition"] = "attachment; filename=transacoes.pdf"
            response.headers["Content-type"] = "application/pdf"
            return response
        else:
            flash("Formato de exportação não suportado.", "danger")
            return redirect(url_for('index'))

    except Exception as e:
        flash(f"Erro ao exportar transações: {e}", "danger")
        print(f"DEBUG: Erro detalhado na exportação: {e}")
        return redirect(url_for('index'))

# --- ROTAS DE ALERTA (agora por utilizador) ---
@app.route('/adicionar_alerta', methods=['POST'])
@login_required
def adicionar_alerta():
    user_id = session['user_id']
    try:
        simbolo_ativo_input = request.form['simbolo_ativo'].upper()
        simbolo_ativo_yf = SYMBOL_MAPPING.get(simbolo_ativo_input, simbolo_ativo_input)

        if simbolo_ativo_yf == simbolo_ativo_input and \
           not any(c in simbolo_ativo_input for c in ['^', '-', '=']) and \
           not any(simbolo_ativo_input.upper().endswith(suf)for suf in ['.SA', '.BA', '.TO', '.L', '.PA', '.AX', '.V', '.F']):
            if 4 <= len(simbolo_ativo_input) <= 6 and simbolo_ativo_input.isalnum():
                simbolo_ativo_yf = f"{simbolo_ativo_input}.SA"
            
        preco_alvo = decimal.Decimal(request.form['preco_alvo'])
        tipo_alerta = request.form['tipo_alerta'].upper()

        with MySQLConnectionManager(DB_CONFIG) as cursor_db:
            sql = """
            INSERT INTO alertas_preco (user_id, simbolo_ativo, preco_alvo, tipo_alerta, status, data_criacao)
            VALUES (%s, %s, %s, %s, 'ATIVO', %s)
            """
            cursor_db.execute(sql, (user_id, simbolo_ativo_yf, preco_alvo, tipo_alerta, datetime.datetime.now()))
        flash("Alerta de preço adicionado com sucesso!", "success")

        if user_id in portfolio_cache:
            del portfolio_cache[user_id]
        for key in list(news_cache.keys()):
            # Adicionado um mapeamento inverso para limpar cache de notícias
            reversed_symbol = REVERSE_SYMBOL_MAPPING.get(key, key)
            if simbolo_ativo_yf == key or simbolo_ativo_yf == reversed_symbol:
                del news_cache[key]
        for key in list(prediction_cache.keys()):
            if simbolo_ativo_yf in key:
                del prediction_cache[key]
        for key in list(market_data_cache.keys()):
            if simbolo_ativo_yf in key:
                del market_data_cache[key]

    except mysql.connector.Error as err:
        flash(f"Erro ao adicionar alerta: {err}", "danger")
    except decimal.InvalidOperation:
        flash("Erro: Preço alvo inválido.", "danger")
    except Exception as e:
        flash(f"Erro inesperado ao adicionar alerta: {e}", "danger")
    return redirect(url_for('index')) 

@app.route('/excluir_alerta', methods=['POST'])
@login_required
def excluir_alerta():
    user_id = session['user_id']
    try:
        alerta_id = request.form.get('id')
        if not alerta_id:
            flash("ID do alerta não fornecido para exclusão.", "danger")
            return redirect(url_for('index'))
            
        # Garante que o cursor retorna um dicionário
        with MySQLConnectionManager(DB_CONFIG, dictionary=True) as cursor_db:
            cursor_db.execute("SELECT simbolo_ativo FROM alertas_preco WHERE id = %s AND user_id = %s", (alerta_id, user_id))
            alerta_to_delete = cursor_db.fetchone()
            
            cursor_db.execute("DELETE FROM alertas_preco WHERE id = %s AND user_id = %s", (alerta_id, user_id))
            if cursor_db.rowcount == 0:
                flash("Alerta não encontrado ou não pertence ao seu utilizador.", "danger")
        flash("Alerta de preço excluído com sucesso!", "success")

        if user_id in portfolio_cache:
            del portfolio_cache[user_id]
        
        # Estas verificações agora funcionarão corretamente com um dicionário
        if alerta_to_delete and alerta_to_delete['simbolo_ativo'] in news_cache:
            del news_cache[alerta_to_delete['simbolo_ativo']]
        if alerta_to_delete:
            simbolo_alerta = alerta_to_delete['simbolo_ativo']
            for key in list(prediction_cache.keys()):
                if simbolo_alerta in key:
                    del prediction_cache[key]
            for key in list(market_data_cache.keys()):
                if simbolo_alerta in key:
                    del market_data_cache[key]

    except mysql.connector.Error as err:
        flash(f"Erro ao excluir alerta: {err}", "danger")
    except Exception as e:
        flash(f"Erro inesperado ao excluir alerta: {e}", "danger")
    return redirect(url_for('index')) 

# --- Rotas de Administração ---
@app.route('/admin_dashboard')
@login_required
@admin_required 
def admin_dashboard():
    print(f"DEBUG: Acedendo admin_dashboard como user_id: {session.get('user_id')}")
    users = get_all_users()
    print(f"DEBUG: admin_dashboard - Utilizadores para o template: {users}")
    return render_template('admin_dashboard.html', users=users)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    admin_user_id = session.get('user_id')
    user_to_delete_username = "Desconhecido" 

    if user_id == admin_user_id:
        flash("Você não pode excluir sua própria conta de administrador.", "danger")
        return redirect(url_for('admin_dashboard'))

    user_to_delete = get_user_by_id(user_id)
    if not user_to_delete:
        flash(f"Utilizador com ID {user_id} não encontrado.", "danger")
        return redirect(url_for('admin_dashboard'))
    user_to_delete_username = user_to_delete['username']

    if user_to_delete['is_admin']:
        admin_count = get_admin_count()
        if admin_count <= 1:
            flash("Não é possível excluir o último utilizador administrador do sistema.", "danger")
            return redirect(url_for('admin_dashboard'))
        elif admin_count == 2 and user_id != admin_user_id:
             flash("Não é possível excluir o outro utilizador administrador se você for o único restante.", "danger")
             return redirect(url_for('admin_dashboard'))
    
    if delete_user_from_db(user_id):
        flash(f"Utilizador {user_to_delete_username} (ID: {user_id}) excluído com sucesso.", "success")
        log_admin_action(admin_user_id, 'USER_DELETED', target_user_id=user_id, details={'username': user_to_delete_username})

        portfolio_cache.clear() 
        news_cache.clear()
        prediction_cache.clear()
        market_data_cache.clear()
    else:
        flash(f"Erro ao excluir utilizador {user_to_delete_username} (ID: {user_id}).", "danger")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/reset_password/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_reset_password(user_id):
    admin_user_id = session.get('user_id')
    new_password = request.form['new_password']
    user_target_username = "Desconhecido" 

    user_target = get_user_by_id(user_id)
    if user_target:
        user_target_username = user_target['username']
    
    if not new_password:
        flash("A nova palavra-passe não pode estar vazia.", "danger")
        return redirect(url_for('admin_dashboard'))

    if update_user_password(user_id, new_password):
        flash(f"Palavra-passe do utilizador {user_target_username} (ID: {user_id}) redefinida com sucesso.", "success")
        log_admin_action(admin_user_id, 'PASSWORD_RESET', target_user_id=user_id, details={'username': user_target_username})
    else:
        flash(f"Erro ao redefinir palavra-passe do utilizador {user_target_username} (ID: {user_id}).", "danger")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/toggle_admin/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_toggle_admin(user_id):
    admin_user_id = session.get('user_id')

    if user_id == admin_user_id:
        flash("Você não pode alterar seu próprio status de administrador através desta interface.", "danger")
        return redirect(url_for('admin_dashboard'))

    user_data = get_user_by_id(user_id)
    if not user_data:
        flash(f"Utilizador com ID {user_id} não encontrado.", "danger")
        return redirect(url_for('admin_dashboard'))

    current_is_admin_status = bool(user_data['is_admin'])
    new_status = not current_is_admin_status 
    user_target_username = user_data['username'] 

    if current_is_admin_status and not new_status: 
        admin_count = get_admin_count()
        if admin_count <= 1: 
            flash("Não é possível remover o status de administrador do último utilizador administrador do sistema.", "danger")
            return redirect(url_for('admin_dashboard'))
        elif admin_count == 2 and user_id != admin_user_id:
             pass

    if toggle_user_admin_status(user_id, new_status):
        status_message = "promovido a administrador" if new_status else "rebaixado a utilizador comum"
        flash(f"Utilizador {user_target_username} foi {status_message} com sucesso.", "success")
        log_admin_action(admin_user_id, 'ADMIN_STATUS_CHANGED', target_user_id=user_id, 
                         details={'username': user_target_username, 'old_status': current_is_admin_status, 'new_status': new_status})
    else:
        flash(f"Erro ao alterar status de administrador para o utilizador {user_target_username}.", "danger")
    
    return redirect(url_for('admin_dashboard'))

# --- Rota para a Página de Perfil do Utilizador ---
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_id = session.get('user_id')
    user = get_user_by_id(user_id)

    if not user:
        flash("Erro: Utilizador não encontrado.", "danger")
        return redirect(url_for('index'))

    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        contact_number = request.form.get('contact_number')

        if not full_name or not email:
            flash("Nome completo e email são obrigatórios.", "danger")
            return redirect(url_for('profile'))
        
        existing_user_with_email = get_user_by_email(email)
        if existing_user_with_email and existing_user_with_email['id'] != user_id:
            flash("Este email já está em uso por outra conta.", "danger")
            return redirect(url_for('profile'))

        if update_user_profile_data(user_id, full_name, email, contact_number):
            flash("Perfil atualizado com sucesso!", "success")
            user = get_user_by_id(user_id) 
            return render_template('profile.html', user=user)
        else:
            flash("Erro ao atualizar o perfil. Tente novamente.", "danger")
            
    return render_template('profile.html', user=user)


# --- ROTA para ver todas as transações ---
@app.route('/view_transactions')
@login_required
def view_transactions():
    """Renderiza a página principal com os filtros para exibir todas as transações."""
    user_id = session.get('user_id')
        
    data_inicio = request.args.get('data_inicio')
    data_fim = request.args.get('data_fim')
    ordenar_por = request.args.get('ordenar_por', 'data_transacao')
    ordem = request.args.get('ordem', 'DESC')
    simbolo_filtro = request.args.get('simbolo_filtro')
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', TRANSACTIONS_PER_PAGE, type=int)

    logged_in_username = session.get('username', 'Convidado') 
    is_admin = session.get('is_admin', False) 

    try:
        transacoes, total_transacoes = buscar_transacoes_filtradas(user_id, data_inicio, data_fim, ordenar_por, ordem, simbolo_filtro, page, per_page)
        
        # Estas informações podem ser opcionais na view_transactions, dependendo do design
        # Mas para simplificar, vamos passar dados vazios se não for a index.
        posicoes_carteira = {}
        total_valor_carteira = 0.0
        total_lucro_nao_realizado = 0.0
        total_prejuizo_nao_realizado = 0.0
        alertas = []
        all_news = []

        total_pages = (total_transacoes + per_page - 1) // per_page
        
    except Exception as e:
        flash(f"Erro ao carregar dados: {e}", "danger")
        transacoes = []
        total_transacoes = 0
        total_pages = 0


    return render_template('index.html',
                           transacoes=transacoes,
                           posicoes_carteira=posicoes_carteira, # Pode ser vazio aqui
                           alertas=alertas, # Pode ser vazio aqui
                           all_news=all_news, # Pode ser vazio aqui
                           REVERSE_SYMBOL_MAPPING=REVERSE_SYMBOL_MAPPING,
                           SYMBOL_MAPPING=SYMBOL_MAPPING,
                           data_inicio=data_inicio,
                           data_fim=data_fim,
                           ordenar_por=ordenar_por,
                           ordem=ordem,
                           simbolo_filtro=simbolo_filtro,
                           page=page,
                           per_page=per_page,
                           total_transacoes=total_transacoes,
                           total_pages=total_pages,
                           total_valor_carteira=total_valor_carteira, # Pode ser zero aqui
                           total_lucro_nao_realizado=total_lucro_nao_realizado, # Pode ser zero aqui
                           total_prejuizo_nao_realizado=total_prejuizo_nao_realizado, # Pode ser zero aqui
                           logged_in_username=logged_in_username,
                           is_admin=is_admin)


# --- ROTA: Exibir Logs de Auditoria ---
@app.route('/admin/audit_logs')
@login_required
@admin_required
def admin_audit_logs_view():
    logs = []
    print("DEBUG: admin_audit_logs_view - Iniciando busca de logs.")
    try:
        with MySQLConnectionManager(DB_CONFIG, dictionary=True, buffered=True) as cursor_db:
            print("DEBUG: admin_audit_logs_view - Cursor bufferizado criado.")
            sql = """
            SELECT 
                al.id, 
                al.action_type, 
                al.timestamp, 
                al.details,
                al.admin_username_at_action AS admin_username,    
                al.target_username_at_action AS target_username,  
                al.admin_user_id, 
                al.target_user_id 
            FROM admin_audit_logs al
            ORDER BY al.timestamp DESC
            LIMIT 50; 
            """
            cursor_db.execute(sql)
            print("DEBUG: admin_audit_logs_view - Consulta executada. Buscando resultados...")
            logs = cursor_db.fetchall() 
            print(f"DEBUG: admin_audit_logs_view - {len(logs)} logs buscados.")
            
            for log in logs:
                try:
                    if log['details']:
                        log['details'] = json.loads(log['details'])
                    else:
                        log['details'] = {}
                except json.JSONDecodeError as json_err:
                    print(f"WARNING: Log ID {log.get('id', 'N/A')} has malformed JSON details: {log.get('details', 'N/A')}. Error: {json_err}")
                    log['details'] = {"error": "Malformed JSON in details", "original_data": log.get('details')}
                except Exception as parse_error:
                    print(f"WARNING: Log ID {log.get('id', 'N/A')} has unexpected error parsing details: {parse_error}. Original data: {log.get('details', 'N/A')}")
                    log['details'] = {"error": f"Failed to parse details: {parse_error}", "original_data": log.get('details')}
        print("DEBUG: admin_audit_logs_view - Contexto de conexão encerrado.")

    except mysql.connector.Error as err:
        flash(f"Erro MySQL ao buscar logs de auditoria: {err}", "danger")
        print(f"ERRO: admin_audit_logs_view - Erro MySQL: {err}")
    except Exception as e:
        flash(f"Erro inesperado ao buscar logs de auditoria: {e}", "danger")
        print(f"ERRO: admin_audit_logs_view - Erro inesperado: {e}")

    return render_template('admin_audit_logs.html', logs=logs)


if __name__ == '__main__':
    # Esta é a forma padrão e mais robusta de rodar o aplicativo Flask.
    # O próprio app.run(debug=True) já lida com o reloader.
    app.run(debug=True, port=5000)