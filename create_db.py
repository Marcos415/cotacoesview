import os
import psycopg2
from urllib.parse import urlparse

def create_tables():
    """
    Conecta ao banco de dados PostgreSQL usando DATABASE_URL
    e cria as tabelas necessárias se elas não existirem.
    """
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("Erro: A variável de ambiente DATABASE_URL não está definida.")
        print("Este script deve ser executado no ambiente do Render.")
        return

    conn = None
    try:
        # Analisa a DATABASE_URL para extrair credenciais
        result = urlparse(database_url)
        username = result.username
        password = result.password
        database = result.path[1:]
        hostname = result.hostname
        port = result.port if result.port else 5432 # Default PostgreSQL port

        conn = psycopg2.connect(
            database=database,
            user=username,
            password=password,
            host=hostname,
            port=port
        )
        cur = conn.cursor()

        # Comandos SQL para criar as tabelas
        # Adicione IF NOT EXISTS para evitar erros se as tabelas já existirem
        # Ajustei o admin_audit_logs para aceitar NULL em admin_user_id e target_user_id
        # Isso é importante para o primeiro cadastro de admin, onde pode não haver um admin_user_id para referenciar
        # Você pode adicionar a restrição FOREIGN KEY depois, se desejar, quando já tiver admins.
        tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(255),
                email VARCHAR(255) UNIQUE,
                contact_number VARCHAR(20),
                is_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS transacoes (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                data_transacao DATE NOT NULL,
                hora_transacao TIME WITHOUT TIME ZONE,
                simbolo_ativo VARCHAR(20) NOT NULL,
                quantidade NUMERIC(15, 6) NOT NULL,
                preco_unitario NUMERIC(15, 6) NOT NULL,
                tipo_operacao VARCHAR(10) NOT NULL CHECK (tipo_operacao IN ('COMPRA', 'VENDA')),
                custos_taxas NUMERIC(10, 4) DEFAULT 0.00,
                observacoes TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS admin_audit_logs (
                id SERIAL PRIMARY KEY,
                admin_user_id INTEGER, -- Permite NULL temporariamente para o primeiro admin
                admin_username_at_action VARCHAR(255) NOT NULL,
                action_type VARCHAR(50) NOT NULL,
                target_user_id INTEGER,
                target_username_at_action VARCHAR(255),
                details JSONB,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
        ]

        for sql_command in tables_sql:
            print(f"Executando SQL: {sql_command.strip().splitlines()[0]}...")
            cur.execute(sql_command)
            print("Comando SQL executado com sucesso.")

        conn.commit()
        print("Tabelas verificadas/criadas com sucesso no PostgreSQL.")

    except psycopg2.Error as e:
        print(f"Erro ao conectar ou criar tabelas no PostgreSQL: {e}")
        if conn:
            conn.rollback() # Reverte em caso de erro
    except Exception as e:
        print(f"Erro inesperado no script create_db.py: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == '__main__':
    print("Iniciando script de criação de tabelas...")
    create_tables()
    print("Script de criação de tabelas finalizado.")