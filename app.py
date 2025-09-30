from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import os

app = Flask(__name__)
app.secret_key = 'supersegredo'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'clientes.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS clientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            endereco TEXT,
            bairro TEXT,
            cep TEXT,
            cidade TEXT,
            uf TEXT,
            cpf TEXT NOT NULL,
            rg TEXT,
            orgao_expedidor TEXT,
            telefone TEXT,
            processo TEXT,
            data_inicial TEXT,
            data_final TEXT,
            justica_vara TEXT,
            whatsapp TEXT
        )
        """
    )
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return redirect(url_for('pesquisar'))

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        id_cliente = request.form.get('id')
        dados = (
            request.form.get('nome'),
            request.form.get('endereco'),
            request.form.get('bairro'),
            request.form.get('cep'),
            request.form.get('cidade'),
            request.form.get('uf'),
            request.form.get('cpf'),
            request.form.get('rg'),
            request.form.get('orgao_expedidor'),
            request.form.get('telefone'),
            request.form.get('processo'),
            request.form.get('data_inicial'),
            request.form.get('data_final'),
            request.form.get('justica_vara'),
            request.form.get('whatsapp'),
        )

        conn = get_connection()
        cursor = conn.cursor()
        if id_cliente:
            cursor.execute(
                '''
                UPDATE clientes SET
                    nome=?, endereco=?, bairro=?, cep=?, cidade=?, uf=?, cpf=?, rg=?,
                    orgao_expedidor=?, telefone=?, processo=?, data_inicial=?, data_final=?,
                    justica_vara=?, whatsapp=?
                WHERE id=?
                ''',
                dados + (id_cliente,),
            )
        else:
            cursor.execute(
                '''
                INSERT INTO clientes (
                    nome, endereco, bairro, cep, cidade, uf, cpf, rg,
                    orgao_expedidor, telefone, processo, data_inicial, data_final,
                    justica_vara, whatsapp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                dados,
            )
        conn.commit()
        conn.close()
        return redirect(url_for('pesquisar'))

    # GET
    return render_template('form.html', cliente=None)

@app.route('/editar/<int:id>', methods=['GET'])
def editar(id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM clientes WHERE id=?', (id,))
    cliente = cursor.fetchone()
    conn.close()
    if not cliente:
        return redirect(url_for('pesquisar'))
    return render_template('form.html', cliente=cliente)

@app.route('/deletar/<int:id>', methods=['POST'])
def deletar(id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM clientes WHERE id=?', (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('pesquisar'))

@app.route('/pesquisar', methods=['GET'])
def pesquisar():
    nome = request.args.get('nome', '').strip()
    cpf = request.args.get('cpf', '').strip()

    conn = get_connection()
    cursor = conn.cursor()

    filtros = []
    params = []
    if nome:
        filtros.append('nome LIKE ?')
        params.append(f'%{nome}%')
    if cpf:
        filtros.append('cpf LIKE ?')
        params.append(f'%{cpf}%')

    where = f"WHERE {' AND '.join(filtros)}" if filtros else ''
    cursor.execute(f'SELECT * FROM clientes {where} ORDER BY id DESC', params)
    clientes = cursor.fetchall()
    conn.close()
    return render_template('pesquisar.html', clientes=clientes, nome_busca=nome, cpf_busca=cpf)

@app.route('/imprimir/<int:id>', methods=['GET'])
def imprimir(id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM clientes WHERE id=?', (id,))
    cliente = cursor.fetchone()
    conn.close()
    if not cliente:
        return redirect(url_for('pesquisar'))
    return render_template('imprimir.html', cliente=cliente)


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)