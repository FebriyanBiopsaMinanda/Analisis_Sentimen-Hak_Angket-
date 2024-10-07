from flask import Flask, redirect, render_template, request, url_for, flash
from flask_mysqldb import MySQL
import os
import numpy as np
import pandas as pd

# Import For Text Proccesing
import nltk
from nltk import ngrams
from collections import Counter
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import re
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

# Metode
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'analisis_sentimen'
mysql = MySQL(app)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# ========================= TEXT PROCCESING =========================


def cleandata():
    # Get Data From Database
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM data")
    data = cur.fetchall()

    # Delete From Database
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM data_bersih")

    # Get Text
    text = []
    for i in data:
        text.append(i[1])

    # Procces
    for i in text:
        # Case Folding
        teksInput = i
        lower_case = teksInput.lower()

        # Removing Special Characters
        result = re.sub(r"\d+", "", lower_case)
        result = result.translate(str.maketrans("", "", string.punctuation))

        # Tokenization
        tokens = nltk.tokenize.word_tokenize(result)

        freq_tokens = nltk.FreqDist(tokens)

        # Removing Stopwords
        list_stopwords = set(stopwords.words('indonesian'))
        tokens_without_stopword = []
        for word in freq_tokens:
            if word not in list_stopwords:
                tokens_without_stopword.append(word)

        # Stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        list_tokens = tokens_without_stopword
        result_stem = [(stemmer.stem(token)) for token in list_tokens]
        stem_text = ' '.join(result_stem)

        cur.execute("INSERT INTO `data_bersih`(`full_text`, `clean_text`) VALUES(%s,%s)", (str(
            teksInput), str(stem_text)))
        mysql.connection.commit()

# ========================= PRORES KLASIFIKASI =========================


def processSentiments():
    # Get Data From Database
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM data_bersih")
    data = cur.fetchall()

    # Delete From Database
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM sentiment")

    teks_bersih = []
    for i in data:
        txt = i[1]
        teks_bersih.append(txt)

    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for teks in teks_bersih:
        score = sia.polarity_scores(teks)
        if score['compound'] > 0:
            sentiments.append('Positif')
        else:
            sentiments.append('Negatif')

    for i, sentimen in enumerate(sentiments):
        txt = teks_bersih[i]
        cur.execute(
            "INSERT INTO sentiment (clean_text, sentimen) VALUES (%s,%s)", (txt, sentimen))
        mysql.connection.commit()


def processTfIdf():
    # Get Data From Database
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM sentiment")
    data = cur.fetchall()

    df = pd.DataFrame(data, columns=['clean_text', 'sentimen'])
    clean_text = df['clean_text']
    sentimen = df['sentimen']

    # Proses TF-IDF
    tfIdfVectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
    tfidf_matrix = tfIdfVectorizer.fit_transform(clean_text)
    feature_names = tfIdfVectorizer.get_feature_names_out()

    df = pd.DataFrame(tfidf_matrix[0].T.todense(
    ), index=tfIdfVectorizer.get_feature_names_out())
    value = df.to_numpy()

    index = 0
    for kata in feature_names:
        cur.execute("INSERT INTO `tfidf`(`kata`, `value`) VALUES(%s,%s)",
                    (str(kata), str(value[index][0])))
        mysql.connection.commit()
        index += 1

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    return tfidf_df, sentimen


def calculate_ngram_weights_manual(text_data, n):
    all_ngrams = []
    for text in text_data:
        tokens = nltk.word_tokenize(text)
        ngram_list = list(ngrams(tokens, n))
        all_ngrams.extend(ngram_list)

    ngram_counts = Counter(all_ngrams)

    return ngram_counts


def N_Gram():
    # Get Data From Database
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM sentiment")
    data = cur.fetchall()

    df = pd.DataFrame(data, columns=['clean_text', 'sentimen'])
    clean_text = df['clean_text']
    sentimen = df['sentimen']

    ngram_weights = calculate_ngram_weights_manual(clean_text, 2)
    for ngram, weight in ngram_weights.items():
        cur.execute(
            "INSERT INTO `ngram`(`kata`, `value`) VALUES(%s,%s)", (str(ngram), weight))
        mysql.connection.commit()

    ngram_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(2, 2))
    ngram_matrix = ngram_vectorizer.fit_transform(clean_text)
    feature_names = ngram_vectorizer.get_feature_names_out()
    ngram_df = pd.DataFrame(ngram_matrix.toarray(), columns=feature_names)
    return ngram_df, sentimen


# ========================= PRORES KLASIFIKASI =========================
def TrainTestSplit_TfIdf():
    tfidf_df, sentimen = processTfIdf()

    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM split")
    split = cur.fetchall()
    test_number = (split[0][1] / 100)

    X = tfidf_df.values
    y = sentimen

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_number, random_state=42)

    return X_train, X_test, y_train, y_test


def TrainTestSplit_NGram():
    ngram_df, sentimen = N_Gram()

    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM split")
    split = cur.fetchall()
    test_number = (split[0][1] / 100)

    X = ngram_df.values
    y = sentimen

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_number, random_state=42)

    return X_train, X_test, y_train, y_test


# ========================= DASHBOARD =========================
@app.route("/")
def login():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM split")
    split = cur.fetchall()
    split_test = split[0][1]
    split_train = split[1][1]
    cur.close()

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM data")
    data = cur.fetchall()
    total = len(data)

    train_nilai = round((split_train/100)*total)
    test_nilai = round((split_test/100)*total)
    cur.close()
    return render_template("index.html", split_test=split_test, split_train=split_train, train_nilai=train_nilai, test_nilai=test_nilai, total=total)


@app.route("/simpan_split", methods=["GET", "POST"])
def simpan_split():
    if request.method == 'POST':
        nilai_train = int(request.form['train'])
        nilai_test = 100 - nilai_train

        cur = mysql.connection.cursor()
        cur.execute("UPDATE split SET nilai=%s WHERE data='train'",
                    [nilai_train])
        mysql.connection.commit()
        cur.execute(
            "UPDATE split SET nilai=%s WHERE data='test'", [nilai_test])
        mysql.connection.commit()
        return redirect(url_for('dashboard'))

# ========================= DATA =========================


@app.route("/data")
def data():
    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM data")
    data = cur.fetchall()

    total = len(data)
    return render_template("data.html", data=data, total=total)


@app.route("/clearData")
def clearData():
    try:
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM data")
        cur.execute("DELETE FROM data_bersih")
        cur.execute("DELETE FROM sentiment")
        cur.execute("DELETE FROM tfidf")
        cur.execute("DELETE FROM ngram")
        mysql.connection.commit()
        flash("Database Berhasil Di Bersihkan", 'success')
        return redirect(url_for('data'))
    except:
        flash("Database Gagal Untuk Di Bersihkan", 'danger')
        return redirect(url_for('data'))


# ========================= Import CSV =========================
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/uploadFile", methods=["POST"])
def uploadFile():
    try:
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            parseCSV(file_path)

            flash(f"{uploaded_file.filename} Berhasil Di Upload", 'success')
            return redirect(url_for('data'))
    except:
        flash(f"File Tersebut Gagal Di Upload", 'danger')
        return redirect(url_for('data'))


def parseCSV(file_path):
    cur = mysql.connection.cursor()

    csvData = pd.read_csv(file_path)
    csvData = csvData['full_text']
    index = 1

    for i in range(len(csvData)):

        cur.execute(
            "INSERT INTO data (id, full_text) VALUES (%s, %s)", (index, csvData[i]))
        mysql.connection.commit()
        index += 1

# ========================= Tambah Text =========================


@app.route("/tambahData", methods=["POST"])
def tambahData():
    try:
        text = request.form['full_text']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM data")
        data = cur.fetchall()

        last = data[int(len(data) - 1)][0]

        index = last + 1
        cur.execute(
            "INSERT INTO data (id, full_text) VALUES (%s, %s)", (index, text))
        mysql.connection.commit()
        flash("Text Berhasil Di Tambahkan", 'success')
        return redirect(url_for('data'))
    except IndexError:
        text = request.form['full_text']
        index = 1
        cur.execute(
            "INSERT INTO data (id, full_text) VALUES (%s, %s)", (index, text))
        mysql.connection.commit()
        flash("Text Berhasil Di Tambahkan", 'success')
        return redirect(url_for('data'))
    except:
        flash("Text Gagal Untuk Di Tambahkan", 'danger')
        return redirect(url_for('data'))

# ========================= DATA TWEET =========================


@app.route("/data_tweet")
def data_tweet():
    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM data")
    data = cur.fetchall()

    total = len(data)
    return render_template("data_tweet.html", data=data, total=total)


@app.route("/hapusTweet/<index>")
def hapusTweet(index):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM data WHERE id = %s", [index])
    mysql.connection.commit()
    cur.close()
    flash("Tweet Berhasil Di Hapus", 'success')
    return redirect(url_for('data_tweet'))

# ========================= DATA BERSIH =========================


@app.route("/data_bersih")
def data_bersih():
    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM data_bersih")
    data = cur.fetchall()
    return render_template("data_bersih.html", data=data)


@app.route("/prosesBersih")
def prosesBersih():
    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM data")
    data = cur.fetchall()
    total = len(data)
    if total != 0:
        cleandata()
        flash("Proses Selesai", 'success')
        return redirect(url_for('data_bersih'))
    else:
        flash("Opss.. Database Kosong", 'warning')
        return redirect(url_for('data_bersih'))

# ========================= KLASIFIKASI =========================


@app.route("/klasifikasi")
def klasifikasi():
    try:
        cur = mysql.connection.cursor()

        cur.execute("SELECT * FROM sentiment")
        data = cur.fetchall()
        total = int(len(data))
        positif = 0
        negatif = 0
        for i in data:
            sentimen = i[1]

            if sentimen == 'Positif':
                positif += 1
            else:
                negatif += 1

        persen_positif = round((positif/total) * 100)
        persen_negatif = round((negatif/total) * 100)

        return render_template("klasifikasi.html", data=data, persen_positif=persen_positif, persen_negatif=persen_negatif, positif=positif, negatif=negatif, total=total)
    except:
        cur = mysql.connection.cursor()

        cur.execute("SELECT * FROM sentiment")
        data = cur.fetchall()

        total = 0
        positif = 0
        negatif = 0
        persen_positif = 0
        persen_negatif = 0
        return render_template("klasifikasi.html", data=data, persen_positif=persen_positif, persen_negatif=persen_negatif, positif=positif, negatif=negatif, total=total)


@app.route("/prosesKlasifikasi")
def prosesKlasifikasi():
    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM data_bersih")
    data = cur.fetchall()
    total = len(data)
    if total != 0:
        processSentiments()
        tfidf_df, sentimen = processTfIdf()
        sentimen = N_Gram()
        flash("Proses Selesai", 'success')
        return redirect(url_for('klasifikasi'))
    else:
        flash("Opss.. Database Kosong", 'warning')
        return redirect(url_for('klasifikasi'))

# ========================= PEMBOBOTAN =========================


@app.route("/bobot")
def bobot():
    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM tfidf")
    tfidf = cur.fetchall()

    cur.execute("SELECT * FROM ngram")
    n_gram = cur.fetchall()
    return render_template("bobot.html", tfidf=tfidf, n_gram=n_gram)


# ========================= HASIL =========================
@app.route("/hasil")
def hasil():
    return render_template("hasil.html")

# ========================= Naive Bayes =========================
@app.route("/nvbayes")
def nvbayes():
    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = TrainTestSplit_TfIdf()
    X_train_ngram, X_test_ngram, y_train_ngram, y_test_ngram = TrainTestSplit_NGram()

    # Naive Bayes TF IDF
    nb_model_tfidf = MultinomialNB()
    nb_model_tfidf.fit(X_train_tfidf, y_train_tfidf)
    y_pred_nb_tfidf = nb_model_tfidf.predict(X_test_tfidf)
    accuracy_nb_tfidf = accuracy_score(y_test_tfidf, y_pred_nb_tfidf)

    akurasi_tfidf = f"{(accuracy_nb_tfidf*100):.0f}"
    akurasi_tfidf = int(akurasi_tfidf)
    selisih_akurasi_tfidf = 100 - akurasi_tfidf

    accuracy_nb_tfidf = f"{(accuracy_nb_tfidf*100):.0f} %"

    label_mapping = {"Positif": 1, "Negatif": 0}

    # Testing
    data_numeric_test_tfidf = [label_mapping[label] for label in y_test_tfidf]
    jumlah_positif_test_tfidf = sum(
        [1 for value in data_numeric_test_tfidf if value == 1])
    jumlah_negatif_test_tfidf = sum(
        [1 for value in data_numeric_test_tfidf if value == 0])

    # Naive Bayes
    data_numeric_nb_tfidf = [label_mapping[label] for label in y_pred_nb_tfidf]
    jumlah_positif_nb_tfidf = sum(
        [1 for value in data_numeric_nb_tfidf if value == 1])
    jumlah_negatif_nb_tfidf = sum(
        [1 for value in data_numeric_nb_tfidf if value == 0])

    # Selisih
    selisih_positif_tfidf = abs(
        jumlah_positif_test_tfidf - jumlah_positif_nb_tfidf)
    selisih_negatif_tfidf = abs(
        jumlah_negatif_test_tfidf - jumlah_negatif_nb_tfidf)

    # Naive Bayes N Gram
    nb_model_ngram = MultinomialNB()
    nb_model_ngram.fit(X_train_ngram, y_train_ngram)
    y_pred_nb_ngram = nb_model_ngram.predict(X_test_ngram)
    accuracy_nb_ngram = accuracy_score(y_test_ngram, y_pred_nb_ngram)

    akurasi_ngram = f"{(accuracy_nb_ngram*100):.0f}"
    akurasi_ngram = int(akurasi_ngram)
    selisih_akurasi_ngram = 100 - akurasi_ngram

    accuracy_nb_ngram = f"{(accuracy_nb_ngram*100):.0f} %"

    label_mapping = {"Positif": 1, "Negatif": 0}

    # Testing
    data_numeric_test_ngram = [label_mapping[label] for label in y_test_ngram]
    jumlah_positif_test_ngram = sum(
        [1 for value in data_numeric_test_ngram if value == 1])
    jumlah_negatif_test_ngram = sum(
        [1 for value in data_numeric_test_ngram if value == 0])

    # Naive Bayes
    data_numeric_nb_ngram = [label_mapping[label] for label in y_pred_nb_ngram]
    jumlah_positif_nb_ngram = sum(
        [1 for value in data_numeric_nb_ngram if value == 1])
    jumlah_negatif_nb_ngram = sum(
        [1 for value in data_numeric_nb_ngram if value == 0])

    # Selisih
    selisih_positif_ngram = abs(
        jumlah_positif_test_ngram - jumlah_positif_nb_ngram)
    selisih_negatif_ngram = abs(
        jumlah_negatif_test_ngram - jumlah_negatif_nb_ngram)

    return render_template("nvbayes.html", accuracy_nb_tfidf=accuracy_nb_tfidf, accuracy_nb_ngram=accuracy_nb_ngram, 
                           jumlah_positif_test_tfidf=jumlah_positif_test_tfidf, jumlah_negatif_test_tfidf=jumlah_negatif_test_tfidf,
                           jumlah_positif_nb_tfidf=jumlah_positif_nb_tfidf, jumlah_negatif_nb_tfidf=jumlah_negatif_nb_tfidf,
                           selisih_positif_tfidf=selisih_positif_tfidf, selisih_negatif_tfidf=selisih_negatif_tfidf,
                           akurasi_tfidf=akurasi_tfidf, selisih_akurasi_tfidf=selisih_akurasi_tfidf, jumlah_positif_test_ngram=jumlah_positif_test_ngram, 
                           jumlah_negatif_test_ngram=jumlah_negatif_test_ngram, jumlah_positif_nb_ngram=jumlah_positif_nb_ngram, 
                           jumlah_negatif_nb_ngram=jumlah_negatif_nb_ngram, selisih_positif_ngram=selisih_positif_ngram, selisih_negatif_ngram=selisih_negatif_ngram,
                           selisih_akurasi_ngram=selisih_akurasi_ngram, akurasi_ngram=akurasi_ngram)

# ========================= Decision  =========================
@app.route("/dctree")
def dctree():
    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = TrainTestSplit_TfIdf()
    X_train_ngram, X_test_ngram, y_train_ngram, y_test_ngram = TrainTestSplit_NGram()

    # Decision Tree TF IDF
    dt_model_tfidf = DecisionTreeClassifier(random_state=42)
    dt_model_tfidf.fit(X_train_tfidf, y_train_tfidf)
    y_pred_dt_tfidf = dt_model_tfidf.predict(X_test_tfidf)
    accuracy_dt_tfidf = accuracy_score(y_test_tfidf, y_pred_dt_tfidf)

    akurasi_tfidf = f"{(accuracy_dt_tfidf*100):.0f}"
    akurasi_tfidf = int(akurasi_tfidf)
    selisih_akurasi_tfidf = 100 - akurasi_tfidf

    accuracy_dt_tfidf = f"{(accuracy_dt_tfidf*100):.0f} %"

    label_mapping = {"Positif": 1, "Negatif": 0}

    # Testing
    data_numeric_test_tfidf = [label_mapping[label] for label in y_test_tfidf]
    jumlah_positif_test_tfidf = sum(
        [1 for value in data_numeric_test_tfidf if value == 1])
    jumlah_negatif_test_tfidf = sum(
        [1 for value in data_numeric_test_tfidf if value == 0])

    data_numeric_dt_tfidf = [label_mapping[label] for label in y_pred_dt_tfidf]
    jumlah_positif_dt_tfidf = sum(
        [1 for value in data_numeric_dt_tfidf if value == 1])
    jumlah_negatif_dt_tfidf = sum(
        [1 for value in data_numeric_dt_tfidf if value == 0])

    selisih_positif_tfidf = abs(
        jumlah_positif_test_tfidf - jumlah_positif_dt_tfidf)
    selisih_negatif_tfidf = abs(
        jumlah_negatif_test_tfidf - jumlah_negatif_dt_tfidf)
    
    # Decision Tree N GRAM
    dt_model_ngram = DecisionTreeClassifier(random_state=42)
    dt_model_ngram.fit(X_train_ngram, y_train_ngram)
    y_pred_dt_ngram = dt_model_ngram.predict(X_test_ngram)
    accuracy_dt_ngram = accuracy_score(y_test_ngram, y_pred_dt_ngram)

    akurasi_ngram = f"{(accuracy_dt_ngram*100):.0f}"
    akurasi_ngram = int(akurasi_ngram)
    selisih_akurasi_ngram = 100 - akurasi_ngram

    accuracy_dt_ngram = f"{(accuracy_dt_ngram*100):.0f} %"

    label_mapping = {"Positif": 1, "Negatif": 0}

    # Testing
    data_numeric_test_ngram = [label_mapping[label] for label in y_test_ngram]
    jumlah_positif_test_ngram = sum(
        [1 for value in data_numeric_test_ngram if value == 1])
    jumlah_negatif_test_ngram = sum(
        [1 for value in data_numeric_test_ngram if value == 0])

    data_numeric_dt_ngram = [label_mapping[label] for label in y_pred_dt_ngram]
    jumlah_positif_dt_ngram = sum(
        [1 for value in data_numeric_dt_ngram if value == 1])
    jumlah_negatif_dt_ngram = sum(
        [1 for value in data_numeric_dt_ngram if value == 0])

    selisih_positif_ngram = abs(
        jumlah_positif_test_ngram - jumlah_positif_dt_ngram)
    selisih_negatif_ngram = abs(
        jumlah_negatif_test_ngram - jumlah_negatif_dt_ngram)

    return render_template("dctree.html", accuracy_dt_tfidf=accuracy_dt_tfidf, accuracy_dt_ngram=accuracy_dt_ngram,
                           jumlah_positif_test_tfidf=jumlah_positif_test_tfidf, jumlah_negatif_test_tfidf=jumlah_negatif_test_tfidf,
                           jumlah_positif_dt_tfidf=jumlah_positif_dt_tfidf, jumlah_negatif_dt_tfidf=jumlah_negatif_dt_tfidf,
                           selisih_positif_tfidf=selisih_positif_tfidf, selisih_negatif_tfidf=selisih_negatif_tfidf,
                           akurasi_tfidf=akurasi_tfidf, selisih_akurasi_tfidf=selisih_akurasi_tfidf,
                           jumlah_positif_test_ngram=jumlah_positif_test_ngram, jumlah_negatif_test_ngram=jumlah_negatif_test_ngram,
                           jumlah_positif_dt_ngram=jumlah_positif_dt_ngram, jumlah_negatif_dt_ngram=jumlah_negatif_dt_ngram,
                           selisih_positif_ngram=selisih_positif_ngram, selisih_negatif_ngram=selisih_negatif_ngram,
                           akurasi_ngram=akurasi_ngram, selisih_akurasi_ngram=selisih_akurasi_ngram)


if __name__ == "__main__":
    app.run(debug=True)
